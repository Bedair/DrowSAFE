"""
DrowSAFE — Feature extraction module.

Computes drowsiness-relevant features from MediaPipe face landmarks:
  - EAR  (Eye Aspect Ratio)    — eye openness
  - MAR  (Mouth Aspect Ratio)  — yawn detection
  - Head pitch angle           — forward nod detection

References
----------
Soukupová & Čech (2016) "Real-Time Eye Blink Detection using Facial Landmarks"
"""

import numpy as np
import cv2
import logging
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger("drowsafe.features")


# ---------------------------------------------------------------------------
# MediaPipe FaceMesh landmark indices
# ---------------------------------------------------------------------------
# Eyes — using the 6-point EAR model (vertical + horizontal spans)
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]

# Mouth — direct vertical opening ratio
# Upper inner lip: 13, Lower inner lip: 14
# Left mouth corner: 78, Right mouth corner: 308
# MAR = vertical_opening / mouth_width
# Resting: ~0.0–0.15  |  Yawning: ~0.5–0.8
MOUTH_TOP    = 13    # Upper inner lip centre
MOUTH_BOTTOM = 14    # Lower inner lip centre
MOUTH_LEFT   = 78    # Left mouth corner
MOUTH_RIGHT  = 308   # Right mouth corner

# 6 canonical 3D-to-2D correspondence points for head pose (PnP solve)
# Indices: nose tip, chin, left eye corner, right eye corner, left mouth, right mouth
HEAD_POSE_LANDMARKS = [1, 152, 263, 33, 287, 57]

# 3D model points of the above landmarks in mm (generic face model)
HEAD_POSE_3D = np.array([
    [   0.0,    0.0,   0.0],   # Nose tip
    [   0.0, -330.0, -65.0],   # Chin
    [-225.0,  170.0, -135.0],  # Left eye corner
    [ 225.0,  170.0, -135.0],  # Right eye corner
    [-150.0, -150.0, -125.0],  # Left mouth corner
    [ 150.0, -150.0, -125.0],  # Right mouth corner
], dtype=np.float64)


@dataclass
class Features:
    """Container for all drowsiness features extracted from one frame."""
    ear:          float          # Mean Eye Aspect Ratio (both eyes)
    ear_left:     float          # Left eye EAR
    ear_right:    float          # Right eye EAR
    mar:          float          # Mouth Aspect Ratio
    head_pitch:   float          # Head pitch in degrees (positive = down)
    head_yaw:     float          # Head yaw in degrees
    head_roll:    float          # Head roll in degrees
    face_visible: bool = True


def _euclidean(p1, p2) -> float:
    """Euclidean distance between two 2D points."""
    return np.linalg.norm(np.array(p1) - np.array(p2))


def _mouth_opening_ratio(landmarks, frame_w: int, frame_h: int) -> float:
    """
    Compute mouth opening as a ratio of vertical gap to mouth width.

    Uses 4 reliable MediaPipe inner-lip landmarks:
      - MOUTH_TOP (13)    upper inner lip centre
      - MOUTH_BOTTOM (14) lower inner lip centre
      - MOUTH_LEFT (78)   left mouth corner
      - MOUTH_RIGHT (308) right mouth corner

    MAR = vertical_gap / mouth_width
    Typical values:
      Resting (closed): 0.00 – 0.15
      Slight open:      0.15 – 0.35
      Yawning:          0.45 – 0.80
    """
    top    = (landmarks[MOUTH_TOP].x    * frame_w, landmarks[MOUTH_TOP].y    * frame_h)
    bottom = (landmarks[MOUTH_BOTTOM].x * frame_w, landmarks[MOUTH_BOTTOM].y * frame_h)
    left   = (landmarks[MOUTH_LEFT].x   * frame_w, landmarks[MOUTH_LEFT].y   * frame_h)
    right  = (landmarks[MOUTH_RIGHT].x  * frame_w, landmarks[MOUTH_RIGHT].y  * frame_h)

    vertical = _euclidean(top, bottom)
    width    = _euclidean(left, right)

    if width < 1e-6:
        return 0.0
    return vertical / width


def _aspect_ratio(landmarks, indices, frame_w: int, frame_h: int) -> float:
    """
    Compute the aspect ratio for a 6-point eye or mouth model.

    The 6 points are arranged as:
      p0 (left), p1 (top-left), p2 (top-right),
      p3 (right), p4 (bottom-right), p5 (bottom-left)

    AR = (|p1–p5| + |p2–p4|) / (2 × |p0–p3|)
    """
    pts = [
        (landmarks[i].x * frame_w, landmarks[i].y * frame_h)
        for i in indices
    ]
    vertical   = _euclidean(pts[1], pts[5]) + _euclidean(pts[2], pts[4])
    horizontal = _euclidean(pts[0], pts[3])
    if horizontal < 1e-6:
        return 0.0
    return vertical / (2.0 * horizontal)


class FeatureExtractor:
    """
    Extracts EAR, MAR, and head pose from MediaPipe landmark list.

    Assumes a fixed camera focal length approximated from frame size.
    Call `update_frame_size()` once when frame dimensions are known.
    """

    def __init__(self, frame_w: int = 1280, frame_h: int = 720):
        self.frame_w = frame_w
        self.frame_h = frame_h
        self._build_camera_matrix()
        log.info("FeatureExtractor ready (%dx%d)", frame_w, frame_h)

    def update_frame_size(self, frame_w: int, frame_h: int):
        self.frame_w = frame_w
        self.frame_h = frame_h
        self._build_camera_matrix()

    def _build_camera_matrix(self):
        """Approximate pinhole camera matrix from frame dimensions."""
        focal = self.frame_w  # focal length ≈ image width in pixels
        cx    = self.frame_w / 2.0
        cy    = self.frame_h / 2.0
        self._camera_matrix = np.array([
            [focal,   0, cx],
            [    0, focal, cy],
            [    0,   0,  1],
        ], dtype=np.float64)
        self._dist_coeffs = np.zeros((4, 1))  # Assume no lens distortion

    def extract(self, landmarks) -> Features:
        """
        Extract all features from a MediaPipe landmark list.

        Parameters
        ----------
        landmarks : list of NormalizedLandmark
            468 landmarks from MediaPipe FaceMesh.

        Returns
        -------
        Features
        """
        w, h = self.frame_w, self.frame_h

        # --- EAR ---
        ear_l = _aspect_ratio(landmarks, LEFT_EYE,  w, h)
        ear_r = _aspect_ratio(landmarks, RIGHT_EYE, w, h)
        ear   = (ear_l + ear_r) / 2.0

        # --- MAR ---
        # Direct vertical opening / mouth width
        # Rises from ~0.05 at rest to ~0.6+ during a yawn
        mar = _mouth_opening_ratio(landmarks, w, h)

        # --- Head pose ---
        pitch, yaw, roll = self._head_pose(landmarks)

        return Features(
            ear          = ear,
            ear_left     = ear_l,
            ear_right    = ear_r,
            mar          = mar,
            head_pitch   = pitch,
            head_yaw     = yaw,
            head_roll    = roll,
            face_visible = True,
        )

    def _head_pose(self, landmarks):
        """
        Estimate head pitch, yaw, roll via solvePnP.

        Returns pitch, yaw, roll in degrees.
        Positive pitch = head tilted downward (nodding).
        """
        image_points = np.array([
            (landmarks[i].x * self.frame_w,
             landmarks[i].y * self.frame_h)
            for i in HEAD_POSE_LANDMARKS
        ], dtype=np.float64)

        success, rvec, tvec = cv2.solvePnP(
            HEAD_POSE_3D,
            image_points,
            self._camera_matrix,
            self._dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            return 0.0, 0.0, 0.0

        rmat, _ = cv2.Rodrigues(rvec)
        # Decompose rotation matrix into Euler angles
        sy = np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
        singular = sy < 1e-6

        if not singular:
            pitch = np.degrees(np.arctan2( rmat[2, 1], rmat[2, 2]))
            yaw   = np.degrees(np.arctan2(-rmat[2, 0], sy))
            roll  = np.degrees(np.arctan2( rmat[1, 0], rmat[0, 0]))
        else:
            pitch = np.degrees(np.arctan2(-rmat[1, 2], rmat[1, 1]))
            yaw   = np.degrees(np.arctan2(-rmat[2, 0], sy))
            roll  = 0.0

        return pitch, yaw, roll
