"""
DrowSAFE — Face detector module.

Runs MediaPipe Face Mesh on each frame and returns the 468 facial
landmarks normalised to [0, 1] in (x, y, z) coordinates.
"""

import cv2
import mediapipe as mp
import logging

log = logging.getLogger("drowsafe.detector")


class FaceDetector:
    """
    Wraps MediaPipe FaceMesh for single-face landmark detection.

    MediaPipe returns 468 landmarks per face, each with:
      - x, y  : normalised [0, 1] position relative to frame dimensions
      - z     : relative depth (smaller = closer to camera)
    """

    def __init__(
        self,
        max_num_faces: int = 1,
        refine_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self._mp_face_mesh = mp.solutions.face_mesh
        self._mp_drawing   = mp.solutions.drawing_utils
        self._mp_styles    = mp.solutions.drawing_styles

        self._face_mesh = self._mp_face_mesh.FaceMesh(
            max_num_faces            = max_num_faces,
            refine_landmarks         = refine_landmarks,
            min_detection_confidence = min_detection_confidence,
            min_tracking_confidence  = min_tracking_confidence,
        )

        log.info(
            "FaceDetector ready (max_faces=%d, refine=%s)",
            max_num_faces, refine_landmarks,
        )

    def process(self, frame):
        """
        Run face mesh detection on a BGR frame.

        Parameters
        ----------
        frame : numpy.ndarray
            BGR image from the camera.

        Returns
        -------
        landmarks : list of mediapipe.framework.formats.landmark_pb2.NormalizedLandmark
            468 landmarks for the first detected face, or None if no face found.
        annotated_frame : numpy.ndarray
            Copy of the input frame with landmarks drawn (for debug display).
        """
        # Convert BGR → RGB for MediaPipe processing
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._face_mesh.process(rgb)
        rgb.flags.writeable = True

        if not results.multi_face_landmarks:
            # Return original BGR frame unchanged
            return None, frame.copy()

        face_landmarks = results.multi_face_landmarks[0]

        # Draw landmarks on the RGB frame (MediaPipe drawing utils expect RGB)
        annotated_rgb = rgb.copy()
        self._mp_drawing.draw_landmarks(
            image                        = annotated_rgb,
            landmark_list                = face_landmarks,
            connections                  = self._mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec        = None,
            connection_drawing_spec      = self._mp_styles
                .get_default_face_mesh_tesselation_style(),
        )
        self._mp_drawing.draw_landmarks(
            image                        = annotated_rgb,
            landmark_list                = face_landmarks,
            connections                  = self._mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec        = None,
            connection_drawing_spec      = self._mp_styles
                .get_default_face_mesh_contours_style(),
        )

        # Convert annotated frame back to BGR for the rest of the pipeline
        annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)

        return face_landmarks.landmark, annotated_bgr

    def close(self):
        self._face_mesh.close()
        log.info("FaceDetector closed.")
