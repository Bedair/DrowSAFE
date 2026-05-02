"""
DrowSAFE — Dashboard UI.

Pygame-based fullscreen dashboard for the Raspberry Pi Touch Display v1.1.
Shows live camera feed, fatigue score, alert level, and key metrics.

Layout (800×480)
----------------
  ┌──────────────────────────────────────────┐
  │  CAMERA FEED (left 60%)   │  METRICS     │
  │                           │  (right 40%) │
  │                           │  Score gauge │
  │                           │  EAR / MAR   │
  │                           │  Head pose   │
  │                           │  PERCLOS     │
  │  ALERT BANNER (bottom)                   │
  └──────────────────────────────────────────┘
"""

import sys
import logging
import numpy as np

log = logging.getLogger("drowsafe.dashboard")

try:
    from config.config import EAR_THRESHOLD, EAR_CONSEC_FRAMES, MAR_THRESHOLD, HEAD_PITCH_THRESHOLD
except ImportError:
    EAR_THRESHOLD        = 0.22
    EAR_CONSEC_FRAMES    = 3
    MAR_THRESHOLD        = 0.45
    HEAD_PITCH_THRESHOLD = 20

try:
    import pygame
    _PYGAME_AVAILABLE = True
except ImportError:
    _PYGAME_AVAILABLE = False
    log.warning("Pygame not available — dashboard disabled.")

# Alert level colours (RGB)
COLOURS = {
    0: (39,  174,  96),   # Green  — ALERT
    1: (243, 156,  18),   # Amber  — WARNING
    2: (231,  76,  60),   # Red    — CRITICAL
}

LABEL_COLOURS = {
    0: "ALERT",
    1: "WARNING ⚠",
    2: "CRITICAL ⛔",
}

BG_COLOUR     = (18,  18,  18)   # Dark background
TEXT_PRIMARY  = (236, 240, 241)
TEXT_SECONDARY= (149, 165, 166)


class Dashboard:
    """Pygame fullscreen dashboard."""

    CAM_W_RATIO = 0.60   # Camera feed takes 60% of display width

    def __init__(self, width: int = 800, height: int = 480, fullscreen: bool = True):
        self._width      = width
        self._height     = height
        self._running    = _PYGAME_AVAILABLE
        self._screen     = None
        self._clock      = None
        self._font_large = None
        self._font_med   = None
        self._font_small = None

        if not _PYGAME_AVAILABLE:
            return

        pygame.init()
        flags = pygame.FULLSCREEN | pygame.NOFRAME if fullscreen else 0
        self._screen = pygame.display.set_mode((width, height), flags)
        pygame.display.set_caption("DrowSAFE")
        self._clock = pygame.time.Clock()

        # Fonts — uses system DejaVu Sans (installed via apt)
        self._font_large = pygame.font.SysFont("dejavusans", 52, bold=True)
        self._font_med   = pygame.font.SysFont("dejavusans", 28)
        self._font_small = pygame.font.SysFont("dejavusans", 20)

        self._ear_low_frames  = 0   # consecutive frames EAR below threshold
        self._ear_high_frames = 0   # consecutive frames EAR above threshold (grace period)

        log.info("Dashboard initialised (%dx%d, fullscreen=%s)", width, height, fullscreen)

    def render(
        self,
        frame,
        score: float,
        alert_level: int,
        features,
        fps: float = None,
        simulated: bool = False,
    ):
        """
        Render one dashboard frame.

        Parameters
        ----------
        frame       : numpy.ndarray (BGR) | None
        score       : float  — fatigue score 0–100
        alert_level : int    — 0 / 1 / 2
        features    : Features | None
        fps         : float | None
        simulated   : bool   — show simulation badge if True
        """
        if not _PYGAME_AVAILABLE or self._screen is None:
            return

        # Handle quit events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._running = False
                return
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self._running = False
                return

        colour = COLOURS[alert_level]
        self._screen.fill(BG_COLOUR)

        cam_w = int(self._width * self.CAM_W_RATIO)
        cam_h = self._height - 60   # Leave space for alert banner

        # --- Camera feed ---
        if frame is not None:
            self._draw_camera(frame, cam_w, cam_h)

        # --- Metrics panel ---
        metrics_x = cam_w + 10
        self._draw_metrics(metrics_x, score, alert_level, features, colour)

        # --- Alert banner (bottom) ---
        self._draw_alert_banner(alert_level, score, colour)

        # --- FPS ---
        if fps is not None:
            fps_surf = self._font_small.render(f"{fps:.1f} fps", True, TEXT_SECONDARY)
            self._screen.blit(fps_surf, (8, 8))

        # --- Simulation badge ---
        if simulated:
            sim_surf = self._font_small.render("⚙ SIMULATION", True, (80, 80, 220))
            self._screen.blit(sim_surf, (self._width - sim_surf.get_width() - 8, 8))

        pygame.display.flip()
        self._clock.tick(60)

    def _draw_camera(self, frame, cam_w: int, cam_h: int):
        """Scale and blit the BGR camera frame to the left panel."""
        import cv2
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Scale to fit panel maintaining aspect ratio
        h, w  = rgb.shape[:2]
        scale = min(cam_w / w, cam_h / h)
        nw, nh = int(w * scale), int(h * scale)
        rgb   = cv2.resize(rgb, (nw, nh))

        surface = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
        # Centre in panel
        ox = (cam_w - nw) // 2
        oy = (cam_h - nh) // 2
        self._screen.blit(surface, (ox, oy))

    def _draw_metrics(self, x: int, score: float, alert_level: int, features, colour):
        """Draw the right-side metrics panel."""
        panel_w = self._width - x - 10
        y = 20

        # Score heading
        s = self._font_small.render("FATIGUE SCORE", True, TEXT_SECONDARY)
        self._screen.blit(s, (x, y)); y += 24

        # Score value — large, coloured
        s = self._font_large.render(f"{int(score)}", True, colour)
        self._screen.blit(s, (x, y)); y += 64

        # Score bar
        bar_h = 14
        pygame.draw.rect(self._screen, (50, 50, 50), (x, y, panel_w, bar_h), border_radius=7)
        fill_w = int(panel_w * score / 100)
        if fill_w > 0:
            pygame.draw.rect(self._screen, colour, (x, y, fill_w, bar_h), border_radius=7)
        y += bar_h + 20

        # Feature rows
        def metric_row(label, value_str, warn=False):
            nonlocal y
            lc = (231, 76, 60) if warn else TEXT_SECONDARY
            vc = (231, 76, 60) if warn else TEXT_PRIMARY
            self._screen.blit(self._font_small.render(label, True, lc), (x, y))
            self._screen.blit(self._font_small.render(value_str, True, vc), (x + 110, y))
            y += 26

        if features:
            # EAR: only warn after sustained closure — ignore normal blinks.
            # Uses a grace period on recovery so a brief EAR spike mid-blink
            # does not reset the counter (MediaPipe noise during blink).
            # Counter only resets after EAR_CONSEC_FRAMES consecutive HIGH frames.
            if features.ear < EAR_THRESHOLD:
                self._ear_low_frames  += 1
                self._ear_high_frames  = 0
            else:
                self._ear_high_frames += 1
                if self._ear_high_frames >= EAR_CONSEC_FRAMES:
                    self._ear_low_frames = 0
            ear_sustained = self._ear_low_frames >= EAR_CONSEC_FRAMES

            metric_row("EAR",       f"{features.ear:.3f}",
                       warn=ear_sustained)
            metric_row("MAR",       f"{features.mar:.3f}",
                       warn=features.mar > MAR_THRESHOLD)
            metric_row("Head pitch",f"{features.head_pitch:+.1f}°",
                       warn=abs(features.head_pitch) > HEAD_PITCH_THRESHOLD)
        else:
            metric_row("EAR",       "—")
            metric_row("MAR",       "—")
            metric_row("Head pitch","—")

        y += 6
        # Alert level badge
        badge_col = colour
        badge_rect = pygame.Rect(x, y, panel_w, 34)
        pygame.draw.rect(self._screen, badge_col, badge_rect, border_radius=8)
        label = LABEL_COLOURS[alert_level]
        ls    = self._font_small.render(label, True, (255, 255, 255))
        lx    = badge_rect.centerx - ls.get_width() // 2
        ly    = badge_rect.centery - ls.get_height() // 2
        self._screen.blit(ls, (lx, ly))

    def _draw_alert_banner(self, alert_level: int, score: float, colour):
        """Draw the bottom alert banner."""
        bh      = 54
        by      = self._height - bh
        banner  = pygame.Rect(0, by, self._width, bh)
        pygame.draw.rect(self._screen, colour, banner)

        if alert_level == 0:
            text = "Driver monitoring active — stay alert"
        elif alert_level == 1:
            text = "⚠  Drowsiness detected — please take a break"
        else:
            text = "⛔  WAKE UP — Pull over immediately!"

        s  = self._font_med.render(text, True, (255, 255, 255))
        sx = self._width  // 2 - s.get_width()  // 2
        sy = by + bh // 2 - s.get_height() // 2
        self._screen.blit(s, (sx, sy))

    def is_running(self) -> bool:
        return self._running

    def quit(self):
        if _PYGAME_AVAILABLE:
            pygame.quit()
        log.info("Dashboard closed.")
