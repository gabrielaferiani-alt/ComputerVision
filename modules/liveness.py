import cv2
import time


class LivenessDetector:
    """
    Detects liveness via two mechanisms:
    1. Eye blink detection (requires 2 blinks within timeout)
    2. Texture analysis via Laplacian variance (anti-photo spoofing)
    """

    def __init__(self, timeout: float = 12.0, blinks_required: int = 2):
        self.timeout = timeout
        self.blinks_required = blinks_required
        self.reset()

    def reset(self):
        self.blink_count = 0
        self._eye_state = "open"   # open | closing | closed
        self._closed_frames = 0
        self._open_frames = 0
        self._start_time = time.time()

    def process_eyes(self, eyes) -> bool:
        """
        Feed detected eyes each frame.
        Returns True the moment a new blink completes.

        Blink cycle: open (≥3 frames) → closing → closed (≥2 frames) → open again.
        The debounce prevents Haar cascade flicker from creating phantom blinks.
        """
        num_eyes = len(eyes)
        blink_detected = False

        if num_eyes >= 2:
            # Eyes clearly open
            if self._eye_state == "closed" and self._closed_frames >= 2:
                self.blink_count += 1
                blink_detected = True

            self._eye_state = "open"
            self._closed_frames = 0
            self._open_frames = min(self._open_frames + 1, 10)

        elif num_eyes == 0:
            # Eyes not detected
            if self._eye_state == "open" and self._open_frames >= 3:
                self._eye_state = "closing"
                self._closed_frames = 1
            elif self._eye_state == "closing":
                self._closed_frames += 1
                if self._closed_frames >= 2:
                    self._eye_state = "closed"
            elif self._eye_state == "closed":
                self._closed_frames += 1

            self._open_frames = 0

        # 1 eye detected: Haar flickering — hold current state

        return blink_detected

    def check_texture(self, face_roi_gray) -> bool:
        """
        Laplacian variance: real skin has more high-frequency texture
        than a flat printed or screen-displayed photo.
        Threshold of 30 is deliberately lenient for varying cameras.
        """
        face_small = cv2.resize(face_roi_gray, (100, 100))
        lap = cv2.Laplacian(face_small, cv2.CV_64F)
        variance = lap.var()
        return float(variance) > 30.0

    def is_timed_out(self) -> bool:
        return time.time() - self._start_time > self.timeout

    def time_remaining(self) -> float:
        return max(0.0, self.timeout - (time.time() - self._start_time))

    @property
    def is_live(self) -> bool:
        return self.blink_count >= self.blinks_required

    @property
    def eye_state(self) -> str:
        return self._eye_state
