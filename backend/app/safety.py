import base64
import logging
import os
import tempfile

from nudenet import NudeDetector

logger = logging.getLogger("safety-monitor")

detector: NudeDetector | None = None

# Classes that constitute NSFW content
NSFW_CLASSES = {
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "BUTTOCKS_EXPOSED",
    "ANUS_EXPOSED",
}

NSFW_CONFIDENCE_THRESHOLD = 0.6
NSFW_HARD_BLOCK_THRESHOLD = 0.9


def init_detector():
    """Load the NudeNet detector model. Call once at startup."""
    global detector
    logger.info("Loading NudeNet detector ...")
    detector = NudeDetector()
    logger.info("NudeNet detector loaded.")


def check_frame(image_base64: str) -> dict | None:
    """Decode base64 image, run nudenet, return flagged detections or None.

    Returns None if the frame is safe. Returns a dict with flagged detections
    if NSFW content is detected above the confidence threshold.
    """
    if detector is None:
        logger.error("NudeNet detector not initialized")
        return None

    tmp_path = None
    try:
        image_bytes = base64.b64decode(image_base64)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(image_bytes)
            tmp_path = f.name

        detections = detector.detect(tmp_path)

        flagged = [
            d for d in detections
            if d["class"] in NSFW_CLASSES and d["score"] >= NSFW_CONFIDENCE_THRESHOLD
        ]

        if flagged:
            return {
                "flagged": True,
                "detections": [
                    {"class": d["class"], "score": round(d["score"], 3)}
                    for d in flagged
                ],
                "hard_block": any(d["score"] >= NSFW_HARD_BLOCK_THRESHOLD for d in flagged),
            }
        return None

    except Exception as e:
        logger.error("NSFW check failed: %s", e)
        return None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
