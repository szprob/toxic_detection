from toxic_detection.img_detection.detection import Detector as ImgToxicDetector
from toxic_detection.text_detection.detection import Detector as TextToxicDetector
from toxic_detection.version import __version__

__all__ = [
    "__version__",
    "ImgToxicDetector",
    "TextToxicDetector",
]
