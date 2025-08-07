"""
Sub-package that houses the three task classes.
"""

from .emotion    import EmotionClassifier
from .hashtags   import HashtagGenerator
from .popularity import PopularityPredictor

__all__ = [
    "EmotionClassifier",
    "HashtagGenerator",
    "PopularityPredictor",
]
