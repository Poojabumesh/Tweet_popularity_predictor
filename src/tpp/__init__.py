"""
tweet-popularity-predictor (tpp) top-level package
"""

__version__ = "0.1.0"

#     from tpp import MTMLModel
from .model import MTMLModel

__all__ = ["MTMLModel", "__version__"]
