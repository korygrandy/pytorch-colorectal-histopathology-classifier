"""Model architectures."""

from .classifier import (
    ColorectalClassifier,
    create_model,
    load_checkpoint
)

__all__ = [
    'ColorectalClassifier',
    'create_model',
    'load_checkpoint'
]
