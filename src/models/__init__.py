"""Model architectures."""

from .classifier import (
    ColorectalClassifier,
    create_model,
    load_checkpoint
)

from .simple_resnet import (
    SimpleResNetClassifier,
    create_simple_resnet,
)

__all__ = [
    'ColorectalClassifier',
    'create_model',
    'load_checkpoint',
    'SimpleResNetClassifier',
    'create_simple_resnet',
]
