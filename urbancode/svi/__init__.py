"""
Street View Image (SVI) analysis module.

This module provides functions for analyzing street view images, including:
- Basic image features (color, edges, etc.)
- Semantic segmentation
- Object detection
- Scene recognition
- Perception (comfort prediction)
"""

from .feature import (
    filename,
    color,
    segmentation,
    object_detection,
    scene_recognition
)

from .perception import comfort

__all__ = [
    'filename',
    'color',
    'segmentation',
    'object_detection',
    'scene_recognition',
    'comfort'
]
