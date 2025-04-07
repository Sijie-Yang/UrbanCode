__version__ = "0.1.0"
__author__ = "Sijie Yang"
__description__ = "A package for universal urban analysis"

# Import svi functions directly into the urbancode namespace
from urbancode.svi import (
    filename,
    color,
    read_image_with_pil,
    compute_colorfulness,
    compute_canny_edges,
    compute_hue_mean_std,
    compute_saturation_mean_std,
    compute_lightness_mean_std,
    compute_contrast,
    compute_sharpness,
    compute_entropy,
    compute_image_variance,
    segmentation,
    object_detection,
    scene_recognition
)