# UrbanCode (v0.2.1)

A Python package for street view image perception analysis, providing tools for feature extraction and comfort prediction.

## Related Research

[Thermal Comfort in Sight: Thermal Affordance and Its Visual Assessment](https://github.com/Sijie-Yang/Thermal-Affordance)

## Features

### Street View Image (SVI) Analysis
- Semantic segmentation
- Object detection
- Color feature extraction
- Scene recognition
- Perception analysis (thermal_comfort, visual_comfort, safety, etc.)

## Examples

### 1. Street View Image Feature Extraction
`examples/test_svi_image_feature.ipynb`
- Demonstrates how to extract various features from street view images
- Includes semantic segmentation, object detection, color analysis, and scene recognition
- Shows how to process multiple images and save results

### 2. Street View Image Comfort Prediction
`examples/test_svi_comfort_prediction.ipynb`
- Shows how to predict comfort scores from street view images
- Demonstrates the use of the comfort function for both single images and folders
- Includes visualization of perception metrics
- Automatically normalizes perception scores to 0-5 range

## Installation

```bash
pip install urbancode
```

## Usage

### Feature Extraction
```python
import urbancode as uc
import pandas as pd

# Process a folder of images
df = uc.svi.filename("path/to/folder")
df = uc.svi.segmentation(df, folder_path="path/to/folder")
df = uc.svi.object_detection(df, folder_path="path/to/folder")
df = uc.svi.color(df, folder_path="path/to/folder")
df = uc.svi.scene_recognition(df, folder_path="path/to/folder")

# Save results
df.to_csv("svi_results.csv", index=False)
```

### Comfort Prediction
```python
import urbancode as uc

# Process a single image
df = uc.svi.comfort("path/to/image.jpg", mode='image')

# Process a folder of images
df = uc.svi.comfort("path/to/folder", mode='folder')

# Save results
df.to_csv("comfort_results.csv", index=False)
```

### Perception Metrics
The comfort function returns a DataFrame with the following perception metrics (normalized to 0-5 range):
- thermal_comfort
- visual_comfort
- temp_intensity
- sun_intensity
- humidity_inference
- wind_inference
- traffic_flow
- greenery_rate
- shading_area
- material_comfort
- imageability
- enclosure
- human_scale
- transparency
- complexity
- safe
- lively
- beautiful
- wealthy
- boring
- depressing
