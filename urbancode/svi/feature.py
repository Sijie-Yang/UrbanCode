"""
Street View Image (SVI) feature extraction module.

This module provides functions for extracting various features from street view images,
including:
- Basic image features (color, edges, contrast, etc.)
- Semantic segmentation using SegFormer
- Object detection using Faster R-CNN
- Scene recognition using ResNet

The module is designed to work with pandas DataFrames containing image filenames
and returns DataFrames with added feature columns.
"""

import os
import pandas as pd
from PIL import Image
import numpy as np
import cv2
from scipy.stats import entropy as shannon_entropy
from tqdm import tqdm
import torch
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet50
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import torchvision.models as models
import torchvision.transforms as transforms
import urllib.request
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

def filename(folder_path, output_filename=None, filename_column='Filename'):
    """
    Scan folder for image files and create a DataFrame with filenames.
    
    Args:
        folder_path (str): Path to the folder containing images
        output_filename (str, optional): CSV filename to save results
        filename_column (str, optional): Name for the filename column. Defaults to 'Filename'
        
    Returns:
        pd.DataFrame: DataFrame with image filenames
    """
    # Get all image files
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    filenames = [f for f in os.listdir(folder_path) 
                if f.lower().endswith(image_extensions)]
    
    # Create DataFrame
    df = pd.DataFrame(filenames, columns=[filename_column])
    
    # Save to CSV if filename provided
    if output_filename:
        df.to_csv(output_filename, index=False)
    
    return df

def color(df, folder_path=None, filename_column='Filename'):
    """
    Extract color features from images and add to DataFrame.
    
    This function extracts various color-based features from images including:
    - Colorfulness
    - Hue statistics
    - Saturation statistics
    - Lightness statistics
    - Contrast
    - Sharpness
    - Entropy
    - Image variance
    
    Args:
        df (pd.DataFrame): DataFrame containing image filenames
        folder_path (str): Path to the folder containing images
        filename_column (str, optional): Name of the column containing filenames. Defaults to 'Filename'
        
    Returns:
        pd.DataFrame: Original DataFrame with added color feature columns
    """
    if folder_path is None:
        raise ValueError("folder_path must be provided")

    # Initialize feature columns
    feature_data = []
    
    # Process each image with tqdm progress bar
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
        image_path = os.path.join(folder_path, row[filename_column])
        try:
            features = extract_features(image_path)
            feature_data.append(features)
        except Exception as e:
            print(f"\nError processing {image_path}: {str(e)}")
            feature_data.append({k: np.nan for k in extract_features.keys()})
    
    # Add features to DataFrame
    features_df = pd.DataFrame(feature_data)
    return pd.concat([df, features_df], axis=1)

def read_image_with_pil(image_path):
    """
    Read an image file using PIL and convert it to OpenCV format.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        numpy.ndarray: Image in BGR format (OpenCV format)
        
    Raises:
        ValueError: If image cannot be read
    """
    try:
        pil_image = Image.open(image_path)
        pil_image = pil_image.convert('RGB')
        image = np.array(pil_image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert to BGR format for OpenCV
        return image
    except Exception as e:
        raise ValueError(f"Failed to read the image with PIL: {str(e)}")

def compute_colorfulness(image):
    """
    Compute the colorfulness metric of an image.
    
    This metric is based on the paper "Measuring colorfulness in natural images"
    by Hasler and Süsstrunk (2003).
    
    Args:
        image (numpy.ndarray): Image in BGR format
        
    Returns:
        float: Colorfulness score
    """
    (B, G, R) = cv2.split(image.astype("float"))
    rg = np.absolute(R - G)
    yb = np.absolute(0.5 * (R + G) - B)
    (rbMean, rbStd) = (np.mean(rg), np.std(rg))
    (ybMean, ybStd) = (np.mean(yb), np.std(yb))
    stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
    meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))
    return stdRoot + (0.3 * meanRoot)

def compute_canny_edges(image):
    """
    Compute the ratio of edge pixels in an image using Canny edge detection.
    
    Args:
        image (numpy.ndarray): Image in BGR format
        
    Returns:
        float: Ratio of edge pixels to total pixels
    """
    edges = cv2.Canny(image, 100, 200)
    edge_ratio = np.sum(edges) / edges.size
    return edge_ratio

def compute_hue_mean_std(image):
    """
    Compute mean and standard deviation of hue channel in HSV color space.
    
    Args:
        image (numpy.ndarray): Image in BGR format
        
    Returns:
        tuple: (mean hue, standard deviation of hue)
    """
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue = hsv_image[:, :, 0]
    hue_mean = np.mean(hue)
    hue_std = np.std(hue)
    return hue_mean, hue_std

def compute_saturation_mean_std(image):
    """
    Compute mean and standard deviation of saturation channel in HSV color space.
    
    Args:
        image (numpy.ndarray): Image in BGR format
        
    Returns:
        tuple: (mean saturation, standard deviation of saturation)
    """
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv_image[:, :, 1]
    saturation_mean = np.mean(saturation)
    saturation_std = np.std(saturation)
    return saturation_mean, saturation_std

def compute_lightness_mean_std(image):
    """
    Compute mean and standard deviation of lightness channel in LAB color space.
    
    Args:
        image (numpy.ndarray): Image in BGR format
        
    Returns:
        tuple: (mean lightness, standard deviation of lightness)
    """
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    lightness = lab_image[:, :, 0]
    lightness_mean = np.mean(lightness)
    lightness_std = np.std(lightness)
    return lightness_mean, lightness_std

def compute_contrast(image):
    """
    Compute the contrast of an image using standard deviation of grayscale values.
    
    Args:
        image (numpy.ndarray): Image in BGR format
        
    Returns:
        float: Contrast value
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contrast = gray_image.std()
    return contrast

def compute_sharpness(image):
    """
    Compute the sharpness of an image using Laplacian variance.
    
    Args:
        image (numpy.ndarray): Image in BGR format
        
    Returns:
        float: Sharpness value
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray_image, cv2.CV_64F).var()
    return laplacian_var

def compute_entropy(image):
    """
    Compute the Shannon entropy of an image.
    
    Args:
        image (numpy.ndarray): Image in BGR format
        
    Returns:
        float: Entropy value
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    entropy = shannon_entropy(gray_image)
    
    # 确保返回的是单一值而不是列表
    if isinstance(entropy, (list, np.ndarray)):
        # 如果是列表或数组，取平均值
        entropy = np.mean(entropy)
    
    return float(entropy)  # 确保返回的是浮点数

def compute_image_variance(image):
    """
    Compute the variance of grayscale pixel values.
    
    Args:
        image (numpy.ndarray): Image in BGR format
        
    Returns:
        float: Variance value
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = np.var(gray_image)
    return variance

def extract_features(image_path):
    """
    Extract all basic image features from a single image.
    
    This is a helper function that combines all basic feature extraction functions
    into a single call.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        dict: Dictionary containing all extracted features
    """
    image = read_image_with_pil(image_path)
    colorfulness = compute_colorfulness(image)
    canny_edges = compute_canny_edges(image)
    hue_mean, hue_std = compute_hue_mean_std(image)
    saturation_mean, saturation_std = compute_saturation_mean_std(image)
    lightness_mean, lightness_std = compute_lightness_mean_std(image)
    contrast = compute_contrast(image)
    sharpness = compute_sharpness(image)
    entropy = compute_entropy(image)
    image_variance = compute_image_variance(image)
    return {
        'Colorfulness': colorfulness,
        'Canny_Edges': canny_edges,
        'Hue_Mean': hue_mean,
        'Hue_Std': hue_std,
        'Saturation_Mean': saturation_mean,
        'Saturation_Std': saturation_std,
        'Lightness_Mean': lightness_mean,
        'Lightness_Std': lightness_std,
        'Contrast': contrast,
        'Sharpness': sharpness,
        'Entropy': entropy,
        'Image_Variance': image_variance
    }

def segmentation(df, folder_path, filename_column='Filename'):
    """
    Perform semantic segmentation on images using SegFormer model.
    
    This function uses the SegFormer model fine-tuned on Cityscapes dataset to
    segment images into 19 different classes including road, building, vegetation, etc.
    
    Args:
        df (pd.DataFrame): DataFrame containing image filenames
        folder_path (str): Path to the folder containing images
        filename_column (str, optional): Name of the column containing filenames. Defaults to 'Filename'
        
    Returns:
        pd.DataFrame: Original DataFrame with added segmentation feature columns
    """
    # Initialize model and feature extractor
    model_name = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
    feature_extractor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    model = SegformerForSemanticSegmentation.from_pretrained(model_name)
    
    # Use MPS if on M1/M2 Mac
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # Cityscapes categories
    classes = [
        'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 
        'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
        'person', 'rider', 'car', 'truck', 'bus', 'train', 
        'motorcycle', 'bicycle'
    ]
    
    results = []
    # Add tqdm progress bar
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Segmenting images"):
        img_path = os.path.join(folder_path, row[filename_column])
        image = Image.open(img_path).convert('RGB')
        
        # Preprocessing
        inputs = feature_extractor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Inference
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            
        # Post-processing
        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=image.size[::-1],
            mode="bilinear",
            align_corners=False
        )
        pred = upsampled_logits.argmax(dim=1)[0].cpu().numpy()
        
        # Calculate pixel ratio for each category
        total_pixels = pred.size
        class_ratios = {}
        for i, class_name in enumerate(classes):
            ratio = (pred == i).sum() / total_pixels
            class_ratios[f'seg_{class_name}'] = ratio
            
        results.append(class_ratios)
    
    # Create new feature DataFrame and merge with original DataFrame
    features_df = pd.DataFrame(results)
    return pd.concat([df, features_df], axis=1)


# Add COCO label definitions
COCO_LABELS = {
    1: 'person',
    2: 'bicycle',
    3: 'car',
    4: 'motorcycle',
    6: 'bus',
    8: 'truck',
    10: 'traffic light',
    11: 'fire hydrant',
    13: 'stop sign',
    15: 'bench'
}

def object_detection(df, folder_path, filename_column='Filename'):
    """
    Perform object detection on images using Faster R-CNN.
    
    This function uses Faster R-CNN with ResNet-50 backbone pre-trained on COCO dataset
    to detect objects in images. It detects common urban objects like people, vehicles,
    traffic signs, etc.
    
    Args:
        df (pd.DataFrame): DataFrame containing image filenames
        folder_path (str): Path to the folder containing images
        filename_column (str, optional): Name of the column containing filenames. Defaults to 'Filename'
        
    Returns:
        pd.DataFrame: Original DataFrame with added object detection feature columns
    """
    # Initialize model
    model = models.detection.fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    )
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Define image transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Detecting objects"):
        img_path = os.path.join(folder_path, row[filename_column])
        
        # Load and transform image
        image = Image.open(img_path).convert("RGB")
        image_tensor = transform(image).to(device)
        
        # Run detection
        with torch.no_grad():
            prediction = model([image_tensor])[0]
        
        # Count detection results
        counts = {f'det_{label}': 0 for label in COCO_LABELS.values()}
        
        for label in prediction['labels']:
            label_name = COCO_LABELS.get(label.item())
            if label_name:
                counts[f'det_{label_name}'] += 1
        
        results.append(counts)
    
    # Create new feature DataFrame and merge with original DataFrame
    features_df = pd.DataFrame(results)
    return pd.concat([df, features_df], axis=1)

# Add scene categories
SCENE_CATEGORIES = [
    'downtown', 'office_building', 'apartment_building/outdoor', 
    'residential_neighborhood', 'food_court', 'parking_lot', 
    'driveway', 'highway', 'plaza', 'market/outdoor', 
    'campus', 'promenade', 'field/wild', 'forest_path', 
    'forest/broadleaf', 'park', 'construction_site', 'industrial_area'
]

def scene_recognition(df, folder_path=None, filename_column='Filename'):
    """
    Perform scene recognition on images using ResNet.
    
    This function uses ResNet-50 pre-trained on ImageNet to classify images into
    different scene categories.
    
    Args:
        df (pd.DataFrame): DataFrame containing image filenames
        folder_path (str): Path to the folder containing images
        filename_column (str, optional): Name of the column containing filenames. Defaults to 'Filename'
        
    Returns:
        pd.DataFrame: Original DataFrame with added scene recognition feature columns
    """

    scene_model_file =  './data/resnet50_places365.pth.tar'
    scene_label_file = './data/categories_places365.txt'

    # Check and create data directory
    os.makedirs('./data', exist_ok=True)

    # Download model file if it doesn't exist
    if not os.path.exists(scene_model_file):
        print("Downloading Places365 model...")
        url = 'http://places2.csail.mit.edu/models_places365/resnet50_places365.pth.tar'
        urllib.request.urlretrieve(url, scene_model_file)

    # Download label file if it doesn't exist
    if not os.path.exists(scene_label_file):
        print("Downloading category labels...")
        url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
        urllib.request.urlretrieve(url, scene_label_file)

    # Initialize model
    model = models.resnet50(num_classes=365)
    checkpoint = torch.load(scene_model_file, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    model.eval()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Load category labels
    with open(scene_label_file) as f:
        categories = [line.strip().split(' ')[0][3:] for line in f]
    specific_category_indices = [categories.index(category) for category in SCENE_CATEGORIES]
    
    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    results = []
    # Show progress with tqdm
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Recognizing scenes"):
        img_path = os.path.join(folder_path, row[filename_column])
        
        # Load and process image
        image = Image.open(img_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0).cpu().numpy()
        
        # Collect probabilities for specific categories
        scene_probs = {
            f'scene_{category}': probabilities[idx] 
            for category, idx in zip(SCENE_CATEGORIES, specific_category_indices)
        }
        
        results.append(scene_probs)
    
    # Create new feature DataFrame and merge with original DataFrame
    features_df = pd.DataFrame(results)
    return pd.concat([df, features_df], axis=1)