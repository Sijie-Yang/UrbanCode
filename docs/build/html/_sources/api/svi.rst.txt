Street View Image (SVI) Analysis
===============================

The SVI module provides tools for analyzing street view images, including feature extraction, semantic segmentation, and object detection.

Feature Extraction
-----------------

.. automodule:: urbancode.svi.feature
   :members:
   :undoc-members:
   :show-inheritance:

The feature extraction module provides various functions for analyzing images:

* ``filename``: Scan a folder for image files and create a DataFrame with filenames
* ``color``: Extract color features from images
* ``compute_colorfulness``: Calculate the colorfulness of an image
* ``compute_canny_edges``: Detect edges in an image using Canny edge detection
* ``compute_hue_mean_std``: Calculate mean and standard deviation of hue
* ``compute_saturation_mean_std``: Calculate mean and standard deviation of saturation
* ``compute_lightness_mean_std``: Calculate mean and standard deviation of lightness
* ``compute_contrast``: Calculate image contrast
* ``compute_sharpness``: Calculate image sharpness
* ``compute_entropy``: Calculate image entropy
* ``compute_image_variance``: Calculate image variance

Semantic Segmentation
--------------------

The segmentation module uses SegFormer for semantic segmentation of street view images.

.. automodule:: urbancode.svi.feature
   :members: segmentation
   :undoc-members:
   :show-inheritance:

Object Detection
--------------

The object detection module uses Faster R-CNN for COCO object detection in street view images.

.. automodule:: urbancode.svi.feature
   :members: object_detection
   :undoc-members:
   :show-inheritance:

Examples
--------

Here are some examples of using the SVI module:

.. code-block:: python

   from urbancode.svi import feature

   # Extract features from a single image
   features = feature.extract_features('path/to/image.jpg')

   # Process a folder of images
   df = feature.filename('path/to/folder')
   df = feature.color(df)

   # Perform semantic segmentation
   segmentation = feature.segmentation('path/to/image.jpg')

   # Detect objects
   objects = feature.object_detection('path/to/image.jpg') 