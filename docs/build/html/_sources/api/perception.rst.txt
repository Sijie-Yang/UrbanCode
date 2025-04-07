Urban Perception
===============

The perception module provides tools for analyzing urban environments using computer vision and machine learning techniques.

Two-Stage Neural Network
-----------------------

.. automodule:: urbancode.perception.TwoStageNN
   :members:
   :undoc-members:
   :show-inheritance:

The Two-Stage Neural Network module implements a deep learning approach for urban perception analysis. It consists of:

1. Feature Extraction Stage
   * Image preprocessing
   * Feature extraction using pre-trained models
   * Feature normalization and augmentation

2. Classification Stage
   * Multi-label classification
   * Confidence scoring
   * Ensemble predictions

Examples
--------

Here are some examples of using the perception module:

.. code-block:: python

   from urbancode.perception import TwoStageNN

   # Initialize the model
   model = TwoStageNN()

   # Load pre-trained weights
   model.load_weights('path/to/weights.pth')

   # Process a single image
   predictions = model.predict('path/to/image.jpg')

   # Process a batch of images
   batch_predictions = model.predict_batch(['image1.jpg', 'image2.jpg'])

   # Get feature embeddings
   features = model.extract_features('path/to/image.jpg') 