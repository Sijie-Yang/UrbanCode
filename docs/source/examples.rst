Examples
========

This section contains example notebooks demonstrating how to use UrbanCode for various urban analysis tasks.

Street View Image Analysis
-------------------------

.. toctree::
   :maxdepth: 1

   examples/test_svi_feature
   examples/test_perception_TwoStageNN

These notebooks demonstrate:

* Feature extraction from street view images
* Color analysis
* Semantic segmentation
* Object detection
* Two-stage neural network for urban perception

To run these examples:

1. Install the development dependencies:

   .. code-block:: bash

      pip install -e ".[dev]"

2. Start Jupyter Notebook:

   .. code-block:: bash

      jupyter notebook

3. Navigate to the ``examples`` directory and open the desired notebook.

Note: Make sure you have sufficient GPU memory for running the neural network models. 