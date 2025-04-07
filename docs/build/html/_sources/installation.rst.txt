Installation
============

UrbanCode can be installed using pip. We recommend using a virtual environment.

Basic Installation
-----------------

To install UrbanCode with basic dependencies:

.. code-block:: bash

   pip install urbancode

Development Installation
-----------------------

For development, you can install UrbanCode in editable mode with all development dependencies:

.. code-block:: bash

   git clone https://github.com/sijieyang/urbancode.git
   cd urbancode
   pip install -e ".[dev]"

Dependencies
-----------

UrbanCode has the following main dependencies:

* Python >= 3.8
* PyTorch >= 2.0.0
* torchvision >= 0.15.0
* networkx >= 2.5
* osmnx >= 1.1.1
* momepy >= 0.5.3
* geopandas >= 0.9.0
* matplotlib >= 3.3.4

For development, additional dependencies include:

* pytest >= 6.0
* pytest-cov >= 2.0
* black >= 21.0
* flake8 >= 3.9.0
* isort >= 5.9.0

System Requirements
-----------------

* Operating System: Windows, macOS, or Linux
* GPU: Recommended for faster processing of computer vision tasks
* RAM: Minimum 8GB, recommended 16GB or more
* Storage: At least 1GB of free space 