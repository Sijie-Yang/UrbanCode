.. UrbanCode documentation master file, created by
   sphinx-quickstart on Tue Dec 5 00:10:00 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to UrbanCode's documentation!
====================================

UrbanCode is a package for universal urban analysis.

**Recommended Import Style**::

    import urbancode as uc
    
    # Basic example
    df = uc.svi.filename("path/to/images")
    
    # Advanced features
    segmentation = uc.svi.segmentation("path/to/image.jpg")

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   svi
   examples
   changelog

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

