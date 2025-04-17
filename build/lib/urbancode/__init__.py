__version__ = "0.1.0"
__author__ = "Sijie Yang"
__description__ = "A package for universal urban analysis"

# Import submodules
from . import svi

# Make svi module available at package level
__all__ = ['svi']

# Remove the direct imports