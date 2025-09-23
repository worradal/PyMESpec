"""
PyMESpec - Python for Modulated Experimental Spectroscopy

PyMESpec is designed to be an encompassing software package for the analysis 
of transient experimental spectroscopic data, with an emphasis in aiding 
laboratory automation.
"""

__version__ = "1.2.1"
__author__ = "worrada"

# Import main modules for easy access
from .core_functionality import *
from .config_files import *

# Make key classes available at package level
try:
    from .core_functionality.spectrum import Spectrum
    from .core_functionality.phase import Phase
    from .core_functionality.rate_data import RateData
    from .core_functionality.chemometrics import Chemometrics
    from .core_functionality.baseline_correction import BaselineCorrection
except ImportError:
    # Handle case where some dependencies might not be available
    pass
