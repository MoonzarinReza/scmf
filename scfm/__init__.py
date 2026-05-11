"""
SCFM
----
Supernova Classification with FPCA and Machine Learning

Functions
---------
fit_lc   : FPCA-based light curve fitting pipeline
classify : Binary transient classification using gradient boosted decision trees to identify potential SNe Ia
"""

from .lc_fitting import fit_lc
from .classify import classify

__version__ = '0.1.0'
__author__  = 'Moonzarin Reza'