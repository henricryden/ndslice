"""
ndslice - Interactive N-dimensional array viewer with FFT support
"""

from .ndslice import ndslice, NDSliceWindow, Domain
from .imageview2d import ImageView2D

__version__ = "0.1.0"
__all__ = ["ndslice", "NDSliceWindow", "ImageView2D", "Domain"]
