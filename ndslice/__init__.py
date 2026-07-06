"""
ndslice - Interactive N-dimensional array viewer with FFT support
"""

from importlib.metadata import PackageNotFoundError, version

from .ndslice import ndslice, NDSliceWindow, Domain
from .imageview2d import ImageView2D

try:
    __version__ = version("ndslice")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = ["ndslice", "NDSliceWindow", "ImageView2D", "Domain", "__version__"]
