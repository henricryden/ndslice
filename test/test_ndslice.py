import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ndslice import ndslice
import numpy as np

# Create some data
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
z = np.linspace(-5, 5, 50)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

mag = np.exp(-(X**2 + Y**2 + Z**2) / 10)
pha = np.pi/4 * (X + Y + Z)
complex_data = mag * np.exp(1j * pha)

ndslice(complex_data, title='3D Complex Gaussian')
ndslice(np.abs(complex_data), block=True, title='3D abs Gaussian')
ndslice(np.angle(complex_data), title='3D phase Gaussian')
