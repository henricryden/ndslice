# ndslice

**Quick interactive visualization for N-dimensional NumPy arrays**

A python package for browsing slices, applying FFTs, and inspecting data.

Quickly checking multi-dimensional data usually means writing the same matplotlib boilerplate over and over. This tool lets you just call `ndslice(data)` and interactively explore what you've got.

## Usage
```python
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
```

![Showcase](docs/images/showcase.gif)

## Features

Data slicing and dimension selection should be intuitive: click the two dimensions you want to show and slice using the spinboxes.

**Centered FFT** - Click dimension labels to apply centered 1D FFT transforms. Useful for checking k-space data in MRI reconstructions or analyzing frequency content.
![FFT](docs/images/fft.gif)

**Line plot** - See 1D slices through your data. Shift+scroll for Y zoom, Ctrl+scroll for X zoom:

![Line plot](docs/images/lineplot.png)

**Scaling**

Log scaling is often good for k-space visualization.
Symmetric log scaling is an extension of the log scale which supports negative values.


**Colormap**
Change colormap:
- `Ctrl+1` Grayscale
- `Ctrl+2` Viridis
- `Ctrl+3` Plasma
- `Ctrl+4` Rainbow

**Non-blocking windows**

By default, windows open in separate processes, allowing multiple simultaneous views:
```python
ndslice(data1)
ndslice(data2) # Both windows appear
```

Use `block=True` to wait for the window to close before continuing:
```python
ndslice(data1, block=True)  # Script pauses here
ndslice(data2)  # Shown after first closes
```
## Installation

### From PyPI

```bash
pip install ndslice
```

### From source

```bash
git clone https://github.com/henricryden/ndslice.git
cd ndslice
pip install -e .
```

## Requirements

- Python >= 3.8
- NumPy >= 1.20.0
- PyQtGraph >= 0.12.0
- PyQt5 >= 5.15.0

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

Built with [PyQtGraph](https://www.pyqtgraph.org/) for high-performance visualization.


---
Henric Rydén

Karolinska University Hospital

Stockholm, Sweden