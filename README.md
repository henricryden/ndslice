# ndslice

Interactive N-dimensional array viewer with FFT support for NumPy arrays.

## Features

- **N-dimensional slicing**: View any 2D slice of your N-dimensional data
- **Multiple view modes**: Image view and line plot modes
- **FFT/IFFT support**: Apply Fourier transforms along any dimension with a click
- **Complex data support**: View real, imaginary, magnitude, or phase components
- **Scale transformations**: Linear and symmetric logarithmic scaling
- **Interactive controls**: Mouse hover for pixel values, dynamic zooming, and panning

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

## Usage

### Basic Usage

The `ndslice()` function opens an interactive window to explore your N-dimensional arrays:

```python
from ndslice import ndslice
import numpy as np

# View a 4D array
data_4d = np.random.randn(10, 20, 30, 40)
ndslice(data_4d)

# View complex FFT data
fft_data = np.fft.fftn(data_4d)
ndslice(fft_data, title='FFT Data')
```

### Interactive Features

- **Dimension Selection**: Click Y/X buttons to choose which dimensions to display
- **Slicing**: Use spinboxes to select the slice index for other dimensions
- **FFT Transforms**: 
  - Left-click dimension labels to apply FFT
  - Right-click to apply inverse FFT
  - Click again to return to native domain
- **Channel Selection** (for complex data): Real, Imaginary, Magnitude, or Phase
- **Scale Options**: Linear or Symmetric Log scaling
- **Display Modes**: Square pixels, square FOV, or fit to window
- **View Modes**: Switch between 2D image view and 1D line plot
- **Complex Data**: View real, imaginary, magnitude, or phase components

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
