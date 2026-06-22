[![Python versions](https://img.shields.io/pypi/pyversions/ndslice.svg)](https://pypi.org/project/ndslice/)
[![PyPI version](https://img.shields.io/pypi/v/ndslice.svg)](https://pypi.org/project/ndslice/)
[![License](https://img.shields.io/github/license/henricryden/ndslice.svg)](https://github.com/henricryden/ndslice/blob/main/LICENSE)
[![Downloads](https://static.pepy.tech/personalized-badge/ndslice?period=total&units=international_system&left_color=black&right_color=green&left_text=downloads)](https://pepy.tech/projects/ndslice)
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

**Video export**
Right-clicking a dimension button to export a video or PNG frames along that dimension.
The video export functionality is optional, and can be installed with

```bash
pip install ndslice[video_export]
```
![Export](docs/images/video_export.gif)


**Scaling**

Log scaling is often good for k-space visualization.
Symmetric log scaling is an extension of the log scale which supports negative values.


**Colormap**
Change colormap:
  - Ctrl+1: Gray
  - Ctrl+2: [Viridis](https://bids.github.io/colormap/)
  - Ctrl+3: [Plasma](https://bids.github.io/colormap/)
  - Ctrl+4: Cyclic rainbow, hides phase wraps
  - Ctrl+5: [Cividis](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0199239)
  - Ctrl+6: [Cubehelix](http://www.mrao.cam.ac.uk/~dag/CUBEHELIX/)
  - Ctrl+7: [Cool](https://d3js.org/d3-scale-chromatic/sequential)
  - Ctrl+8: [Warm](https://d3js.org/d3-scale-chromatic/sequential)


**Axis flipping**
Click arrow icons (⬇️/⬆️ and ⬅️/➡️) next to dimension labels to flip axes.
Default orientation is image-style (origin lower-left).
Flip the primary axis for matrix-style (origin upper-left).

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

If Qt is already initialized in the current process, `ndslice(..., block=False)`
cannot safely fork a child process. In that case:

- In IPython/Jupyter with `%gui qt`, ndslice opens inline in the current process (still non-blocking).
- Without an active Qt event loop, ndslice falls back to blocking mode and emits a warning.

### Command Line

```bash
ndslice data.npy # Numpy file
ndslice image.nii.gz
ndslice image.dcm
ndslice some_dicom_dir/ # Automatically attemps to form an nd-array from DICOM the files
ndslice --help   # Show all options
```

**File support**
ndslice has CLI support and can conveniently display:
| Format | File suffix | Requirement |
|---|---:|---|
| NumPy | `.npy`, `.npz` | NumPy |
| MATLAB | `.mat` | scipy |
| HDF5 | `.h5`, `.hdf5` | h5py |
| [BART](https://mrirecon.github.io/bart/) | `.cfl` + `.hdr` | — |
| Philips REC | `.REC` + `.xml` | — |
| [NIfTI](https://nifti.nimh.nih.gov/) | `.nii`, `.nii.gz` | nibabel |
| DICOM file | `.dcm` | pydicom |
| DICOM directory | directory containing `.dcm` files | pydicom, nibabel, `dcm2niix` on `PATH` |

HDF5 files can be compound complex dtype, or real/imag fields.

For DICOM directories, ndslice does not infer series dimensions itself. It runs `dcm2niix`, then loads the produced NIfTI volume.

If there are multiple datasets in the file, a selection GUI appears which highlights arrays supported by ndslice (essentially numeric).
Double click to open.

![Selector](docs/images/selector.png)


## Installation

### From PyPI

```bash
pip install ndslice
# Or, if you want just the package without pulling in dependencies:
pip install --no-deps ndslice
```

For DICOM directories you also need the external `dcm2niix` binary available on `PATH`. A practical install route is:

```bash
conda install -c conda-forge dcm2niix
```

If `dcm2niix` is missing or conversion fails, ndslice reports a clear error instead of trying to infer the series layout itself.

### From source

```bash
git clone https://github.com/henricryden/ndslice.git
cd ndslice

# Use directly without installing
python -m ndslice data.npy

pip install -e .
```


## Requirements

- Python >= 3.8
- NumPy >= 1.20.0
- PyQtGraph >= 0.12.0
- PyQt5 >= 5.15.0
- h5py >= 3.0.0
- scipy >= 1.7.0
- pydicom >= 2.4.0
- nibabel >= 4.0.0
- imageio >= 2.9.0
- imageio-ffmpeg >= 0.4.2
- Pillow >= 8.0.0

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