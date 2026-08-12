# Changelog


## [Unreleased]

### Added
- Native light/dark appearance support with persistent System, Light, and Dark settings.
- Array metadata dialog and DICOM tag browser with fuzzy search.
- Configurable DICOM/NIfTI intensity scaling (scaled/raw pixel values)

### Changed
- PyQt6 6.8 or newer is now required (PyQt5 support removed)
- ndslice now uses the native platform widget style instead of forcing Fusion.
- Adopted SPEC 0 with Python 3.12, NumPy 2.2, and SciPy 1.15 minimums.

### Fixed
- Improved Linux dark-mode contrast for controls, text, and toolbar symbols.


## [0.9.0]

### Added
- **Mask overlays** — Added mask overlays from Python (`mask=`) and CLI (`--mask`).
- **Video export ranges** — Added start/end slice selection.

### Changed
- Video export now defaults to MP4 when available and uses the displayed aspect ratio by default.
- Settings tooltip shows ndslice version.

### Fixed
- Restored `ndslice.__version__` from package metadata.
- Removed duplicate MP4 `-pix_fmt` ffmpeg warning.


## [0.8.0]

### Added
- **Persistent viewer settings** — Added a settings menu backed by a cross-platform TOML config file.
  - **Startup defaults** - Configure initial indices (`First`, `Center`, `Last`) and initial origin (`Upper left`, `Lower left`, `Upper right`, `Lower right`).
  - **Display defaults** - Configure default channel, default colormap, angle-specific colormap, and default pixel-ratio display mode.
- **Line/bar slice preview** - Hovering the Line Plot/Bar Plot tab now highlights the corresponding row or column in the image view when the plotted dimension matches an image axis.
 **When that preview line is visible, the mouse wheel updates the corresponding slice index instead of switching tabs**.
- **Dimension labels** - Labels dimensions from NIfTI/DICOM metadata and HDF5 `DIMENSION_LABELS`.
- **Voxel spacing** - Support `voxel_spacing` for spatial dimension. Automatically read from nifti, DICOM, and Riesling HDF5 files.

### Fixed
- **Qt startup and non-blocking windows** _— Made viewer windows open more reliably across Qt bindings, especially from notebooks and interactive Python sessions. Thanks Thomas Roos.
- **Fix missing file extension in video export**


## [0.7.0]

### Added
- **DICOM directory loading** — `ndslice some_dicom_dir/` now converts a directory of `.dcm` files via `dcm2niix` and loads the produced NIfTI volume. Single `.dcm` files still load through `pydicom`.
- **Save current array as NumPy** — Ctrl+S now saves the current array state to `.npy`, with a range-selection dialog and optional singleton-dimension squeezing.

## [0.6.1]

### Changed
- **Default channel selection** — Real-valued data now defaults to 'real' channel. Complex data defaults to 'abs' channel

### Fixed
- **macOS emoji rendering** — Fixed emoji glyphs on macOS

## [0.6.0]

### Added
- **File monitoring & live reload** — Watch for file changes; click warning icon (⚠️) to reload
- **Cross-platform CI** — Automated testing on Ubuntu, macOS, Windows with Python 3.12–3.14

### Changed
- **Fusion Qt style**

### Fixed
- **File locking** — Close HDF5/NPZ files immediately after data load to prevent locks blocking other processes

## [0.5.1] - 2026-04-09

### Fixed
- **Colormap switching** — Fixed error when switching back to gray colormap on PyQt5 systems without matplotlib.

## [0.5.0] - 2026-02-18

### Added
- **PyQt6 support** — works with both PyQt5 (default) and PyQt6 via optional dependency: `pip install ndslice[pyqt6]`
- **HiDPI display support**
- **Colormaps** — Added colormaps with keyboard shortcuts:
  - Ctrl+1: Gray
  - Ctrl+2: [Viridis](https://bids.github.io/colormap/)
  - Ctrl+3: [Plasma](https://bids.github.io/colormap/)
  - Ctrl+4: PAL-relaxed (cyclic, hides phase wraps)
  - Ctrl+5: [Cividis](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0199239)
  - Ctrl+6: [Cubehelix](http://www.mrao.cam.ac.uk/~dag/CUBEHELIX/)
  - Ctrl+7: [Cool](https://d3js.org/d3-scale-chromatic/sequential)
  - Ctrl+8: [Warm](https://d3js.org/d3-scale-chromatic/sequential)
- **Video export**:
  - GIF, WebM, MP4, PNG (frames)
  - Window/Level can be per-slice or fixed
  
- **Update pyqtgraph to 0.14.0**

### Fixed
- Window/Level reset on re-clicking `linear` / `symlog`
- MATLAB v7.3 file loading — falls back to HDF5 loader when scipy.io.loadmat fails
