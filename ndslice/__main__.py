#!/usr/bin/env python3
"""
Command-line interface for ndslice.
"""
import argparse
import numpy as np
from pathlib import Path
from .ndslice import ndslice
from .selectors import h5_selector_for_path, NpzDatasetSelector, MatDatasetSelector
from .file_interpreters import load_path


def main():
    parser = argparse.ArgumentParser(
        prog='ndslice',
        description='Interactive N-dimensional array viewer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ndslice data.npy                      # View single file
  ndslice data.h5 data2.npy data3.npz   # View multiple files
  ndslice scan.REC                      # View Philips REC/XML pair
  ndslice ref.cfl                       # View BART CFL/HDR pair
    ndslice dicomdir/                     # Convert DICOM directory via dcm2niix, then view
  ndslice scan.dcm                      # View DICOM file
  ndslice scan.nii                      # View NIfTI file
  ndslice data.txt                      # View text file with numeric data
  
For files with multiple datasets (HDF5, NPZ, MAT), a GUI selector will automatically appear.
        """
    )
    parser.add_argument('files', type=str, nargs='+', 
                        help='Path(s) to data files or DICOM directories')
    
    args = parser.parse_args()
    
    for file_arg in args.files:
        filepath = Path(file_arg)
        
        if not filepath.exists():
            print(f"Error: File not found: {filepath}")
            continue
        
        try:
            suffix = ''.join(filepath.suffixes).lower()
            # Single-dataset formats and DICOM directories are handled by file_interpreters.load_path
            if filepath.is_dir() or suffix in ['.npy', '.rec', '.cfl', '.dcm', '.nii', '.nii.gz', '.txt']:
                loaded = load_path(filepath)
                title = filepath.name or str(filepath)
                detected_format = loaded.metadata.get('detected_format')
                if detected_format:
                    title = f"{title} [{detected_format}]"
                ndslice(data=loaded.data, title=title, block=False, filepath=filepath,
                        dim_labels=loaded.metadata.get('dim_labels'),
                        voxel_spacing=loaded.metadata.get('voxel_spacing'))
                continue
            
            # Multi-dataset formats - use selectors
            selector = None
            if suffix in ['.h5', '.hdf5']:
                selector = h5_selector_for_path(filepath)
            elif suffix == '.npz':
                selector = NpzDatasetSelector(filepath)
            elif suffix == '.mat':
                selector = MatDatasetSelector(filepath)
            else:
                print(f"Unsupported file type: {suffix}. Supported types: directories with DICOM .dcm files, .h5, .hdf5, .npy, .npz, .mat, .REC, .cfl, .dcm, .nii, .nii.gz, .txt")
                continue
            
            # Select and view dataset (shows GUI if multiple datasets)
            if not selector.view(block=False):
                print(f"No compatible datasets found in {filepath}")
            
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            import traceback
            traceback.print_exc()
            continue


if __name__ == '__main__':
    main()
