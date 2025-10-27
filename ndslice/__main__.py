#!/usr/bin/env python3
"""
Command-line interface for ndslice.
"""
import sys
import argparse
import numpy as np
from pathlib import Path
from .ndslice import ndslice
from .selectors import H5DatasetSelector, NpzDatasetSelector, MatDatasetSelector


def main():
    parser = argparse.ArgumentParser(
        prog='ndslice',
        description='Interactive N-dimensional array viewer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ndslice data.npy                      # View single file
  ndslice data.h5 data2.npy data3.npz   # View multiple files
  
For files with multiple datasets (HDF5, NPZ, MAT), a GUI selector will automatically appear.
        """
    )
    parser.add_argument('files', type=str, nargs='+', 
                        help='Path(s) to data file(s) (.h5, .hdf5, .npy, .npz, .mat)')
    
    args = parser.parse_args()
    
    for file_arg in args.files:
        filepath = Path(file_arg)
        
        if not filepath.exists():
            print(f"Error: File not found: {filepath}")
            continue
        
        try:
            suffix = filepath.suffix.lower()
            
            # Simple NPY file - load directly
            if suffix == '.npy':
                data = np.load(filepath)
                ndslice(data=data, title=filepath.name, block=False)
                continue
            
            # Create appropriate selector based on file type
            selector = None
            if suffix in ['.h5', '.hdf5']:
                selector = H5DatasetSelector(filepath)
            elif suffix == '.npz':
                selector = NpzDatasetSelector(filepath)
            elif suffix == '.mat':
                selector = MatDatasetSelector(filepath)
            else:
                print(f"Unsupported file type: {suffix}. Supported types: .h5, .hdf5, .npy, .npz, .mat")
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
