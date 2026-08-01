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
from .config import load_config


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
    parser.add_argument('--mask', type=str, default=None,
                        help='Optional mask volume to overlay on a single main file')
    
    args = parser.parse_args()
    viewer_config = load_config()

    mask_path = None
    if args.mask is not None:
        if len(args.files) != 1:
            parser.error('--mask requires exactly one main file')

        mask_path = Path(args.mask)
        if not mask_path.exists():
            parser.error(f'Mask file not found: {mask_path}')
    
    for file_arg in args.files:
        filepath = Path(file_arg)
        
        if not filepath.exists():
            print(f"Error: File not found: {filepath}")
            continue
        
        try:
            suffix = ''.join(filepath.suffixes).lower()
            # Single-dataset formats and DICOM directories are handled by file_interpreters.load_path
            if filepath.is_dir() or suffix in ['.npy', '.rec', '.cfl', '.dcm', '.nii', '.nii.gz', '.txt']:
                loaded = load_path(
                    filepath,
                    apply_scaling=viewer_config.apply_scaling,
                )
                title = filepath.name or str(filepath)
                detected_format = loaded.metadata.get('detected_format')
                if detected_format:
                    title = f"{title} [{detected_format}]"
                mask_data = (
                    load_path(mask_path, apply_scaling=False).data
                    if mask_path is not None
                    else None
                )
                ndslice(data=loaded.data, title=title, block=False, filepath=filepath,
                        dim_labels=loaded.metadata.get('dim_labels'),
                        voxel_spacing=loaded.metadata.get('voxel_spacing'),
                        mask=mask_data,
                        metadata=loaded.metadata)
                continue
            
            # Multi-dataset formats - use selectors
            selector = None
            if suffix in ['.h5', '.hdf5']:
                if mask_path is not None:
                    print("--mask is not supported for multi-dataset main files yet")
                    continue
                selector = h5_selector_for_path(filepath)
            elif suffix == '.npz':
                if mask_path is not None:
                    print("--mask is not supported for multi-dataset main files yet")
                    continue
                selector = NpzDatasetSelector(filepath)
            elif suffix == '.mat':
                if mask_path is not None:
                    print("--mask is not supported for multi-dataset main files yet")
                    continue
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
