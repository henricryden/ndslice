#!/usr/bin/env python3
"""
Command-line interface for ndslice.
"""
import sys
import argparse
import numpy as np
from pathlib import Path
from .ndslice import ndslice # .import because we're inside the package


def load_h5_file(filepath):
    try:
        import h5py as h5
    except ImportError:
        print("Error: h5py is required to read HDF5 files.")
        print("Install it with: pip install h5py")
        sys.exit(1)
    
    def find_multidim_datasets(f):
        """Find all datasets in the file that are multidimensional (ndim > 1)."""
        multidim = []
        for name in f.keys():
            dset = f[name]
            if hasattr(dset, 'ndim') and dset.ndim >= 1:
                multidim.append((name, dset))
        return multidim

    def select_most_dimensional(multidim):
        """Select the dataset with the most dimensions."""
        if not multidim:
            raise ValueError("No array dataset found.")
        # Sort by number of dimensions, then by shape size if tied
        multidim.sort(key=lambda x: (x[1].ndim, np.prod(x[1].shape)), reverse=True)
        return multidim[0]

    def interpret_dataset(name, dset, f):
        """Interpret the dataset as real or complex, including compound datasets."""
        # If dataset is native complex type (HDF5 1.10+)
        if np.iscomplexobj(dset):
            return np.array(dset)

        # Check for compound dtype with real/imag fields
        if dset.dtype.names is not None:
            names = set(dset.dtype.names)
            real_names = {'real', 'realdata', 'r'}
            imag_names = {'imag', 'i'}
            found_real = names & real_names
            found_imag = names & imag_names
            if found_real and found_imag:
                real_field = list(found_real)[0]
                imag_field = list(found_imag)[0]
                data = dset[real_field] + 1j * dset[imag_field]
                return data
            elif found_real:
                return dset[list(found_real)[0]]
            else:
                raise ValueError(f"Compound dataset '{name}' does not have expected real/imag fields.")
        
        # If there are two datasets with real/imag names
        if set(f.keys()) & {'real', 'realdata', 'r'} and set(f.keys()) & {'imag', 'i'}:
            real_name = list(set(f.keys()) & {'real', 'realdata', 'r'})[0]
            imag_name = list(set(f.keys()) & {'imag', 'i'})[0]
            data = np.array(f[real_name]) + 1j * np.array(f[imag_name])
            return data
        
        # Assume dataset is real type
        return np.array(dset)

    with h5.File(filepath, 'r') as f:
        multidim = find_multidim_datasets(f)
        if not multidim:
            raise ValueError("No array dataset found in file.")
        name, dset = select_most_dimensional(multidim)
        print(f"Loading dataset: '{name}' with shape {dset.shape}")
        data = interpret_dataset(name, dset, f)
        return np.transpose(data)


def load_npz_file(filepath):
    """.npz files can contain multiple arrays."""
    data = np.load(filepath)
    keys = list(data.keys())
    if len(keys) == 1:
        return data[keys[0]]
    else:
        print(f"Found multiple arrays in .npz file: {keys}. Picking the first one: {keys[0]}.")
        return data[keys[0]]


def load_file(filepath):
    suffix = filepath.suffix.lower()
    
    if suffix in ['.h5', '.hdf5']:
        return load_h5_file(filepath)
    elif suffix == '.npy':
        return np.load(filepath) # .npy files can only contain a single array.
    elif suffix == '.npz':
        return load_npz_file(filepath)
    else:
        raise ValueError(f"Unsupported file type: {suffix}. Supported types: .h5, .hdf5, .npy, .npz")


def main():
    parser = argparse.ArgumentParser(
        prog='ndslice',
        description='Interactive N-dimensional array viewer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ndslice data.npy                      # View single file
  ndslice data.h5 data2.npy data3.npz   # View multiple files (works with glob patterns too)
        """
    )
    parser.add_argument('files', type=str, nargs='+', 
                        help='Path(s) to data file(s) (.h5, .hdf5, .npy, .npz)')
    
    args = parser.parse_args()
    
    for file_arg in args.files:
        filepath = Path(file_arg)
        
        if not filepath.exists():
            print(f"Error: File not found: {filepath}")
            continue
        
        try:
            ndslice(data=load_file(filepath), title=filepath.name, block=False)
            
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            continue


if __name__ == '__main__':
    main()
