from .ndslice import ndslice
import numpy as np
import h5py
import sys

def main():
    fname = sys.argv[1]
    with h5py.File(fname) as hf:
        a = np.array(hf[sys.argv[2]])
        ndslice(a, title=sys.argv[2])
