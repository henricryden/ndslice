import xml.etree.ElementTree as ET
import numpy as np
import struct
from pathlib import Path


def remove_trailing_singletons(data):
    shape = data.shape
    for dim in reversed(range(len(shape))):
        if shape[dim] != 1:
            break
    
    return data.reshape(shape[:dim+1])


class PhilipsRECLoader:
    def __init__(self, rec_path):
        self.rec_path = Path(rec_path)
        self.xml_path = self.rec_path.with_suffix('.xml')
        
        if not self.xml_path.exists():
            raise FileNotFoundError(f"XML metadata file not found: {self.xml_path}")
        
        # Validate this is a Philips XLMREC file
        with open(self.xml_path, 'r') as f:
            first_line = f.readline().strip()
            if not first_line.startswith('<PRIDE_'):
                raise ValueError(f"Not a Philips XLMREC file: {self.xml_path}")
        
        self._parse_metadata()
        self._setup_dimensions()

        # Pre-allocate
        self.data = np.zeros(
            (self.nx, self.ny, self.n_slices, self.n_echoes, 
             self.n_grad_orients, self.n_b_values, self.n_phases, self.n_dynamics),
            dtype=self.dtype
        )
        
    def _parse_metadata(self):
        tree = ET.parse(self.xml_path)
        root = tree.getroot()
        
        def parse_attribute(attr_elem):
            type_str = attr_elem.get('Type')
            text = attr_elem.text
            
            if text is None:
                return None
            
            # Handle array attributes (space-separated values)
            if attr_elem.get('ArraySize'):
                values = [float(v) if type_str in ['Float', 'Double'] else int(v) 
                         for v in text.split()]
                return values
            
            # Single value
            if type_str in ['Int32', 'Int16', 'UInt16']:
                return int(text)
            elif type_str in ['Float', 'Double']:
                return float(text)
            else:
                return text
        
        # Extract series-level metadata
        self.general_info = {}
        for attr in root.findall('.//Series_Info/Attribute'):
            name = attr.get('Name')
            self.general_info[name] = {
                'Value': parse_attribute(attr), 
                'Type': attr.get('Type')
            }
        
        # Extract image-level metadata (only elements with Key children)
        self.image_infos = []
        for img_info in root.findall('.//Image_Info'):
            if img_info.find('Key') is None:
                continue
                
            img_meta = {}
            
            # Parse Key attributes (slice indices)
            key_elem = img_info.find('Key')
            for attr in key_elem.findall('.//Attribute'):
                name = attr.get('Name')
                img_meta[name] = {
                    'Value': parse_attribute(attr), 
                    'Type': attr.get('Type')
                }
            
            # Parse image attributes (resolution, scaling, etc.)
            for attr in img_info.findall('.//Attribute'):
                name = attr.get('Name')
                img_meta[name] = {
                    'Value': parse_attribute(attr), 
                    'Type': attr.get('Type')
                }
            
            self.image_infos.append(img_meta)
        
        if not self.image_infos:
            raise ValueError("No valid image metadata found in XML")
        
        # Validate resolution consistency
        nx = self.image_infos[0]['Resolution X']['Value']
        ny = self.image_infos[0]['Resolution Y']['Value']
        
        for idx, img_meta in enumerate(self.image_infos[1:], start=1):
            img_nx = img_meta['Resolution X']['Value']
            img_ny = img_meta['Resolution Y']['Value']
            if img_nx != nx or img_ny != ny:
                raise ValueError(
                    f"Resolution mismatch: Image 0 is {nx}x{ny}, "
                    f"but image {idx} is {img_nx}x{img_ny}"
                )
    
    def _setup_dimensions(self):
        """Set up array dimensions and data type as class members."""
        # Image dimensions
        self.nx = self.image_infos[0]['Resolution X']['Value']
        self.ny = self.image_infos[0]['Resolution Y']['Value']
        self.pixel_size = self.image_infos[0]['Pixel Size']['Value']
        
        # Array dimensions
        self.n_slices = self.general_info['Max No Slices']['Value']
        self.n_echoes = self.general_info['Max No Echoes']['Value']
        self.n_grad_orients = self.general_info['Max No Gradient Orients']['Value']
        self.n_b_values = self.general_info['Max No B Values']['Value']
        self.n_phases = self.general_info['Max No Phases']['Value']
        self.n_dynamics = self.general_info['Max No Dynamics']['Value']
        
        # Binary reading parameters
        self.bytes_per_pixel = (self.pixel_size - 1) // 8 + 1
        self.n_pixels_per_image = self.nx * self.ny
        
        # Detect data types and determine output dtype
        data_types = set(img['Type']['Value'] for img in self.image_infos)
        self.has_real = 'R' in data_types
        self.has_imag = 'I' in data_types
        self.has_mag = 'M' in data_types
        self.has_phase = 'P' in data_types
        
        is_complex = (self.has_real and self.has_imag) or (self.has_mag and self.has_phase)
        self.dtype = np.complex64 if is_complex else np.float32
    
    def _next_slice(self, fid, img_idx):
        img_meta = self.image_infos[img_idx]
        
        # Unpack to unsigned short (16-bit)
        data_bytes = fid.read(self.bytes_per_pixel * self.n_pixels_per_image)
        format_string = f'@{self.n_pixels_per_image}H'
        raw_slice = np.array(struct.unpack_from(format_string, data_bytes), dtype=np.float32)
        raw_slice = raw_slice.reshape((self.ny, self.nx))
        
        # Apply rescaling
        ri = img_meta['Rescale Intercept']['Value']
        rs = img_meta['Rescale Slope']['Value']
        ss = img_meta['Scale Slope']['Value']
        scaled_slice = (1.0 / ss) * raw_slice + ri / (rs * ss)
                
        # Get position indices (convert from 1-based to 0-based)
        slice_idx = img_meta['Slice']['Value'] - 1
        echo_idx = img_meta['Echo']['Value'] - 1
        grad_idx = img_meta['Grad Orient']['Value'] - 1
        bval_idx = img_meta['BValue']['Value'] - 1
        phase_idx = img_meta['Phase']['Value'] - 1
        dyn_idx = img_meta['Dynamic']['Value'] - 1
        
        img_type = img_meta['Type']['Value']
        
        if self.has_real and self.has_imag: # R+I
            if img_type == 'R':
                self.data.real[:, :, slice_idx, echo_idx, grad_idx, bval_idx, phase_idx, dyn_idx] = scaled_slice
            elif img_type == 'I':
                self.data.imag[:, :, slice_idx, echo_idx, grad_idx, bval_idx, phase_idx, dyn_idx] = scaled_slice
        elif self.has_mag:
            if img_type == 'M':
                self.data[:, :, slice_idx, echo_idx, grad_idx, bval_idx, phase_idx, dyn_idx] = scaled_slice
            elif img_type == 'P' and self.has_phase: # M+P 
                scale_factor = 1.0
                if img_meta.get('Contrast Type', {}).get('Value') == 'FLOW_ENCODED':
                    if 'Phase Encoding Velocity' in self.general_info:
                        venc_vector = self.general_info['Phase Encoding Velocity']['Value']
                        if isinstance(venc_vector, (list, np.ndarray)):
                            venc = np.linalg.norm(venc_vector)
                            if venc > 0:
                                scale_factor = np.pi / venc
                # M * np.exp(i*P)
                self.data[:, :, slice_idx, echo_idx, grad_idx, bval_idx, phase_idx, dyn_idx] *= \
                    np.exp(1j * scaled_slice * scale_factor)
    
    def load(self):
        with open(self.rec_path, 'rb') as fid:
            for img_idx in range(len(self.image_infos)):
                self._next_slice(fid, img_idx)

        return remove_trailing_singletons(self.data)


class BartLoader:
    def __init__(self, cfl_path):
        self.cfl_path = Path(cfl_path)
        self.hdr_path = self.cfl_path.with_suffix('.hdr')
        
        if not self.hdr_path.exists():
            raise FileNotFoundError(f"BART header file not found: {self.hdr_path}")
        
        # Parse dimensions from .hdr file
        self._parse_header()
    
    def _parse_header(self):
        with open(self.hdr_path, 'r') as h:
            h.readline()  # Skip first line (comment)
            dims_line = h.readline()
            self.dims = tuple(int(i) for i in dims_line.split())
        
        if not self.dims:
            raise ValueError(f"No dimensions found in {self.hdr_path}")
    
    def load(self):
        data = np.memmap( 
            self.cfl_path,
            dtype=np.complex64,
            mode='r',
            shape=self.dims,
            order='F'  # BART files are stored in Fortran order
        )
        
        # Copy to memory and remove trailing singleton dimensions
        return remove_trailing_singletons(np.array(data))


class DicomLoader:
    def __init__(self, dcm_path):
        self.dcm_path = Path(dcm_path)
        
        if not self.dcm_path.exists():
            raise FileNotFoundError(f"DICOM file not found: {self.dcm_path}")
    
    def load(self):
        try:
            import pydicom
        except ImportError:
            raise ImportError(
                "pydicom is required to read DICOM files.\n"
                "Install it with: pip install pydicom"
            )
        
        dcm = pydicom.dcmread(self.dcm_path)
        return remove_trailing_singletons(dcm.pixel_array)


class NiftiLoader:
    def __init__(self, file_path):
        self.file_path = Path(file_path)
        
        if not self.file_path.exists():
            raise FileNotFoundError(f"NIfTI file not found: {self.file_path}")

    def load(self):
        try:
            import nibabel as nib
        except ImportError:
            raise ImportError(
                "nibabel is required to read NIfTI files.\n"
                "Install it with: pip install nibabel"
            )
        
        data = nib.load(self.file_path).get_fdata()
        return remove_trailing_singletons(data)





def load_file(filepath):
    """
    Generic file loader - automatically detects format and returns NumPy array.
    
    Supported formats:
    - .npy: NumPy binary format
    - .REC: Philips XLM+REC (requires xml file)
    - .cfl: BART format (requires hdr file)
    - .dcm: DICOM format (requires pydicom)
    
    Args:
        filepath: Path to data file
        
    Returns:
        NumPy array
        
    """
    filepath = Path(filepath)
    suffix = ''.join(Path(filepath).suffixes).lower()
    
    if suffix == '.npy':
        return np.load(filepath)
    
    elif suffix == '.rec':
        loader = PhilipsRECLoader(filepath)
        return loader.load()
    
    elif suffix == '.cfl':
        loader = BartLoader(filepath)
        return loader.load()
    
    elif suffix == '.dcm':
        loader = DicomLoader(filepath)
        return loader.load()
    elif suffix in ['.nii', '.nii.gz']:
        loader = NiftiLoader(filepath)
        return loader.load()
    
    else:
        raise ValueError(f"Unsupported file format: {suffix}")