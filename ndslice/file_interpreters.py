import xml.etree.ElementTree as ET
import numpy as np
import struct
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
import shutil


@dataclass
class LoadedPath:
    data: np.ndarray
    metadata: dict


def remove_trailing_singletons(data):
    shape = data.shape
    for dim in reversed(range(len(shape))):
        if shape[dim] != 1:
            break
    
    return data.reshape(shape[:dim+1])


def _nifti_dim_labels(ndim):
    labels = [''] * ndim
    for dim, label in enumerate(('X', 'Y', 'Z', 'T')):
        if dim < ndim:
            labels[dim] = label
    return labels


def _append_dim_label(dim_labels, label):
    labels = list(dim_labels or [])
    labels.append(label or '')
    return labels


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
        self.metadata = {
            'source_path': str(self.dcm_path),
            'detected_format': 'dicom_file',
        }
        
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
        if not hasattr(dcm, 'pixel_array'):
            raise ValueError(
                f"DICOM file does not contain pixel data: {self.dcm_path}"
            )

        if getattr(dcm, 'SamplesPerPixel', 1) != 1:
            raise ValueError(
                "Single-file DICOM loading currently supports grayscale pixel data only. "
                f"Unsupported SamplesPerPixel={dcm.SamplesPerPixel} for {self.dcm_path}"
            )

        data = remove_trailing_singletons(np.asarray(dcm.pixel_array))
        self.metadata.update({
            'shape': tuple(data.shape),
            'dtype': str(data.dtype),
            'dicom_metadata': {
                'patient_name': str(getattr(dcm, 'PatientName', '')),
                'series_description': str(getattr(dcm, 'SeriesDescription', '')),
                'modality': str(getattr(dcm, 'Modality', '')),
                'pixel_spacing': list(getattr(dcm, 'PixelSpacing', [])) if hasattr(dcm, 'PixelSpacing') else None,
                'image_position_patient': list(getattr(dcm, 'ImagePositionPatient', [])) if hasattr(dcm, 'ImagePositionPatient') else None,
            },
        })
        return data


def _find_dicom_files(directory_path):
    directory_path = Path(directory_path)
    return sorted(
        path for path in directory_path.rglob('*')
        if path.is_file() and path.suffix.lower() == '.dcm'
    )


def _json_sidecar_for_nifti(nifti_path):
    nifti_path = Path(nifti_path)
    json_path = nifti_path.with_suffix('.json')
    if nifti_path.suffix == '.gz':
        json_path = nifti_path.with_suffix('').with_suffix('.json')
    return json_path if json_path.exists() else None


def _pick_converted_niftis(output_dir):
    output_dir = Path(output_dir)
    nifti_paths = sorted(output_dir.glob('*.nii')) + sorted(output_dir.glob('*.nii.gz'))

    if not nifti_paths:
        raise ValueError("dcm2niix did not produce a NIfTI file")

    return [(nifti_path, _json_sidecar_for_nifti(nifti_path)) for nifti_path in nifti_paths]


def _read_json_sidecar(json_path):
    if json_path is None:
        return {}

    import json

    with open(json_path, 'r') as json_file:
        return json.load(json_file)


def _describe_stacking_axis(sidecar_metadata):
    axis_candidates = [
        ('EchoNumber', 'echoes'),
        ('FlipAngle', 'flip angles'),
        ('DiffusionBValue', 'b-values'),
        ('SeriesNumber', 'series'),
        ('AcquisitionNumber', 'acquisitions'),
    ]

    for key, label in axis_candidates:
        values = [metadata.get(key) for metadata in sidecar_metadata]
        if any(value is not None for value in values) and len(set(values)) > 1:
            return label, key, values

    names = [metadata.get('ProtocolName') or metadata.get('SeriesDescription') for metadata in sidecar_metadata]
    if any(name for name in names) and len(set(names)) > 1:
        return 'series', 'SeriesDescription', names

    return 'series', None, None


def _run_dcm2niix(directory_path, output_dir):
    dcm2niix_path = shutil.which('dcm2niix')
    if dcm2niix_path is None:
        raise RuntimeError(
            "dcm2niix is required to load DICOM directories. Install it, for example, with "
            "'conda install -c conda-forge dcm2niix', and ensure it is on PATH."
        )

    command = [
        dcm2niix_path,
        '-b', 'y',
        '-z', 'n',
        '-f', 'ndslice_%s',
        '-o', str(output_dir),
        str(directory_path),
    ]
    completed = subprocess.run(command, capture_output=True)
    if completed.returncode != 0:
        stdout = completed.stdout.decode('utf-8', errors='replace').strip()
        stderr = completed.stderr.decode('utf-8', errors='replace').strip()
        error_output = '\n'.join(part for part in [stdout, stderr] if part)
        raise RuntimeError(
            "dcm2niix failed to convert the DICOM directory. "
            f"Input: {directory_path}\n{error_output}"
        )

    return _pick_converted_niftis(output_dir)


class DicomDirectoryLoader:
    def __init__(self, directory_path):
        self.directory_path = Path(directory_path)
        self.metadata = {
            'source_path': str(self.directory_path),
            'detected_format': 'dicom_directory',
        }

        if not self.directory_path.exists():
            raise FileNotFoundError(f"DICOM directory not found: {self.directory_path}")
        if not self.directory_path.is_dir():
            raise NotADirectoryError(f"Expected a directory, got: {self.directory_path}")

    def load(self):
        dicom_files = _find_dicom_files(self.directory_path)
        if not dicom_files:
            raise ValueError(
                f"No DICOM files with suffix .dcm were found under directory: {self.directory_path}"
            )

        with tempfile.TemporaryDirectory(prefix='ndslice_dicom_') as temp_dir:
            nifti_outputs = _run_dcm2niix(self.directory_path, temp_dir)
            arrays = []
            nifti_paths = []
            json_paths = []
            sidecar_metadata = []
            nifti_dim_labels = []

            for nifti_path, json_path in nifti_outputs:
                nifti_loader = NiftiLoader(nifti_path)
                arrays.append(nifti_loader.load())
                nifti_paths.append(str(nifti_path))
                json_paths.append(str(json_path) if json_path else None)
                sidecar_metadata.append(_read_json_sidecar(json_path))
                nifti_dim_labels.append(nifti_loader.metadata.get('dim_labels', []))

            if len(arrays) == 1:
                data = arrays[0]
                dim_labels = list(nifti_dim_labels[0]) if nifti_dim_labels else _nifti_dim_labels(data.ndim)
                stacking_label = None
                stacking_key = None
                stacking_values = None
            else:
                shapes = [tuple(array.shape) for array in arrays]
                if len(set(shapes)) != 1:
                    paths = ', '.join(Path(path).name for path in nifti_paths)
                    raise ValueError(
                        "dcm2niix produced multiple NIfTI files with different shapes, so ndslice "
                        f"cannot stack them automatically. Outputs: {paths}"
                    )

                stacking_label, stacking_key, stacking_values = _describe_stacking_axis(sidecar_metadata)
                print(f"Dataset stacked automatically along new dimension: {stacking_label}")
                if stacking_key is not None and stacking_values is not None:
                    print(f"  {stacking_key}: {stacking_values}")
                if stacking_key == 'SeriesNumber':
                    series_descriptions = [metadata.get('SeriesDescription') for metadata in sidecar_metadata]
                    if any(description is not None for description in series_descriptions):
                        print(f"  SeriesDescription: {series_descriptions}")
                data = np.stack(arrays, axis=-1)
                base_dim_labels = list(nifti_dim_labels[0]) if nifti_dim_labels else _nifti_dim_labels(arrays[0].ndim)
                dim_labels = _append_dim_label(base_dim_labels, stacking_label)

            self.metadata.update({
                'shape': tuple(data.shape),
                'dtype': str(data.dtype),
                'dim_labels': dim_labels,
                'nifti_output_path': nifti_paths[0] if len(nifti_paths) == 1 else nifti_paths,
                'sidecar_json_path': json_paths[0] if len(json_paths) == 1 else json_paths,
                'stacked_dimension': stacking_label,
                'stacked_dimension_key': stacking_key,
                'stacked_dimension_values': stacking_values,
            })
            return data


class NiftiLoader:
    def __init__(self, file_path):
        self.file_path = Path(file_path)
        self.metadata = {
            'source_path': str(self.file_path),
            'detected_format': 'nifti',
        }
        
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
        data = remove_trailing_singletons(data)
        self.metadata.update({
            'shape': tuple(data.shape),
            'dtype': str(data.dtype),
            'dim_labels': _nifti_dim_labels(data.ndim),
        })
        return data


class TextLoader:
    def __init__(self, file_path):
        self.file_path = Path(file_path)
        self.metadata = {
            'source_path': str(self.file_path),
            'detected_format': 'text',
        }
        
        if not self.file_path.exists():
            raise FileNotFoundError(f"Text file not found: {self.file_path}")
    
    def load(self):
        """
        Load simple text files with numeric data.
        - Skips non-numeric header lines
        - Supports comma, tab, semicolon, and whitespace delimiters
        - Handles up to 2D data
        """
        numeric_lines = []
        
        with open(self.file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Replace common delimiters with spaces for uniform parsing
                for delimiter in [',', '\t', ';', '|']:
                    line = line.replace(delimiter, ' ')
                
                # Try to parse as numeric values
                try:
                    values = [float(x) for x in line.split()]
                    if values:  # Only add non-empty lines
                        numeric_lines.append(values)
                except ValueError:
                    # Skip non-numeric lines (headers/comments)
                    continue
        
        if not numeric_lines:
            raise ValueError(f"No numeric data found in {self.file_path}")
        
        # Check if all rows have the same length (2D) or if it's 1D
        row_lengths = [len(row) for row in numeric_lines]
        
        if len(set(row_lengths)) == 1:
            # All rows have same length - create 2D array
            data = np.array(numeric_lines, dtype=np.float32)
        else:
            # Different row lengths - flatten to 1D
            data = np.array([val for row in numeric_lines for val in row], dtype=np.float32)
        
        data = remove_trailing_singletons(data)
        self.metadata.update({
            'shape': tuple(data.shape),
            'dtype': str(data.dtype),
        })
        return data


def load_path(filepath):
    filepath = Path(filepath)

    if filepath.is_dir():
        loader = DicomDirectoryLoader(filepath)
        data = loader.load()
        return LoadedPath(data=data, metadata=loader.metadata)

    suffix = ''.join(filepath.suffixes).lower()

    if suffix == '.npy':
        data = np.load(filepath)
        return LoadedPath(
            data=data,
            metadata={
                'source_path': str(filepath),
                'detected_format': 'numpy',
                'shape': tuple(data.shape),
                'dtype': str(data.dtype),
            },
        )

    if suffix == '.rec':
        loader = PhilipsRECLoader(filepath)
    elif suffix == '.cfl':
        loader = BartLoader(filepath)
    elif suffix == '.dcm':
        loader = DicomLoader(filepath)
    elif suffix in ['.nii', '.nii.gz']:
        loader = NiftiLoader(filepath)
    elif suffix == '.txt':
        loader = TextLoader(filepath)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")

    data = loader.load()
    metadata = {
        'source_path': str(filepath),
        'detected_format': suffix.lstrip('.'),
        'shape': tuple(data.shape),
        'dtype': str(data.dtype),
    }
    metadata.update(getattr(loader, 'metadata', {}))
    return LoadedPath(data=data, metadata=metadata)





def load_file(filepath):
    """
    Generic path loader - automatically detects format and returns a NumPy array.
    
    Supported formats:
    - directory: DICOM directory via dcm2niix
    - .npy: NumPy binary format
    - .REC: Philips XLM+REC (requires xml file)
    - .cfl: BART format (requires hdr file)
    - .dcm: DICOM format (requires pydicom)
    - .nii/.nii.gz: NIfTI format (requires nibabel)
    - .txt: Simple text files with numeric data
    
    Args:
        filepath: Path to data file or DICOM directory
        
    Returns:
        NumPy array
        
    """
    return load_path(filepath).data