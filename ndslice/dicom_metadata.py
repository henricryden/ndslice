import hashlib
from pathlib import Path

import numpy as np


DICOM_CURATED_TAGS = (
    'Modality',
    'SeriesDescription',
    'ProtocolName',
    'SeriesNumber',
    'InstanceNumber',
    'SOPClassUID',
    'ImageType',
    'AcquisitionNumber',
    'TemporalPositionIdentifier',
    'EchoNumbers',
    'EchoTime',
    'RepetitionTime',
    'FlipAngle',
    'DiffusionBValue',
    'ImagePositionPatient',
    'ImageOrientationPatient',
    'PixelSpacing',
    'SliceThickness',
    'SpacingBetweenSlices',
    'Rows',
    'Columns',
    'NumberOfFrames',
    'SamplesPerPixel',
    'PhotometricInterpretation',
    'BitsAllocated',
    'BitsStored',
    'HighBit',
    'PixelRepresentation',
    'Manufacturer',
    'ManufacturerModelName',
    'MagneticFieldStrength',
    'SoftwareVersions',
)

DICOM_NAVIGATION_TAGS = (
    'Rows',
    'Columns',
    'NumberOfFrames',
    'InstanceNumber',
    'SeriesNumber',
    'EchoNumbers',
    'EchoTime',
    'InversionTime',
    'AcquisitionNumber',
    'TemporalPositionIdentifier',
    'ImagePositionPatient',
    'ImageOrientationPatient',
    'PixelSpacing',
    'SeriesDescription',
    'ProtocolName',
    'FlipAngle',
    'DiffusionBValue',
)


DICOM_VOLUME_FIELDS = (
    ('SeriesNumber', 'series_number', 'series'),
    ('EchoNumber', 'echo_number', 'echoes'),
    ('EchoTime', 'echo_time', 'echo times'),
    ('FlipAngle', 'flip_angle', 'flip angles'),
    ('DiffusionBValue', 'diffusion_b_value', 'b-values'),
    ('InversionTime', 'inversion_time', 'inversion times'),
    ('TemporalPositionIdentifier', 'temporal_position', 'time points'),
    ('AcquisitionNumber', 'acquisition_number', 'acquisitions'),
)

DICOM_BINARY_VRS = {'OB', 'OD', 'OF', 'OL', 'OV', 'OW', 'UN'}


def dicom_tag_text(tag):
    return f"({int(tag.group):04X},{int(tag.element):04X})"


def dicom_value_text(element):
    value = element.value
    if element.VR in DICOM_BINARY_VRS or isinstance(value, bytes):
        try:
            size = len(value)
        except TypeError:
            size = 0
        return f"<{size} bytes>"
    text = str(value)
    if len(text) > 2048:
        return text[:2045] + "..."
    return text


def _dicom_value_signature(element):
    value = element.value
    if element.VR in DICOM_BINARY_VRS or isinstance(value, bytes):
        try:
            payload = bytes(value)
        except (TypeError, ValueError):
            payload = str(value).encode('utf-8', errors='replace')
        return ('binary', len(payload), hashlib.sha256(payload).hexdigest())
    return (str(element.VR), str(value))


def _dicom_all_tag_entries(dataset, prefix=''):
    entries = []
    for element in dataset:
        if element.keyword == 'PixelData' or int(element.tag) == 0x7FE00010:
            continue
        tag_text = prefix + dicom_tag_text(element.tag)
        if element.VR == 'SQ':
            entries.append((
                tag_text,
                element.name,
                f"{len(element.value)} item(s)",
                ('sequence', len(element.value)),
            ))
            for item_index, item in enumerate(element.value):
                item_prefix = f"{tag_text}[{item_index + 1}] / "
                entries.extend(_dicom_all_tag_entries(item, item_prefix))
            continue
        entries.append((
            tag_text,
            element.name,
            dicom_value_text(element),
            _dicom_value_signature(element),
        ))
    return entries


def dicom_dataset_entries(dataset):
    entries = []
    file_meta = getattr(dataset, 'file_meta', None)
    if file_meta is not None:
        entries.extend(_dicom_all_tag_entries(file_meta, "File meta / "))
    entries.extend(_dicom_all_tag_entries(dataset))
    return entries


def curated_tag_rows(dataset):
    rows = []
    for keyword in DICOM_CURATED_TAGS:
        try:
            element = dataset.data_element(keyword)
        except Exception:
            element = None
        if element is None or element.keyword == 'PixelData':
            continue
        rows.append((dicom_tag_text(element.tag), element.name, dicom_value_text(element)))
    return rows


def compare_dicom_tags(paths, cancelled=None):
    import pydicom

    ordered_tags = []
    tag_names = {}
    signatures = {}
    presence_counts = {}
    readable_count = 0
    unreadable_count = 0

    for path in paths:
        if cancelled is not None and cancelled():
            return None
        try:
            dataset = pydicom.dcmread(path, stop_before_pixels=True)
            entries = dicom_dataset_entries(dataset)
        except Exception:
            unreadable_count += 1
            continue

        readable_count += 1
        seen_tags = set()
        for tag, name, _display_value, signature in entries:
            if tag not in tag_names:
                ordered_tags.append(tag)
                tag_names[tag] = name
            signatures.setdefault(tag, set()).add(signature)
            if tag not in seen_tags:
                presence_counts[tag] = presence_counts.get(tag, 0) + 1
                seen_tags.add(tag)

    varying_tags = [
        tag for tag in ordered_tags
        if len(signatures.get(tag, ())) > 1
        or presence_counts.get(tag, 0) < readable_count
    ]
    return {
        'varying_tags': varying_tags,
        'tag_names': tag_names,
        'readable_count': readable_count,
        'unreadable_count': unreadable_count,
        'total_count': len(paths),
    }


def find_dicom_files(directory_path):
    directory_path = Path(directory_path)
    return sorted(
        path for path in directory_path.rglob('*')
        if path.is_file() and path.suffix.lower() == '.dcm'
    )


def _dicom_plain_value(value):
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return f"<{len(value)} bytes>"
    if isinstance(value, np.generic):
        return value.item()
    try:
        values = list(value)
    except TypeError:
        return str(value)
    return [_dicom_plain_value(item) for item in values]


def dicom_header_record(path, directory_path=None, dataset=None):
    path = Path(path)
    relative_path = path.name
    if directory_path is not None:
        try:
            relative_path = str(path.relative_to(directory_path))
        except ValueError:
            pass

    record = {
        'path': str(path),
        'relative_path': relative_path,
        'navigable': False,
        'navigation_reason': 'DICOM geometry has not been mapped',
        'in_plane_dims': [],
        'fixed_indices': [],
    }

    try:
        if dataset is None:
            import pydicom
            dataset = pydicom.dcmread(
                path,
                stop_before_pixels=True,
                specific_tags=list(DICOM_NAVIGATION_TAGS),
            )
    except Exception as exc:
        record['navigation_reason'] = f"Unable to read DICOM header: {exc}"
        return record

    value_fields = {
        'rows': 'Rows',
        'columns': 'Columns',
        'number_of_frames': 'NumberOfFrames',
        'instance_number': 'InstanceNumber',
        'series_number': 'SeriesNumber',
        'echo_number': 'EchoNumbers',
        'echo_time': 'EchoTime',
        'acquisition_number': 'AcquisitionNumber',
        'temporal_position': 'TemporalPositionIdentifier',
        'inversion_time': 'InversionTime',
        'image_position_patient': 'ImagePositionPatient',
        'image_orientation_patient': 'ImageOrientationPatient',
        'pixel_spacing': 'PixelSpacing',
        'series_description': 'SeriesDescription',
        'protocol_name': 'ProtocolName',
        'flip_angle': 'FlipAngle',
        'diffusion_b_value': 'DiffusionBValue',
    }
    for field, keyword in value_fields.items():
        record[field] = _dicom_plain_value(getattr(dataset, keyword, None))
    return record


def single_dicom_file_record(path, dataset, data):
    record = dicom_header_record(path, dataset=dataset)
    if data.ndim == 2:
        record.update({
            'navigable': True,
            'navigation_reason': '',
            'in_plane_dims': [0, 1],
            'fixed_indices': [],
            'output_index': 0,
            'group_label': '',
        })
    else:
        record['navigation_reason'] = (
            'Multi-frame single-file DICOM navigation is not available'
        )
    return record


def _normalized_match_value(value):
    if isinstance(value, list):
        normalized = tuple(_normalized_match_value(item) for item in value)
        return normalized[0] if len(normalized) == 1 else normalized
    if value is None:
        return None
    try:
        return round(float(value), 6)
    except (TypeError, ValueError):
        return str(value).strip().casefold()


def _dicom_output_index(record, output_metadata):
    if len(output_metadata) == 1:
        return 0

    match_fields = (
        ('series_number', 'SeriesNumber', True),
        ('echo_number', 'EchoNumber', True),
        ('acquisition_number', 'AcquisitionNumber', True),
        ('flip_angle', 'FlipAngle', True),
        ('diffusion_b_value', 'DiffusionBValue', True),
        ('series_description', 'SeriesDescription', False),
        ('protocol_name', 'ProtocolName', False),
    )
    candidates = []
    for output_index, metadata in enumerate(output_metadata):
        matches = 0
        conflicts = 0
        for record_key, sidecar_key, strict in match_fields:
            source_value = _normalized_match_value(record.get(record_key))
            output_value = _normalized_match_value(metadata.get(sidecar_key))
            if source_value is None or output_value is None:
                continue
            if source_value == output_value:
                matches += 1
            elif strict:
                conflicts += 1
        if conflicts == 0 and matches > 0:
            candidates.append((matches, output_index))

    if not candidates:
        return None
    best_score = max(score for score, _ in candidates)
    best = [index for score, index in candidates if score == best_score]
    return best[0] if len(best) == 1 else None


def _lps_to_ras(vector):
    values = np.asarray(vector, dtype=float)
    return np.array([-values[0], -values[1], values[2]], dtype=float)


def _dominant_voxel_axis(vector, inverse_linear, spatial_ndim):
    voxel_vector = inverse_linear @ _lps_to_ras(vector)
    magnitudes = np.abs(voxel_vector[:spatial_ndim])
    axis = int(np.argmax(magnitudes))
    if magnitudes[axis] < 0.5:
        return None
    other_magnitude = float(np.sum(magnitudes) - magnitudes[axis])
    if other_magnitude > 0.1 * float(magnitudes[axis]):
        return None
    return axis


def _map_dicom_plane(record, output_shape, affine):
    frame_count = record.get('number_of_frames')
    if frame_count not in (None, '', 1, '1'):
        return None, 'Multi-frame DICOM files do not map to one ndarray plane'

    if len(output_shape) > 3:
        return None, 'The converted output has an unmapped higher-dimensional index'

    orientation = record.get('image_orientation_patient')
    position = record.get('image_position_patient')
    pixel_spacing = record.get('pixel_spacing')
    if not (
        isinstance(orientation, list) and len(orientation) >= 6
        and isinstance(position, list) and len(position) >= 3
        and isinstance(pixel_spacing, list) and len(pixel_spacing) >= 2
    ):
        return None, 'Required DICOM position, orientation, or pixel spacing is missing'

    try:
        affine = np.asarray(affine, dtype=float)
        if affine.shape != (4, 4):
            return None, 'Converted NIfTI affine is unavailable or invalid'
        inverse_affine = np.linalg.inv(affine)
        inverse_linear = np.linalg.inv(affine[:3, :3])
        orientation = np.asarray(orientation, dtype=float)
        row_spacing, column_spacing = (float(pixel_spacing[0]), float(pixel_spacing[1]))
    except (TypeError, ValueError, IndexError, np.linalg.LinAlgError):
        return None, 'DICOM or NIfTI geometry is invalid'

    spatial_ndim = min(3, len(output_shape))
    column_axis = _dominant_voxel_axis(
        orientation[:3] * column_spacing,
        inverse_linear,
        spatial_ndim,
    )
    row_axis = _dominant_voxel_axis(
        orientation[3:6] * row_spacing,
        inverse_linear,
        spatial_ndim,
    )
    if row_axis is None or column_axis is None or row_axis == column_axis:
        return None, 'DICOM plane is not aligned with the converted ndarray axes'

    rows = record.get('rows')
    columns = record.get('columns')
    try:
        if rows is not None and int(rows) != int(output_shape[row_axis]):
            return None, 'DICOM rows do not match the converted ndarray plane'
        if columns is not None and int(columns) != int(output_shape[column_axis]):
            return None, 'DICOM columns do not match the converted ndarray plane'
    except (TypeError, ValueError):
        return None, 'DICOM row or column metadata is invalid'

    in_plane_dims = [row_axis, column_axis]
    slice_axes = [axis for axis in range(spatial_ndim) if axis not in in_plane_dims]
    fixed_indices = []
    if slice_axes:
        if len(slice_axes) != 1:
            return None, 'DICOM plane does not identify one converted slice axis'
        slice_axis = slice_axes[0]
        world_position = np.append(_lps_to_ras(position[:3]), 1.0)
        voxel_position = inverse_affine @ world_position
        slice_position = float(voxel_position[slice_axis])
        slice_index = int(round(slice_position))
        if abs(slice_position - slice_index) > 0.25:
            return None, 'DICOM position does not land on a converted ndarray slice'
        if slice_index < 0 or slice_index >= int(output_shape[slice_axis]):
            return None, 'Mapped DICOM slice is outside the converted ndarray'
        fixed_indices.append([slice_axis, slice_index])

    return {
        'in_plane_dims': in_plane_dims,
        'fixed_indices': fixed_indices,
    }, ''


def _dicom_group_label(record, output_index, stacking_label, stacking_values):
    if stacking_label and stacking_values and output_index < len(stacking_values):
        return f"{stacking_label}: {stacking_values[output_index]}"
    description = record.get('series_description') or record.get('protocol_name')
    if description:
        return str(description)
    series_number = record.get('series_number')
    return f"series {series_number}" if series_number is not None else ''


def _uniform_output_values(records, output_count, record_key):
    values = []
    for output_index in range(output_count):
        output_values = {
            _normalized_match_value(record.get(record_key))
            for record in records
            if record.get('output_index') == output_index
            and record.get(record_key) is not None
        }
        if len(output_values) != 1:
            return None

        normalized_value = next(iter(output_values))
        values.append(next(
            record[record_key]
            for record in records
            if record.get('output_index') == output_index
            and _normalized_match_value(record.get(record_key)) == normalized_value
        ))
    return values


def describe_dicom_volume_stack(output_metadata, records):
    """Describe dcm2niix outputs from their source DICOM headers when possible."""
    output_count = len(output_metadata)
    if output_count < 2:
        return None, None, None

    for sidecar_key, record_key, label in DICOM_VOLUME_FIELDS:
        values = _uniform_output_values(records, output_count, record_key)
        if values is not None and len({
            _normalized_match_value(value) for value in values
        }) > 1:
            return label, sidecar_key, values

    for sidecar_key, _record_key, label in DICOM_VOLUME_FIELDS:
        values = [metadata.get(sidecar_key) for metadata in output_metadata]
        if any(value is not None for value in values) and len({
            _normalized_match_value(value) for value in values
        }) > 1:
            return label, sidecar_key, values

    names = [
        metadata.get('ProtocolName') or metadata.get('SeriesDescription')
        for metadata in output_metadata
    ]
    if any(name for name in names) and len(set(names)) > 1:
        return 'series', 'SeriesDescription', names

    return 'series', None, None


def set_dicom_record_group_labels(records, stacking_label, stacking_values):
    """Update DICOM record labels after the converted outputs are described."""
    for record in records:
        output_index = record.get('output_index')
        if not isinstance(output_index, int):
            continue
        record['group_label'] = _dicom_group_label(
            record,
            output_index,
            stacking_label,
            stacking_values,
        )


def build_dicom_file_records(
        dicom_files, directory_path, output_shapes, output_affines,
        output_metadata, final_shape, stacking_label, stacking_values):
    records = [
        dicom_header_record(path, directory_path=directory_path)
        for path in dicom_files
    ]
    stacked_output_dim = len(final_shape) - 1 if len(output_shapes) > 1 else None

    for record in records:
        if record.get('navigation_reason', '').startswith('Unable to read'):
            continue
        output_index = _dicom_output_index(record, output_metadata)
        if output_index is None:
            record['navigation_reason'] = 'DICOM file could not be matched to one converted output'
            continue

        mapping, reason = _map_dicom_plane(
            record,
            output_shapes[output_index],
            output_affines[output_index],
        )
        record['output_index'] = output_index
        record['group_label'] = _dicom_group_label(
            record,
            output_index,
            stacking_label,
            stacking_values,
        )
        if mapping is None:
            record['navigation_reason'] = reason
            continue

        fixed_indices = list(mapping['fixed_indices'])
        if stacked_output_dim is not None:
            fixed_indices.append([stacked_output_dim, output_index])
        mapped_dims = {int(dim) for dim, _ in fixed_indices}
        required_fixed_dims = set(range(len(final_shape))) - set(mapping['in_plane_dims'])
        if mapped_dims != required_fixed_dims:
            record['navigation_reason'] = 'Not every fixed ndarray dimension could be mapped'
            continue

        record.update({
            'navigable': True,
            'navigation_reason': '',
            'in_plane_dims': mapping['in_plane_dims'],
            'fixed_indices': fixed_indices,
        })

    def sort_key(record):
        if not record.get('navigable'):
            return (1, str(record.get('relative_path', '')).casefold())
        fixed_indices = {
            int(dim): int(index)
            for dim, index in record.get('fixed_indices', [])
        }
        output_index = int(record.get('output_index', 0))
        slice_indices = tuple(
            fixed_indices[dim]
            for dim in sorted(fixed_indices)
            if dim != stacked_output_dim
        )
        return (
            0,
            output_index,
            slice_indices,
            str(record.get('relative_path', '')).casefold(),
        )

    return sorted(records, key=sort_key)
