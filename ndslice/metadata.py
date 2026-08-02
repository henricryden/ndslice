import math
from pathlib import Path

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets


MASK_LABEL_MAX = 32

_SPATIAL_UNIT_LABELS = {
    'm': 'm',
    'mm': 'mm',
    'um': 'um',
}
_SPATIAL_UNIT_TO_MM = {
    'm': 1000.0,
    'mm': 1.0,
    'um': 0.001,
}
_CONSUMED_METADATA_KEYS = {
    'shape',
    'ndim',
    'dtype',
    'dim_labels',
    'voxel_spacing',
    'spatial_unit',
    'source_path',
    'detected_format',
    'dataset_path',
    'dicom_file_count',
    'dicom_files',
    'nifti_output_path',
    'sidecar_json_path',
    'converted_output_names',
    'stacked_dimension',
    'stacked_dimension_key',
    'stacked_dimension_values',
    'applied_scale_slope',
    'applied_scale_intercept',
    'applied_scale_axis',
    'applied_scale_transforms',
}


def plain_metadata_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.size <= 64:
            return plain_metadata_value(value.tolist())
        return f"<array shape={tuple(value.shape)} dtype={value.dtype}>"
    if isinstance(value, bytes):
        return f"<{len(value)} bytes>"
    if isinstance(value, dict):
        return {
            str(key): plain_metadata_value(item)
            for key, item in value.items()
            if str(key).replace(' ', '').lower() not in {
                'pixeldata', '(7fe0,0010)', '7fe00010'
            }
        }
    if isinstance(value, (list, tuple)):
        return [plain_metadata_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def clean_metadata(metadata):
    if not isinstance(metadata, dict):
        return {}
    return plain_metadata_value(metadata)


def _format_byte_size(byte_count):
    value = float(byte_count)
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{int(value)} {unit}" if unit == "B" else f"{value:.2f} {unit}"
        value /= 1024


def _metadata_value_text(value):
    if isinstance(value, list):
        return ", ".join("-" if item is None else str(item) for item in value)
    if value is None:
        return "-"
    return str(value)


def _format_statistic_value(value):
    return f"{float(value):,.6g}"


def _shape_text(shape):
    return " x ".join(str(size) for size in shape)


def _format_percentage(value, compact=False):
    if value == 0:
        return "0%"
    if not compact and value < 0.01:
        return f"{value:.4f}%"
    if value < 1:
        return f"{value:.2f}%"
    return f"{value:.1f}%"


def _friendly_format_name(value):
    names = {
        'dicom_directory': 'DICOM directory',
        'dicom_file': 'DICOM',
        'nifti': 'NIfTI',
        'numpy': 'NumPy',
        'text': 'Text',
        'rec': 'Philips REC',
        'cfl': 'BART CFL',
    }
    return names.get(str(value), str(value).replace('_', ' ').upper())


def _base_mask_label_counts(mask_data):
    counts = np.zeros(MASK_LABEL_MAX + 1, dtype=np.int64)
    if mask_data.size == 0:
        return counts
    return np.bincount(
        np.ravel(mask_data),
        minlength=MASK_LABEL_MAX + 1,
    )[:MASK_LABEL_MAX + 1]


def _mask_physical_statistics(
        data_shape, mask_data, voxel_spacing, has_voxel_spacing_metadata,
        spatial_unit):
    spatial_dims = [
        dim for dim, spacing in enumerate(voxel_spacing)
        if spacing is not None
    ]
    if not has_voxel_spacing_metadata or not spatial_dims:
        return None, "voxel spacing is unavailable"
    if len(spatial_dims) > 3:
        return None, "more than three dimensions have spatial spacing"

    varying_unspaced_dims = [
        dim for dim, spacing in enumerate(voxel_spacing)
        if spacing is None and mask_data.shape[dim] > 1
    ]
    if varying_unspaced_dims:
        dims = ", ".join(str(dim) for dim in varying_unspaced_dims)
        return None, f"mask varies along dimensions without spatial spacing: {dims}"

    counts = _base_mask_label_counts(mask_data)
    spatial_broadcast_factor = math.prod(
        data_shape[dim] // mask_data.shape[dim]
        for dim in spatial_dims
    )
    counts *= spatial_broadcast_factor

    return {
        'counts': counts,
        'dimensions': spatial_dims,
        'power': len(spatial_dims),
        'element_measure': math.prod(voxel_spacing[dim] for dim in spatial_dims),
        'spatial_unit': spatial_unit if spatial_unit in _SPATIAL_UNIT_LABELS else None,
    }, ''


def _format_physical_measure(value, power, spatial_unit, include_ml=True):
    unit = _SPATIAL_UNIT_LABELS.get(spatial_unit, "units")
    unit_text = unit if power == 1 else f"{unit}^{power}"
    text = f"{_format_statistic_value(value)} {unit_text}"
    factor_to_mm = _SPATIAL_UNIT_TO_MM.get(spatial_unit)
    if include_ml and factor_to_mm is not None and power == 3:
        millilitres = value * factor_to_mm ** 3 / 1000.0
        text += f" ({_format_statistic_value(millilitres)} mL)"
    return text


def _mask_metadata_model(
        data_shape, mask_data, mask_positive_labels, dim_labels,
        voxel_spacing, has_voxel_spacing_metadata, spatial_unit,
        label_colors):
    physical, unavailable_reason = _mask_physical_statistics(
        data_shape,
        mask_data,
        voxel_spacing,
        has_voxel_spacing_metadata,
        spatial_unit,
    )
    counts = physical['counts'] if physical is not None else _base_mask_label_counts(mask_data)
    total_count = int(np.sum(counts))
    labeled_count = int(np.sum(counts[1:]))
    coverage = 100 * labeled_count / total_count if total_count else 0.0

    broadcast_dims = [
        dim for dim, (mask_size, data_size) in enumerate(zip(mask_data.shape, data_shape))
        if mask_size == 1 and data_size > 1
    ]
    mask_shape = [
        size for dim, size in enumerate(mask_data.shape)
        if dim not in broadcast_dims
    ] or [1]

    power = physical['power'] if physical is not None else None
    count_header = {2: 'Pixels', 3: 'Voxels'}.get(power, 'Elements')
    summary_rows = [
        ('Shape', _shape_text(mask_shape)),
        ('Labels', str(len(mask_positive_labels))),
        (
            'Coverage',
            f"{labeled_count:,} / {total_count:,} "
            f"{count_header.lower()} ({_format_percentage(coverage)})",
        ),
    ]
    if broadcast_dims:
        broadcast_parts = []
        for dim in broadcast_dims:
            label = dim_labels[dim] or f"axis {dim}"
            broadcast_parts.append(f"{label} x{data_shape[dim]}")
        summary_rows.append(('Broadcast', ', '.join(broadcast_parts)))

    measure_header = None
    if physical is None:
        summary_rows.append(('Physical size', f"Unavailable ({unavailable_reason})"))
    else:
        element_measure = physical['element_measure']
        unit = physical['spatial_unit']
        measure_name = {1: 'Length', 2: 'Area', 3: 'Volume'}[power]
        element_name = {1: 'Step length', 2: 'Pixel area', 3: 'Voxel volume'}[power]
        summary_rows.append((
            element_name,
            _format_physical_measure(element_measure, power, unit, include_ml=False),
        ))
        summary_rows.append((
            f"Total {measure_name.lower()}",
            _format_physical_measure(labeled_count * element_measure, power, unit),
        ))
        unit_text = _SPATIAL_UNIT_LABELS.get(unit, "units")
        if power != 1:
            unit_text += f"^{power}"
        measure_header = f"{measure_name} ({unit_text})"

    labels = []
    for value in mask_positive_labels:
        label = int(value)
        label_count = int(counts[label])
        share = 100 * label_count / labeled_count if labeled_count else 0.0
        label_row = {
            'label': str(label),
            'count': f"{label_count:,}",
            'share': _format_percentage(share, compact=True),
            'color': label_colors[label],
        }
        if physical is not None:
            label_row['measure'] = _format_statistic_value(
                label_count * physical['element_measure']
            )
        labels.append(label_row)

    return {
        'summary_rows': summary_rows,
        'count_header': count_header,
        'measure_header': measure_header,
        'labels': labels,
    }


def _scaling_rows(metadata):
    slope = metadata.get('applied_scale_slope')
    intercept = metadata.get('applied_scale_intercept')
    if slope is not None and intercept is not None:
        try:
            scale_text = f"slope {float(slope):g}; intercept {float(intercept):g}"
        except (TypeError, ValueError):
            scale_text = f"slope {slope}; intercept {intercept}"
        return [('Scaling', scale_text)]

    transforms = metadata.get('applied_scale_transforms')
    if isinstance(transforms, list) and transforms:
        applied_count = sum(
            pair != [1.0, 0.0]
            for pair in transforms
            if isinstance(pair, list) and len(pair) == 2
        )
        if applied_count:
            return [('Scaling', f"varies across {len(transforms)} converted volumes")]
    return []


def build_metadata_model(
        data, metadata, dim_labels, voxel_spacing,
        has_voxel_spacing_metadata=False, mask_data=None,
        mask_positive_labels=(), label_colors=None):
    metadata = clean_metadata(metadata)
    spatial_unit = metadata.get('spatial_unit')
    model = {
        'array_rows': [
            ('Shape', _shape_text(data.shape)),
            ('Type', str(data.dtype)),
            ('Memory', _format_byte_size(data.nbytes)),
        ],
        'axis_rows': [],
    }

    unit_label = _SPATIAL_UNIT_LABELS.get(spatial_unit)
    for dim, size in enumerate(data.shape):
        spacing = voxel_spacing[dim]
        spacing_text = '-' if spacing is None else f"{spacing:g}"
        if spacing is not None and unit_label:
            spacing_text += f" {unit_label}"
        model['axis_rows'].append([
            str(dim),
            dim_labels[dim] or '-',
            f"{size:,}",
            spacing_text,
        ])

    source_rows = []
    detected_format = metadata.get('detected_format')
    if detected_format not in (None, ''):
        source_rows.append(('Format', _friendly_format_name(detected_format)))
    source_path = metadata.get('source_path')
    if source_path not in (None, ''):
        source_rows.append(('Source', _metadata_value_text(source_path)))
    dataset_path = metadata.get('dataset_path')
    if dataset_path not in (None, ''):
        source_rows.append(('Dataset', _metadata_value_text(dataset_path)))
    model['source_rows'] = source_rows

    conversion_rows = []
    dicom_count = metadata.get('dicom_file_count')
    output_names = metadata.get('converted_output_names')
    if dicom_count not in (None, ''):
        try:
            dicom_text = f"{int(dicom_count):,} files"
        except (TypeError, ValueError):
            dicom_text = f"{dicom_count} files"
        if isinstance(output_names, list) and output_names:
            volume_word = 'volume' if len(output_names) == 1 else 'volumes'
            dicom_text += f" to {len(output_names)} {volume_word}"
        conversion_rows.append(('DICOM', dicom_text))

    stacked_dimension = metadata.get('stacked_dimension')
    stacking_key = metadata.get('stacked_dimension_key')
    stacking_values = metadata.get('stacked_dimension_values')
    if stacked_dimension not in (None, ''):
        stack_text = str(stacked_dimension)
        if stacking_key not in (None, ''):
            stack_text += f"; {stacking_key}: {_metadata_value_text(stacking_values)}"
        conversion_rows.append(('Stack', stack_text))
    conversion_rows.extend(_scaling_rows(metadata))
    model['conversion_rows'] = conversion_rows

    if mask_data is None:
        model['mask'] = None
    else:
        model['mask'] = _mask_metadata_model(
            data.shape,
            mask_data,
            mask_positive_labels,
            dim_labels,
            voxel_spacing,
            has_voxel_spacing_metadata,
            spatial_unit,
            label_colors or {},
        )

    model['additional_metadata'] = {
        key: value
        for key, value in metadata.items()
        if key not in _CONSUMED_METADATA_KEYS
    }
    model['copy_text'] = metadata_copy_text(model)
    return model


def _append_additional_copy_lines(lines, key, value, indent=0):
    prefix = "  " * indent
    if isinstance(value, dict):
        lines.append(f"{prefix}{key}:")
        for child_key, child_value in value.items():
            _append_additional_copy_lines(lines, child_key, child_value, indent + 1)
    elif isinstance(value, list):
        lines.append(f"{prefix}{key}:")
        for index, item in enumerate(value):
            _append_additional_copy_lines(lines, f"[{index}]", item, indent + 1)
    else:
        lines.append(f"{prefix}{key}: {_metadata_value_text(value)}")


def metadata_copy_text(model):
    sections = []

    def add_rows(title, rows):
        if not rows:
            return
        sections.append(title.upper())
        sections.extend(f"{name}: {value}" for name, value in rows)

    add_rows('Array', model['array_rows'])
    add_rows('Source', model.get('source_rows', []))

    sections.append('AXES')
    for axis, label, size, spacing in model['axis_rows']:
        label_text = f" {label}" if label != '-' else ''
        spacing_text = f", {spacing}" if spacing != '-' else ''
        sections.append(f"{axis}{label_text}: {size}{spacing_text}")

    add_rows('Conversion', model.get('conversion_rows', []))

    mask = model.get('mask')
    if mask is not None:
        add_rows('Mask', mask['summary_rows'])
        header = ["Label", mask['count_header'], "% of mask"]
        if mask.get('measure_header'):
            header.append(mask['measure_header'])
        sections.append(" | ".join(header))
        for label in mask['labels']:
            row = [label['label'], label['count'], label['share']]
            if mask.get('measure_header'):
                row.append(label['measure'])
            sections.append(" | ".join(row))

    additional = model.get('additional_metadata', {})
    if additional:
        sections.append('ADDITIONAL METADATA')
        for key, value in additional.items():
            _append_additional_copy_lines(sections, key, value)

    return "\n".join(sections)


class MetadataDialog(QtWidgets.QDialog):
    def __init__(self, model, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Array information")
        self.resize(680, 560)
        self._model = model

        layout = QtWidgets.QVBoxLayout(self)
        tabs = QtWidgets.QTabWidget(self)
        tabs.addTab(self._create_overview_tab(), "Overview")
        if model.get('mask') is not None:
            tabs.addTab(self._create_mask_tab(), "Mask")
        if model.get('additional_metadata'):
            tabs.addTab(self._create_additional_tab(), "Additional metadata")
        layout.addWidget(tabs, 1)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Close,
            parent=self,
        )
        copy_button = buttons.addButton(
            "Copy",
            QtWidgets.QDialogButtonBox.ButtonRole.ActionRole,
        )
        copy_icon = QtGui.QIcon.fromTheme("edit-copy")
        if not copy_icon.isNull():
            copy_button.setIcon(copy_icon)
        copy_button.clicked.connect(self._copy_metadata)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @staticmethod
    def _section_label(text, parent):
        label = QtWidgets.QLabel(text, parent)
        font = label.font()
        font.setBold(True)
        label.setFont(font)
        return label

    @staticmethod
    def _add_form_rows(layout, rows, parent):
        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(5)
        for row, (name, value) in enumerate(rows):
            name_label = QtWidgets.QLabel(name, parent)
            value_label = QtWidgets.QLabel(str(value), parent)
            value_label.setTextInteractionFlags(
                QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
                | QtCore.Qt.TextInteractionFlag.TextSelectableByKeyboard
            )
            value_label.setWordWrap(True)
            grid.addWidget(name_label, row, 0, alignment=QtCore.Qt.AlignmentFlag.AlignTop)
            grid.addWidget(value_label, row, 1)
        grid.setColumnStretch(1, 1)
        layout.addLayout(grid)

    @staticmethod
    def _table(headers, rows, parent):
        table = QtWidgets.QTableWidget(len(rows), len(headers), parent)
        table.setHorizontalHeaderLabels(headers)
        table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        table.setAlternatingRowColors(True)
        table.setShowGrid(False)
        table.verticalHeader().setVisible(False)
        for row, values in enumerate(rows):
            for column, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(str(value))
                item.setToolTip(str(value))
                table.setItem(row, column, item)
        return table

    def _create_overview_tab(self):
        tab = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setSpacing(10)

        layout.addWidget(self._section_label("Array", tab))
        self._add_form_rows(layout, self._model['array_rows'], tab)

        source_rows = self._model.get('source_rows', [])
        if source_rows:
            layout.addWidget(self._section_label("Source", tab))
            self._add_form_rows(layout, source_rows, tab)

        layout.addWidget(self._section_label("Axes", tab))
        axes_table = self._table(
            ["Axis", "Label", "Size", "Spacing"],
            self._model['axis_rows'],
            tab,
        )
        axes_header = axes_table.horizontalHeader()
        axes_header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        axes_header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.Stretch)
        axes_header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        axes_header.setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(axes_table, 1)

        conversion_rows = self._model.get('conversion_rows', [])
        if conversion_rows:
            layout.addWidget(self._section_label("Conversion", tab))
            self._add_form_rows(layout, conversion_rows, tab)
        return tab

    def _create_mask_tab(self):
        mask = self._model['mask']
        tab = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setSpacing(10)

        layout.addWidget(self._section_label("Mask", tab))
        self._add_form_rows(layout, mask['summary_rows'], tab)
        layout.addWidget(self._section_label("Labels", tab))

        headers = ["Label", mask['count_header'], "% of mask"]
        if mask.get('measure_header'):
            headers.append(mask['measure_header'])
        row_values = []
        for label in mask['labels']:
            values = [label['label'], label['count'], label['share']]
            if mask.get('measure_header'):
                values.append(label['measure'])
            row_values.append(values)

        label_table = self._table(headers, row_values, tab)
        label_header = label_table.horizontalHeader()
        label_header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        label_header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.Stretch)
        label_header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        if mask.get('measure_header'):
            label_header.setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeMode.Stretch)
        for row, label in enumerate(mask['labels']):
            pixmap = QtGui.QPixmap(14, 14)
            pixmap.fill(QtGui.QColor(*label['color']))
            label_table.item(row, 0).setIcon(QtGui.QIcon(pixmap))
        layout.addWidget(label_table, 1)
        return tab

    def _create_additional_tab(self):
        tab = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout(tab)
        tree = QtWidgets.QTreeWidget(tab)
        tree.setColumnCount(2)
        tree.setHeaderLabels(["Key", "Value"])
        tree.setAlternatingRowColors(True)
        tree.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        tree.header().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        tree.header().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.Stretch)
        for key, value in self._model['additional_metadata'].items():
            self._add_tree_value(tree, str(key), value)
        layout.addWidget(tree)
        return tab

    @classmethod
    def _add_tree_value(cls, parent, key, value):
        if isinstance(value, dict):
            item = QtWidgets.QTreeWidgetItem(parent, [key, ""])
            for child_key, child_value in value.items():
                cls._add_tree_value(item, str(child_key), child_value)
            return
        if isinstance(value, list):
            item = QtWidgets.QTreeWidgetItem(parent, [key, f"{len(value)} item(s)"])
            for index, child_value in enumerate(value):
                cls._add_tree_value(item, f"[{index}]", child_value)
            return
        text = _metadata_value_text(value)
        item = QtWidgets.QTreeWidgetItem(parent, [key, text])
        item.setToolTip(1, text)

    def _copy_metadata(self):
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.clipboard().setText(self._model['copy_text'])
