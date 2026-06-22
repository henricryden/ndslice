import numpy as np
import pyqtgraph as pg
import pyqtgraph.Qt as Qt
from pyqtgraph.Qt import QtWidgets, QtGui
import os
import math
import platform
from enum import Enum
from pathlib import Path
from .imageview2d import ImageView2D
from .video_export import VideoExportWorker, VideoExportDialog, VideoExportSettingsDialog
import multiprocessing as mp
import warnings

try:
    from IPython import get_ipython
except ImportError:
    def get_ipython():
        return None

def symlog(data, C = 0):
    return np.sign(data) * np.log10( 1 + np.abs(data) / 10**C)

def asinh(data, linear_width = 1):
    return np.arcsinh( data / linear_width ) * linear_width

def getNumberOfDecimalPlaces(number):
    if isinstance(number, (int, np.integer)):
        return int(0)
    else:
        return int(max(1, (number.as_integer_ratio()[1]).bit_length()))

class Domain(Enum):
    INV_FOURIER=-1
    NATIVE=0
    FOURIER=1

QT_SIGNAL = getattr(Qt.QtCore, "Signal", None)
if QT_SIGNAL is None:
    QT_SIGNAL = getattr(Qt.QtCore, "pyqtSignal", None)
if QT_SIGNAL is None:
    raise AttributeError("Could not find Qt signal class: expected Signal or pyqtSignal")

class RangeSlider(QtWidgets.QWidget):
    """A minimal horizontal two-handle slider for inclusive integer ranges."""

    valuesChanged = QT_SIGNAL(int, int)

    def __init__(self, parent=None, minimum=0, maximum=0):
        super().__init__(parent)
        self._minimum = int(minimum)
        self._maximum = max(int(minimum), int(maximum))
        self._lower_value = self._minimum
        self._upper_value = self._maximum
        self._active_handle = None
        self.setMouseTracking(True)
        self.setMinimumHeight(30)

    def sizeHint(self):
        return Qt.QtCore.QSize(240, 30)

    def values(self):
        return self._lower_value, self._upper_value

    def setValues(self, lower_value, upper_value):
        lower_value = max(self._minimum, min(int(lower_value), self._maximum))
        upper_value = max(self._minimum, min(int(upper_value), self._maximum))
        if lower_value > upper_value:
            lower_value, upper_value = upper_value, lower_value

        if (lower_value, upper_value) == (self._lower_value, self._upper_value):
            return

        self._lower_value = lower_value
        self._upper_value = upper_value
        self.valuesChanged.emit(self._lower_value, self._upper_value)
        self.update()

    def paintEvent(self, event):
        del event
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        palette = self.palette()
        groove_rect = self._groove_rect()
        lower_center = self._handle_center(self._lower_value)
        upper_center = self._handle_center(self._upper_value)

        painter.setPen(Qt.QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(palette.midlight())
        painter.drawRoundedRect(groove_rect, 3, 3)

        selected_rect = Qt.QtCore.QRectF(
            lower_center.x(),
            groove_rect.top(),
            max(upper_center.x() - lower_center.x(), 1),
            groove_rect.height(),
        )
        painter.setBrush(palette.highlight())
        painter.drawRoundedRect(selected_rect, 3, 3)

        self._paint_handle(painter, lower_center, self._active_handle == 'lower')
        self._paint_handle(painter, upper_center, self._active_handle == 'upper')

    def mousePressEvent(self, event):
        if event.button() != Qt.QtCore.Qt.MouseButton.LeftButton:
            event.ignore()
            return

        point = self._event_point(event)
        self._active_handle = self._closest_handle(point)
        self._move_active_handle(point.x())
        event.accept()

    def mouseMoveEvent(self, event):
        point = self._event_point(event)
        if self._active_handle is not None:
            self._move_active_handle(point.x())
            event.accept()
            return

        self.setCursor(
            QtGui.QCursor(
                Qt.QtCore.Qt.CursorShape.SizeHorCursor if self._is_over_handle(point) else Qt.QtCore.Qt.CursorShape.ArrowCursor
            )
        )
        event.ignore()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.QtCore.Qt.MouseButton.LeftButton:
            self._active_handle = None
            self.update()
        event.accept()

    def leaveEvent(self, event):
        self.setCursor(QtGui.QCursor(Qt.QtCore.Qt.CursorShape.ArrowCursor))
        super().leaveEvent(event)

    def _event_point(self, event):
        if hasattr(event, 'position'):
            return event.position()
        if hasattr(event, 'localPos'):
            return event.localPos()
        return Qt.QtCore.QPointF(event.pos())

    def _handle_radius(self):
        return 8.0

    def _groove_rect(self):
        margin = self._handle_radius() + 4.0
        return Qt.QtCore.QRectF(
            margin,
            self.height() / 2.0 - 3.0,
            max(self.width() - 2.0 * margin, 1.0),
            6.0,
        )

    def _handle_center(self, value):
        groove_rect = self._groove_rect()
        span = max(self._maximum - self._minimum, 1)
        ratio = (value - self._minimum) / span
        return Qt.QtCore.QPointF(groove_rect.left() + groove_rect.width() * ratio, groove_rect.center().y())

    def _handle_rect(self, value):
        center = self._handle_center(value)
        radius = self._handle_radius()
        return Qt.QtCore.QRectF(center.x() - radius, center.y() - radius, radius * 2.0, radius * 2.0)

    def _paint_handle(self, painter, center, active):
        radius = self._handle_radius()
        palette = self.palette()
        rect = Qt.QtCore.QRectF(center.x() - radius, center.y() - radius, radius * 2.0, radius * 2.0)
        painter.setPen(QtGui.QPen(palette.shadow().color(), 1.0))
        painter.setBrush(palette.highlight() if active else palette.button())
        painter.drawEllipse(rect)

        inner_rect = rect.adjusted(3.0, 3.0, -3.0, -3.0)
        painter.setPen(Qt.QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(palette.base())
        painter.drawEllipse(inner_rect)

    def _closest_handle(self, point):
        lower_distance = abs(point.x() - self._handle_center(self._lower_value).x())
        upper_distance = abs(point.x() - self._handle_center(self._upper_value).x())
        return 'lower' if lower_distance <= upper_distance else 'upper'

    def _is_over_handle(self, point):
        expanded_lower = self._handle_rect(self._lower_value).adjusted(-4, -4, 4, 4)
        expanded_upper = self._handle_rect(self._upper_value).adjusted(-4, -4, 4, 4)
        return expanded_lower.contains(point) or expanded_upper.contains(point)

    def _value_from_position(self, position_x):
        groove_rect = self._groove_rect()
        clamped_x = min(max(position_x, groove_rect.left()), groove_rect.right())
        ratio = (clamped_x - groove_rect.left()) / max(groove_rect.width(), 1.0)
        return int(round(self._minimum + ratio * (self._maximum - self._minimum)))

    def _move_active_handle(self, position_x):
        value = self._value_from_position(position_x)
        if self._active_handle == 'lower':
            self.setValues(min(value, self._upper_value), self._upper_value)
        elif self._active_handle == 'upper':
            self.setValues(self._lower_value, max(value, self._lower_value))


class SaveRangeDialog(QtWidgets.QDialog):
    """Dialog for selecting an inclusive index range along each dimension."""

    def __init__(self, parent, data_shape):
        super().__init__(parent)
        self.setWindowTitle('Save as NumPy')
        self._controls = []
        self._data_shape = tuple(data_shape)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        content = QtWidgets.QWidget()
        content_layout = QtWidgets.QVBoxLayout(content)
        content_layout.setContentsMargins(4, 4, 4, 4)
        content_layout.setSpacing(10)

        for dim, size in enumerate(data_shape):
            max_index = max(0, size - 1)
            row_widget = QtWidgets.QWidget()
            row_layout = QtWidgets.QGridLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setHorizontalSpacing(8)
            row_layout.setVerticalSpacing(4)

            dimension_label = QtWidgets.QLabel(f'Dim {dim}')
            dimension_label.setStyleSheet('QLabel { font-size: 9pt; font-weight: bold; }')
            slider = RangeSlider(row_widget, 0, max_index)

            start_spinbox = QtWidgets.QSpinBox()
            end_spinbox = QtWidgets.QSpinBox()
            for spinbox in (start_spinbox, end_spinbox):
                spinbox.setRange(0, max_index)
                spinbox.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
                spinbox.setStyleSheet(NDSliceWindow.SPINBOX_STYLE)
                spinbox.setFixedWidth(70)

            end_spinbox.setValue(max_index)
            start_spinbox.valueChanged.connect(lambda value, end_box=end_spinbox: end_box.setMinimum(value))
            end_spinbox.valueChanged.connect(lambda value, start_box=start_spinbox: start_box.setMaximum(value))
            start_spinbox.valueChanged.connect(lambda value, slider_widget=slider, end_box=end_spinbox: slider_widget.setValues(value, end_box.value()))
            end_spinbox.valueChanged.connect(lambda value, slider_widget=slider, start_box=start_spinbox: slider_widget.setValues(start_box.value(), value))
            start_spinbox.valueChanged.connect(lambda _value: self._update_output_shape_label())
            end_spinbox.valueChanged.connect(lambda _value: self._update_output_shape_label())
            slider.valuesChanged.connect(lambda lower, upper, start_box=start_spinbox, end_box=end_spinbox: self._sync_spinboxes(start_box, end_box, lower, upper))

            slider.setValues(0, max_index)

            row_layout.addWidget(dimension_label, 0, 0)
            row_layout.addWidget(slider, 0, 1, 1, 4)
            row_layout.addWidget(QtWidgets.QLabel('start'), 1, 0)
            row_layout.addWidget(start_spinbox, 1, 1)
            row_layout.addItem(QtWidgets.QSpacerItem(24, 1, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum), 1, 2)
            row_layout.addWidget(QtWidgets.QLabel('end'), 1, 3)
            row_layout.addWidget(end_spinbox, 1, 4)
            row_layout.setColumnStretch(1, 0)
            row_layout.setColumnStretch(2, 1)
            row_layout.setColumnStretch(4, 0)
            content_layout.addWidget(row_widget)
            self._controls.append((slider, start_spinbox, end_spinbox))

        content_layout.addStretch(1)

        scroll_area.setWidget(content)
        scroll_area.setMaximumHeight(min(440, 84 * max(len(data_shape), 1)))
        layout.addWidget(scroll_area)

        footer_layout = QtWidgets.QHBoxLayout()
        footer_layout.setContentsMargins(0, 0, 0, 0)
        footer_layout.setSpacing(12)
        self._squeeze_checkbox = QtWidgets.QCheckBox('Squeeze singleton dimensions')
        self._squeeze_checkbox.setChecked(True)
        self._squeeze_checkbox.toggled.connect(self._update_output_shape_label)
        self._output_shape_label = QtWidgets.QLabel()
        self._output_shape_label.setStyleSheet('QLabel { font-size: 9pt; color: palette(windowText); }')
        footer_layout.addWidget(self._squeeze_checkbox)
        footer_layout.addStretch(1)
        footer_layout.addWidget(self._output_shape_label)
        layout.addLayout(footer_layout)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Save | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._update_output_shape_label()
        self.resize(640, min(560, 150 + 84 * len(data_shape)))

    def _sync_spinboxes(self, start_spinbox, end_spinbox, lower_value, upper_value):
        start_spinbox.blockSignals(True)
        end_spinbox.blockSignals(True)
        start_spinbox.setValue(lower_value)
        end_spinbox.setValue(upper_value)
        start_spinbox.blockSignals(False)
        end_spinbox.blockSignals(False)
        self._update_output_shape_label()

    def _selected_shape(self):
        selected_shape = [max(end.value() - start.value() + 1, 0) for _, start, end in self._controls]
        if self.should_squeeze():
            squeezed_shape = [size for size in selected_shape if size != 1]
            return squeezed_shape or [1]
        return selected_shape

    def _update_output_shape_label(self):
        self._output_shape_label.setText(f'Output shape: {self._selected_shape()}')

    def get_ranges(self):
        """Return Python slice bounds as (start, stop) tuples."""
        return [(start.value(), end.value() + 1) for _, start, end in self._controls]

    def should_squeeze(self):
        return self._squeeze_checkbox.isChecked()

class NDSliceWindow(QtWidgets.QMainWindow):
    # Styling constants — use pt (point) units so font sizes are DPI-independent
    DIMENSION_LABEL_STYLE = "QLabel { font-size: 9pt; padding: 1px; margin: 2px; }"
    FLIP_ICON_STYLE = "QLabel { font-size: 15pt; padding: 0px; margin: 0px; color: palette(text); }"
    BUTTON_STYLE = "QPushButton { font-size: 9pt; padding: 2px; margin: 2px; } QPushButton:disabled { color: palette(mid); }"
    SPINBOX_STYLE = "QSpinBox { font-size: 9pt; } QSpinBox:disabled { color: palette(mid); }"
    RADIO_BUTTON_STYLE = "QRadioButton { font-size: 9pt; }"
    GROUPBOX_BASE_STYLE = "QGroupBox { font-size: 9pt; font-weight: bold; border: 1px solid palette(mid); border-radius: 3px; margin-top: 1.4ex; padding-top: 3pt; } QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 3px; }"
    
    @staticmethod
    def _set_emoji_font(widget):
        if platform.system() == 'Darwin':
            font = widget.font()
            font.setFamily('Apple Color Emoji')
            widget.setFont(font)

    def __init__(self, data, complex_dim=None, filepath=None, dataset_path=None, selector_class_name=None):
        super(NDSliceWindow, self).__init__()
        self.resize(800,800)

        self.data = data
        self.singleton = [e == 1 for e in list(data.shape)]
        self.selected_indices = []
        self.channel = None
        self.scale = None
        self._force_autolevel = False
        self._filepath = filepath
        self._dataset_path = dataset_path
        self._selector_class_name = selector_class_name
        
        self.axis_flipped = [False] * data.ndim  # Track flip state so that one can toggle dims and come back to the same flip state
        
        # If data is real-valued and has size-2 dimensions, ndslice can combine them as complex (ISMRMD uses this for real/imag parts)
        if np.iscomplexobj(data):
            self.can_combine_as_complex = [False] * data.ndim
        else:
            self.can_combine_as_complex = [data.shape[i] == 2 for i in range(data.ndim)]
        self.combined_as_complex = [False] * data.ndim
        
        # Store complex_dim for later use (after widgets are created)
        self._initial_complex_dim = complex_dim
        
        for dim in range(0,data.ndim):
            if self.singleton[dim] is False and len(self.selected_indices) < 2:
                self.selected_indices.append(dim)
        # For 1D arrays, we only need one dimension; for 2D+, ensure we have two
        if len(self.selected_indices) < 2 and data.ndim >= 2:
            self.selected_indices = [0, 1]
        elif len(self.selected_indices) == 0:
            # Edge case: if all dimensions are singleton, pick first one
            self.selected_indices = [0]
        
        # Line plot mode uses a single selected dimension
        self.line_plot_dimension = 0  # Default to first non-singleton dimension
        for dim in range(data.ndim):
            if not self.singleton[dim]:
                self.line_plot_dimension = dim
                break
                
        self.domain = [Domain.NATIVE for _ in range(data.ndim)]
        self.widgets = {
            'buttons': {
                'primary': [QtWidgets.QPushButton(str(i), checkable=True) for i in range(data.ndim)],
                'secondary': [QtWidgets.QPushButton(str(i), checkable=True) for i in range(data.ndim)],
                'channel': {
                    'real': QtWidgets.QRadioButton('real', enabled=np.iscomplexobj(self.data)),
                    'imag': QtWidgets.QRadioButton('imag', enabled=np.iscomplexobj(self.data)),
                    'abs': QtWidgets.QRadioButton('abs', enabled=np.iscomplexobj(self.data)),
                    'angle': QtWidgets.QRadioButton('angle', enabled=np.iscomplexobj(self.data)),
                },
                'processing': {
                    #'log': QtWidgets.QRadioButton('log', checkable=True),
                    'linear': QtWidgets.QRadioButton('linear', checkable=True, checked=True),
                    'symlog': QtWidgets.QRadioButton('symlog', checkable=True),
                    #'asinh': QtWidgets.QRadioButton('asinh', checkable=True)
                    
                },
                'display': {
                    'square_pixels': QtWidgets.QRadioButton('Square pixels', checkable=True, checked=True),
                    'square_fov': QtWidgets.QRadioButton('Square FOV', checkable=True),
                    'fit': QtWidgets.QRadioButton('Fit', checkable=True)
                }
            },
            'labels': {
                'dims': [QtWidgets.QLabel('[' + str(data.shape[i]) + ']', alignment=Qt.QtCore.Qt.AlignmentFlag.AlignCenter) for i in range(data.ndim)],
                'flip': [QtWidgets.QLabel('', alignment=Qt.QtCore.Qt.AlignmentFlag.AlignCenter) for i in range(data.ndim)],
                'complex': [QtWidgets.QLabel('', alignment=Qt.QtCore.Qt.AlignmentFlag.AlignCenter) for i in range(data.ndim)],
                'primary': QtWidgets.QLabel('Y'),
                'secondary': QtWidgets.QLabel('X'),
                'slice': QtWidgets.QLabel('Slice'),
                'dimensions': QtWidgets.QLabel('Dimensions'),
                'pixelValue': QtWidgets.QLabel(''),
                'arrayInfo': QtWidgets.QLabel('')
            },
            'spins': {
                'slice_indices': [QtWidgets.QSpinBox(minimum=0, maximum=data.shape[i]-1) for i in range(data.ndim)]
            }
        }
        
        # Create a button group for the channel radio buttons
        self.channel_button_group = QtWidgets.QButtonGroup()
        self.channel_button_group.addButton(self.widgets['buttons']['channel']['real'])
        self.channel_button_group.addButton(self.widgets['buttons']['channel']['imag'])
        self.channel_button_group.addButton(self.widgets['buttons']['channel']['abs'])
        self.channel_button_group.addButton(self.widgets['buttons']['channel']['angle'])
        self._update_channel_controls()
            
        self.scale_button_group = QtWidgets.QButtonGroup()
        #self.scale_button_group.addButton(self.widgets['buttons']['processing']['log'])
        self.scale_button_group.addButton(self.widgets['buttons']['processing']['linear'])
        self.scale_button_group.addButton(self.widgets['buttons']['processing']['symlog'])
        #self.scale_button_group.addButton(self.widgets['buttons']['processing']['asinh'])
        
        
        self.display_button_group = QtWidgets.QButtonGroup()
        self.display_button_group.addButton(self.widgets['buttons']['display']['square_pixels'])
        self.display_button_group.addButton(self.widgets['buttons']['display']['square_fov'])
        self.display_button_group.addButton(self.widgets['buttons']['display']['fit'])
        
        self.layouts = {
            'main': QtWidgets.QVBoxLayout(),
            'top': QtWidgets.QVBoxLayout(),
            'topUp': QtWidgets.QHBoxLayout(),
            'topDown': QtWidgets.QHBoxLayout(),
            'bot': QtWidgets.QHBoxLayout(),
            'botLeft': QtWidgets.QVBoxLayout(),
            'botRight': QtWidgets.QVBoxLayout(),
            'dims': QtWidgets.QHBoxLayout(),
            'primary': QtWidgets.QHBoxLayout(),
            'secondary': QtWidgets.QHBoxLayout(),
            'slice': QtWidgets.QHBoxLayout(),
            'hover': QtWidgets.QHBoxLayout()
        }
        
        for i, label in enumerate(self.widgets['labels']['dims']):
            label.mousePressEvent = lambda event, i=i, l=label: self.dimClicked(event, l, i)
            label.setCursor(QtGui.QCursor(Qt.QtCore.Qt.CursorShape.PointingHandCursor))
            label.setToolTip(f"Apply centered FFT along dim {i}")

        
        # Set up flip labels with click handlers
        for i, flip_label in enumerate(self.widgets['labels']['flip']):
            flip_label.mousePressEvent = lambda event, i=i: self.flipAxisClicked(event, i)
            flip_label.setStyleSheet(self.FLIP_ICON_STYLE)
            flip_label.setAlignment(Qt.QtCore.Qt.AlignmentFlag.AlignLeft | Qt.QtCore.Qt.AlignmentFlag.AlignVCenter)
            self._set_emoji_font(flip_label)
        
        # Set up complex indicator labels with click handlers
        for i, complex_label in enumerate(self.widgets['labels']['complex']):
            complex_label.mousePressEvent = lambda event, i=i: self.complexOrRealClicked(event, i)
            complex_label.setStyleSheet(self.FLIP_ICON_STYLE)
            complex_label.setAlignment(Qt.QtCore.Qt.AlignmentFlag.AlignRight | Qt.QtCore.Qt.AlignmentFlag.AlignVCenter)
            self._set_emoji_font(complex_label)
        
        # Apply compact styling to dimension control widgets
        for label in self.widgets['labels']['dims']:
            label.setStyleSheet(self.DIMENSION_LABEL_STYLE)
            label.setMinimumHeight(24)
            label.setMinimumWidth(30)
        
        for btn in self.widgets['buttons']['primary'] + self.widgets['buttons']['secondary']:
            btn.setStyleSheet(self.BUTTON_STYLE)
            
        for spin in self.widgets['spins']['slice_indices']:
            spin.setStyleSheet(self.SPINBOX_STYLE)
        
        # Set minimal spacing for all control layouts
        self.layouts['dims'].setSpacing(2)  # Extra tight for dimensions row
        self.layouts['dims'].setContentsMargins(0, 0, 0, 0)  # Remove all margins
        self.layouts['primary'].setSpacing(2)  # Tighter spacing
        self.layouts['primary'].setContentsMargins(0, 1, 0, 1)  # Minimal margins
        self.layouts['secondary'].setSpacing(2)  # Tighter spacing
        self.layouts['secondary'].setContentsMargins(0, 1, 0, 1)  # Minimal margins
        self.layouts['slice'].setSpacing(2)  # Tighter spacing
        self.layouts['slice'].setContentsMargins(0, 1, 0, 1)  # Minimal margins
        
        # For each dimension, create a QVBoxLayout with flip icon + dim label above primary button
        dim_containers = []
        for i in range(data.ndim):
            # Create a container widget for this dimension column
            container = QtWidgets.QWidget()
            container_layout = QtWidgets.QVBoxLayout()
            container_layout.setSpacing(0)
            container_layout.setContentsMargins(0, 0, 0, 0)
            
            # Create horizontal layout for flip icon + dimension label
            label_row = QtWidgets.QWidget()
            label_layout = QtWidgets.QHBoxLayout()
            label_layout.setSpacing(0)
            label_layout.setContentsMargins(0, 0, 0, 0)
            
            # Add flip icon (left-aligned)
            label_layout.addWidget(self.widgets['labels']['flip'][i])
            # Add dimension label (centered, takes remaining space)
            self.widgets['labels']['dims'][i].setAlignment(Qt.QtCore.Qt.AlignmentFlag.AlignCenter)
            label_layout.addWidget(self.widgets['labels']['dims'][i], 1)
            # Add complex indicator (ℝ/ℂ)
            label_layout.addWidget(self.widgets['labels']['complex'][i])
            
            label_row.setLayout(label_layout)
            container_layout.addWidget(label_row)
            
            container.setLayout(container_layout)
            dim_containers.append(container)
            self.layouts['dims'].addWidget(container)
        
        # Store containers for later access
        self.dim_containers = dim_containers
        
        # Add all buttons and spinboxes to their vertical containers
        for i in range(data.ndim):
            w = self.widgets['buttons']['primary'][i]
            self.dim_containers[i].layout().addWidget(w)
            w.clicked.connect(lambda checked, i=i : self.changedIndex(checked, 0, i))
            
            w = self.widgets['buttons']['secondary'][i]
            self.dim_containers[i].layout().addWidget(w)
            w.clicked.connect(lambda checked, i=i: self.changedIndex(checked, 1, i))
            
            w = self.widgets['spins']['slice_indices'][i]
            self.dim_containers[i].layout().addWidget(w)
            w.valueChanged.connect(self.update)
        
        self._setup_export_context_menus()
        self._save_shortcut = QtGui.QShortcut(QtGui.QKeySequence.StandardKey.Save, self)
        self._save_shortcut.activated.connect(self._save_current_numpy_file)
        
        # Create a single compact control panel with all radio buttons
        controls_widget = QtWidgets.QWidget()
        controls_layout = QtWidgets.QVBoxLayout()
        
        # Channel controls - horizontal layout to save space
        channel_group = QtWidgets.QGroupBox("Channel")
        channel_group.setStyleSheet(self.GROUPBOX_BASE_STYLE)
        channel_layout = QtWidgets.QHBoxLayout()
        channel_layout.setSpacing(5)
        channel_layout.setContentsMargins(3, 3, 3, 3)
        
        buttons = list(self.widgets['buttons']['channel'].values())
        for btn in buttons:
            btn.setStyleSheet(self.RADIO_BUTTON_STYLE)
            self.channel_button_group.addButton(btn)
            channel_layout.addWidget(btn)
            btn.clicked.connect(self.update)
        
        channel_group.setLayout(channel_layout)
        controls_layout.addWidget(channel_group)
        
        # Processing controls - horizontal layout
        processing_group = QtWidgets.QGroupBox("Scale")
        processing_group.setStyleSheet(self.GROUPBOX_BASE_STYLE)
        processing_layout = QtWidgets.QHBoxLayout()
        processing_layout.setSpacing(5)
        processing_layout.setContentsMargins(3, 3, 3, 3)
        
        proc_buttons = list(self.widgets['buttons']['processing'].values())
        for btn in proc_buttons:
            btn.setStyleSheet(self.RADIO_BUTTON_STYLE)
            processing_layout.addWidget(btn)
            # When a processing button is pressed while already active, force auto-level
            btn.pressed.connect(lambda b=btn: self._processing_pressed(b))
            btn.clicked.connect(self.update)
        
        processing_group.setLayout(processing_layout)
        controls_layout.addWidget(processing_group)
        
        # Display controls - horizontal layout
        self.display_group = QtWidgets.QGroupBox("Display")
        self.display_group.setStyleSheet(self.GROUPBOX_BASE_STYLE)
        display_layout = QtWidgets.QHBoxLayout()
        display_layout.setSpacing(5)
        display_layout.setContentsMargins(3, 3, 3, 3)
        
        disp_buttons = list(self.widgets['buttons']['display'].values())
        for btn in disp_buttons:
            btn.setStyleSheet(self.RADIO_BUTTON_STYLE)
            display_layout.addWidget(btn)
            btn.clicked.connect(self.update_display_mode)
        
        # Optional: tooltip hints
        self.widgets['buttons']['display']['square_pixels'].setToolTip('Lock to 1:1 pixel aspect')
        self.widgets['buttons']['display']['square_fov'].setToolTip('Lock aspect to image width/height ratio')
        self.widgets['buttons']['display']['fit'].setToolTip('Always fit entire image in viewport')
        
        self.display_group.setLayout(display_layout)
        controls_layout.addWidget(self.display_group)
        
        # Add stretch to push everything to the top
        controls_layout.addStretch()
        
        controls_widget.setLayout(controls_layout)
        self.layouts['botRight'].addWidget(controls_widget)
        
        # Create tab widget for switching between image and line plot views
        self.tab_widget = QtWidgets.QTabWidget()
        
        # Create image view tab
        self.image_tab = QtWidgets.QWidget()
        self.image_tab_layout = QtWidgets.QVBoxLayout()
        
        self.img_view = ImageView2D()
        self.image_tab_layout.addWidget(self.img_view)
        self.image_tab.setLayout(self.image_tab_layout)
        self.img_view.getView().scene().sigMouseMoved.connect(lambda pos: self.getPixel(pos))
        
        # Connect to view range changes to update aspect ratio in fit mode
        self.img_view.getView().sigRangeChanged.connect(self._on_view_range_changed)
        
        # Create line plot tab
        self.plot_tab = QtWidgets.QWidget()
        self.plot_tab_layout = QtWidgets.QVBoxLayout()
        
        class AxisConstrainedViewBox(pg.ViewBox):
            def __init__(self, owner, *a, **k):
                super().__init__(*a, **k)
                self._owner = owner
            def wheelEvent(self, ev):
                # Only apply custom behavior in line plot mode
                if self._owner is None or not self._owner.is_line_plot_mode():
                    return super().wheelEvent(ev)
                modifiers = ev.modifiers()
                ctrl = modifiers & Qt.QtCore.Qt.KeyboardModifier.ControlModifier
                shift = modifiers & Qt.QtCore.Qt.KeyboardModifier.ShiftModifier
                # Angle delta: use y for typical vertical wheel (Qt6: angleDelta)
                delta = ev.delta() if hasattr(ev, 'delta') else ev.angleDelta().y()
                if delta == 0:
                    return
                # Base scaling factor (slightly super-linear for smoother feel)
                step = 1.0015 ** abs(delta)
                if delta < 0:
                    scale_factor = step  # zoom out
                else:
                    scale_factor = 1.0 / step  # zoom in
                # Current ranges
                (x0, x1), (y0, y1) = self.viewRange()
                xspan = (x1 - x0)
                yspan = (y1 - y0)
                # Determine mouse position in data coords for cursor-centered zoom
                try:
                    if hasattr(ev, 'scenePos'):
                        scene_pos = ev.scenePos()
                    elif hasattr(ev, 'scenePosition'):
                        scene_pos = ev.scenePosition()
                    else:
                        scene_pos = ev.pos()  # Fallback; may be in local coords
                    anchor_pt = self.mapSceneToView(scene_pos)
                    ax = anchor_pt.x()
                    ay = anchor_pt.y()
                except Exception:
                    # Fallback to center if mapping failed
                    ax = (x0 + x1) * 0.5
                    ay = (y0 + y1) * 0.5
                # Clamp anchor inside current view range (prevents wild jumps)
                if not (x0 <= ax <= x1):
                    ax = (x0 + x1) * 0.5
                if not (y0 <= ay <= y1):
                    ay = (y0 + y1) * 0.5
                # Fractions of anchor within current span
                fx = 0 if xspan == 0 else (ax - x0) / max(1e-12, xspan)
                fy = 0 if yspan == 0 else (ay - y0) / max(1e-12, yspan)
                # Apply axis-constrained scaling around the cursor
                if ctrl and not shift:
                    # X only zoom around cursor
                    new_xspan = xspan * scale_factor
                    x0n = ax - fx * new_xspan
                    x1n = x0n + new_xspan
                    self.setXRange(x0n, x1n, padding=0)
                elif shift and not ctrl:
                    # Y only zoom around cursor
                    new_yspan = yspan * scale_factor
                    y0n = ay - fy * new_yspan
                    y1n = y0n + new_yspan
                    self.setYRange(y0n, y1n, padding=0)
                else:
                    # Both axes zoom around cursor
                    new_xspan = xspan * scale_factor
                    new_yspan = yspan * scale_factor
                    x0n = ax - fx * new_xspan
                    x1n = x0n + new_xspan
                    y0n = ay - fy * new_yspan
                    y1n = y0n + new_yspan
                    self.setXRange(x0n, x1n, padding=0)
                    self.setYRange(y0n, y1n, padding=0)
                ev.accept()
            
        self.plot_widget = pg.PlotWidget(viewBox=AxisConstrainedViewBox(self))
        self.plot_tab_layout.addWidget(self.plot_widget)
        self.plot_tab.setLayout(self.plot_tab_layout)
        # Enable antialiasing globally for nicer lines
        pg.setConfigOptions(antialias=True)
        self.current_line_data = None  # store latest 1D line data for hover
        # Vertical crosshair line (hidden until hover)
        self.plot_crosshair = pg.InfiniteLine(angle=90, movable=False,
                                              pen=pg.mkPen((150, 0, 0, 180), width=2))
        self.plot_crosshair.setVisible(False)
        self.plot_widget.addItem(self.plot_crosshair)
        # Connect scene mouse move for hover on plot
        self.plot_widget.scene().sigMouseMoved.connect(self._on_plot_hover)
        # Dynamic line thickness parameters
        self.line_curve = None
        self.line_base_range = None  # (x_span) captured after first plot
        self.line_base_pen_width = 3.0
        self.line_min_pen = 2.0
        self.line_max_pen = 6.0
        self.line_color = (50, 100, 200)
        # Listen to range changes to adapt thickness
        self.plot_widget.getViewBox().sigRangeChanged.connect(self._on_plot_range_changed)
        
        # Add tabs to tab widget
        self.tab_widget.addTab(self.image_tab, "Image View")
        self.tab_widget.addTab(self.plot_tab, "Line Plot")
        
        # Track plot style (line vs bar)
        self.plot_style = 'line'  # 'line' or 'bar'
        
        if data.ndim == 1:
            self.tab_widget.setTabEnabled(0, False)  # Disable Image View tab
            self.tab_widget.setCurrentIndex(1)  # Set line plot as default
        
        # Connect tab change handler
        self.tab_widget.currentChanged.connect(self.on_tab_changed)
        
        self.tab_widget.tabBar().installEventFilter(self)
        
        # Add tab widget to the main layout
        self.layouts['topDown'].addWidget(self.tab_widget)

        # Reload / file-changed button (⟳ by default, ⚠️ when file changes on disk)
        self._reload_btn = QtWidgets.QPushButton("⟳")
        self._reload_btn.setStyleSheet("QPushButton { font-size: 18pt; padding: 1px 2px; margin: 0px; border: none; background: transparent; }")
        self._reload_btn.setToolTip("Reload file")
        self._reload_btn.setFlat(True)
        self._reload_btn.setFixedSize(28, 20)
        self._reload_btn.clicked.connect(self._reload_file)
        self._reload_btn.setVisible(filepath is not None)
        self._set_emoji_font(self._reload_btn)
        self.layouts['topUp'].addWidget(self._reload_btn)
        self.layouts['topUp'].addWidget(self.widgets['labels']['pixelValue'])
        self.layouts['topUp'].addWidget(self.widgets['labels']['arrayInfo'])

        self.layouts['botLeft'].addLayout(self.layouts['dims'])
        
        
        # Create container widgets for proper alignment
        left_container = QtWidgets.QWidget()
        left_container.setLayout(self.layouts['botLeft'])
        
        right_container = QtWidgets.QWidget()
        right_container.setLayout(self.layouts['botRight'])
        
        # Set both containers to expand vertically the same way
        left_container.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        right_container.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        
        self.layouts['top'].addLayout(self.layouts['topUp'],1)
        self.layouts['top'].addLayout(self.layouts['topDown'],20)  # Give viewport much more space
        
        self.layouts['bot'].addWidget(left_container, 12)
        self.layouts['bot'].addWidget(right_container, 1)

        # Give the viewport (top) much more weight so it expands when window grows
        self.layouts['main'].addLayout(self.layouts['top'], 10)  # Viewport gets most space
        self.layouts['main'].addLayout(self.layouts['bot'], 1)   # Controls get minimal fixed space
        tmp = QtWidgets.QWidget()
        tmp.setLayout(self.layouts['main'])
        self.setCentralWidget(tmp)
        
        # Initialize complex indicators for size-2 real dimensions
        self.update_complex_indicators()
        
        if complex_dim is not None: # user requested combining as complex
            if complex_dim < 0 or complex_dim >= data.ndim:
                print(f"Warning: complex_dim={complex_dim} is out of range for {data.ndim}D array. Ignoring.")
            elif np.iscomplexobj(data):
                print(f"Warning: Data is already complex. Ignoring complex_dim={complex_dim}.")
            elif data.shape[complex_dim] != 2:
                print(f"Warning: Dimension {complex_dim} has shape {data.shape[complex_dim]}, not 2. Cannot combine as complex. Ignoring.")
            else:
                self.combineAsComplex(complex_dim) # valid
        
        # Initialize dimension controls based on data dimensions
        if len(self.selected_indices) >= 1:
            self.changedIndex(True, 0, self.selected_indices[0], update=False)
        if len(self.selected_indices) >= 2:
            self.changedIndex(True, 1, self.selected_indices[1], update=False)
        self.update_dimension_controls()  # Initialize dimension controls properly
        self.update()
        self.show()

        # Set up file watcher if a filepath was provided (QFileSystemWatcher uses
        # OS-native events: inotify on Linux, FSEvents on macOS, ReadDirectoryChanges on Windows)
        self._file_watcher = None
        if filepath is not None:
            self._file_watcher = Qt.QtCore.QFileSystemWatcher([str(filepath)])
            self._file_watcher.fileChanged.connect(self._on_file_changed)




    
    def dimClicked(self, event, label, dim):
        if self.singleton[dim]:
            return
    
        p = QtGui.QPalette()
        
        # If already transformed, any click returns to native
        if self.domain[dim] == Domain.FOURIER:
            # From FFT domain, go back to native (undo)
            self.domain[dim] = Domain.NATIVE
            p.setColor(QtGui.QPalette.ColorRole.WindowText, QtGui.QColor('black'))
            label.setStyleSheet("font-weight: normal;")
            self._apply_ifft(dim)  # Undo the FFT by applying IFFT
        elif self.domain[dim] == Domain.INV_FOURIER:
            # From IFFT domain, go back to native (undo)
            self.domain[dim] = Domain.NATIVE
            p.setColor(QtGui.QPalette.ColorRole.WindowText, QtGui.QColor('black'))
            label.setStyleSheet("font-weight: normal;")
            self._apply_fft(dim)  # Undo the IFFT by applying FFT
        elif event.button() == Qt.QtCore.Qt.MouseButton.RightButton:
            # Right click from native: apply IFFT
            self.domain[dim] = Domain.INV_FOURIER
            p.setColor(QtGui.QPalette.ColorRole.WindowText, QtGui.QColor('green'))
            label.setStyleSheet("font-weight: bold; color: green;")
            self._apply_ifft(dim)
        else:
            # Left click from native: apply FFT
            self.domain[dim] = Domain.FOURIER
            p.setColor(QtGui.QPalette.ColorRole.WindowText, QtGui.QColor('blue'))
            label.setStyleSheet("font-weight: bold; color: blue;")
            self._apply_fft(dim)

        label.setPalette(p)
        self.update_image_view()
        self.update_line_plot()
        
    def _apply_fft(self, dim):
        """Apply forward FFT along specified dimension"""
        self.data = np.fft.ifftshift(np.fft.ifft(np.fft.fftshift(self.data), axis=dim, norm='ortho'))
        self._update_channel_controls()
        
    def _apply_ifft(self, dim):
        """Apply inverse FFT along specified dimension"""
        self.data = np.fft.ifftshift(np.fft.fft(np.fft.fftshift(self.data), axis=dim, norm='ortho'))
        self._update_channel_controls()
    
    def flipAxisClicked(self, event, dim):
        """Handle click on flip axis icon"""
        # Only respond if this dimension is currently selected
        if dim not in self.selected_indices:
            return
        
        # Toggle flip
        self.axis_flipped[dim] = not self.axis_flipped[dim]
        
        self.update_flip_icons()
        self.apply_axis_flips()
        
    def update_flip_icons(self):
        for i, flip_label in enumerate(self.widgets['labels']['flip']):
            if i in self.selected_indices:
                # In line plot mode, only show horizontal flip icon for the plot dimension
                if self.is_line_plot_mode():
                    if i == self.line_plot_dimension:
                        flip_label.setCursor(QtGui.QCursor(Qt.QtCore.Qt.CursorShape.SizeHorCursor))
                        flip_label.setToolTip("Flip X axis")
                        if self.axis_flipped[i]:
                            flip_label.setText('⬅️')    
                        else:
                            flip_label.setText('➡️')
                    else:
                        flip_label.setText('')  # Hide flip icons for non-plot dimensions
                        flip_label.setToolTip('')
                # In image view mode, show vertical flip for primary, horizontal for secondary
                elif i == self.selected_indices[0]:
                    flip_label.setCursor(QtGui.QCursor(Qt.QtCore.Qt.CursorShape.SizeVerCursor))
                    flip_label.setToolTip("Flip Y")
                    if self.axis_flipped[i]:
                        flip_label.setText('⬇️')
                    else:
                        flip_label.setText('⬆️')
                elif len(self.selected_indices) > 1 and i == self.selected_indices[1]:
                    flip_label.setCursor(QtGui.QCursor(Qt.QtCore.Qt.CursorShape.SizeHorCursor))
                    flip_label.setToolTip("Flip X")
                    if self.axis_flipped[i]:
                        flip_label.setText('⬅️')
                    else:
                        flip_label.setText('➡️')
                else:
                    flip_label.setText('')  # Clear for dimensions not in primary/secondary
                    flip_label.setToolTip('')
            else:
                flip_label.setText('')  # Clear for unselected dimensions
                flip_label.setToolTip('')
    
    def apply_axis_flips(self):

        
        if self.is_line_plot_mode():
            plot_view = self.plot_widget.getViewBox()
            plot_dim = self.line_plot_dimension
            plot_view.invertX(self.axis_flipped[plot_dim])
        else:
            if self.data.ndim == 1: # Shouldn't ever be in view mode for 1D data
                return
            
            view = self.img_view.getView()
            
            y_dim = self.selected_indices[0]
            view.invertY(self.axis_flipped[y_dim])

            if len(self.selected_indices) > 1:
                x_dim = self.selected_indices[1]
                view.invertX(self.axis_flipped[x_dim])

    def update_complex_indicators(self):
        """Initialize or update ℝ/ℂ indicators for dimensions that can be combined as complex"""
        for i in range(self.data.ndim):
            indicator = self.widgets['labels']['complex'][i]
            
            if self.combined_as_complex[i]:
                indicator.setText('ℂ')
                indicator.setStyleSheet(self.FLIP_ICON_STYLE + " QLabel {font-weight: bold; }")
                indicator.setCursor(QtGui.QCursor(Qt.QtCore.Qt.CursorShape.PointingHandCursor))
                indicator.setToolTip(f'Split to real')
            elif self.can_combine_as_complex[i]:
                indicator.setText('ℝ')
                indicator.setStyleSheet(self.FLIP_ICON_STYLE + " QLabel {font-weight: bold; }")
                indicator.setCursor(QtGui.QCursor(Qt.QtCore.Qt.CursorShape.PointingHandCursor))
                indicator.setToolTip(f'Combine as complex')
            else:
                # No indicator, already-complex data or non-size-2 dimensions
                indicator.setText('')
                indicator.setToolTip('')
                indicator.setCursor(QtGui.QCursor(Qt.QtCore.Qt.CursorShape.ArrowCursor))

    def _update_channel_controls(self):
        """Keep channel options in sync with the current array dtype."""
        is_complex = np.iscomplexobj(self.data)
        channel_buttons = self.widgets['buttons']['channel']
        enabled_channels = {
            'real': True,
            'abs': True,
            'imag': is_complex,
            'angle': is_complex,
        }

        checked_channel = next(
            (name for name, button in channel_buttons.items() if button.isChecked()),
            None,
        )

        for name, button in channel_buttons.items():
            button.setEnabled(enabled_channels[name])

        if checked_channel not in enabled_channels or not enabled_channels[checked_channel]:
            checked_channel = 'abs' if is_complex else 'real'

        channel_buttons[checked_channel].setChecked(True)
    
    def complexOrRealClicked(self, event, dim):
        if self.can_combine_as_complex[dim] and not self.combined_as_complex[dim]:
            # ℝ clicked - combine to complex
            self.combineAsComplex(dim)
        elif self.combined_as_complex[dim]:
            # ℂ clicked - split back to real
            self.splitToReal(dim)
    
    def combineAsComplex(self, dim):
        """Combine a size-2 real dimension into complex (real+imag), keeping singleton dimension (makes indexing easier)"""
        if not self.can_combine_as_complex[dim] or self.combined_as_complex[dim]:
            return
        
        # Build slices to extract real and imaginary parts
        real_slice = [slice(None)] * self.data.ndim
        imag_slice = [slice(None)] * self.data.ndim
        real_slice[dim] = 0
        imag_slice[dim] = 1
        
        self.data = np.expand_dims(self.data[tuple(real_slice)] + 1j * self.data[tuple(imag_slice)], axis=dim)

        # Update state
        self.combined_as_complex[dim] = True
        self.can_combine_as_complex[dim] = False
        self.singleton[dim] = True  # Now it's a singleton dimension
        
        # Once data is complex, no other dimension can be combined (all other size-2 dims become invalid)
        for i in range(self.data.ndim):
            if i != dim:
                self.can_combine_as_complex[i] = False
        
        # If the converted dimension was in selected_indices or line_plot_dimension, we need to find a new valid selection
        if dim in self.selected_indices:
            # Find a new non-singleton dimension to replace it
            for new_dim in range(self.data.ndim):
                if not self.singleton[new_dim] and new_dim not in self.selected_indices:
                    idx = self.selected_indices.index(dim)
                    self.selected_indices[idx] = new_dim
                    break
        
        # Same problem can happen with line plot.
        if self.line_plot_dimension == dim:
            for new_dim in range(self.data.ndim):
                if not self.singleton[new_dim]:
                    self.line_plot_dimension = new_dim
                    break
        
        self._update_channel_controls()
        
        # Update UI
        self.update_complex_indicators()
        self.update_dimension_controls()
        self.update()
    
    def splitToReal(self, dim):
        """Split a complex dimension back to real (real+imag as separate slices)"""
        if not self.combined_as_complex[dim]:
            return
        
        # Restore data to real with size-2 dimension
        self.data = np.stack([np.real(self.data).squeeze(dim), 
                              np.imag(self.data).squeeze(dim)], axis=dim)
        
        # Update state
        self.combined_as_complex[dim] = False
        self.can_combine_as_complex[dim] = True
        self.singleton[dim] = False
        
        # Data is now real - re-enable combining for all size-2 dimensions
        for i in range(self.data.ndim):
            if self.data.shape[i] == 2:
                self.can_combine_as_complex[i] = True
        
        # Update max value for the spinbox for this dimension
        self.widgets['spins']['slice_indices'][dim].setMaximum(self.data.shape[dim] - 1)
        
        self._update_channel_controls()
        
        # Update UI
        self.update_complex_indicators()
        self.update_dimension_controls()
        self.update()
    
    def getPixel(self, pos):
        img = self.img_view.image
        container = self.img_view.getView()
        if container.sceneBoundingRect().contains(pos): 
            mousePoint = container.mapSceneToView(pos) 
            x_i = math.floor(mousePoint.x()) 
            y_i = math.floor(mousePoint.y()) 
            if x_i >= 0 and x_i < img.shape [ 0 ] and y_i >= 0 and y_i < img.shape[1]:
                decimal_places = getNumberOfDecimalPlaces(abs(img[x_i ,y_i]))
                if decimal_places > 5:
                    self.widgets['labels']['pixelValue'].setText("({}, {}) = {:.3e}".format (x_i, y_i, img[x_i ,y_i]))
                else:
                    self.widgets['labels']['pixelValue'].setText("({}, {}) = {:.{}f}".format (x_i, y_i, img[x_i ,y_i], decimal_places))

    
    def update_image_view(self):
        if self.data.ndim == 1: # No image view for 1D data
            return
            
        prev_levels = None
        al = True
        def_levels = None
        old_channel = self.channel
        oldscale = self.scale
        
        # Determine channel transformation
        channel_func = None
        if self.widgets['buttons']['channel']['abs'].isChecked():
            self.channel = 'abs'
            channel_func = np.abs
        elif self.widgets['buttons']['channel']['angle'].isChecked():
            self.channel = 'angle'
            def_levels = [-np.pi, np.pi]
            channel_func = np.angle
        elif self.widgets['buttons']['channel']['real'].isChecked():
            self.channel = 'real'
            channel_func = np.real
        elif self.widgets['buttons']['channel']['imag'].isChecked():
            self.channel = 'imag'
            channel_func = np.imag
        
        # Determine processing transformation
        processing_func = None
        if self.widgets['buttons']['processing']['symlog'].isChecked():
            processing_func = symlog
            self.scale = 'symlog'
        else:
            self.scale = None
        
        changed_channel = old_channel != self.channel
        changed_scale = oldscale != self.scale
        al = changed_scale or changed_channel or getattr(self, '_force_autolevel', False)
        # reset the one-shot flag after using it
        if getattr(self, '_force_autolevel', False):
            self._force_autolevel = False
        
        prev_levels = None
        if not al:
            prev_levels = self.img_view.imageItem.levels
        elif def_levels is not None:
            prev_levels = def_levels

        
        try:
            # Get the sliced data
            image_data = self.data[tuple(self.slice)]
            
            # Apply transformations in sequence
            if channel_func is not None:
                image_data = channel_func(image_data)
                
            if processing_func is not None:
                image_data = processing_func(image_data)
            
            # Handle transpose if needed
            if self.selected_indices[0] < self.selected_indices[1]:
                image_data = np.transpose(image_data)
            
            # Final processing and display
            image_data = np.nan_to_num(np.squeeze(image_data))
            self.img_view.setImage(image_data, autoLevels=al, levels=prev_levels)
            
            # Apply axis flips after setting the image
            self.apply_axis_flips()
            
        except Exception as e:
            print(f'Image update failed: {e}')
    
    def update_display_mode(self):
        """Update the display mode for the image view"""
        if self.widgets['buttons']['display']['square_pixels'].isChecked():
            self.img_view.setDisplayMode('square_pixels')
        elif self.widgets['buttons']['display']['square_fov'].isChecked():
            self.img_view.setDisplayMode('square_fov')
        elif self.widgets['buttons']['display']['fit'].isChecked():
            self.img_view.setDisplayMode('fit')

    def _processing_pressed(self, btn):
        """Called on processing button press; if the button is already checked
        the user is re-clicking it and we should force an auto-level on next update."""
        try:
            if btn.isChecked():
                self._force_autolevel = True
            else:
                self._force_autolevel = False
        except Exception:
            self._force_autolevel = False

        # Update the display group title
        self._update_display_group_title()

        # Force update to ensure view changes immediately
        self.update_image_view()

    def _update_display_group_title(self):
        """Update the display group title with aspect ratio information."""
        mode = self.img_view.displayMode
        aspect_str = ''
        
        if mode == 'square_pixels': # Simple
            self.display_group.setTitle('Display (1:1)')
            return
        
        if mode == 'fit': #use the viewport aspect ratio
            aspect_str = ''
            try:
                if hasattr(self.img_view, 'image') and self.img_view.image is not None:
                    view = self.img_view.getView()
                    
                    img_height, img_width = self.img_view.image.shape
                    widget_ratio = view.size().width() / view.size().height()
                    img_ratio = img_width / img_height
                    ratio = img_ratio * widget_ratio
                    
                    if abs(ratio - 1.0) < 1e-2:
                        aspect_str = '(1:1)'
                    else:
                        aspect_str = f'({ratio:.2f}:1)'
            finally:
                self.display_group.setTitle(f'Display {aspect_str}')
            
        elif mode == 'square_fov':
            # For square FOV, use the image aspect ratio
            if hasattr(self.img_view, 'image') and self.img_view.image is not None:
                shape = self.img_view.image.shape
                if len(shape) == 2:
                    height, width = shape
                    ratio = width / height
                    if abs(ratio - 1.0) < 1e-2:
                        aspect_str = '(1:1)'
                    else:
                        aspect_str = f'({ratio:.2f}:1)'
                else:
                    aspect_str = ''
            self.display_group.setTitle(f'Display {aspect_str}')
        else:
            self.display_group.setTitle('Display')
    
    def update_line_plot(self):
        """Update the line plot with 1D data slices"""
        # Determine channel transformation
        channel_func = None
        if self.widgets['buttons']['channel']['abs'].isChecked():
            channel_func = np.abs
        elif self.widgets['buttons']['channel']['angle'].isChecked():
            channel_func = np.angle
        elif self.widgets['buttons']['channel']['real'].isChecked():
            channel_func = np.real
        elif self.widgets['buttons']['channel']['imag'].isChecked():
            channel_func = np.imag
        
        # Determine processing transformation
        processing_func = None
        if self.widgets['buttons']['processing']['symlog'].isChecked():
            processing_func = symlog
        
        # Clear previous plots but preserve crosshair reference (we will re-add after clear)
        self.plot_widget.clear()
        # Re-add crosshair item if it exists
        if hasattr(self, 'plot_crosshair') and self.plot_crosshair is not None:
            self.plot_widget.addItem(self.plot_crosshair)
            self.plot_crosshair.setVisible(False)
        
        try:
            # Create slice for line plot mode
            line_slice = [slice(None)] * self.data.ndim
            
            # For all dimensions except the one we're plotting along, use the spinbox values
            for dim in range(self.data.ndim):
                if dim == self.line_plot_dimension:
                    # Plot along this dimension - use full range
                    line_slice[dim] = slice(None)
                else:
                    # Use the slice specified by the spinbox
                    val = self.widgets['spins']['slice_indices'][dim].value()
                    line_slice[dim] = slice(val, val+1)
            
            # Extract and transform the 1D data
            line_data = self.data[tuple(line_slice)]
            
            # Apply transformations in sequence
            if channel_func is not None:
                line_data = channel_func(line_data)
                
            if processing_func is not None:
                line_data = processing_func(line_data)
            
            # Squeeze out singleton dimensions
            line_data = np.squeeze(line_data)
            
            # Make sure we have 1D data
            if line_data.ndim == 1:
                # Remember for hover queries
                self.current_line_data = line_data
                
                if self.plot_style == 'bar':
                    x = np.arange(len(line_data))
                    brush = pg.mkBrush(50, 100, 200, 180)
                    pen = pg.mkPen(50, 100, 200)
                    self.line_curve = pg.BarGraphItem(x=x, height=line_data, width=0.8, brush=brush, pen=pen)
                    self.plot_widget.addItem(self.line_curve)
                    self.tab_widget.setTabText(1, "Bar Plot")
                else:
                    pen = pg.mkPen(color=(50, 100, 200), width=2)
                    self.line_curve = self.plot_widget.plot(line_data, pen=pen, name='')
                    self.tab_widget.setTabText(1, "Line Plot")
                
                # Capture base x-span after first valid plot for scaling reference
                if self.line_base_range is None:
                    x_min = 0
                    x_max = len(line_data)-1
                    self.line_base_range = max(1.0, (x_max - x_min))
            else:
                print(f'Warning: Expected 1D data but got {line_data.ndim}D data with shape {line_data.shape}')
            
            self.plot_widget.setLabel('bottom', f'Index along dim {self.line_plot_dimension}')
            
        except Exception as e:
            print(f'Line plot update failed: {e}')
            self.current_line_data = None

    def _on_plot_range_changed(self, vb, ranges):
        """Adapt line pen thickness based on horizontal zoom.

        As user zooms in (smaller x-span), increase thickness up to max; zooming out decreases thickness.
        """
        if self.line_curve is None or self.line_base_range is None:
            return
        
        if self.plot_style == 'bar':
            return

        try:
            (x_range, _) = ranges  # ranges is ((xMin,xMax),(yMin,yMax))
            x_span = max(1e-9, x_range[1] - x_range[0])
            # Scale factor inverse to span relative to base
            scale = self.line_base_range / x_span
            # Mild logarithmic dampening to avoid extremes
            adj = math.log10(1 + scale)
            new_width = self.line_base_pen_width * adj
            # Clamp
            new_width = max(self.line_min_pen, min(self.line_max_pen, new_width))
            current_pen = self.line_curve.opts['pen']
            if current_pen.widthF() != new_width:
                new_pen = pg.mkPen(self.line_color, width=new_width)
                self.line_curve.setPen(new_pen)
        except Exception as e:
            print(f"Thickness update failed: {e}")

    def _on_view_range_changed(self):
        """Update display group title when view range changes (for fit mode)."""
        self._update_display_group_title()

    def _on_plot_hover(self, pos):
        """Handle mouse hover over the line plot: update pixelValue label."""
        # Only react when on the line plot tab
        if not self.is_line_plot_mode():
            return
        if self.current_line_data is None:
            return
        vb = self.plot_widget.getViewBox()
        if vb.sceneBoundingRect().contains(pos):
            mouse_point = vb.mapSceneToView(pos)
            x = mouse_point.x()
            # Index is along x
            idx = int(round(x))
            if 0 <= idx < len(self.current_line_data):
                val = self.current_line_data[idx]
                # Determine formatting similar to image hover
                decimal_places = getNumberOfDecimalPlaces(abs(val)) if np.isfinite(val) else 3
                if decimal_places > 5:
                    text = f"[{idx}] = {val:.3e}"
                else:
                    # Clamp decimal places to reasonable range
                    dp = min(8, max(0, decimal_places))
                    fmt = f"{{val:.{dp}f}}"
                    text = f"[{idx}] = {val:.{dp}f}"
                self.widgets['labels']['pixelValue'].setText(text)
                # Move and show crosshair
                if self.plot_crosshair is not None:
                    self.plot_crosshair.setValue(idx)
                    if not self.plot_crosshair.isVisible():
                        self.plot_crosshair.setVisible(True)
            else:
                # Outside data range; optionally clear or ignore
                if self.plot_crosshair is not None and self.plot_crosshair.isVisible():
                    self.plot_crosshair.setVisible(False)
        else:
            if self.plot_crosshair is not None and self.plot_crosshair.isVisible():
                self.plot_crosshair.setVisible(False)
    
    def on_tab_changed(self, index):
        """Handle tab change between Image View and Line Plot"""
        self.update_dimension_controls()
        self.update()
        # Hide crosshair if leaving line plot
        if not self.is_line_plot_mode() and hasattr(self, 'plot_crosshair') and self.plot_crosshair is not None:
            self.plot_crosshair.setVisible(False)
    
    def is_line_plot_mode(self):
        """Check if currently in line plot mode"""
        return self.tab_widget.currentIndex() == 1

    def transposeView(self, event):
        old_primary = self.selected_indices[0]
        old_secondary = self.selected_indices[1]

        # Swap the flip states along with the dimensions (prevents the axes from appearing flipped after transpose)
        self.axis_flipped[old_primary], self.axis_flipped[old_secondary] = self.axis_flipped[old_secondary], self.axis_flipped[old_primary]
        
        self.changedIndex(event, 0, old_secondary, update=False)
        self.changedIndex(event, 1, old_primary, update=True)

    def update(self):
        self.update_slice()
        self.update_image_view()
        self.update_line_plot()

    def update_slice(self):
        """Update slice for image view mode"""
        self.slice = [slice(None)] * self.data.ndim
        for dim in range(0, self.data.ndim):
            if dim in self.selected_indices:
                self.slice[dim] = slice(None) # pass?
            else:
                val = self.widgets['spins']['slice_indices'][dim].value()
                self.slice[dim] = slice(val, val+1)

    def changedIndex(self, checked, which_one, idx, update=True):
        if self.is_line_plot_mode():
            # In line plot mode, only use primary button to select the dimension to plot along
            if which_one == 0:  # Primary button clicked
                self.line_plot_dimension = idx
        else:
            # In image view mode, use the original two-dimension selection
            # For 1D arrays, we don't have a second dimension
            if which_one < len(self.selected_indices):
                self.selected_indices[which_one] = idx
        
        self.update_dimension_controls()
        if update is True:
            self.update()
    
    def update_dimension_controls(self):
        """Update button and spinbox states based on current mode"""
        if self.is_line_plot_mode():
            # Line plot mode: single dimension selection, all other spinboxes enabled
            for i, w in enumerate(self.widgets['spins']['slice_indices']):
                bPrim = self.widgets['buttons']['primary'][i]
                bSecondary = self.widgets['buttons']['secondary'][i]
                
                if self.singleton[i]:
                    w.setEnabled(False)
                    bPrim.setEnabled(False)
                    bSecondary.setEnabled(False)
                    bPrim.setChecked(False)
                    bSecondary.setChecked(False)
                elif i == self.line_plot_dimension:
                    # This is the dimension we're plotting along
                    w.setEnabled(False)
                    bPrim.setEnabled(True)
                    bSecondary.setEnabled(False)
                    bPrim.setChecked(True)
                    bSecondary.setChecked(False)
                else:
                    # All other dimensions: enable spinbox to select slice
                    w.setEnabled(True)
                    bPrim.setEnabled(True)
                    bSecondary.setEnabled(False)
                    bPrim.setChecked(False)
                    bSecondary.setChecked(False)
        else:
            # Image view mode: original two-dimension selection behavior
            for i, w in enumerate(self.widgets['spins']['slice_indices']):
                bPrim = self.widgets['buttons']['primary'][i]
                bSecondary = self.widgets['buttons']['secondary'][i]
                if self.singleton[i] == True:
                    w.setEnabled(False)
                    bPrim.setEnabled(False)
                    bSecondary.setEnabled(False)
                elif i in self.selected_indices:
                    w.setEnabled(False)
                    if i == self.selected_indices[0]:
                        bPrim.setChecked(True)
                        bSecondary.setChecked(False)
                    else:
                        bPrim.setChecked(False)
                        bSecondary.setChecked(True)
                    bPrim.setEnabled(False)
                    bSecondary.setEnabled(False)
                else:
                    w.setEnabled(True)
                    bPrim.setEnabled(True)
                    bSecondary.setEnabled(True)
                    bPrim.setChecked(False)
                    bSecondary.setChecked(False)
                    
            # Only set buttons if we have at least 2 selected indices
            if len(self.selected_indices) >= 1:
                self.widgets['buttons']['primary'][self.selected_indices[0]].setChecked(True)
            if len(self.selected_indices) >= 2:
                self.widgets['buttons']['secondary'][self.selected_indices[1]].setChecked(True)
        
        self.update_flip_icons()
    
    def keyPressEvent(self, event):
        """Handle keyboard shortcuts"""
        key = event.key()
        modifiers = event.modifiers()
        
        # Check for 'T' key to transpose view (swap X and Y dimensions)
        if key == Qt.QtCore.Qt.Key.Key_T and modifiers == Qt.QtCore.Qt.KeyboardModifier.NoModifier:
            if not self.is_line_plot_mode() and len(self.selected_indices) >= 2:
                self.transposeView(event)
                event.accept()
                return
        
        # Check CTRL+number for colormap changes
        if modifiers == Qt.QtCore.Qt.KeyboardModifier.ControlModifier:
            if key == Qt.QtCore.Qt.Key.Key_1:
                self.setColormap('gray')
                event.accept()
                return
            elif key == Qt.QtCore.Qt.Key.Key_2:
                self.setColormap('viridis')
                event.accept()
                return
            elif key == Qt.QtCore.Qt.Key.Key_3:
                self.setColormap('plasma')
                event.accept()
                return
            elif key == Qt.QtCore.Qt.Key.Key_4:
                self.setColormap('PAL-relaxed')
                event.accept()
                return
            elif key == Qt.QtCore.Qt.Key.Key_5:
                self.setColormap('cividis')
                event.accept()
                return
            elif key == Qt.QtCore.Qt.Key.Key_6:
                self.setColormap('CET-CBL1')
                event.accept()
                return
            elif key == Qt.QtCore.Qt.Key.Key_7:
                self.setColormap('d3-cool')
                event.accept()
                return
            elif key == Qt.QtCore.Qt.Key.Key_8:
                self.setColormap('d3-warm')
                event.accept()
                return
                
        # Pass event to parent if not handled
        super().keyPressEvent(event)
    
    def setColormap(self, colormap_name):
        """Set the colormap for the image view"""
        try:
            if colormap_name == 'gray':
                colormap = self._create_gray_colormap()
            elif colormap_name == 'viridis':
                colormap = pg.colormap.get('viridis')
            elif colormap_name == 'PAL-relaxed':
                colormap = pg.colormap.get('PAL-relaxed')
            elif colormap_name == 'plasma':
                colormap = pg.colormap.get('plasma')
            elif colormap_name == 'd3-warm':
                colormap = self._create_d3_warm_colormap()
            elif colormap_name == 'd3-cool':
                colormap = self._create_d3_cool_colormap()
            elif colormap_name == 'CET-CBL1':
                colormap = pg.colormap.get('CET-CBL1')
            elif colormap_name == 'cividis':
                colormap = pg.colormap.get('cividis')
            else:
                print(f"Unknown colormap: {colormap_name}")
                return
            
            if colormap is None:
                print(f"Failed to load colormap '{colormap_name}': returned None")
                return
            
            # Apply colormap to the image view
            self.img_view.setColorMap(colormap)
            #self.current_colormap = colormap_name
            
        except Exception as e:
            print(f"Failed to set colormap {colormap_name}: {e}")
    
    
    def _create_gray_colormap(self):
        """Create a grayscale colormap matching pyqtgraph's built-in default (black to white)"""
        return pg.ColorMap(pos=[0.0, 1.0], color=[[0, 0, 0, 255], [255, 255, 255, 255]])
    
    def _create_d3_warm_colormap(self):
        """Create D3.js interpolateWarm colormap (Niccoli's perceptual rainbow, 180° rotation)"""
        # D3 uses cubehelix interpolation: cubehelix(-100, 0.75, 0.35) to cubehelix(80, 1.50, 0.8)
        # Uses "long" interpolation (linear, not shortest path)
        colors = []
        positions = []
        n_samples = 256
        
        # D3 cubehelix constants from d3-color
        A = -0.14861
        B = +1.78277
        C = -0.29227
        D = -0.90649
        E = +1.97294
        
        for i in range(n_samples):
            t = i / (n_samples - 1)
            # Linear interpolation of cubehelix parameters (not shortest path)
            h = -100 + t * (80 - (-100))  # hue: -100 to 80
            s = 0.75 + t * (1.50 - 0.75)  # saturation: 0.75 to 1.50
            l = 0.35 + t * (0.8 - 0.35)   # lightness: 0.35 to 0.8
            
            # Convert cubehelix to RGB using D3 formula
            h_rad = (h + 120) * np.pi / 180
            a = s * l * (1 - l)
            cosh = np.cos(h_rad)
            sinh = np.sin(h_rad)
            
            r = l + a * (A * cosh + B * sinh)
            g = l + a * (C * cosh + D * sinh)
            b = l + a * (E * cosh)
            
            # Convert to 0-255 range (D3 multiplies by 255)
            r = np.clip(r * 255, 0, 255)
            g = np.clip(g * 255, 0, 255)
            b = np.clip(b * 255, 0, 255)
            
            colors.append((int(r), int(g), int(b)))
            positions.append(t)
        
        return pg.ColorMap(pos=np.array(positions), color=np.array(colors))
    
    def _create_d3_cool_colormap(self):
        """Create D3.js interpolateCool colormap (Niccoli's perceptual rainbow)"""
        # D3 uses cubehelix interpolation: cubehelix(260, 0.75, 0.35) to cubehelix(80, 1.50, 0.8)
        # Uses "long" interpolation (linear, not shortest path)
        colors = []
        positions = []
        n_samples = 256
        
        # D3 cubehelix constants from d3-color
        A = -0.14861
        B = +1.78277
        C = -0.29227
        D = -0.90649
        E = +1.97294
        
        for i in range(n_samples):
            t = i / (n_samples - 1)
            # Linear interpolation of cubehelix parameters (not shortest path)
            h = 260 + t * (80 - 260)      # hue: 260 to 80
            s = 0.75 + t * (1.50 - 0.75)  # saturation: 0.75 to 1.50
            l = 0.35 + t * (0.8 - 0.35)   # lightness: 0.35 to 0.8
            
            # Convert cubehelix to RGB using D3 formula
            h_rad = (h + 120) * np.pi / 180
            a = s * l * (1 - l)
            cosh = np.cos(h_rad)
            sinh = np.sin(h_rad)
            
            r = l + a * (A * cosh + B * sinh)
            g = l + a * (C * cosh + D * sinh)
            b = l + a * (E * cosh)
            
            # Convert to 0-255 range (D3 multiplies by 255)
            r = np.clip(r * 255, 0, 255)
            g = np.clip(g * 255, 0, 255)
            b = np.clip(b * 255, 0, 255)
            
            colors.append((int(r), int(g), int(b)))
            positions.append(t)
        
        return pg.ColorMap(pos=np.array(positions), color=np.array(colors))
    
    def eventFilter(self, obj, event):
        if obj == self.tab_widget.tabBar():
            if event.type() == Qt.QtCore.QEvent.Type.MouseButtonDblClick:
                # which tab was double-clicked
                tab_bar = self.tab_widget.tabBar()
                clicked_index = tab_bar.tabAt(event.pos())
                
                # Check if line/bar tab (index 1)
                if clicked_index == 1:
                    if self.plot_style == 'line':
                        self.plot_style = 'bar'
                    else:
                        self.plot_style = 'line'
                    self.update_line_plot()
                    event.accept()
                    return True 
        return super().eventFilter(obj, event)
    
    def _setup_export_context_menus(self):
        for i in range(self.data.ndim):
            prim_btn = self.widgets['buttons']['primary'][i]
            sec_btn = self.widgets['buttons']['secondary'][i]
            
            prim_btn.setContextMenuPolicy(Qt.QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
            sec_btn.setContextMenuPolicy(Qt.QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
            
            prim_btn.customContextMenuRequested.connect(lambda pos, btn=prim_btn, dim=i: self._show_export_context_menu(pos, btn, dim))
            sec_btn.customContextMenuRequested.connect( lambda pos, btn=sec_btn,  dim=i: self._show_export_context_menu(pos, btn, dim))
    
    def _show_export_context_menu(self, pos, btn, dim):
        menu = QtWidgets.QMenu()
        
        export_action = menu.addAction("Export along this dimension...")
        export_action.triggered.connect(lambda: self._start_export(dim))
        
        # Show menu at cursor position
        menu.exec(btn.mapToGlobal(pos))
    
    def _start_export(self, export_dim):
        """Initiate video export workflow"""
        if self.singleton[export_dim]:
            QtWidgets.QMessageBox.warning(self, "Cannot Export", 
                f"Dimension {export_dim} has size 1 and cannot be exported.")
            return
        
        # Get export settings from dialog
        settings_dialog = VideoExportSettingsDialog(parent=self, export_dim=export_dim, data_shape=self.data.shape)
        if settings_dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        
        settings = settings_dialog.get_settings()
        
        # Get file save path (for PNG frames we ask for a directory)
        file_path = None
        if settings['format'] == 'png':
            dir_path = QtWidgets.QFileDialog.getExistingDirectory(
                self, "Export frames to directory", os.path.expanduser("~")
            )
            if not dir_path:
                return
            file_path = dir_path
        else:
            file_filter = f"{settings['format'].upper()} files (*.{settings['format']})"
            file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, f"Export Video as {settings['format'].upper()}", 
                f"export.{settings['format']}", file_filter
            )
            if not file_path:
                return
        
        # Determine transpose flag to match on-screen orientation (same condition as update_image_view)
        transpose = True if len(self.selected_indices) >= 2 and self.selected_indices[0] > self.selected_indices[1] else False
        
        # Capture display mode and widget aspect ratio
        display_mode = getattr(self.img_view, 'displayMode', 'square_pixels')
        view = self.img_view.getView()
        widget_ratio = view.size().width() / view.size().height() if view.size().height() != 0 else 1.0
        
        # Prepare transformation functions
        channel_func = None
        if self.widgets['buttons']['channel']['abs'].isChecked():
            channel_func = np.abs
        elif self.widgets['buttons']['channel']['angle'].isChecked():
            channel_func = np.angle
        elif self.widgets['buttons']['channel']['real'].isChecked():
            channel_func = np.real
        elif self.widgets['buttons']['channel']['imag'].isChecked():
            channel_func = np.imag
        
        processing_func = None
        if self.widgets['buttons']['processing']['symlog'].isChecked():
            processing_func = symlog
        
        # Capture current display levels for consistent frame scaling
        levels = None
        if hasattr(self.img_view, 'imageItem') and self.img_view.imageItem.levels is not None:
            levels = tuple(self.img_view.imageItem.levels)

        # Capture current color map LUT (256x3) to apply in export
        lut = None
        try:
            if hasattr(self.img_view, 'histogram') and hasattr(self.img_view.histogram, 'gradient'):
                cm = self.img_view.histogram.gradient.colorMap()
                if cm is not None:
                    lut = cm.getLookupTable(0.0, 1.0, 256, alpha=False)
        except Exception:
            lut = None
        
        # Create worker thread
        worker = VideoExportWorker(
            data=self.data,
            export_dim=export_dim,
            output_path=file_path,
            fps=settings['fps'],
            format_type=settings['format'],
            window_level_mode=settings.get('window_level', 'displayed'),
            channel_func=channel_func,
            processing_func=processing_func,
            slice_indices=self.slice,
            selected_indices=self.selected_indices,
            singleton=self.singleton,
            levels=levels,
            transpose=transpose,
            pixel_ratio_mode=settings.get('pixel_ratio', 'square_pixels'),
            display_mode=display_mode,
            widget_ratio=widget_ratio,
            axis_flipped=self.axis_flipped,
            lut=lut
        )
        
        # Show progress dialog
        progress_dialog = VideoExportDialog(self)
        progress_dialog.start_export(worker, self.data.shape[export_dim])

    def _save_current_numpy_file(self):
        """Save the currently displayed array state to a NumPy .npy file."""
        range_dialog = SaveRangeDialog(self, self.data.shape)
        if range_dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return

        ranges = range_dialog.get_ranges()
        sliced_data = self.data[tuple(slice(start, stop) for start, stop in ranges)]
        output_data = np.squeeze(sliced_data) if range_dialog.should_squeeze() else sliced_data

        default_name = 'ndslice.npy'
        if self._filepath is not None:
            source_path = Path(self._filepath)
            source_name = source_path.name
            if source_name.lower().endswith('.nii.gz'):
                default_name = source_name[:-7] + '.npy'
            else:
                default_name = f"{source_path.stem}.npy"

        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save current array as NumPy file",
            default_name,
            "NumPy files (*.npy)"
        )
        if not file_path:
            return

        if not file_path.lower().endswith('.npy'):
            file_path += '.npy'

        try:
            np.save(file_path, output_data)
            print(f"Saved array {list(output_data.shape)} to {file_path}")
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Save Error", f"Failed to save NumPy file:\n{e}")

    def _on_file_changed(self, path):
        """Called by QFileSystemWatcher when the source file changes on disk."""
        self._reload_btn.setText("⚠️")
        self._reload_btn.setToolTip("File changed — click to reload")
        # Re-add the path: handles atomic replacement where the original inode disappears
        if self._file_watcher:
            self._file_watcher.addPath(path)

    def _reload_file(self):
        """Reload data from the source file, preserving slice positions where possible."""
        if self._filepath is None:
            return
        try:
            new_data = None
            new_dataset_path = self._dataset_path

            if self._selector_class_name is None:
                from .file_interpreters import load_file
                new_data = load_file(self._filepath)
            else:
                from .selectors import H5DatasetSelector, NpzDatasetSelector, MatDatasetSelector
                selector_map = {
                    'H5DatasetSelector': H5DatasetSelector,
                    'NpzDatasetSelector': NpzDatasetSelector,
                    'MatDatasetSelector': MatDatasetSelector,
                }
                selector_cls = selector_map.get(self._selector_class_name)
                if selector_cls is None:
                    return
                selector = selector_cls(self._filepath)
                compatible_keys = {d[0] for d in selector.compatible_datasets}
                if self._dataset_path is not None and self._dataset_path in compatible_keys:
                    new_data = selector.load_data(self._dataset_path)
                    selector.close()
                elif selector.requires_gui():
                    selected = selector.show()
                    if selected is None:
                        selector.close()
                        return  # User cancelled — keep ⚠️ visible
                    new_data = selector.load_data(selected)
                    new_dataset_path = selected
                    selector.close()
                else:
                    result = selector.get_single_data()
                    selector.close()
                    if result is None:
                        return
                    new_dataset_path, new_data = result

            if new_data is None:
                return

            self._dataset_path = new_dataset_path
            self._reset_data(new_data)
            self._reload_btn.setText("⟳")
            self._reload_btn.setToolTip("Reload file")
            if self._file_watcher and self._filepath:
                self._file_watcher.addPath(str(self._filepath))
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Reload Error", f"Failed to reload:\n{e}")

    def _reset_data(self, new_data):
        """Replace the displayed data, clamping slice positions to the new shape."""
        old_ndim = self.data.ndim
        new_ndim = new_data.ndim

        if new_ndim != old_ndim:
            # Per-dimension widgets were built for old_ndim and cannot be rebuilt in-place.
            # Open a fresh window with the new data and close this one.
            win = NDSliceWindow(new_data,
                                filepath=self._filepath,
                                dataset_path=self._dataset_path,
                                selector_class_name=self._selector_class_name)
            win.setWindowTitle(self.windowTitle())
            win.show()
            self.close()
            return

        self.data = new_data
        self.singleton = [e == 1 for e in new_data.shape]

        if np.iscomplexobj(new_data):
            self.can_combine_as_complex = [False] * new_ndim
        else:
            self.can_combine_as_complex = [new_data.shape[i] == 2 for i in range(new_ndim)]
        self.combined_as_complex = [False] * new_ndim

        # Reset FFT domain state and dim label styling
        self.domain = [Domain.NATIVE for _ in range(new_ndim)]
        for i, label in enumerate(self.widgets['labels']['dims']):
            label.setStyleSheet(self.DIMENSION_LABEL_STYLE)
            label.setText(f'[{new_data.shape[i]}]')

        # Update spinbox maximums (auto-clamps current value)
        for i in range(new_ndim):
            self.widgets['spins']['slice_indices'][i].setMaximum(new_data.shape[i] - 1)

        # Clamp selected_indices: keep existing choices where still valid, fill from valid dims
        valid_dims = [i for i in range(new_ndim) if not self.singleton[i]]
        clamped = [i for i in self.selected_indices if i < new_ndim and not self.singleton[i]]
        for d in valid_dims:
            if len(clamped) >= 2:
                break
            if d not in clamped:
                clamped.append(d)
        if not clamped and valid_dims:
            clamped = [valid_dims[0]]
        self.selected_indices = clamped[:2]

        # Clamp line_plot_dimension
        if self.line_plot_dimension >= new_ndim or self.singleton[self.line_plot_dimension]:
            for d in range(new_ndim):
                if not self.singleton[d]:
                    self.line_plot_dimension = d
                    break

        self.axis_flipped = [False] * new_ndim

        self._update_channel_controls()
        self.update_complex_indicators()
        if len(self.selected_indices) >= 1:
            self.changedIndex(True, 0, self.selected_indices[0], update=False)
        if len(self.selected_indices) >= 2:
            self.changedIndex(True, 1, self.selected_indices[1], update=False)
        self.update_dimension_controls()
        self._force_autolevel = True
        self.update()


def _prepare_qt_environment():
    """Apply conservative Qt workarounds before QApplication creation."""
    # On XWayland/XCB, Qt's MIT-SHM path can fail with:
    #   qt.qpa.xcb: xcb_shm_create_segment() failed
    #   The X11 connection broke
    # This must be set before QApplication / the QPA plugin is initialized.
    if os.environ.get("QT_QPA_PLATFORM") == "xcb":
        os.environ.setdefault("QT_X11_NO_MITSHM", "1")


def _qt_application_exists():
    """Return True if Qt has already created a QApplication in this process."""
    try:
        return QtWidgets.QApplication.instance() is not None
    except Exception:
        return False
    
def _enable_ipython_qt_event_loop():
    """Try to enable IPython's Qt input hook for responsive inline windows.

    Returns True if we are in IPython and Qt GUI integration is active or was
    successfully enabled. Returns False otherwise.
    """
    ip = get_ipython()

    if ip is None:
        return False

    # If already active, good.
    active_eventloop = getattr(ip, "active_eventloop", None)
    if active_eventloop in {"qt", "qt5", "qt6"}:
        return True

    try:
        ip.enable_gui("qt")
    except Exception:
        return False

    active_eventloop = getattr(ip, "active_eventloop", None)
    return active_eventloop in {"qt", "qt5", "qt6"}

def _retain_window_reference(app, win):
    """Keep inline windows alive for as long as they are open.

    Without a strong Python reference, some Qt bindings may garbage-collect the
    wrapper even while the native window is visible.
    """
    live_windows = app.property("_ndslice_live_windows")
    if not isinstance(live_windows, list):
        live_windows = []

    live_windows.append(win)
    app.setProperty("_ndslice_live_windows", live_windows)

    def _release_reference(_=None, w=win, qapp=app):
        refs = qapp.property("_ndslice_live_windows")
        if not isinstance(refs, list):
            return
        try:
            refs.remove(w)
        except ValueError:
            pass
        qapp.setProperty("_ndslice_live_windows", refs)

    win.destroyed.connect(_release_reference)

def _create_window(data, title='', complex_dim=None, filepath=None,
                   dataset_path=None, selector_class_name=None):
    _prepare_qt_environment()

    app = pg.mkQApp()
    app.setStyle('Fusion')

    win = NDSliceWindow(data, complex_dim=complex_dim, filepath=filepath,
                        dataset_path=dataset_path,
                        selector_class_name=selector_class_name)
    win.setWindowTitle(title)
    win.show()

    return app, win

def _run_window(data, title='', complex_dim=None, filepath=None,
                dataset_path=None, selector_class_name=None):
    """Open a viewer window in this process and block on the Qt event loop."""
    try:
        app, win = _create_window(
            data, title=title, complex_dim=complex_dim,
            filepath=filepath, dataset_path=dataset_path,
            selector_class_name=selector_class_name,
        )
        return app.exec()
    except BaseException:
        import traceback
        traceback.print_exc()
        raise

def _show_window_inline(data, title='', complex_dim=None, filepath=None,
                        dataset_path=None, selector_class_name=None):
    """Open a viewer window in this process without starting app.exec()."""
    app, win = _create_window(
        data, title=title, complex_dim=complex_dim,
        filepath=filepath, dataset_path=dataset_path,
        selector_class_name=selector_class_name,
    )

    _retain_window_reference(app, win)

    return win

def ndslice(data, title='', block=False, complex_dim=None, filepath=None,
            dataset_path=None, selector_class_name=None):
    if not isinstance(data, np.ndarray):
        raise TypeError("data must be a numpy array")
    if data.ndim < 1:
        raise ValueError("data must have at least 1 dimension")

    kwargs = {
        "filepath": filepath,
        "dataset_path": dataset_path,
        "selector_class_name": selector_class_name,
    }

    if block:
        return _run_window(data, title, complex_dim, **kwargs)

    if _qt_application_exists():
        if _enable_ipython_qt_event_loop():
            return _show_window_inline(data, title, complex_dim, **kwargs)

        warnings.warn(
            "Qt is already initialized, so ndslice cannot safely fork a child "
            "process for non-blocking display. No active IPython Qt event-loop "
            "integration was detected, so an inline non-blocking window may freeze. "
            "Falling back to blocking mode in the current process. To avoid this, "
            "use ndslice(..., block=True) explicitly, or in IPython run `%gui qt` "
            "before calling ndslice(...).",
            RuntimeWarning,
            stacklevel=2,
        )
        return _run_window(data, title, complex_dim, **kwargs)

    p = mp.Process(target=_run_window, args=(data, title, complex_dim),
                   kwargs=kwargs)
    p.start()
    return p