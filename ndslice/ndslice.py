import numpy as np
import pyqtgraph as pg
import pyqtgraph.Qt as Qt
from pyqtgraph.Qt import QtWidgets, QtGui
import math
from enum import Enum
from .imageview2d import ImageView2D
import multiprocessing as mp

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

class NDSliceWindow(QtWidgets.QMainWindow):

    def __init__(self, data, *args, **kwargs):
        super(NDSliceWindow, self).__init__(*args, **kwargs)
        self.resize(800,800)

        self.data = data
        self.singleton = [e == 1 for e in list(data.shape)]
        self.selected_indices = []
        self.channel = None
        self.scale = None
        
        # Detect dark mode
        self.is_dark_mode = self._detect_dark_mode()
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
                'dims': [QtWidgets.QLabel('[' + str(data.shape[i]) + ']', alignment=Qt.QtCore.Qt.AlignCenter) for i in range(data.ndim)],
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
        
        # Set default checked radio button
        if self.widgets['buttons']['channel']['abs'].isEnabled():
            self.widgets['buttons']['channel']['abs'].setChecked(True)
            #self.channel = 'abs'
        else:
            self.widgets['buttons']['channel']['real'].setChecked(True)
            #self.channel = 'real'
            
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
        self.widgets['labels']['primary'].mousePressEvent = self.transposeView
        
        # Apply compact styling to dimension control widgets
        self.widgets['labels']['dimensions'].setStyleSheet("QLabel { font-size: 11px; font-weight: bold; padding: 1px; margin: 2px; }")
        self.widgets['labels']['dimensions'].setFixedHeight(18)  # Match the dimension labels height
        self.widgets['labels']['primary'].setStyleSheet("QLabel { font-size: 11px; font-weight: bold; }")
        self.widgets['labels']['secondary'].setStyleSheet("QLabel { font-size: 11px; font-weight: bold; }")
        self.widgets['labels']['slice'].setStyleSheet("QLabel { font-size: 11px; font-weight: bold; }")
        
        # Make dimension labels much more compact
        for label in self.widgets['labels']['dims']:
            label.setStyleSheet("QLabel { font-size: 11px; padding: 1px; margin: 2px; }")
            label.setFixedHeight(18)  # Force a much smaller height
            label.setMinimumWidth(30)  # Ensure minimum readability
        
        # Make dimension buttons more compact
        for btn in self.widgets['buttons']['primary'] + self.widgets['buttons']['secondary']:
            btn.setStyleSheet("QPushButton { font-size: 11px; padding: 3px; margin: 2px; }")
            
        # Make spinboxes more compact
        for spin in self.widgets['spins']['slice_indices']:
            spin.setStyleSheet("QSpinBox { font-size: 11px; }")
        
        # Set minimal spacing for all control layouts
        self.layouts['dims'].setSpacing(2)  # Extra tight for dimensions row
        self.layouts['dims'].setContentsMargins(0, 0, 0, 0)  # Remove all margins
        self.layouts['primary'].setSpacing(2)  # Tighter spacing
        self.layouts['primary'].setContentsMargins(0, 1, 0, 1)  # Minimal margins
        self.layouts['secondary'].setSpacing(2)  # Tighter spacing
        self.layouts['secondary'].setContentsMargins(0, 1, 0, 1)  # Minimal margins
        self.layouts['slice'].setSpacing(2)  # Tighter spacing
        self.layouts['slice'].setContentsMargins(0, 1, 0, 1)  # Minimal margins
        
        self.layouts['dims'].addWidget(self.widgets['labels']['dimensions'])
        [self.layouts['dims'].addWidget(w) for w in self.widgets['labels']['dims']]
        
        self.layouts['primary'].addWidget(self.widgets['labels']['primary'])
        for i, w in enumerate(self.widgets['buttons']['primary']):
            self.layouts['primary'].addWidget(w)
            w.clicked.connect(lambda checked, i=i : self.changedIndex(checked, 0, i))
        
        self.layouts['secondary'].addWidget(self.widgets['labels']['secondary'])
        for i, w in enumerate(self.widgets['buttons']['secondary']):
            self.layouts['secondary'].addWidget(w)
            w.clicked.connect(lambda checked, i=i: self.changedIndex(checked, 1, i))

        self.layouts['slice'].addWidget(self.widgets['labels']['slice'])
        for i, w in enumerate(self.widgets['spins']['slice_indices']):
            self.layouts['slice'].addWidget(w)
            w.valueChanged.connect(self.update)
        
        # Create a single compact control panel with all radio buttons
        controls_widget = QtWidgets.QWidget()
        controls_layout = QtWidgets.QVBoxLayout()
        controls_layout.setSpacing(2)  # Reduce spacing between groups
        controls_layout.setContentsMargins(5, 5, 5, 5)  # Reduce margins
        
        # Channel controls - horizontal layout to save space
        channel_group = QtWidgets.QGroupBox("Channel")
        
        if not self.is_dark_mode:
            channel_group.setStyleSheet("QGroupBox { background-color: #F7FFE6; font-size: 11px; font-weight: bold; padding-top: 8px; margin-top: 3px; } QGroupBox::title { subcontrol-origin: margin; left: 5px; }")
        else:
            channel_group.setStyleSheet("QGroupBox { font-size: 11px; font-weight: bold; padding-top: 8px; margin-top: 3px; } QGroupBox::title { subcontrol-origin: margin; left: 5px; }")
        channel_layout = QtWidgets.QHBoxLayout()
        channel_layout.setSpacing(5)
        channel_layout.setContentsMargins(3, 3, 3, 3)
        
        buttons = list(self.widgets['buttons']['channel'].values())
        for btn in buttons:
            btn.setStyleSheet("QRadioButton { font-size: 11px; }")
            self.channel_button_group.addButton(btn)
            channel_layout.addWidget(btn)
            btn.clicked.connect(self.update)
        
        channel_group.setLayout(channel_layout)
        controls_layout.addWidget(channel_group)
        
        # Processing controls - horizontal layout
        processing_group = QtWidgets.QGroupBox("Scale")
        if not self.is_dark_mode:
            processing_group.setStyleSheet("QGroupBox { background-color: #F4E6FF; font-size: 11px; font-weight: bold; padding-top: 8px; margin-top: 3px; } QGroupBox::title { subcontrol-origin: margin; left: 5px; }")
        else:
            processing_group.setStyleSheet("QGroupBox { font-size: 11px; font-weight: bold; padding-top: 8px; margin-top: 3px; } QGroupBox::title { subcontrol-origin: margin; left: 5px; }")
        processing_layout = QtWidgets.QHBoxLayout()
        processing_layout.setSpacing(5)
        processing_layout.setContentsMargins(3, 3, 3, 3)
        
        proc_buttons = list(self.widgets['buttons']['processing'].values())
        for btn in proc_buttons:
            btn.setStyleSheet("QRadioButton { font-size: 11px; }")
            processing_layout.addWidget(btn)
            btn.clicked.connect(self.update)
        
        processing_group.setLayout(processing_layout)
        controls_layout.addWidget(processing_group)
        
        # Display controls - horizontal layout
        self.display_group = QtWidgets.QGroupBox("Display")
        if not self.is_dark_mode:
            self.display_group.setStyleSheet("QGroupBox { background-color: #e6f3ff; font-size: 11px; font-weight: bold; padding-top: 8px; margin-top: 3px; } QGroupBox::title { subcontrol-origin: margin; left: 5px; }")
        else:
            self.display_group.setStyleSheet("QGroupBox { font-size: 11px; font-weight: bold; padding-top: 8px; margin-top: 3px; } QGroupBox::title { subcontrol-origin: margin; left: 5px; }")
        display_layout = QtWidgets.QHBoxLayout()
        display_layout.setSpacing(5)
        display_layout.setContentsMargins(3, 3, 3, 3)
        
        disp_buttons = list(self.widgets['buttons']['display'].values())
        for btn in disp_buttons:
            btn.setStyleSheet("QRadioButton { font-size: 11px; }")
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
        
        if data.ndim == 1:
            self.tab_widget.setTabEnabled(0, False)  # Disable Image View tab
            self.tab_widget.setCurrentIndex(1)  # Set line plot as default
        
        # Connect tab change handler
        self.tab_widget.currentChanged.connect(self.on_tab_changed)
        
        # Add tab widget to the main layout
        self.layouts['topDown'].addWidget(self.tab_widget)
        
        self.layouts['topUp'].addWidget(self.widgets['labels']['pixelValue'])
        self.layouts['topUp'].addWidget(self.widgets['labels']['arrayInfo'])

        # Make the left control panel much more compact with minimal spacing
        self.layouts['botLeft'].setSpacing(1)  # Minimal spacing between rows
        self.layouts['botLeft'].setContentsMargins(5, 2, 5, 2)  # Smaller top/bottom margins
        
        self.layouts['botLeft'].addLayout(self.layouts['dims'])
        self.layouts['botLeft'].addLayout(self.layouts['primary'])
        self.layouts['botLeft'].addLayout(self.layouts['secondary'])
        self.layouts['botLeft'].addLayout(self.layouts['slice'])
        
        # Add stretch to prevent the control panel from expanding vertically
        self.layouts['botLeft'].addStretch()
        
        self.layouts['top'].addLayout(self.layouts['topUp'],1)
        self.layouts['top'].addLayout(self.layouts['topDown'],20)  # Give viewport much more space
        
        self.layouts['bot'].addLayout(self.layouts['botLeft'],12)
        self.layouts['bot'].addLayout(self.layouts['botRight'],1)

        # Give the viewport (top) much more weight so it expands when window grows
        self.layouts['main'].addLayout(self.layouts['top'], 10)  # Viewport gets most space
        self.layouts['main'].addLayout(self.layouts['bot'], 1)   # Controls get minimal fixed space
        tmp = QtWidgets.QWidget()
        tmp.setLayout(self.layouts['main'])
        self.setCentralWidget(tmp)
        
        # Initialize dimension controls based on data dimensions
        if len(self.selected_indices) >= 1:
            self.changedIndex(True, 0, self.selected_indices[0], update=False)
        if len(self.selected_indices) >= 2:
            self.changedIndex(True, 1, self.selected_indices[1], update=False)
        self.update_dimension_controls()  # Initialize dimension controls properly
        self.update()
        self.show()
    
    def _detect_dark_mode(self):
        try:
            palette = QtWidgets.QApplication.instance().palette()
            bg_color = palette.color(QtGui.QPalette.Window)
            return bg_color.lightness() < 128
        except:
            return False # assume light
    
    def dimClicked(self, event, label, dim):
        if self.singleton[dim]:
            return
    
        p = QtGui.QPalette()
        
        # If already transformed, any click returns to native
        if self.domain[dim] == Domain.FOURIER:
            # From FFT domain, go back to native (undo)
            self.domain[dim] = Domain.NATIVE
            p.setColor(QtGui.QPalette.WindowText, QtGui.QColor('black'))
            label.setStyleSheet("font-weight: normal;")
            self._apply_ifft(dim)  # Undo the FFT by applying IFFT
        elif self.domain[dim] == Domain.INV_FOURIER:
            # From IFFT domain, go back to native (undo)
            self.domain[dim] = Domain.NATIVE
            p.setColor(QtGui.QPalette.WindowText, QtGui.QColor('black'))
            label.setStyleSheet("font-weight: normal;")
            self._apply_fft(dim)  # Undo the IFFT by applying FFT
        elif event.button() == Qt.QtCore.Qt.RightButton:
            # Right click from native: apply IFFT
            self.domain[dim] = Domain.INV_FOURIER
            p.setColor(QtGui.QPalette.WindowText, QtGui.QColor('green'))
            label.setStyleSheet("font-weight: bold; color: green;")
            self._apply_ifft(dim)
        else:
            # Left click from native: apply FFT
            self.domain[dim] = Domain.FOURIER
            p.setColor(QtGui.QPalette.WindowText, QtGui.QColor('blue'))
            label.setStyleSheet("font-weight: bold; color: blue;")
            self._apply_fft(dim)

        label.setPalette(p)
        self.update_image_view()
        self.update_line_plot()
        
    def _apply_fft(self, dim):
        """Apply forward FFT along specified dimension"""
        self.data = np.fft.ifftshift(np.fft.ifft(np.fft.fftshift(self.data), axis=dim, norm='ortho'))
        
    def _apply_ifft(self, dim):
        """Apply inverse FFT along specified dimension"""
        self.data = np.fft.ifftshift(np.fft.fft(np.fft.fftshift(self.data), axis=dim, norm='ortho'))
            
    
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
        al = changed_scale or changed_channel
        
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
            try:
                view = self.img_view.getView()
                view_rect = view.viewRect()
                if view_rect.width() > 0 and view_rect.height() > 0:
                    # Get the actual widget size to calculate viewport aspect ratio
                    widget_size = view.size()
                    viewport_ratio = widget_size.width() / widget_size.height()
                    if abs(viewport_ratio - 1.0) < 1e-2:
                        aspect_str = '(1:1)'
                    else:
                        aspect_str = f'({viewport_ratio:.2f}:1)'
                else:
                    aspect_str = ''
            except:
                aspect_str = ''
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
                # Thicker anti-aliased line
                pen = pg.mkPen(color=(50, 100, 200), width=2)
                # Keep a reference to the curve so we could later update data without clearing
                self.line_curve = self.plot_widget.plot(line_data, pen=pen, name='')
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
        self.changedIndex(event, 0, self.selected_indices[1], update=False)
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
    
    def keyPressEvent(self, event):
        """Handle keyboard shortcuts"""
        key = event.key()
        modifiers = event.modifiers()
        
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
            
        
        # Pass event to parent if not handled
        super().keyPressEvent(event)
    
    def setColormap(self, colormap_name):
        """Set the colormap for the image view"""
        try:
            if colormap_name == 'gray':
                colormap = pg.colormap.get('gray', source='matplotlib')
            elif colormap_name == 'viridis':
                colormap = pg.colormap.get('viridis')
            elif colormap_name == 'PAL-relaxed':
                colormap = pg.colormap.get('PAL-relaxed')
            elif colormap_name == 'plasma':
                colormap = pg.colormap.get('plasma')
            else:
                print(f"Unknown colormap: {colormap_name}")
                return
            
            # Apply colormap to the image view
            self.img_view.setColorMap(colormap)
            #self.current_colormap = colormap_name
            
        except Exception as e:
            print(f"Failed to set colormap {colormap_name}: {e}")
        
def _run_window(data, title):
    """Opens a window in a separate process which is blocked on exec()"""
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    win = NDSliceWindow(data)
    win.setWindowTitle(title)
    win.show()
    app.exec()

def ndslice(data, title='', block=False):
    if not isinstance(data, np.ndarray):
        raise TypeError("data must be a numpy array")
    if data.ndim < 1:
        raise ValueError("data must have at least 1 dimension")
    
    if block:
        _run_window(data, title)
    else:
        process = mp.Process(target=_run_window, args=(data, title)).start()
    