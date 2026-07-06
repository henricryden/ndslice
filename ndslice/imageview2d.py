import numpy as np
from pyqtgraph.Qt import QtGui, QtWidgets
import pyqtgraph as pg
from pyqtgraph.graphicsItems.ImageItem import ImageItem
from pyqtgraph.graphicsItems.ViewBox import ViewBox


class ImageView2D(QtWidgets.QWidget):
    """
    Simplified widget for displaying 2D image data.
    
    Features:
    - 2D image display via ImageItem
    - Zoom/pan via ViewBox
    - Histogram with level controls
    - Auto-ranging and level adjustment
    """
    
    def __init__(self, parent=None, view=None, imageItem=None):
        """
        Parameters
        ----------
        parent : QWidget
            Parent widget
        view : ViewBox
            If specified, this ViewBox will be used for display
        imageItem : ImageItem
            If specified, this ImageItem will be used for display
        """
        super().__init__(parent)
        
        self.image = None
        self.imageDisp = None
        self.mask = None
        self.maskDisp = None
        self.maskLookupTable = None
        self.levelMin = None
        self.levelMax = None
        self.displayMode = 'square_pixels'  # Default to square pixels
        self.autoAspectRatio = None
        
        # Create the UI layout
        self.setupUI()
        
        # Create view if not provided
        if view is None:
            self.view = ViewBox()
        else:
            self.view = view
        self.graphicsView.setCentralItem(self.view)
        self.view.setAspectLocked(True)
        self.view.invertY()
        
        # Create image item if not provided
        if imageItem is None:
            self.imageItem = ImageItem()
        else:
            self.imageItem = imageItem
        self.view.addItem(self.imageItem)

        self.maskImageItem = ImageItem()
        self.maskImageItem.setZValue(10)
        self.maskImageItem.setOpacity(0.5)
        self.maskImageItem.setVisible(False)
        self.view.addItem(self.maskImageItem)
        
        # Setup histogram
        self.histogram.setImageItem(self.imageItem)
        self.histogram.setLevelMode('mono')  # Force mono mode for scalar values
        
        # Initialize levels
        self.levelMin = 0.0
        self.levelMax = 1.0
        
    def setupUI(self):
        """Create the user interface"""
        # Main layout
        self.layout = QtWidgets.QHBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)
        
        # Graphics view for image display
        self.graphicsView = pg.GraphicsView()
        self.layout.addWidget(self.graphicsView, 1)  # Give it most of the space
        
        # Histogram widget
        self.histogram = pg.HistogramLUTWidget()
        self.layout.addWidget(self.histogram)
        
    def setImage(self, img, autoRange=True, autoLevels=True, levels=None, 
                 pos=None, scale=None, transform=None, autoHistogramRange=True):
        """
        Set the image to be displayed.
        
        Parameters
        ----------
        img : np.ndarray
            2D image data to display
        autoRange : bool
            Whether to auto-scale the view to fit the image
        autoLevels : bool
            Whether to auto-adjust the histogram levels
        levels : tuple
            (min, max) levels for the histogram
        pos : tuple
            Position offset for the image
        scale : tuple  
            Scale factors for the image
        transform : QTransform
            Transform to apply to the image
        autoHistogramRange : bool
            Whether to auto-scale the histogram range
        """
        if not isinstance(img, np.ndarray):
            raise TypeError("Image must be a numpy array")
            
        if img.ndim != 2:
            raise ValueError("ImageView2D only supports 2D images")
            
        self.image = img
        self.imageDisp = None
        
        # Update the image display
        self.updateImage(autoHistogramRange=autoHistogramRange)
        
        # Set levels
        if levels is None and autoLevels:
            self.autoLevels()
        elif levels is not None:
            if isinstance(levels, (list, tuple)) and len(levels) == 2:
                self.setLevels(levels[0], levels[1])
            else:
                self.setLevels(*levels)
            
        # Set transform
        if transform is None:
            if pos is not None or scale is not None:
                if pos is None:
                    pos = (0, 0)
                if scale is None:
                    scale = (1, 1)
                transform = QtGui.QTransform()
                transform.translate(pos[0], pos[1])
                transform.scale(scale[0], scale[1])
        
        if transform is not None:
            self.imageItem.setTransform(transform)
            self.maskImageItem.setTransform(transform)
            
        # Update aspect ratio based on display mode
        self._updateAspectRatio()
        
        # Auto range the view
        if autoRange:
            self.autoRange()
            
    def updateImage(self, autoHistogramRange=True):
        """Update the displayed image"""
        if self.image is None:
            return
            
        # For 2D images, we can display directly
        self.imageDisp = self.image
        
        # Calculate min/max levels from the image data for histogram
        self._updateImageLevels()
        
        # Set the image data
        self.imageItem.setImage(self.imageDisp, autoLevels=False)
        
        # Update histogram range if requested
        if autoHistogramRange:
            self.histogram.setHistogramRange(self.levelMin, self.levelMax)

    def setMaskImage(self, img, levels=None):
        if img is None:
            self.clearMask()
            return

        if not isinstance(img, np.ndarray):
            raise TypeError("Mask image must be a numpy array")

        if img.ndim != 2:
            raise ValueError("Mask image must be 2D")

        self.mask = img
        self.maskDisp = None
        self.updateMaskImage()

        if levels is None:
            levels = (0, 32)
        self.maskImageItem.setLevels(levels)

        self.maskImageItem.setVisible(True)

    def updateMaskImage(self):
        if self.mask is None:
            return

        self.maskDisp = self.mask
        self.maskImageItem.setImage(
            self.maskDisp,
            autoLevels=False,
            lut=self.maskLookupTable,
            levels=(0, 32),
        )
            
    def autoRange(self):
        """Auto scale and pan the view to fit the image"""
        if self.imageDisp is not None:
            self.view.autoRange()
            
    def _updateImageLevels(self):
        """Update the min/max levels from the current image data"""
        if self.imageDisp is not None:
            # Use the same approach as the original ImageView
            finite_data = self.imageDisp[np.isfinite(self.imageDisp)]
            if len(finite_data) > 0:
                self.levelMin = float(np.min(finite_data))
                self.levelMax = float(np.max(finite_data))
            else:
                self.levelMin = 0.0
                self.levelMax = 1.0

    def autoLevels(self):
        """Automatically set the histogram levels based on image data"""
        if self.imageDisp is not None:
            self._updateImageLevels()
            self.setLevels(self.levelMin, self.levelMax)
                
    def setLevels(self, min_level, max_level):
        """Set the histogram levels"""
        self.histogram.setLevels(min_level, max_level)
        
    def getLevels(self):
        """Get the current histogram levels"""
        return self.histogram.getLevels()

    def setHistogramRange(self, min_val, max_val):
        """Set the range of the histogram"""
        self.histogram.setHistogramRange(min_val, max_val)
        
    def getProcessedImage(self):
        """Get the processed image data"""
        return self.imageDisp
        
    def getView(self):
        """Get the ViewBox containing the image"""
        return self.view
        
    def getImageItem(self):
        """Get the ImageItem"""
        return self.imageItem
        
    def getHistogramWidget(self):
        """Get the histogram widget"""
        return self.histogram
        
    def clear(self):
        """Clear the displayed image"""
        self.image = None
        self.imageDisp = None
        self.imageItem.clear()
        
    def setColorMap(self, colormap):
        """Set the color map for the histogram"""
        self.histogram.gradient.setColorMap(colormap)

    def setMaskLookupTable(self, lut):
        self.maskLookupTable = lut
        self.maskImageItem.setLookupTable(lut)
        self.maskImageItem.setLevels((0, 32))
        if self.mask is not None:
            self.updateMaskImage()

    def setMaskOpacity(self, opacity):
        self.maskImageItem.setOpacity(float(opacity))

    def setMaskVisible(self, visible):
        has_mask = self.mask is not None
        self.maskImageItem.setVisible(bool(visible) and has_mask)

    def clearMask(self):
        self.mask = None
        self.maskDisp = None
        self.maskImageItem.clear()
        self.maskImageItem.setVisible(False)
        
    def setDisplayMode(self, mode, auto_aspect_ratio=None):
        """Set the display mode.

        Modes:
        - 'square_pixels': force square pixel display (aspect ratio 1.0)
        - 'square_fov'   : lock aspect ratio to image width/height (field of view square)
        - 'auto'         : lock to provided physical aspect, or fit if unavailable
        """
        if mode == 'fit':
            mode = 'auto'
        if mode not in ('square_pixels', 'square_fov', 'auto'):
            raise ValueError(f"Unknown display mode: {mode}")
        self.displayMode = mode
        self.autoAspectRatio = auto_aspect_ratio
        self._updateAspectRatio()

    def setAutoAspectRatio(self, auto_aspect_ratio):
        self.autoAspectRatio = auto_aspect_ratio
        if self.displayMode == 'auto':
            self._updateAspectRatio()
        
    def _updateAspectRatio(self):
        """Update the aspect ratio based on display mode"""
        if self.image is None:
            return
            
        if self.displayMode == 'square_pixels':
            # Square pixels: maintain 1:1 aspect ratio
            self.view.setAspectLocked(True, ratio=1.0)
        elif self.displayMode == 'square_fov':
            # Square FOV: adjust aspect ratio based on image dimensions
            height, width = self.image.shape
            aspect_ratio = width / height
            self.view.setAspectLocked(True, ratio=aspect_ratio)
        elif self.displayMode == 'auto':
            if self.autoAspectRatio is None:
                self.view.setAspectLocked(False)
                self.view.autoRange()
            else:
                self.view.setAspectLocked(True, ratio=self.autoAspectRatio)
        
        # Trigger a refresh of the view
        if hasattr(self, 'imageItem') and self.imageItem is not None:
            self.view.autoRange()

    # --- Qt Events -----------------------------------------------------
    def resizeEvent(self, event):
        """On resize, if in auto-fit mode keep the image fully visible."""
        super().resizeEvent(event)
        if self.displayMode == 'auto' and self.autoAspectRatio is None and self.image is not None:
            self.view.autoRange()
