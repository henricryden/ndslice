from PyQt6 import QtCore, QtGui, QtWidgets


class RangeSlider(QtWidgets.QWidget):
    """A minimal horizontal two-handle slider for inclusive integer ranges."""

    valuesChanged = QtCore.pyqtSignal(int, int)

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
        return QtCore.QSize(240, 30)

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

        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(palette.midlight())
        painter.drawRoundedRect(groove_rect, 3, 3)

        selected_rect = QtCore.QRectF(
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
        if event.button() != QtCore.Qt.MouseButton.LeftButton:
            event.ignore()
            return

        point = event.position()
        self._active_handle = self._closest_handle(point)
        self._move_active_handle(point.x())
        event.accept()

    def mouseMoveEvent(self, event):
        point = event.position()
        if self._active_handle is not None:
            self._move_active_handle(point.x())
            event.accept()
            return

        self.setCursor(
            QtGui.QCursor(
                QtCore.Qt.CursorShape.SizeHorCursor if self._is_over_handle(point) else QtCore.Qt.CursorShape.ArrowCursor
            )
        )
        event.ignore()

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._active_handle = None
            self.update()
        event.accept()

    def leaveEvent(self, event):
        self.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.ArrowCursor))
        super().leaveEvent(event)

    def _handle_radius(self):
        return 8.0

    def _groove_rect(self):
        margin = self._handle_radius() + 4.0
        return QtCore.QRectF(
            margin,
            self.height() / 2.0 - 3.0,
            max(self.width() - 2.0 * margin, 1.0),
            6.0,
        )

    def _handle_center(self, value):
        groove_rect = self._groove_rect()
        span = max(self._maximum - self._minimum, 1)
        ratio = (value - self._minimum) / span
        return QtCore.QPointF(groove_rect.left() + groove_rect.width() * ratio, groove_rect.center().y())

    def _handle_rect(self, value):
        center = self._handle_center(value)
        radius = self._handle_radius()
        return QtCore.QRectF(center.x() - radius, center.y() - radius, radius * 2.0, radius * 2.0)

    def _paint_handle(self, painter, center, active):
        radius = self._handle_radius()
        palette = self.palette()
        rect = QtCore.QRectF(center.x() - radius, center.y() - radius, radius * 2.0, radius * 2.0)
        painter.setPen(QtGui.QPen(palette.shadow().color(), 1.0))
        painter.setBrush(palette.highlight() if active else palette.button())
        painter.drawEllipse(rect)

        inner_rect = rect.adjusted(3.0, 3.0, -3.0, -3.0)
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
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
