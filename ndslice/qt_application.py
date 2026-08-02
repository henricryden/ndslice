"""Qt application ownership and appearance handling."""

from PyQt6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

from .config import (
    COLOR_SCHEME_DARK,
    COLOR_SCHEME_LIGHT,
    COLOR_SCHEME_SYSTEM,
    SUPPORTED_COLOR_SCHEMES,
    load_config,
)


_OWNS_APPLICATION_PROPERTY = "_ndslice_owns_qt_application"
_COLOR_SCHEME_PROPERTY = "_ndslice_color_scheme"
_SYSTEM_SCHEME_CONNECTED_PROPERTY = "_ndslice_system_scheme_connected"
_APPEARANCE_REFRESH_PENDING_PROPERTY = "_ndslice_appearance_refresh_pending"


def application_is_ndslice_owned(app=None):
    app = app or QtWidgets.QApplication.instance()
    return app is not None and bool(app.property(_OWNS_APPLICATION_PROPERTY))


def _refresh_widget_styles(app):
    for widget in app.allWidgets():
        style = widget.style()
        style.unpolish(widget)
        style.polish(widget)
        widget.update()


def _refresh_application_appearance(app):
    app.setProperty(_APPEARANCE_REFRESH_PENDING_PROPERTY, False)
    _refresh_widget_styles(app)


def _schedule_appearance_refresh(app):
    if app.property(_APPEARANCE_REFRESH_PENDING_PROPERTY):
        return
    app.setProperty(_APPEARANCE_REFRESH_PENDING_PROPERTY, True)
    QtCore.QTimer.singleShot(
        0,
        lambda owned_app=app: _refresh_application_appearance(owned_app),
    )


def _palette_matches_color_scheme(app, color_scheme):
    window_is_dark = (
        app.palette().color(QtGui.QPalette.ColorRole.Window).lightness() < 128
    )
    return window_is_dark == (color_scheme == QtCore.Qt.ColorScheme.Dark)


def _release_system_color_scheme_refresh(app):
    if app.property(_COLOR_SCHEME_PROPERTY) == COLOR_SCHEME_SYSTEM:
        app.styleHints().unsetColorScheme()


def _on_color_scheme_changed(app, color_scheme):
    if (
        app.property(_COLOR_SCHEME_PROPERTY) == COLOR_SCHEME_SYSTEM
        and color_scheme in (
            QtCore.Qt.ColorScheme.Light,
            QtCore.Qt.ColorScheme.Dark,
        )
        and not _palette_matches_color_scheme(app, color_scheme)
    ):
        app.styleHints().setColorScheme(color_scheme)
        QtCore.QTimer.singleShot(
            0,
            lambda owned_app=app: _release_system_color_scheme_refresh(owned_app),
        )
    _schedule_appearance_refresh(app)


class _ApplicationPaletteChangeFilter(QtCore.QObject):
    def eventFilter(self, watched, event):
        if (
            watched is self.parent()
            and event.type() == QtCore.QEvent.Type.ApplicationPaletteChange
        ):
            _schedule_appearance_refresh(watched)
        return super().eventFilter(watched, event)


def apply_color_scheme(app, color_scheme):
    """Apply a saved color scheme to an ndslice-owned application."""
    if not application_is_ndslice_owned(app):
        return False

    if color_scheme not in SUPPORTED_COLOR_SCHEMES:
        color_scheme = COLOR_SCHEME_SYSTEM

    app.setProperty(_COLOR_SCHEME_PROPERTY, color_scheme)
    style_hints = app.styleHints()
    if color_scheme == COLOR_SCHEME_SYSTEM:
        style_hints.unsetColorScheme()
    elif color_scheme == COLOR_SCHEME_LIGHT:
        style_hints.setColorScheme(QtCore.Qt.ColorScheme.Light)
    elif color_scheme == COLOR_SCHEME_DARK:
        style_hints.setColorScheme(QtCore.Qt.ColorScheme.Dark)
    _schedule_appearance_refresh(app)
    return True


def get_qapplication(config_path=None):
    """Return QApplication and mark it owned when ndslice creates it."""
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = pg.mkQApp()
        app.setProperty(_OWNS_APPLICATION_PROPERTY, True)

    if application_is_ndslice_owned(app):
        if not app.property(_SYSTEM_SCHEME_CONNECTED_PROPERTY):
            app.styleHints().colorSchemeChanged.connect(
                lambda color_scheme, owned_app=app: _on_color_scheme_changed(
                    owned_app,
                    color_scheme,
                )
            )
            palette_change_filter = _ApplicationPaletteChangeFilter(app)
            app.installEventFilter(palette_change_filter)
            app._ndslice_palette_change_filter = palette_change_filter
            app.setProperty(_SYSTEM_SCHEME_CONNECTED_PROPERTY, True)
        config = load_config(config_path)
        apply_color_scheme(app, config.color_scheme)
    return app
