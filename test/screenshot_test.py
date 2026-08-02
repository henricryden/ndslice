"""Screenshot smoke-test for ndslice.

Directly instantiates NDSliceWindow (bypassing multiprocessing), renders it
headlessly, and saves a PNG. Run via:
  xvfb-run -a python test/screenshot_test.py   # Linux
  python test/screenshot_test.py               # macOS / Windows
"""
import sys
import numpy as np
from pathlib import Path


def make_data():
    """3D complex Gaussian from the README."""
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    z = np.linspace(-5, 5, 50)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    mag = np.exp(-(X**2 + Y**2 + Z**2) / 10)
    pha = np.pi / 4 * (X + Y + Z)
    return (mag * np.exp(1j * pha)).astype(np.complex64)


def take_screenshot(win, path: Path):
    """Grab the window contents and save as PNG."""
    pixmap = win.grab()
    assert not pixmap.isNull(), "grab() returned a null pixmap"
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = pixmap.save(str(path), "PNG")
    assert ok, f"Failed to save screenshot to {path}"
    size = path.stat().st_size
    assert size > 1000, f"Screenshot suspiciously small ({size} bytes)"
    print(f"Screenshot saved: {path}  ({size} bytes)")


def take_widget_composite_screenshot(widgets, path: Path):
    """Grab visible widgets and stitch them into one screenshot."""
    from PyQt6 import QtGui, QtWidgets

    pixmaps = []
    for widget in widgets:
        if widget is None or not widget.isVisible():
            continue
        pixmap = widget.grab()
        assert not pixmap.isNull(), f"grab() returned a null pixmap for {widget!r}"
        pixmaps.append(pixmap)

    assert pixmaps, "No visible widgets available for screenshot"

    spacing = 8
    width = max(pixmap.width() for pixmap in pixmaps)
    height = sum(pixmap.height() for pixmap in pixmaps) + spacing * (len(pixmaps) - 1)
    combined = QtGui.QPixmap(width, height)
    combined.fill(
        QtWidgets.QApplication.palette().color(QtGui.QPalette.ColorRole.Window)
    )

    painter = QtGui.QPainter(combined)
    y = 0
    for pixmap in pixmaps:
        painter.drawPixmap(0, y, pixmap)
        y += pixmap.height() + spacing
    painter.end()

    path.parent.mkdir(parents=True, exist_ok=True)
    ok = combined.save(str(path), "PNG")
    assert ok, f"Failed to save screenshot to {path}"
    size = path.stat().st_size
    assert size > 1000, f"Screenshot suspiciously small ({size} bytes)"
    print(f"Screenshot saved: {path}  ({size} bytes)")


def flush_qt_events(app, count=5):
    for _ in range(count):
        app.processEvents()


def main():
    import argparse
    import pyqtgraph as pg
    from ndslice.ndslice import NDSliceWindow

    parser = argparse.ArgumentParser()
    parser.add_argument('--style', default=None,
                        help='Optional Qt style override, e.g. Fusion or Windows')
    parser.add_argument('--out', default=None,
                        help='Output directory for screenshot (default: test/screenshots/)')
    args = parser.parse_args()

    data = make_data()

    app = pg.mkQApp()
    if args.style:
        app.setStyle(args.style)
    win = NDSliceWindow(data)
    style_name = args.style or app.style().objectName()
    win.setWindowTitle(f"CI test — {sys.platform} — style: {style_name}")
    win.resize(800, 800)
    win.show()

    # Let Qt process events so the image actually renders
    flush_qt_events(app)

    out_dir = Path(args.out) if args.out else Path(__file__).parent / "screenshots"
    out = out_dir / f"screenshot_{sys.platform}.png"
    take_screenshot(win, out)

    button = win._settings_btn
    menu_pos = button.mapToGlobal(button.rect().bottomLeft())
    win._settings_menu.popup(menu_pos)
    flush_qt_events(app)
    assert win._settings_menu.isVisible(), "Settings menu did not open"

    combo = win._colormap_combo
    combo.showPopup()
    flush_qt_events(app)

    settings_out = out_dir / f"screenshot_settings_menu_{sys.platform}.png"
    take_widget_composite_screenshot(
        [win._settings_menu, combo.view().window()],
        settings_out,
    )

    combo.hidePopup()
    win._settings_menu.hide()
    flush_qt_events(app)

    win.close()
    app.quit()
    print("All checks passed.")


if __name__ == "__main__":
    main()
