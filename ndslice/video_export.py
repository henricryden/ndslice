import numpy as np
import os
from pyqtgraph.Qt import QtWidgets, QtCore, QtGui
from .range_slider import RangeSlider

# Compatibility for PyQt5/PySide6 signal naming
try:
    Signal = QtCore.pyqtSignal
except AttributeError:
    Signal = QtCore.Signal

# Try to import imageio for MP4/WebM support
try:
    import imageio
    import imageio_ffmpeg
    HAS_IMAGEIO = True
except ImportError:
    print("imageio not available. MP4/WebM export will be disabled.")
    HAS_IMAGEIO = False


def _message_box_enum(enum_group, enum_name, fallback_name=None):
    """Return a QMessageBox enum value across Qt5 and Qt6 bindings."""
    message_box = QtWidgets.QMessageBox
    group = getattr(message_box, enum_group, None)
    if group is not None and hasattr(group, enum_name):
        return getattr(group, enum_name)
    return getattr(message_box, fallback_name or enum_name)


def _exec_dialog(dialog):
    """Execute a dialog across Qt5 and Qt6 bindings."""
    if hasattr(dialog, "exec"):
        return dialog.exec()
    return dialog.exec_()


class VideoExportWorker(QtCore.QThread):
    """Worker thread for video export with progress signals"""
    progress_updated = Signal(int, str)  # (current_frame, status_text)
    export_finished = Signal(bool, str)  # (success, message)
    
    def __init__(self, data, export_dim, output_path, fps, format_type, 
                 channel_func, processing_func, slice_indices, selected_indices, 
                 singleton, levels=None, transpose=False, pixel_ratio_mode='square_pixels',
                 display_mode='square_pixels', widget_ratio=1.0, axis_flipped=None,
                 lut=None, window_level_mode='displayed', mask_data=None,
                 mask_lut=None, mask_enabled=False, mask_opacity=0.5,
                 frame_start=0, frame_stop=None, display_aspect_ratio=None):
        super().__init__()
        self.transpose = transpose
        self.pixel_ratio_mode = pixel_ratio_mode
        self.display_mode = display_mode
        self.widget_ratio = widget_ratio
        self.display_aspect_ratio = display_aspect_ratio
        self.axis_flipped = axis_flipped or []
        self.lut = lut
        self.mask_data = mask_data
        self.mask_lut = mask_lut
        self.mask_enabled = bool(mask_enabled)
        self.mask_opacity = max(0.0, min(float(mask_opacity), 1.0))
        self.data = data
        self.export_dim = export_dim
        self.output_path = output_path
        self.fps = fps
        self.format_type = format_type
        self.channel_func = channel_func
        self.processing_func = processing_func
        self.slice_indices = slice_indices
        self.selected_indices = selected_indices
        self.singleton = singleton
        self.levels = levels
        self.window_level_mode = (window_level_mode or 'displayed')
        data_frame_count = self.data.shape[self.export_dim]
        self.frame_start = max(0, min(int(frame_start), data_frame_count - 1))
        self.frame_stop = data_frame_count if frame_stop is None else int(frame_stop)
        self.frame_stop = max(self.frame_start + 1, min(self.frame_stop, data_frame_count))
        self._is_running = True
        
    def run(self):
        """Main export routine"""
        try:
            frame_indices = list(self._frame_indices())
            total_frames = len(frame_indices)
            frames = []
            
            # Generate all frames first to compute consistent levels if needed
            for progress_idx, frame_idx in enumerate(frame_indices, start=1):
                if not self._is_running:
                    self.export_finished.emit(False, "Export cancelled")
                    return

                frames.append(self._render_frame(frame_idx))
                
                status = f"Processing frame {progress_idx}/{total_frames}"
                self.progress_updated.emit(progress_idx, status)
            
            # Save video file
            self.progress_updated.emit(total_frames, "Encoding video...")
            if self.format_type == 'gif':
                self._save_gif(frames)
            elif self.format_type == 'png':
                self._save_png_frames(frames)
            elif self.format_type in ('mp4', 'webm'):
                if not HAS_IMAGEIO:
                    raise RuntimeError(f"imageio not installed. Cannot save {self.format_type.upper()} files. "
                                     f"Install with: pip install imageio[ffmpeg]")
                self._save_video(frames)
            
            self.export_finished.emit(True, f"Video exported successfully to {self.output_path}")
            
        except Exception as e:
            self.export_finished.emit(False, f"Export failed: {str(e)}")

    def _frame_indices(self):
        return range(self.frame_start, self.frame_stop)

    def frame_count(self):
        return max(0, self.frame_stop - self.frame_start)

    def _render_frame(self, frame_idx):
        frame_slice = list(self.slice_indices)
        frame_slice[self.export_dim] = slice(frame_idx, frame_idx + 1)
        frame_slice = tuple(frame_slice)

        frame_data = self.data[frame_slice]
        if self.channel_func is not None:
            frame_data = self.channel_func(frame_data)
        if self.processing_func is not None:
            frame_data = self.processing_func(frame_data)

        frame_data = np.squeeze(frame_data)
        if (str(self.window_level_mode).lower() == 'displayed') and (self.levels is not None):
            vmin, vmax = self.levels
        else:
            vmin = np.nanmin(frame_data)
            vmax = np.nanmax(frame_data)

        if vmax > vmin:
            normalized = (frame_data - vmin) / (vmax - vmin)
        else:
            normalized = np.zeros_like(frame_data)

        frame_uint8 = np.clip(normalized * 255, 0, 255).astype(np.uint8)
        frame_rgb = self._colorize_frame(frame_uint8)
        frame_rgb = self._orient_rgb_frame(frame_rgb)

        mask_frame = self._mask_frame(frame_slice, frame_rgb.shape[:2])
        frame_rgb = self._composite_mask(frame_rgb, mask_frame)
        return self._apply_pixel_ratio(frame_rgb)

    def _colorize_frame(self, frame_uint8):
        lut = getattr(self, 'lut', None)
        if frame_uint8.ndim == 2 and lut is not None:
            try:
                lut_arr = np.asarray(lut)
                if lut_arr.shape[1] >= 3:
                    lut_rgb = np.asarray(lut_arr[:, :3], dtype=np.uint8)
                    return lut_rgb[frame_uint8]
            except Exception:
                pass

        if frame_uint8.ndim == 2:
            return np.stack([frame_uint8] * 3, axis=2)
        return frame_uint8

    def _orient_rgb_frame(self, frame_rgb):
        if self.transpose:
            frame_rgb = np.transpose(frame_rgb, (1, 0, 2))
        return self._apply_axis_flips(frame_rgb)

    def _apply_axis_flips(self, frame):
        try:
            primary = self.selected_indices[0]
            if not self.axis_flipped[primary]:
                frame = np.flipud(frame)
            secondary = self.selected_indices[1]
            if self.axis_flipped[secondary]:
                frame = np.fliplr(frame)
        except Exception:
            pass
        return frame

    def _mask_frame(self, frame_slice, image_shape):
        if not self.mask_enabled or self.mask_data is None or self.mask_lut is None:
            return None

        mask_frame = np.squeeze(self.mask_data[frame_slice])
        if mask_frame.ndim != 2:
            shape_before_orientation = tuple(reversed(image_shape)) if self.transpose else image_shape
            mask_frame = np.broadcast_to(mask_frame, shape_before_orientation)
        if self.transpose:
            mask_frame = np.transpose(mask_frame)
        return self._apply_axis_flips(mask_frame)

    def _composite_mask(self, frame_rgb, mask_frame):
        if mask_frame is None:
            return frame_rgb

        mask_lut = np.asarray(self.mask_lut)
        if mask_lut.ndim != 2 or mask_lut.shape[1] < 3:
            return frame_rgb

        mask_indices = np.asarray(mask_frame, dtype=np.intp)
        valid = (mask_indices >= 0) & (mask_indices < len(mask_lut))
        clipped = np.clip(mask_indices, 0, len(mask_lut) - 1)
        mask_rgba = mask_lut[clipped]
        mask_rgb = np.asarray(mask_rgba[..., :3], dtype=np.float32)
        if mask_rgba.shape[-1] >= 4:
            alpha = np.asarray(mask_rgba[..., 3], dtype=np.float32) / 255.0
        else:
            alpha = (mask_indices != 0).astype(np.float32)
        alpha = alpha * self.mask_opacity * valid.astype(np.float32)
        alpha = alpha[..., np.newaxis]

        frame_float = frame_rgb.astype(np.float32)
        blended = frame_float * (1.0 - alpha) + mask_rgb * alpha
        return np.clip(np.rint(blended), 0, 255).astype(np.uint8)
    
    def _save_gif(self, frames):
        """Save frames as GIF using PIL"""
        try:
            from PIL import Image
        except ImportError:
            raise RuntimeError("PIL not available. GIF export requires Pillow. "
                             "Install with: pip install Pillow")
        
        pil_frames = [Image.fromarray(frame) for frame in frames]
        pil_frames[0].save(
            self.output_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=int(1000 / self.fps),
            loop=0,
            optimize=False
        )
    
    def _save_video(self, frames):
        """Save frames as MP4/WebM using imageio"""
        
        try:
        # Compute required padding so width/height are divisible by 16 (common macro_block_size)
            mb = 16
            h, w = frames[0].shape[:2]
            pad_h = (mb - (h % mb)) % mb
            pad_w = (mb - (w % mb)) % mb

            if pad_h != 0 or pad_w != 0:
                padded_frames = []
                for f in frames:
                    # center pad so image remains centered rather than shifted
                    top = pad_h // 2
                    bottom = pad_h - top
                    left = pad_w // 2
                    right = pad_w - left
                    pad_cfg = ((top, bottom), (left, right), (0, 0))
                    f_padded = np.pad(f, pad_cfg, mode='constant', constant_values=0)
                    padded_frames.append(f_padded)
                frames_to_write = padded_frames
            else:
                frames_to_write = frames

            # Ensure frames are uint8 contiguous arrays
            proc_frames = [np.ascontiguousarray(f.astype(np.uint8)) for f in frames_to_write]

            # Select writer options by container to ensure compatible codecs
            writer = None
            if self.format_type == 'mp4':
                # Use H.264 and set pixel format for widest compatibility
                try:
                    # Use libx264, request yuv420p pixel format, and a reasonable CRF for quality
                    writer = imageio.get_writer(
                        self.output_path,
                        fps=self.fps,
                        codec='libx264',
                        ffmpeg_params=['-pix_fmt', 'yuv420p', '-preset', 'medium', '-crf', '23']
                    )
                except Exception:
                    # Try without explicit codec if libx264 isn't available
                    writer = imageio.get_writer(self.output_path, fps=self.fps, ffmpeg_params=['-pix_fmt', 'yuv420p', '-preset', 'medium', '-crf', '23'])
            elif self.format_type == 'webm':
                # WebM only supports VP8/VP9/AV1 — try VP9 then fall back to VP8
                try:
                    # Prefer VP9 with constrained quality (CRF) and bitrate=0 for constant-quality mode
                    writer = imageio.get_writer(
                        self.output_path,
                        fps=self.fps,
                        codec='libvpx-vp9',
                        ffmpeg_params=['-crf', '30', '-b:v', '0']
                    )
                except Exception:
                    try:
                        # Fallback to VP8; give a reasonable target bitrate
                        writer = imageio.get_writer(self.output_path, fps=self.fps, codec='libvpx', ffmpeg_params=['-b:v', '1M'])
                    except Exception:
                        # Last-resort: default writer (may fail)
                        writer = imageio.get_writer(self.output_path, fps=self.fps)
            else:
                # Generic fallback
                writer = imageio.get_writer(self.output_path, fps=self.fps)

            for frame in proc_frames:
                writer.append_data(frame)
            writer.close()
        except Exception:
            # Fallback: try forcing macro_block_size=1 (may reduce compatibility)
            try:
                proc_frames = [np.ascontiguousarray(f.astype(np.uint8)) for f in frames]
                writer = imageio.get_writer(self.output_path, fps=self.fps, macro_block_size=1)
                for frame in proc_frames:
                    writer.append_data(frame)
                writer.close()
            except Exception as e:
                raise RuntimeError(f"Failed to write video: {e}")


    def _save_png_frames(self, frames):
        """Save each frame as an individual PNG file into the output directory."""
        try:
            from PIL import Image
        except ImportError:
            raise RuntimeError("PIL not available. PNG export requires Pillow. "
                               "Install with: pip install Pillow")

        out_dir = getattr(self, 'output_path', None)
        if not out_dir:
            raise RuntimeError("No output directory specified for PNG frames.")

        # Create output directory if it doesn't exist
        try:
            os.makedirs(out_dir, exist_ok=True)
        except Exception as e:
            raise RuntimeError(f"Failed to create output directory: {e}")

        total = len(frames)
        digits = max(4, len(str(total)))

        for idx, frame in enumerate(frames, start=1):
            try:
                img = Image.fromarray(np.ascontiguousarray(frame.astype(np.uint8)))
                fname = f"frame_{idx:0{digits}d}.png"
                out_path = os.path.join(out_dir, fname)
                img.save(out_path)
            except Exception as e:
                raise RuntimeError(f"Failed to save PNG frame {idx}: {e}")

    
    def stop(self):
        """Stop the export process"""
        self._is_running = False

    def _apply_pixel_ratio(self, frame_rgb):
        """Scale frame according to requested pixel ratio/display mode."""
        try:
            h, w = frame_rgb.shape[:2]
            target_w, target_h = w, h

            mode = (self.pixel_ratio_mode or 'square_pixels').lower()
            if mode == 'square_fov':
                side = max(w, h)
                target_w = target_h = side
            elif mode == 'displayed':
                dm = (self.display_mode or 'square_pixels').lower()
                if dm == 'square_fov':
                    side = max(w, h)
                    target_w = target_h = side
                elif dm == 'auto' and self.display_aspect_ratio is not None:
                    pixel_ratio = self.display_aspect_ratio if self.display_aspect_ratio > 0 else 1.0
                    ratio = (w / h) * pixel_ratio if h > 0 else pixel_ratio
                    target_w = max(w, h)
                    target_h = max(1, int(target_w / ratio))
                elif dm in ('fit', 'auto'):
                    ratio = self.widget_ratio if self.widget_ratio > 0 else 1.0
                    target_w = max(w, h)
                    target_h = max(1, int(target_w / ratio))
                else:
                    # square_pixels
                    target_w, target_h = w, h
            # square_pixels default leaves as-is

            if target_w == w and target_h == h:
                return frame_rgb

            try:
                from PIL import Image
                img = Image.fromarray(frame_rgb)
                img = img.resize((int(target_w), int(target_h)), Image.Resampling.BILINEAR)
                return np.array(img)
            except Exception:
                # Fallback: simple numpy repeat
                scale_w = max(1, int(round(target_w / w)))
                scale_h = max(1, int(round(target_h / h)))
                return np.repeat(np.repeat(frame_rgb, scale_h, axis=0), scale_w, axis=1)
        except Exception:
            return frame_rgb


class VideoExportDialog(QtWidgets.QDialog):
    """Progress dialog for video export"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Exporting Video")
        self.setModal(True)
        self.setMinimumWidth(400)
        self.worker = None
        self.setup_ui()
    
    def setup_ui(self):
        layout = QtWidgets.QVBoxLayout()
        
        # Status label
        self.status_label = QtWidgets.QLabel("Initializing...")
        layout.addWidget(self.status_label)
        
        # Progress bar
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        layout.addWidget(self.progress_bar)
        
        # Details text
        self.details_label = QtWidgets.QLabel("")
        self.details_label.setWordWrap(True)
        layout.addWidget(self.details_label)
        
        # Cancel button
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addStretch()
        self.cancel_button = QtWidgets.QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.cancel_export)
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def start_export(self, worker, total_frames):
        """Start export with worker thread"""
        self.worker = worker
        self.total_frames = total_frames
        self.progress_bar.setRange(0, total_frames)
        self.progress_bar.setValue(0)
        
        # Connect signals
        self.worker.progress_updated.connect(self.on_progress_updated)
        self.worker.export_finished.connect(self.on_export_finished)
        
        # Start worker
        self.worker.start()
        
        # Show dialog
        _exec_dialog(self)
    
    def on_progress_updated(self, frame_idx, status_text):
        """Update progress display"""
        self.status_label.setText(status_text)
        self.progress_bar.setValue(frame_idx)
        percent = int(100 * frame_idx / self.total_frames) if self.total_frames > 0 else 0
        self.details_label.setText(f"{frame_idx}/{self.total_frames} frames ({percent}%)")

        QtWidgets.QApplication.processEvents()
    
    def on_export_finished(self, success, message):
        """Handle export completion"""
        self.worker.wait()  # Wait for thread to finish
        
        if success:
            # Show a message box with optional buttons to open dir or file
            mb = QtWidgets.QMessageBox(self)
            mb.setIcon(_message_box_enum("Icon", "Information"))
            mb.setWindowTitle("Export Complete")
            mb.setText(message)
            action_role = _message_box_enum("ButtonRole", "ActionRole")
            open_dir_btn = mb.addButton("Open directory", action_role)
            open_file_btn = mb.addButton("Open video", action_role)
            ok_btn = mb.addButton(_message_box_enum("StandardButton", "Ok"))
            _exec_dialog(mb)

            clicked = mb.clickedButton()
            try:
                out_path = getattr(self.worker, 'output_path', None)
                if clicked == open_dir_btn and out_path:
                    dir = QtCore.QFileInfo(out_path).absolutePath()
                    QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(dir))
                elif clicked == open_file_btn and out_path:
                    QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(out_path))
            except Exception:
                pass
        else:
            QtWidgets.QMessageBox.critical(self, "Export Failed", message)

        self.accept()
    
    def cancel_export(self):
        """Cancel the export"""
        if self.worker:
            self.worker.stop()
        self.reject()


class VideoExportSettingsDialog(QtWidgets.QDialog):
    """Dialog to configure export settings"""
    
    def __init__(self, parent=None, export_dim=None, data_shape=None,
                 has_mask=False, mask_visible=False, mask_opacity=0.5,
                 preview_callback=None):
        super().__init__(parent)
        self.setWindowTitle("Export Video Settings")
        self.setModal(True)
        self.export_dim = export_dim
        self.data_shape = data_shape
        self.preview_callback = preview_callback
        self.has_mask = bool(has_mask)
        self.mask_visible = bool(mask_visible)
        self.mask_opacity = max(0.0, min(float(mask_opacity), 1.0))
        self.mask_checkbox = None
        self.mask_opacity_slider = None
        self.mask_opacity_label = None
        self.range_slider = None
        self.start_spinbox = None
        self.end_spinbox = None
        self.info_label = None
        self.setup_ui()
    
    def setup_ui(self):
        layout = QtWidgets.QVBoxLayout()
        
        # Format selection
        format_layout = QtWidgets.QHBoxLayout()
        format_layout.addWidget(QtWidgets.QLabel("Format:"))
        self.format_combo = QtWidgets.QComboBox()
        
        # Offer PNG frames export (saves each frame as a separate PNG)
        if (HAS_IMAGEIO):
            format_options = [
                ("PNG frames", "png", True),
                ("GIF", "gif", True),
                ("MP4", "mp4", True),
                ("WebM", "webm", True),
            ]
        else:
            format_options = [
                ("PNG frames", "png", True),
                ("GIF", "gif", True),
                ("MP4 (requires imageio-ffmpeg)", "mp4", False),
                ("WebM (requires imageio-ffmpeg)", "webm", False),
            ]


        for label, fmt, enabled in format_options:
            self.format_combo.addItem(label, fmt)
            idx = self.format_combo.count() - 1
            if not enabled:
                item = self.format_combo.model().item(idx)
                if item is not None:
                    item.setEnabled(False)

        if HAS_IMAGEIO:
            mp4_index = self.format_combo.findData("mp4")
            if mp4_index >= 0:
                self.format_combo.setCurrentIndex(mp4_index)

        format_layout.addWidget(self.format_combo)
        # Disable/enable other options depending on chosen format
        self.format_combo.currentIndexChanged.connect(self._on_format_changed)
        # Initialize UI state based on default selection
        self._on_format_changed()
        layout.addLayout(format_layout)

        # FPS setting
        fps_layout = QtWidgets.QHBoxLayout()
        fps_layout.addWidget(QtWidgets.QLabel("FPS:"))
        self.fps_spinbox = QtWidgets.QSpinBox()
        self.fps_spinbox.setRange(1, 60)
        self.fps_spinbox.setValue(10)
        fps_layout.addWidget(self.fps_spinbox)
        layout.addLayout(fps_layout)

        # Pixel ratio selection
        ratio_layout = QtWidgets.QHBoxLayout()
        ratio_layout.addWidget(QtWidgets.QLabel("Pixel ratio:"))
        self.ratio_combo = QtWidgets.QComboBox()
        self.ratio_combo.addItems(["Square pixels", "Square FOV", "Displayed"])
        self.ratio_combo.setCurrentText("Displayed")
        ratio_layout.addWidget(self.ratio_combo)
        layout.addLayout(ratio_layout)

        # Window/Level behavior
        wl_layout = QtWidgets.QHBoxLayout()
        wl_layout.addWidget(QtWidgets.QLabel("Window/Level:"))
        self.wl_combo = QtWidgets.QComboBox()
        # Displayed: use the currently displayed levels (default); Rescale: auto-adjust per frame
        self.wl_combo.addItem("Displayed", "displayed")
        self.wl_combo.addItem("Rescale", "rescale")
        wl_layout.addWidget(self.wl_combo)
        layout.addLayout(wl_layout)

        range_layout = QtWidgets.QGridLayout()
        range_layout.setHorizontalSpacing(8)
        range_layout.setVerticalSpacing(4)
        max_index = max(0, self.data_shape[self.export_dim] - 1)

        self.range_slider = RangeSlider(self, 0, max_index)
        self.start_spinbox = QtWidgets.QSpinBox()
        self.end_spinbox = QtWidgets.QSpinBox()
        for spinbox in (self.start_spinbox, self.end_spinbox):
            spinbox.setRange(0, max_index)
            spinbox.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
            spinbox.setFixedWidth(70)

        self.end_spinbox.setValue(max_index)
        self.start_spinbox.valueChanged.connect(lambda value: self.end_spinbox.setMinimum(value))
        self.end_spinbox.valueChanged.connect(lambda value: self.start_spinbox.setMaximum(value))
        self.start_spinbox.valueChanged.connect(
            lambda value: self.range_slider.setValues(value, self.end_spinbox.value())
        )
        self.end_spinbox.valueChanged.connect(
            lambda value: self.range_slider.setValues(self.start_spinbox.value(), value)
        )
        self.start_spinbox.valueChanged.connect(lambda _value: self._update_frame_count_label())
        self.end_spinbox.valueChanged.connect(lambda _value: self._update_frame_count_label())
        self.start_spinbox.valueChanged.connect(self._preview_frame)
        self.end_spinbox.valueChanged.connect(self._preview_frame)
        self.range_slider.valuesChanged.connect(self._sync_range_spinboxes)
        self.range_slider.setValues(0, max_index)

        range_layout.addWidget(QtWidgets.QLabel("Slices:"), 0, 0)
        range_layout.addWidget(self.range_slider, 0, 1, 1, 4)
        range_layout.addWidget(QtWidgets.QLabel("start"), 1, 0)
        range_layout.addWidget(self.start_spinbox, 1, 1)
        range_layout.addItem(
            QtWidgets.QSpacerItem(24, 1, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum),
            1,
            2,
        )
        range_layout.addWidget(QtWidgets.QLabel("end"), 1, 3)
        range_layout.addWidget(self.end_spinbox, 1, 4)
        range_layout.setColumnStretch(2, 1)
        layout.addLayout(range_layout)

        if self.has_mask:
            mask_layout = QtWidgets.QHBoxLayout()
            self.mask_checkbox = QtWidgets.QCheckBox("Mask")
            self.mask_checkbox.setChecked(self.mask_visible)
            self.mask_checkbox.toggled.connect(self._on_mask_toggled)
            mask_layout.addWidget(self.mask_checkbox)

            self.mask_opacity_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            self.mask_opacity_slider.setRange(0, 100)
            self.mask_opacity_slider.setValue(int(round(self.mask_opacity * 100)))
            self.mask_opacity_slider.setFixedWidth(90)
            self.mask_opacity_slider.valueChanged.connect(self._on_mask_opacity_changed)
            mask_layout.addWidget(self.mask_opacity_slider)

            self.mask_opacity_label = QtWidgets.QLabel(f"{int(round(self.mask_opacity * 100))}%")
            self.mask_opacity_label.setMinimumWidth(36)
            mask_layout.addWidget(self.mask_opacity_label)
            layout.addLayout(mask_layout)
            self._on_mask_toggled(self.mask_checkbox.isChecked())
        
        # Info
        self.info_label = QtWidgets.QLabel()
        self.info_label.setStyleSheet("color: gray; font-size: 11px;")
        layout.addWidget(self.info_label)
        self._update_frame_count_label()
        
        # Buttons
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addStretch()
        
        self.ok_button = QtWidgets.QPushButton("Export")
        self.ok_button.clicked.connect(self.accept)
        button_layout.addWidget(self.ok_button)
        
        self.cancel_button = QtWidgets.QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_button)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
    
    def get_settings(self):
        """Get user-selected settings"""
        return {
            'format': (self.format_combo.currentData() or 'gif'),
            'fps': self.fps_spinbox.value(),
            'pixel_ratio': self.ratio_combo.currentText().lower().replace(' ', '_'),
            'window_level': (self.wl_combo.currentData() or 'displayed'),
            'frame_start': self.start_spinbox.value(),
            'frame_stop': self.end_spinbox.value() + 1,
            'mask_enabled': bool(self.has_mask and self.mask_checkbox is not None and self.mask_checkbox.isChecked()),
            'mask_opacity': (
                float(self.mask_opacity_slider.value()) / 100.0
                if self.has_mask and self.mask_opacity_slider is not None
                else 0.0
            ),
        }

    def _on_format_changed(self, *_args):
        """Adjust UI when format changes (disable FPS for PNG frames)."""
        try:
            fmt = self.format_combo.currentData()
            if fmt == 'png':
                self.fps_spinbox.setEnabled(False)
            else:
                self.fps_spinbox.setEnabled(True)
        except Exception:
            pass

    def _on_mask_toggled(self, checked):
        if self.mask_opacity_slider is not None:
            self.mask_opacity_slider.setEnabled(bool(checked))
        if self.mask_opacity_label is not None:
            self.mask_opacity_label.setEnabled(bool(checked))

    def _on_mask_opacity_changed(self, value):
        if self.mask_opacity_label is not None:
            self.mask_opacity_label.setText(f"{int(value)}%")

    def _sync_range_spinboxes(self, lower_value, upper_value):
        old_lower = self.start_spinbox.value()
        old_upper = self.end_spinbox.value()
        self.start_spinbox.blockSignals(True)
        self.end_spinbox.blockSignals(True)
        self.start_spinbox.setValue(lower_value)
        self.end_spinbox.setValue(upper_value)
        self.start_spinbox.blockSignals(False)
        self.end_spinbox.blockSignals(False)
        self._update_frame_count_label()
        if lower_value != old_lower:
            self._preview_frame(lower_value)
        elif upper_value != old_upper:
            self._preview_frame(upper_value)

    def _selected_frame_count(self):
        return max(0, self.end_spinbox.value() - self.start_spinbox.value() + 1)

    def _update_frame_count_label(self):
        if self.info_label is not None:
            self.info_label.setText(
                f"Exporting dimension {self.export_dim} ({self._selected_frame_count()} frames)"
            )

    def _preview_frame(self, frame_idx):
        if self.preview_callback is None:
            return
        try:
            self.preview_callback(int(frame_idx))
        except Exception:
            pass
