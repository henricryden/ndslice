from pathlib import Path

from PyQt6 import QtCore, QtGui, QtWidgets

from .dicom_metadata import (
    compare_dicom_tags,
    curated_tag_rows,
    dicom_dataset_entries,
)


DICOM_TAG_VIEW_LIMITED = 'limited'
DICOM_TAG_VIEW_FULL = 'full'
DICOM_TAG_VIEW_VARYING = 'varying'
DICOM_SEARCH_MATCH_ROLE = int(QtCore.Qt.ItemDataRole.UserRole) + 1

_ACTIVE_VARIATION_WORKERS = set()


def _folded_text(text):
    folded = []
    source_positions = []
    for index, character in enumerate(text):
        character_folded = character.casefold()
        folded.append(character_folded)
        source_positions.extend([index] * len(character_folded))
    return ''.join(folded), source_positions


def _contiguous_match(text, folded, source_positions, term):
    if folded == term:
        return (4, 0, 0), set(source_positions)

    best = None
    start = folded.find(term)
    while start >= 0:
        source_start = source_positions[start]
        word_start = source_start == 0 or not text[source_start - 1].isalnum()
        score = (3 if word_start else 2, 0, -start)
        positions = set(source_positions[start:start + len(term)])
        candidate = score, positions
        if best is None or candidate[0] > best[0]:
            best = candidate
        start = folded.find(term, start + 1)
    return best


def _gapped_match(folded, source_positions, term):
    search_start = 0
    best = None
    while search_start < len(folded):
        term_index = 0
        text_index = search_start
        while text_index < len(folded) and term_index < len(term):
            if folded[text_index] == term[term_index]:
                term_index += 1
            text_index += 1
        if term_index < len(term):
            break

        term_index = len(term) - 1
        match_index = text_index - 1
        positions = []
        while term_index >= 0:
            if folded[match_index] == term[term_index]:
                positions.append(match_index)
                term_index -= 1
            match_index -= 1
        positions.reverse()

        gaps = positions[-1] - positions[0] + 1 - len(term)
        score = (1, -gaps, -positions[0])
        source_matches = {source_positions[index] for index in positions}
        candidate = score, source_matches
        if best is None or candidate[0] > best[0]:
            best = candidate
        search_start = positions[0] + 1
    return best


def _fuzzy_cell_match(text, term):
    folded, source_positions = _folded_text(text)
    if not folded or not term:
        return None

    contiguous = _contiguous_match(text, folded, source_positions, term)
    if contiguous is not None:
        return contiguous
    return _gapped_match(folded, source_positions, term)


def _filtered_tag_rows(rows, query):
    terms = [term.casefold() for term in query.split() if term]
    if not terms or (len(rows) == 1 and not rows[0][0]):
        return [(row, ((), (), ())) for row in rows]

    ranked_rows = []
    for original_index, row in enumerate(rows):
        column_matches = [set(), set(), set()]
        term_scores = []
        for term in terms:
            matches = []
            for column, value in enumerate(row):
                match = _fuzzy_cell_match(value, term)
                if match is None:
                    continue
                score, positions = match
                matches.append((score, -column))
                column_matches[column].update(positions)
            if not matches:
                break
            term_scores.append(max(matches)[0])
        else:
            quality = sum(score[0] for score in term_scores)
            exact_matches = sum(score[0] == 4 for score in term_scores)
            boundary_matches = sum(score[0] == 3 for score in term_scores)
            contiguous_matches = sum(score[0] == 2 for score in term_scores)
            gaps = -sum(score[1] for score in term_scores)
            starts = -sum(score[2] for score in term_scores)
            sort_key = (
                -quality,
                -exact_matches,
                -boundary_matches,
                -contiguous_matches,
                gaps,
                starts,
                original_index,
            )
            ranked_rows.append((sort_key, row, column_matches))

    ranked_rows.sort(key=lambda result: result[0])
    return [
        (row, tuple(tuple(sorted(matches)) for matches in column_matches))
        for _sort_key, row, column_matches in ranked_rows
    ]


def _elided_source_positions(source, displayed):
    if source == displayed:
        return list(range(len(source)))

    prefix_length = 0
    prefix_limit = min(len(source), len(displayed))
    while (
            prefix_length < prefix_limit
            and source[prefix_length] == displayed[prefix_length]):
        prefix_length += 1

    suffix_length = 0
    suffix_limit = min(
        len(source) - prefix_length,
        len(displayed) - prefix_length,
    )
    while (
            suffix_length < suffix_limit
            and source[-suffix_length - 1] == displayed[-suffix_length - 1]):
        suffix_length += 1

    positions = list(range(prefix_length))
    positions.extend([None] * (len(displayed) - prefix_length - suffix_length))
    if suffix_length:
        positions.extend(range(len(source) - suffix_length, len(source)))
    return positions


class DicomSearchHighlightDelegate(QtWidgets.QStyledItemDelegate):
    def paint(self, painter, option, index):
        matches = index.data(DICOM_SEARCH_MATCH_ROLE)
        if not matches:
            super().paint(painter, option, index)
            return

        styled_option = QtWidgets.QStyleOptionViewItem(option)
        self.initStyleOption(styled_option, index)
        text = styled_option.text
        style = (
            styled_option.widget.style()
            if styled_option.widget is not None
            else QtWidgets.QApplication.style()
        )
        text_rect = style.subElementRect(
            QtWidgets.QStyle.SubElement.SE_ItemViewItemText,
            styled_option,
            styled_option.widget,
        )
        displayed_text = styled_option.fontMetrics.elidedText(
            text,
            styled_option.textElideMode,
            text_rect.width(),
        )
        displayed_positions = _elided_source_positions(text, displayed_text)
        highlighted = {
            displayed_index
            for displayed_index, source_index in enumerate(displayed_positions)
            if source_index in matches
        }

        styled_option.text = ''
        style.drawControl(
            QtWidgets.QStyle.ControlElement.CE_ItemViewItem,
            styled_option,
            painter,
            styled_option.widget,
        )

        selected = bool(
            option.state & QtWidgets.QStyle.StateFlag.State_Selected
        )
        palette = styled_option.palette
        text_role = (
            QtGui.QPalette.ColorRole.HighlightedText
            if selected else QtGui.QPalette.ColorRole.Text
        )
        match_background_role = (
            QtGui.QPalette.ColorRole.Base
            if selected else QtGui.QPalette.ColorRole.Highlight
        )
        match_text_role = (
            QtGui.QPalette.ColorRole.Text
            if selected else QtGui.QPalette.ColorRole.HighlightedText
        )

        formats = []
        base_format = QtGui.QTextLayout.FormatRange()
        base_format.start = 0
        base_format.length = len(displayed_text)
        base_format.format.setForeground(palette.color(text_role))
        formats.append(base_format)

        highlighted = sorted(highlighted)
        range_start = None
        previous = None
        for displayed_index in highlighted + [None]:
            if (
                    range_start is not None
                    and displayed_index != previous + 1):
                match_format = QtGui.QTextLayout.FormatRange()
                match_format.start = range_start
                match_format.length = previous - range_start + 1
                match_format.format.setBackground(
                    palette.color(match_background_role)
                )
                match_format.format.setForeground(palette.color(match_text_role))
                formats.append(match_format)
                range_start = None
            if displayed_index is not None and range_start is None:
                range_start = displayed_index
            previous = displayed_index

        text_layout = QtGui.QTextLayout(displayed_text, styled_option.font)
        text_layout.setFormats(formats)
        text_layout.beginLayout()
        line = text_layout.createLine()
        if line.isValid():
            line.setLineWidth(text_rect.width())
        text_layout.endLayout()

        painter.save()
        painter.setClipRect(text_rect)
        vertical_offset = max(
            0.0,
            (text_rect.height() - text_layout.boundingRect().height()) / 2,
        )
        text_layout.draw(
            painter,
            QtCore.QPointF(text_rect.x(), text_rect.y() + vertical_offset),
        )
        painter.restore()


def _retain_variation_worker(worker):
    _ACTIVE_VARIATION_WORKERS.add(worker)

    def release():
        _ACTIVE_VARIATION_WORKERS.discard(worker)
        worker.deleteLater()

    worker.finished.connect(release)


class DicomVariationWorker(QtCore.QThread):
    comparison_ready = QtCore.pyqtSignal(object)

    def __init__(self, paths):
        super().__init__()
        self.paths = list(paths)

    def run(self):
        result = compare_dicom_tags(
            self.paths,
            cancelled=self.isInterruptionRequested,
        )
        if result is not None and not self.isInterruptionRequested():
            self.comparison_ready.emit(result)


class DicomTagsDialog(QtWidgets.QDialog):
    def __init__(
            self, parent, records, initial_index, initial_status='',
            inspected_callback=None, follow_callback=None):
        super().__init__(parent)
        self.records = records
        self.current_index = max(0, min(int(initial_index), len(records) - 1))
        self._inspected_callback = inspected_callback
        self._follow_callback = follow_callback
        self._selection_status = initial_status
        self._record_status = ''
        self._tag_view_status = ''
        self._dataset = None
        self._variation_result = None
        self._variation_worker = None
        self._variation_restore_state = None

        self.setWindowTitle("DICOM tags")
        self.resize(820, 620)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, True)

        layout = QtWidgets.QVBoxLayout(self)
        navigation_layout = QtWidgets.QHBoxLayout()

        self.follow_slice_checkbox = QtWidgets.QCheckBox("Follow slice", self)
        self.follow_slice_checkbox.setToolTip(
            "Keep ndslice synchronized with the selected DICOM file"
        )
        self.follow_slice_checkbox.toggled.connect(self._on_follow_slice_toggled)
        navigation_layout.addWidget(self.follow_slice_checkbox)

        self.file_slider = QtWidgets.QSlider(
            QtCore.Qt.Orientation.Horizontal,
            self,
        )
        self.file_slider.setRange(0, max(0, len(records) - 1))
        self.file_slider.setSingleStep(1)
        self.file_slider.setPageStep(1)
        self.file_slider.setTracking(True)
        self.file_slider.setEnabled(len(records) > 1)
        self.file_slider.setToolTip("DICOM file")
        self.file_slider.valueChanged.connect(self._on_file_slider_changed)
        navigation_layout.addWidget(self.file_slider, 1)

        self.position_label = QtWidgets.QLabel(self)
        navigation_layout.addWidget(self.position_label)
        navigation_layout.addStretch(1)
        self.group_label = QtWidgets.QLabel(self)
        self.group_label.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight
            | QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        navigation_layout.addWidget(self.group_label)
        layout.addLayout(navigation_layout)

        self.file_label = QtWidgets.QLabel(self)
        self.file_label.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.file_label.setWordWrap(True)
        layout.addWidget(self.file_label)

        search_layout = QtWidgets.QHBoxLayout()
        search_label = QtWidgets.QLabel("Search:", self)
        self.search_field = QtWidgets.QLineEdit(self)
        self.search_field.setPlaceholderText("Search tags…")
        self.search_field.setClearButtonEnabled(True)
        self.search_field.setAccessibleName("Search DICOM tags")
        self.search_field.textChanged.connect(self._on_search_changed)
        search_label.setBuddy(self.search_field)
        search_layout.addWidget(search_label)
        search_layout.addWidget(self.search_field, 1)
        layout.addLayout(search_layout)

        self.tag_table = QtWidgets.QTableWidget(self)
        self.tag_table.setColumnCount(3)
        self.tag_table.setHorizontalHeaderLabels(["Tag", "Name", "Value"])
        self.tag_table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.tag_table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.tag_table.setAlternatingRowColors(True)
        self.tag_table.setItemDelegate(
            DicomSearchHighlightDelegate(self.tag_table)
        )
        self.tag_table.verticalHeader().setVisible(False)
        header = self.tag_table.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.tag_table, 1)

        options_layout = QtWidgets.QHBoxLayout()
        options_layout.addWidget(QtWidgets.QLabel("Tags:", self))
        self.tag_view_combo = QtWidgets.QComboBox(self)
        self.tag_view_combo.addItem("Limited", DICOM_TAG_VIEW_LIMITED)
        self.tag_view_combo.addItem("Full", DICOM_TAG_VIEW_FULL)
        self.tag_view_combo.addItem("Varying", DICOM_TAG_VIEW_VARYING)
        self.tag_view_combo.currentIndexChanged.connect(self._on_tag_view_changed)
        options_layout.addWidget(self.tag_view_combo)
        options_layout.addStretch(1)
        layout.addLayout(options_layout)

        self.status_label = QtWidgets.QLabel(self)
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Close,
            parent=self,
        )
        button_box.rejected.connect(self.close)
        layout.addWidget(button_box)

        self._previous_shortcut = QtGui.QShortcut(
            QtGui.QKeySequence(QtCore.Qt.Key.Key_Left),
            self,
        )
        self._previous_shortcut.activated.connect(self.show_previous)
        self._next_shortcut = QtGui.QShortcut(
            QtGui.QKeySequence(QtCore.Qt.Key.Key_Right),
            self,
        )
        self._next_shortcut.activated.connect(self.show_next)
        self._search_shortcut = QtGui.QShortcut(
            QtGui.QKeySequence.StandardKey.Find,
            self,
        )
        self._search_shortcut.activated.connect(self._focus_search)

        self._show_record(self.current_index, preserve_status=True)

    @staticmethod
    def _load_dataset(path):
        import pydicom
        return pydicom.dcmread(path, stop_before_pixels=True)

    def _current_tag_view(self):
        return self.tag_view_combo.currentData() or DICOM_TAG_VIEW_LIMITED

    def _capture_table_position(self):
        selected_row = self.tag_table.currentRow()
        selected_tag = None
        if selected_row >= 0:
            tag_item = self.tag_table.item(selected_row, 0)
            if tag_item is not None:
                selected_tag = tag_item.text()
        return {
            'selected_tag': selected_tag,
            'selected_row': selected_row,
            'selected_column': self.tag_table.currentColumn(),
            'scroll_value': self.tag_table.verticalScrollBar().value(),
            'table_had_focus': self.tag_table.hasFocus(),
        }

    def _rows_for_current_view(self):
        tag_view = self._current_tag_view()
        if tag_view == DICOM_TAG_VIEW_LIMITED:
            return curated_tag_rows(self._dataset)
        if tag_view == DICOM_TAG_VIEW_FULL:
            return [entry[:3] for entry in dicom_dataset_entries(self._dataset)]

        if self._variation_result is None:
            return [("", "Comparing tags...", "")]

        varying_tags = self._variation_result['varying_tags']
        if not varying_tags:
            return [("", "No varying tags", "")]

        current_entries = {
            entry[0]: entry
            for entry in dicom_dataset_entries(self._dataset)
        }
        tag_names = self._variation_result['tag_names']
        return [
            current_entries[tag][:3]
            if tag in current_entries
            else (tag, tag_names.get(tag, "Unknown tag"), "<missing>")
            for tag in varying_tags
        ]

    def _restore_table_position(self, position_state):
        if position_state is None:
            self.tag_table.scrollToTop()
            return

        selected_tag = position_state['selected_tag']
        selected_row = position_state['selected_row']
        selected_column = position_state['selected_column']
        restored_row = -1
        if selected_tag is not None:
            for row in range(self.tag_table.rowCount()):
                item = self.tag_table.item(row, 0)
                if item is not None and item.text() == selected_tag:
                    restored_row = row
                    break
        if restored_row < 0 and selected_row >= 0 and self.tag_table.rowCount():
            restored_row = min(selected_row, self.tag_table.rowCount() - 1)
        if restored_row >= 0:
            restored_column = max(0, min(selected_column, self.tag_table.columnCount() - 1))
            self.tag_table.setCurrentCell(restored_row, restored_column)
        self.tag_table.verticalScrollBar().setValue(position_state['scroll_value'])
        if position_state['table_had_focus']:
            self.tag_table.setFocus()

    def _populate_table(self, preserve_position=False, position_state=None):
        if position_state is None and preserve_position:
            position_state = self._capture_table_position()

        self.tag_table.setRowCount(0)
        if self._dataset is None:
            return

        rows = _filtered_tag_rows(
            self._rows_for_current_view(),
            self.search_field.text(),
        )
        if self.search_field.text().strip() and not rows:
            rows = [(('', 'No matching tags', ''), ((), (), ()))]

        self.tag_table.setRowCount(len(rows))
        for row, (values, column_matches) in enumerate(rows):
            for column, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(value)
                item.setToolTip(value)
                if column_matches[column]:
                    item.setData(
                        DICOM_SEARCH_MATCH_ROLE,
                        column_matches[column],
                    )
                self.tag_table.setItem(row, column, item)
        self._restore_table_position(position_state)

    def _on_search_changed(self, _text):
        if self._dataset is not None:
            self._populate_table()

    def _focus_search(self):
        self.search_field.setFocus()
        self.search_field.selectAll()

    def _update_status_label(self):
        statuses = [
            status for status in (self._record_status, self._tag_view_status)
            if status
        ]
        self.status_label.setText("\n".join(statuses))

    def _variation_status_text(self):
        result = self._variation_result
        if result is None:
            return f"Comparing tags across {len(self.records)} files..."

        readable_count = result['readable_count']
        unreadable_count = result['unreadable_count']
        varying_count = len(result['varying_tags'])
        if readable_count == 0:
            status = "No readable DICOM files are available for comparison"
        elif readable_count == 1:
            status = "No tags vary across one readable DICOM file"
        else:
            status = f"{varying_count} varying tags across {readable_count} readable files"
        if unreadable_count:
            status += f"; {unreadable_count} of {result['total_count']} files could not be read"
        return status

    def _start_variation_comparison(self):
        if self._variation_worker is not None:
            return
        paths = [record.get('path') or '' for record in self.records]
        worker = DicomVariationWorker(paths)
        _retain_variation_worker(worker)
        self._variation_worker = worker
        self.tag_view_combo.setEnabled(False)
        self._tag_view_status = self._variation_status_text()
        self._update_status_label()
        worker.comparison_ready.connect(self._on_variation_ready)
        worker.finished.connect(
            lambda worker=worker: self._on_variation_worker_finished(worker)
        )
        worker.start()

    def _on_variation_ready(self, result):
        self._variation_result = result
        self.tag_view_combo.setEnabled(True)
        self._tag_view_status = self._variation_status_text()
        restore_state = self._variation_restore_state
        self._variation_restore_state = None
        if self._current_tag_view() == DICOM_TAG_VIEW_VARYING:
            self._populate_table(position_state=restore_state)
        self._update_status_label()

    def _on_variation_worker_finished(self, worker):
        if self._variation_worker is worker:
            self._variation_worker = None

    def _show_record(self, index, preserve_status=False, preserve_table_position=False):
        self.current_index = max(0, min(int(index), len(self.records) - 1))
        record = self.records[self.current_index]
        if self._inspected_callback is not None:
            self._inspected_callback(record.get('path'))
        self.position_label.setText(f"{self.current_index + 1} / {len(self.records)}")
        self.file_label.setText(str(record.get('relative_path') or record.get('path', '')))
        self.group_label.setText(str(record.get('group_label') or ''))
        if self.file_slider.value() != self.current_index:
            self.file_slider.blockSignals(True)
            self.file_slider.setValue(self.current_index)
            self.file_slider.blockSignals(False)

        path = record.get('path')
        try:
            if not path or not Path(path).is_file():
                raise FileNotFoundError(path or 'DICOM path is missing')
            self._dataset = self._load_dataset(path)
            self._populate_table(preserve_position=preserve_table_position)
            load_error = ''
        except Exception as exc:
            self._dataset = None
            self.tag_table.setRowCount(1)
            self.tag_table.setItem(0, 0, QtWidgets.QTableWidgetItem("Error"))
            self.tag_table.setItem(0, 1, QtWidgets.QTableWidgetItem("Unable to read file"))
            self.tag_table.setItem(0, 2, QtWidgets.QTableWidgetItem(str(exc)))
            load_error = f"Unable to read DICOM tags: {exc}"

        navigation_reason = str(record.get('navigation_reason') or '')
        navigable = bool(record.get('navigable'))
        self.follow_slice_checkbox.setEnabled(navigable)
        self.follow_slice_checkbox.setToolTip(
            "Keep ndslice synchronized with the selected DICOM file"
            if navigable
            else navigation_reason or "This DICOM file cannot be mapped to ndslice"
        )
        if not preserve_status:
            self._selection_status = ''
        follow_error = self._follow_current_record() if navigable else ''
        self._record_status = "\n".join(
            dict.fromkeys(
                status for status in (
                    load_error,
                    follow_error,
                    navigation_reason,
                    self._selection_status,
                ) if status
            )
        )
        self._update_status_label()

    def show_previous(self):
        if self.current_index > 0:
            self.file_slider.setValue(self.current_index - 1)

    def show_next(self):
        if self.current_index + 1 < len(self.records):
            self.file_slider.setValue(self.current_index + 1)

    def _on_file_slider_changed(self, index):
        if index != self.current_index:
            self._show_record(index, preserve_table_position=True)

    def select_record_path(self, path, status=''):
        if not path:
            return False
        for index, record in enumerate(self.records):
            if record.get('path') != path:
                continue
            if index != self.current_index:
                self._selection_status = status
                self._show_record(
                    index,
                    preserve_status=True,
                    preserve_table_position=True,
                )
            return True
        return False

    def _on_tag_view_changed(self, _index):
        position_state = self._capture_table_position()
        if self._current_tag_view() == DICOM_TAG_VIEW_VARYING:
            if self._variation_result is None:
                self._variation_restore_state = position_state
                self._start_variation_comparison()
            else:
                self._tag_view_status = self._variation_status_text()
            self._populate_table(position_state=position_state)
        else:
            self._tag_view_status = ''
            self._populate_table(position_state=position_state)
        self._update_status_label()

    def _on_follow_slice_toggled(self, enabled):
        if not enabled:
            return
        follow_error = self._follow_current_record()
        if follow_error:
            self._record_status = follow_error
            self._update_status_label()

    def _follow_current_record(self):
        if not self.follow_slice_checkbox.isChecked() or self._follow_callback is None:
            return ''
        record = self.records[self.current_index]
        success, message = self._follow_callback(record)
        return '' if success else message

    def closeEvent(self, event):
        worker = self._variation_worker
        if worker is not None and worker.isRunning():
            try:
                worker.comparison_ready.disconnect(self._on_variation_ready)
            except (TypeError, RuntimeError):
                pass
            worker.requestInterruption()
            self._variation_worker = None
        super().closeEvent(event)
