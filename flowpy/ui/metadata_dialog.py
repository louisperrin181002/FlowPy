"""Sample metadata editor — a spreadsheet of samples × columns (Group,
Condition, Strain, Timepoint…), filled by typing, pasting from Excel,
splitting file names or importing a CSV."""

# ---------------------------------------------------------------------------
# HOW THIS DIALOG WORKS
# ---------------------------------------------------------------------------
# A spreadsheet of samples (rows) against metadata columns, and four ways
# to fill it: type, paste from Excel, split the file names, or import a
# sheet.
#
# The central trick is that the table widget is never the source of
# truth for a *structural* change. Adding, renaming or removing a column
# all follow the same three steps:
#
#     vals = self._current_values()   # read the grid into a dict
#     ...change self.columns...       # change the structure
#     self._fill(vals)                # rebuild the grid from the dict
#
# Rebuilding is cheap at these sizes, and it means column operations
# cannot leave the header and the cells disagreeing — which is the usual
# way a table like this goes wrong.
#
# Rows are keyed by index, never by sample name: two FCS files in
# different folders may legitimately have the same name, and keying by
# name would silently merge their metadata.
# ---------------------------------------------------------------------------

from __future__ import annotations

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTableWidget,
    QTableWidgetItem, QHeaderView, QInputDialog, QMessageBox, QFileDialog,
    QApplication, QFormLayout, QComboBox, QLineEdit, QSpinBox,
    QDialogButtonBox, QListWidget, QAbstractItemView,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QKeySequence

from pathlib import Path

from ..core import groups, sheets
from .errors import show_error

#: Label shown in the combo -> the separator it means. The labels quote
#: the character and name it, because " - " and "-" are indistinguishable
#: in a drop-down otherwise, and they split a file name quite differently.
SEPARATORS = {"' - '  (dash with spaces)": " - ", "'_'  (underscore)": "_",
              "'-'  (dash)": "-", "' '  (space)": " ", "'.'  (dot)": "."}


class _Sheet(QTableWidget):
    """Table that pastes blocks of cells (from Excel) and clears with
    Delete, like a spreadsheet. Column 0 (sample name) is read-only."""

    def keyPressEvent(self, event):
        if event.matches(QKeySequence.StandardKey.Paste):
            self.paste(QApplication.clipboard().text())
            return
        if event.matches(QKeySequence.StandardKey.Copy):
            self.copy_selection()
            return
        # The EditingState guard matters: without it, Backspace inside a
        # cell being edited would wipe the whole selection instead of
        # deleting one character.
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace) and \
                self.state() != QAbstractItemView.State.EditingState:
            for it in self.selectedItems():
                if it.column() > 0:     # column 0 is the sample name
                    it.setText("")
            return
        super().keyPressEvent(event)

    def paste(self, text: str):
        rows = groups.parse_table_text(text)
        if not rows:
            return
        # Paste starts at the current cell, but never in column 0 — the
        # sample name is read-only, and a block pasted from Excel would
        # otherwise be written one column to the left of where it belongs.
        r0, c0 = max(self.currentRow(), 0), max(self.currentColumn(), 1)
        sel = self.selectedRanges()
        # one value pasted onto a selection fills every selected cell
        if len(rows) == 1 and len(rows[0]) == 1 and sel:
            for rng in sel:
                for r in range(rng.topRow(), rng.bottomRow() + 1):
                    for c in range(max(rng.leftColumn(), 1),
                                   rng.rightColumn() + 1):
                        self.item(r, c).setText(rows[0][0].strip())
            return
        # Block paste. Anything falling outside the table is dropped
        # rather than growing it: a sheet with more rows than there are
        # samples usually means the wrong file, and silently adding rows
        # for samples that do not exist would hide that.
        for dr, row in enumerate(rows):
            for dc, val in enumerate(row):
                r, c = r0 + dr, c0 + dc
                if r < self.rowCount() and 1 <= c < self.columnCount():
                    self.item(r, c).setText(val.strip())

    def copy_selection(self):
        """Selection to the clipboard as tab-separated text, Excel-style."""
        sel = self.selectedRanges()
        if not sel:
            return
        # Only the first range: Qt allows several disjoint selections,
        # and there is no sensible rectangular text for those.
        rng = sel[0]
        lines = ["\t".join(self.item(r, c).text()
                           for c in range(rng.leftColumn(),
                                          rng.rightColumn() + 1))
                 for r in range(rng.topRow(), rng.bottomRow() + 1)]
        QApplication.clipboard().setText("\n".join(lines))


class _FromNamesDialog(QDialog):
    """Fill a column from a part of each file name."""

    def __init__(self, names: list[str], columns: list[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Fill From File Names")
        self.resize(520, 440)
        self._names = names
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Split each file name and copy one part into a "
                             "column — e.g. <b>126 - M9 Medium</b> → part 1 "
                             "<b>126</b>, part 2 <b>M9 Medium</b>."))
        form = QFormLayout()
        self.column = QComboBox()
        self.column.setEditable(True)
        self.column.addItems(columns)
        self.column.setToolTip("Pick a column or type a new name")
        form.addRow("Column:", self.column)
        self.sep = QComboBox()
        self.sep.setEditable(True)
        self.sep.addItems(SEPARATORS)
        self.sep.currentTextChanged.connect(self._preview)
        form.addRow("Split on:", self.sep)
        # Negative parts count from the end, so -1 is the last piece
        # however many the name splits into — which is what you want when
        # the interesting part is the suffix and the prefix varies.
        self.part = QSpinBox()
        self.part.setRange(-9, 9)
        self.part.setValue(1)
        self.part.setToolTip("1 = first part, 2 = second… −1 = last part")
        self.part.valueChanged.connect(self._preview)
        form.addRow("Part:", self.part)
        lay.addLayout(form)
        self._list = QListWidget()
        lay.addWidget(self._list, 1)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                              QDialogButtonBox.StandardButton.Cancel)
        bb.button(QDialogButtonBox.StandardButton.Ok).setText("Fill")
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self._preview()

    def separator(self) -> str:
        t = self.sep.currentText()
        return SEPARATORS.get(t, t)

    def values(self) -> list[str]:
        """The chosen part of every file name, in sample order."""
        # 0 is neither the first part nor the last; the spin box can reach
        # it on the way between -1 and 1, so it means "nothing".
        if self.part.value() == 0:
            return [""] * len(self._names)
        return [groups.split_name(n, self.separator(), self.part.value())
                for n in self._names]

    def _preview(self, *_):
        """Show what each file name would become, live as settings change.

        The preview is the whole point of the dialog: splitting on the
        wrong separator is easy and the result is only obvious when you
        see it against the original names.
        """
        self._list.clear()
        for n, v in zip(self._names, self.values()):
            self._list.addItem(f"{n}   →   {v or '(empty)'}")


class MetadataDialog(QDialog):
    """Edit sample metadata. On OK, writes sample.annotations and
    exposes the final column list as ``columns``."""

    def __init__(self, samples, columns: list[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Sample Metadata")
        self.resize(900, 560)
        self._samples = samples
        self.columns = list(columns) or list(groups.DEFAULT_COLUMNS)
        # Display names, keyed by row. Kept apart from sample.name so an
        # imported "Name" column can rename samples in the table without
        # touching the files on disk.
        self._names = {r: s.name for r, s in enumerate(samples)}

        lay = QVBoxLayout(self)
        info = QLabel(
            "Describe your samples — <b>Group</b>, condition, strain, "
            "timepoint… These columns are used by <b>Compare Groups</b> and "
            "<b>Export Data</b>. Type in the cells, paste a block from Excel "
            "(⌘V), or fill a column from the file names.")
        info.setWordWrap(True)
        lay.addWidget(info)

        tools = QHBoxLayout()
        for text, slot, tip in (
                ("Add Column", self._add_column, "New metadata column"),
                ("Rename Column", self._rename_column,
                 "Rename the selected column"),
                ("Remove Column", self._remove_column,
                 "Delete the selected column"),
                ("Fill From File Names…", self._from_names,
                 "Split the file names into columns"),
                ("Set Selected Cells…", self._set_selected,
                 "Type one value for all selected cells"),
                ("Import Excel / CSV…", self._import_csv,
                 "A table (first column = file name or well) or a plate "
                 "map (A–H × 1–12). A 'Name' column renames samples."),
                ("Export CSV…", self._export_csv, "")):
            b = QPushButton(text)
            if tip:
                b.setToolTip(tip)
            b.clicked.connect(slot)
            tools.addWidget(b)
        tools.addStretch()
        lay.addLayout(tools)

        self._table = _Sheet()
        self._table.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection)
        self._table.verticalHeader().setVisible(False)
        lay.addWidget(self._table, 1)

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                              QDialogButtonBox.StandardButton.Cancel)
        bb.button(QDialogButtonBox.StandardButton.Ok).setObjectName(
            "Primary")
        bb.accepted.connect(self._accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

        self._fill({r: dict(s.annotations) for r, s in enumerate(samples)})

    # ------------------------------------------------------------------

    # --- read the grid out, change the structure, write the grid back ---

    def _current_values(self) -> dict[int, dict[str, str]]:
        """{row: {column: value}} (rows, not names: two files may share a
        name)."""
        out = {}
        for r, s in enumerate(self._samples):
            out[r] = {c: self._table.item(r, k + 1).text().strip()
                           for k, c in enumerate(self.columns)}
        return out

    def _fill(self, values: dict[int, dict[str, str]]):
        t = self._table
        t.clear()
        t.setRowCount(len(self._samples))
        t.setColumnCount(len(self.columns) + 1)
        t.setHorizontalHeaderLabels(["Sample"] + self.columns)
        for r, s in enumerate(self._samples):
            name = QTableWidgetItem(self._names.get(r, s.name))
            # The full path in the tooltip is how two samples with the
            # same name are told apart.
            name.setToolTip(s.filepath)
            # &~ clears the editable bit: the name column is a label, and
            # editing it here would not rename anything.
            name.setFlags(name.flags() & ~Qt.ItemFlag.ItemIsEditable)
            f = name.font()
            f.setBold(True)
            name.setFont(f)
            t.setItem(r, 0, name)
            for k, c in enumerate(self.columns):
                t.setItem(r, k + 1, QTableWidgetItem(
                    values.get(r, {}).get(c, "")))
        # The sample column sizes to its contents; the metadata columns
        # are left at a fixed width the user can drag, because
        # ResizeToContents on them would make the columns jump about as
        # values are typed.
        hh = t.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        for k in range(len(self.columns)):
            hh.setSectionResizeMode(k + 1, QHeaderView.ResizeMode.Interactive)
            t.setColumnWidth(k + 1, 150)

    def _selected_column(self) -> int | None:
        """Index into self.columns of the selected column, or None.

        Offset by one throughout: table column 0 is the sample name, so
        table column k+1 is self.columns[k].
        """
        c = self._table.currentColumn()
        return c - 1 if c >= 1 else None

    def _ask_name(self, title: str, text: str = "") -> str | None:
        name, ok = QInputDialog.getText(self, title, "Column name:",
                                        text=text)
        name = name.strip()
        if not ok or not name:
            return None
        # "Sample" is reserved: it is the header of column 0, and a
        # second column with that name would be ambiguous on export.
        if name in self.columns or name == "Sample":
            QMessageBox.information(self, title,
                                    f"There is already a column '{name}'.")
            return None
        return name

    def _add_column(self):
        name = self._ask_name("Add Column")
        if name:
            vals = self._current_values()
            self.columns.append(name)
            self._fill(vals)

    def _rename_column(self):
        k = self._selected_column()
        if k is None:
            QMessageBox.information(self, "Rename Column",
                                    "Click a cell of the column first.")
            return
        old = self.columns[k]
        name = self._ask_name("Rename Column", old)
        if name:
            vals = self._current_values()
            # Move each row's value across to the new key before the
            # rebuild, or the column would come back empty.
            for v in vals.values():
                v[name] = v.pop(old, "")
            self.columns[k] = name
            self._fill(vals)

    def _remove_column(self):
        k = self._selected_column()
        if k is None:
            QMessageBox.information(self, "Remove Column",
                                    "Click a cell of the column first.")
            return
        vals = self._current_values()
        del self.columns[k]
        self._fill(vals)

    def _from_names(self):
        # The file name on disk, not the display name: the display name
        # may already have been changed by an import, and the point here
        # is to read the structure the instrument wrote.
        dlg = _FromNamesDialog([Path(s.filepath).name for s in self._samples],
                               self.columns,
                               self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        col = dlg.column.currentText().strip()
        if not col:
            return
        vals = self._current_values()
        if col not in self.columns:
            self.columns.append(col)
        for r, v in enumerate(dlg.values()):
            vals[r][col] = v
        self._fill(vals)

    def _set_selected(self):
        cells = [it for it in self._table.selectedItems() if it.column() > 0]
        if not cells:
            QMessageBox.information(self, "Set Selected Cells",
                                    "Select some cells first (drag, or "
                                    "⇧/⌘-click).")
            return
        val, ok = QInputDialog.getText(self, "Set Selected Cells",
                                       f"Value for {len(cells)} cell(s):")
        if ok:
            for it in cells:
                it.setText(val.strip())

    def _import_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Sample Sheet", "",
            "Excel or CSV (*.xlsx *.xlsm *.xls *.csv *.tsv *.txt)")
        if path:
            self.import_file(path)

    def import_file(self, path: str):
        """Read a sample sheet or plate map and merge it into the table.

        Two shapes are accepted and told apart by sheets.import_metadata:
        a table whose first column identifies the sample, or a plate map
        laid out as A-H by 1-12. Whatever it finds is merged into the
        existing values rather than replacing them, so a sheet carrying
        one column does not wipe the others.
        """
        try:
            from PyQt6.QtCore import QSettings
            # The same "wells from" rule the Plate Layout dialog is set
            # to, so a plate map matches samples to wells the way the
            # user has already said their files are named.
            src = QSettings("FlowPy", "FlowPy").value("plate/source", None)
            res = sheets.import_metadata(path, self._samples, src)
        except Exception as exc:
            show_error(self, "Import Error", exc)
            return
        # Nothing matched: almost always the identifier column, so the
        # message names what was looked for rather than just failing.
        if not res.n_matched:
            hint = ("the wells in the plate map (FlowPy reads each file's "
                    "well from the FCS file or its name, e.g. …_B03.fcs)"
                    if res.kind == "plate map" else
                    "the first column (file names, with or without .fcs, "
                    "or wells such as B3)")
            QMessageBox.warning(self, "Import Sample Sheet",
                                f"No loaded sample matched {hint}.")
            return
        vals = self._current_values()
        for c in res.columns:
            if c not in self.columns:
                self.columns.append(c)
        for r, row in res.values.items():
            vals[r].update(row)
        self._names.update(res.names)
        self._fill(vals)
        msg = (f"Read a {res.kind} — {res.n_matched} of "
               f"{len(self._samples)} samples matched by {res.matched_by}."
               f"\n\nColumns: {', '.join(res.columns) or '—'}")
        if res.names:
            msg += (f"\n{len(res.names)} sample(s) renamed from the Name "
                    "column (the file names don't change).")
        if res.unmatched and res.unmatched_note and \
                len(res.unmatched) > 12:
            msg += f"\n\n{res.unmatched_note}."
        elif res.unmatched:
            what = "Wells" if res.kind == "plate map" else "Rows"
            msg += (f"\n\n{what} with no loaded sample: "
                    + ", ".join(res.unmatched[:20])
                    + (" …" if len(res.unmatched) > 20 else ""))
        QMessageBox.information(self, "Import Sample Sheet", msg)

    def _export_csv(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Metadata", "sample_metadata.csv", "CSV (*.csv)")
        if not path:
            return
        vals = self._current_values()

        class _Row:    # write_metadata_csv reads .name / .annotations
            def __init__(self, name, ann):
                self.name, self.annotations = name, ann
        try:
            groups.write_metadata_csv(
                path, [_Row(s.name, vals[r])
                       for r, s in enumerate(self._samples)], self.columns)
        except Exception as exc:
            show_error(self, "Export Error", exc)

    def _accept(self):
        vals = self._current_values()
        for r, s in enumerate(self._samples):
            s.annotations = {c: v for c, v in vals[r].items() if v}
            s.name = self._names.get(r, s.name)
        self.accept()
