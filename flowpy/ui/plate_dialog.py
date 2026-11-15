"""Plate Layout — where each sample sits on the plate (from its name, its
FCS file or a Well column), drawn as a 6- to 384-well grid coloured by any
metadata column."""

# ---------------------------------------------------------------------------
# HOW THIS DIALOG WORKS
# ---------------------------------------------------------------------------
# Everything is driven by refresh(), which re-runs end to end whenever any
# of the four combo boxes changes. There is no incremental update: a plate
# is at most 384 cells, so redrawing the lot is cheaper than tracking what
# moved.
#
#   refresh()
#     1. wells()        -> {sample: "A01"} from the chosen source
#     2. plate_format() -> "96-well" etc, guessed if set to Auto
#     3. sort the samples into by_well, and into three problem lists:
#        no well found, well outside this plate, two samples in one well
#     4. build a colour per value of the "Colour by" column
#     5. _fill() one or two grids, then write the legend and the summary
#
# The three problem lists are the point of the dialog as much as the grid
# is: a plate that looks right but has two samples in B2 is the failure
# this is meant to catch before any statistics are run on it.
# ---------------------------------------------------------------------------

from __future__ import annotations

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QComboBox,
    QPushButton, QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox,
    QAbstractItemView, QWidget, QSplitter,
)
from PyQt6.QtCore import Qt, QSettings, pyqtSignal
from PyQt6.QtGui import QColor, QBrush, QFont

from ..core import sheets
from ..core.sheets import (PLATE_FORMATS, AUTO_FORMAT, WELL_SOURCES,
                           WELL_COLUMN, sample_well, detect_format, fits)
from ..core.groups import group_order
from .plotting import PALETTES

NONE = "None"
VIEWS = ["Two plates — names | groups", "One plate"]
_ROWS = "ABCDEFGHIJKLMNOP"


def _text_on(bg: QColor) -> QColor:
    """Black or white text, whichever stays readable on *bg*.

    The colour of a well comes from the data, so it cannot be known in
    advance whether dark or light text will show on it. lightness() is
    the L of HSL; 150 of 255 is about where black stops being legible.
    """
    return QColor("#0f172a" if bg.lightness() > 150 else "#ffffff")


class PlateDialog(QDialog):
    """Signals: sample_requested(sample) on double-click; metadata_changed
    after wells were saved as metadata (read .columns)."""

    sample_requested = pyqtSignal(object)
    metadata_changed = pyqtSignal()

    def __init__(self, samples, meta_columns: list[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Plate Layout")
        self.resize(1100, 640)
        self._samples = samples
        self.columns = list(meta_columns)
        self._settings = QSettings("FlowPy", "FlowPy")

        lay = QVBoxLayout(self)
        form = QHBoxLayout()
        fl = QFormLayout()
        self._source = QComboBox()
        self._source.addItems(WELL_SOURCES)
        self._source.setCurrentText(self._settings.value(
            "plate/source", WELL_SOURCES[0]))
        self._source.setToolTip("Where each sample's well comes from. A "
                                "'Well' column in Sample Metadata always "
                                "wins (type a well there to correct one).")
        fl.addRow("Wells from:", self._source)
        self._format = QComboBox()
        self._format.addItems([AUTO_FORMAT] + list(PLATE_FORMATS))
        self._format.setCurrentText(self._settings.value(
            "plate/format", AUTO_FORMAT))
        fl.addRow("Plate:", self._format)
        self._colour = QComboBox()
        fl.addRow("Colour by:", self._colour)
        self._view = QComboBox()
        self._view.addItems(VIEWS)
        self._view.setCurrentText(self._settings.value("plate/view",
                                                       VIEWS[0]))
        self._view.setToolTip("Two plates: sample names on the left, the "
                              "'Colour by' column (e.g. Group) on the right")
        fl.addRow("Show:", self._view)
        form.addLayout(fl)
        form.addStretch()
        side = QVBoxLayout()
        self._save_btn = QPushButton("Save Wells as Metadata")
        self._save_btn.setObjectName("Primary")
        self._save_btn.setToolTip(
            "Add Well, Row and Column metadata columns, so you can group or "
            "export by them (and fix single wells by hand)")
        self._save_btn.clicked.connect(self._save_wells)
        side.addWidget(self._save_btn)
        side.addStretch()
        form.addLayout(side)
        lay.addLayout(form)

        # --- the plate grids ------------------------------------------
        # Two are built even though usually only one is shown: the "two
        # plates" view puts sample names beside the colour column, and
        # building both once is simpler than creating and destroying the
        # second every time the view changes. refresh() hides the spare.
        self._grids, self._titles = [], []
        split = QSplitter(Qt.Orientation.Horizontal)
        for _ in range(2):
            box = QWidget()
            bl = QVBoxLayout(box)
            bl.setContentsMargins(0, 0, 0, 0)
            t = QLabel()
            t.setObjectName("Breadcrumb")
            g = QTableWidget()
            g.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
            g.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
            # No eliding: a well is better blank than showing "LP1…",
            # which looks like a name and is not one.
            g.setWordWrap(True)
            g.setTextElideMode(Qt.TextElideMode.ElideNone)
            g.cellDoubleClicked.connect(self._open_well)
            bl.addWidget(t)
            bl.addWidget(g, 1)
            split.addWidget(box)
            self._grids.append(g)
            self._titles.append(t)
        self._grid = self._grids[0]
        split.setChildrenCollapsible(False)
        self._split = split
        lay.addWidget(split, 1)
        self._legend = QLabel()
        self._legend.setTextFormat(Qt.TextFormat.RichText)
        self._legend.setWordWrap(True)
        lay.addWidget(self._legend)
        self._info = QLabel()
        self._info.setWordWrap(True)
        self._info.setObjectName("Hint")
        lay.addWidget(self._info)
        hint = QLabel("Double-click a well to open that sample.")
        hint.setObjectName("Hint")
        lay.addWidget(hint)

        self._fill_colour_choices()
        for c in (self._source, self._format, self._colour, self._view):
            c.currentIndexChanged.connect(self.refresh)
        self.refresh()

    # ------------------------------------------------------------------

    def set_samples(self, samples, meta_columns: list[str]):
        self._samples = samples
        self.columns = list(meta_columns)
        self._fill_colour_choices()
        self.refresh()

    def _fill_colour_choices(self):
        """Repopulate "Colour by" after the metadata columns change.

        blockSignals around the rebuild stops clear()/addItems() firing
        currentIndexChanged, which is wired to refresh() — without it,
        this would redraw the plate several times mid-rebuild, once
        against an empty combo.
        """
        cur = self._colour.currentText()
        self._colour.blockSignals(True)
        self._colour.clear()
        # Row and Column are pushed to the end: they are derived from the
        # well itself, so colouring by them only ever draws stripes, and
        # they would otherwise crowd out the real metadata at the top.
        opts = [NONE] + [c for c in self.columns
                         if c not in (WELL_COLUMN, "Row", "Column")] + \
            [c for c in ("Row", "Column") if c in self.columns]
        self._colour.addItems(opts)
        default = next((c for c in self.columns if c not in
                        (WELL_COLUMN, "Row", "Column")), NONE)
        # Keep the user's choice if it still exists, else fall back to
        # the first real metadata column rather than to None — a plate
        # coloured by nothing is the least useful default.
        self._colour.setCurrentText(cur if cur in opts else default)
        self._colour.blockSignals(False)

    def wells(self) -> dict:
        """{sample: well or None} for the chosen source."""
        src = self._source.currentText()
        return {s: sample_well(s, src) for s in self._samples}

    def plate_format(self, wells=None) -> str:
        """The plate size to draw: the chosen one, or guessed from the wells."""
        fmt = self._format.currentText()
        if fmt == AUTO_FORMAT:
            fmt = detect_format((wells or self.wells()).values())
        return fmt

    def refresh(self, *_):
        """Rebuild everything from the current settings. See the map above.

        Takes *_ because it is connected to currentIndexChanged, which
        passes the new index; the argument is never used.
        """
        # Remembered per machine, not per workspace: which naming scheme a
        # lab's cytometer uses is a property of the instrument, so the
        # next file opened should start from the same guess.
        self._settings.setValue("plate/source", self._source.currentText())
        self._settings.setValue("plate/format", self._format.currentText())
        self._settings.setValue("plate/view", self._view.currentText())
        wells = self.wells()
        fmt = self.plate_format(wells)
        nr, nc = PLATE_FORMATS[fmt]
        # --- sort the samples onto the plate ---------------------------
        # by_well maps to a *list*, not a sample: two samples claiming one
        # well is a mistake worth showing rather than one silently
        # overwriting the other.
        by_well: dict[str, list] = {}
        missing, outside = [], []
        for s, w in wells.items():
            if not w:
                missing.append(s)
            elif not fits(w, fmt):
                outside.append((s, w))
            else:
                by_well.setdefault(w, []).append(s)

        # --- one colour per value --------------------------------------
        # Only the placed samples contribute values, so the legend never
        # lists a group that is not on the plate. A blank annotation
        # becomes the em dash, which then gets its own grey.
        col = self._colour.currentText()
        values = ([s.annotations.get(col, "").strip() or "—"
                   for ws in by_well.values() for s in ws]
                  if col != NONE else [])
        # group_order sorts naturally, so "Sample 2" comes before
        # "Sample 10" instead of after it as a plain sort would have it.
        order = group_order(values) if values else []
        # FlowPy's own palette is the more legible but only has eight
        # colours; past that, Tableau's twenty avoid repeats.
        pal = PALETTES["Tableau"] if len(order) > 8 else PALETTES["FlowPy"]
        colours = {v: QColor(pal[i % len(pal)]) for i, v in enumerate(order)}
        colours["—"] = QColor("#cbd5e1")

        # Two plates only make sense if there is something to show on the
        # second one, so the view falls back to one plate when nothing is
        # selected to colour by.
        two = self._view.currentText() == VIEWS[0] and col != NONE
        modes = ["name", "value"] if two else ["both"]
        # Widen the window for two plates, but never past the screen.
        # self.screen() is None before the dialog has been shown, hence
        # the fallback.
        scr = self.screen().availableGeometry().width() if self.screen() \
            else 1500
        self.resize(min(max(self.width(), 1500 if two else 1100), scr - 40),
                    self.height())
        for k, g in enumerate(self._grids):
            box = g.parentWidget()
            box.setVisible(k < len(modes))
            if k < len(modes):
                self._titles[k].setText(
                    {"name": "Sample names", "value": col,
                     "both": f"Sample names · {col}" if col != NONE
                     else "Sample names"}[modes[k]])
                self._fill(g, modes[k], nr, nc, by_well, col, colours)
        if two:
            half = max(self._split.width(), 1000) // 2
            self._split.setSizes([half, half])
        self._by_well = by_well

        if col != NONE and order:
            self._legend.setText(" &nbsp; ".join(
                f"<span style='background:{colours[v].lighter(150).name()};"
                f"border:1px solid {colours[v].name()};'>&nbsp;&nbsp;&nbsp;"
                f"</span> {v}" for v in order))
        else:
            self._legend.setText("")
        info = [f"{len(self._samples) - len(missing) - len(outside)} of "
                f"{len(self._samples)} samples placed on a {fmt} plate."]
        # --- the three things worth warning about ----------------------
        dup = [w for w, ss in by_well.items() if len(ss) > 1]
        if dup:
            info.append("Same well for several samples: " + ", ".join(dup)
                        + " — fix it in the Well column of Sample "
                        "Metadata.")
        if outside:
            info.append("Outside this plate: " + ", ".join(
                f"{s.name} ({w})" for s, w in outside[:10]))
        if missing:
            info.append(f"No well found ({len(missing)}): " + ", ".join(
                s.name for s in missing[:12]) +
                (" …" if len(missing) > 12 else "") +
                " — try another 'Wells from' option, or type the well in "
                "Sample Metadata.")
        self._info.setText("\n".join(info))

    def _fill(self, g, mode, nr, nc, by_well, col, colours):
        """One plate grid; *mode*: 'name' (sample names), 'value' (the
        colour column's value) or 'both'."""
        g.clear()
        g.setRowCount(nr)
        g.setColumnCount(nc)
        g.setHorizontalHeaderLabels([str(c + 1) for c in range(nc)])
        g.setVerticalHeaderLabels(list(_ROWS[:nr]))
        # Past 96 wells (8 rows) there is no room for text in a cell, so
        # the labels are dropped and the colour and tooltip carry the
        # information instead. The font still shrinks a little, and is
        # floored at 7pt so it never collapses to nothing.
        small = nr > 8
        f = QFont(g.font())
        f.setPointSizeF(max(f.pointSizeF() - (2 if small else 1), 7))
        for r in range(nr):
            for c in range(nc):
                w = f"{_ROWS[r]}{c + 1}"
                ss = by_well.get(w, [])
                it = QTableWidgetItem()
                it.setFont(f)
                it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if not ss:
                    # Empty well: pale, with its own name in grey so the
                    # plate can still be read as a coordinate grid.
                    it.setBackground(QBrush(QColor("#f8fafc")))
                    it.setForeground(QBrush(QColor("#cbd5e1")))
                    it.setText("" if small else w)
                else:
                    s = ss[0]
                    if len(ss) > 1:
                        # Red and counted, not one of the names: the point
                        # is that the plate map is wrong, and showing one
                        # of the two names would hide exactly that.
                        it.setText(f"{len(ss)} samples!")
                        it.setBackground(QBrush(QColor("#fee2e2")))
                        it.setForeground(QBrush(QColor("#b91c1c")))
                    else:
                        v = s.annotations.get(col, "").strip() \
                            if col != NONE else ""
                        if col != NONE and mode != "name":
                            # Lighten dark colours so the text on top
                            # stays readable; pale ones are left alone,
                            # since lightening them further would wash
                            # the well out to white.
                            bg = QColor(colours[v or "—"])
                            bg.setAlpha(255)
                            light = QColor(bg).lighter(150) \
                                if bg.lightness() < 200 else bg
                            it.setBackground(QBrush(light))
                            it.setForeground(QBrush(_text_on(light)))
                        else:
                            it.setBackground(QBrush(QColor("#e8effd")))
                        name = sheets.stem(s.name)
                        label = {"name": name, "value": v or "—",
                                 "both": f"{name}\n{v}" if v else name
                                 }[mode]
                        it.setText("" if small else label)
                    # The tooltip carries everything the cell cannot: all
                    # samples in the well and all their annotations. This
                    # is what makes a 384-well plate usable at all, since
                    # none of its cells can show text.
                    it.setToolTip("\n".join(
                        [f"{w}"] + [f"{x.name}" + "".join(
                            f"\n  {k}: {v}" for k, v in x.annotations.items()
                            if v) for x in ss]))
                    # The well name rides along on the item so a
                    # double-click can find its samples again.
                    it.setData(Qt.ItemDataRole.UserRole, w)
                g.setItem(r, c, it)
        # Stretch both ways so the plate fills the dialog and the wells
        # stay square-ish at any window size.
        hh, vh = g.horizontalHeader(), g.verticalHeader()
        hh.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        vh.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

    def _open_well(self, r, c):
        """Double-click: ask the main window to show that well's sample."""
        # Rebuilt from the row and column rather than read from the item,
        # so an empty cell resolves to a well name too and simply finds
        # nothing in _by_well.
        w = f"{_ROWS[r]}{c + 1}"
        ss = getattr(self, "_by_well", {}).get(w)
        if ss:
            self.sample_requested.emit(ss[0])

    def _save_wells(self):
        """Write each sample's well into its metadata as Well/Row/Column.

        Until this is pressed the well is derived on the fly from the
        sample name and exists only in this dialog. Saving it makes it an
        ordinary metadata column: groupable, exportable, and editable by
        hand when the guess got one wrong.
        """
        wells = self.wells()
        n = 0
        for s, w in wells.items():
            if w:
                s.annotations[WELL_COLUMN] = w
                # Split so that a row or a column can be grouped on its
                # own — "is there an edge effect?" is a question about
                # Row, not about Well. w[0] is the letter, w[1:] the
                # number, kept as text so "01" and "1" stay as written.
                s.annotations["Row"] = w[0]
                s.annotations["Column"] = w[1:]
                n += 1
        for c in (WELL_COLUMN, "Row", "Column"):
            if c not in self.columns:
                self.columns.append(c)
        self._fill_colour_choices()
        self.refresh()
        self.metadata_changed.emit()
        QMessageBox.information(
            self, "Plate Layout",
            f"Saved the well of {n} sample(s) as Well, Row and Column "
            "metadata. You can now group, colour or export by row or "
            "column, and correct a well by hand in Sample Metadata.")
