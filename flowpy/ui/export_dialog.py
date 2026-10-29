"""Data export — a table of chosen statistics, for chosen channels and
populations, across chosen samples (e.g. the median of YG5 in the gated
cells of every sample), exported as CSV or copied for Excel / Prism.
"""

from __future__ import annotations
import pandas as pd

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QPushButton, QLineEdit, QCheckBox, QComboBox, QDoubleSpinBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QFileDialog, QMessageBox,
    QApplication, QSplitter, QWidget,
)
from PyQt6.QtCore import Qt, QTimer, QSettings

from ..core.protocol import (
    export_table, population_path, COUNT_STATS, _bfs,
)
from ..core.statistics import STAT_NAMES

LAYOUTS = ["One row per sample", "One row per sample × population"]
_SETTINGS = "export_table/"


def _section(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setObjectName("SectionHeader")
    return lab


class _CheckList(QListWidget):
    """List of checkable entries with helpers."""

    def fill(self, entries: list[str], checked: set[str]):
        self.blockSignals(True)
        self.clear()
        for text in entries:
            it = QListWidgetItem(text)
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Checked if text in checked
                             else Qt.CheckState.Unchecked)
            self.addItem(it)
        self.blockSignals(False)

    def checked(self) -> list[str]:
        return [self.item(i).text() for i in range(self.count())
                if self.item(i).checkState() == Qt.CheckState.Checked]

    def set_all(self, on: bool, visible_only: bool = False):
        self.blockSignals(True)
        for i in range(self.count()):
            it = self.item(i)
            if not (visible_only and it.isHidden()):
                it.setCheckState(Qt.CheckState.Checked if on
                                 else Qt.CheckState.Unchecked)
        self.blockSignals(False)
        if self.count():
            self.itemChanged.emit(self.item(0))


def _all_none(lst: _CheckList, visible_only: bool = False) -> QHBoxLayout:
    row = QHBoxLayout()
    row.setSpacing(4)
    for text, on in (("All", True), ("None", False)):
        b = QPushButton(text)
        b.setMinimumWidth(56)
        b.clicked.connect(lambda _=False, on=on:
                          lst.set_all(on, visible_only))
        row.addWidget(b)
    row.addStretch()
    return row


class ExportDataDialog(QDialog):
    """Pick samples × populations × channels × statistics, preview, export."""

    def __init__(self, samples, current=None, parent=None,
                 meta_columns: list[str] | None = None):
        super().__init__(parent)
        # metadata columns that are filled in for at least one sample
        self._meta = [c for c in (meta_columns or [])
                      if any(s.annotations.get(c) for s in samples)]
        self.setWindowTitle("Export Data")
        self.resize(1180, 700)
        self._samples = samples
        self._current = current or (samples[0] if samples else None)
        self._df = pd.DataFrame()
        self._settings = QSettings("FlowPy", "FlowPy")

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(150)
        self._timer.timeout.connect(self._recompute)

        root = QVBoxLayout(self)
        intro = QLabel("Choose what to export — for example the <b>Median</b>"
                       " of <b>YG5</b> in your gated population for every "
                       "sample. Your choice is remembered for next time.")
        intro.setWordWrap(True)
        root.addWidget(intro)

        split = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(split, 1)
        split.addWidget(self._build_selection())

        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(6, 0, 0, 0)
        top = QHBoxLayout()
        top.addWidget(_section("PREVIEW"))
        top.addStretch()
        top.addWidget(QLabel("Table:"))
        self._layout = QComboBox()
        self._layout.addItems(LAYOUTS)
        self._layout.currentIndexChanged.connect(self._schedule)
        top.addWidget(self._layout)
        self._meta_cb = QCheckBox("Include metadata")
        self._meta_cb.setToolTip("Add the sample metadata columns ("
                                 + (", ".join(self._meta) or "none yet — "
                                    "see Samples → Sample Metadata")
                                 + ") after the sample name")
        self._meta_cb.setChecked(bool(self._meta))
        self._meta_cb.setEnabled(bool(self._meta))
        self._meta_cb.toggled.connect(self._schedule)
        top.addWidget(self._meta_cb)
        rv.addLayout(top)
        self._table = QTableWidget()
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.verticalHeader().setVisible(False)
        self._table.setShowGrid(False)
        rv.addWidget(self._table, 1)
        self._summary = QLabel("")
        self._summary.setObjectName("Hint")
        self._summary.setWordWrap(True)
        rv.addWidget(self._summary)

        buttons = QHBoxLayout()
        buttons.addStretch()
        copy = QPushButton("Copy to Clipboard")
        copy.setToolTip("Paste straight into Excel, Prism or Numbers")
        copy.clicked.connect(self._copy)
        buttons.addWidget(copy)
        export = QPushButton("Export CSV…")
        export.setObjectName("Primary")
        export.clicked.connect(self._export)
        buttons.addWidget(export)
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        buttons.addWidget(close)
        rv.addLayout(buttons)
        split.addWidget(right)
        split.setStretchFactor(1, 1)
        split.setSizes([380, 800])

        self._restore()
        self._recompute()

    # ------------------------------------------------------------------
    # Selection panel
    # ------------------------------------------------------------------

    def _build_selection(self) -> QWidget:
        w = QWidget()
        w.setMinimumWidth(330)
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 6, 0)

        lay.addWidget(_section("SAMPLES"))
        self._sample_list = _CheckList()
        self._sample_list.fill([s.name for s in self._samples],
                               {s.name for s in self._samples})
        self._sample_list.itemChanged.connect(self._schedule)
        lay.addWidget(self._sample_list, 2)
        lay.addLayout(_all_none(self._sample_list))

        lay.addWidget(_section("POPULATIONS"))
        self._pop_list = _CheckList()
        self._pop_list.itemChanged.connect(self._schedule)
        lay.addWidget(self._pop_list, 2)

        lay.addWidget(_section("CHANNELS"))
        self._filter = QLineEdit()
        self._filter.setPlaceholderText("Search channels, e.g. YG5")
        self._filter.setClearButtonEnabled(True)
        self._filter.textChanged.connect(self._apply_filter)
        lay.addWidget(self._filter)
        self._chan_list = _CheckList()
        self._chan_list.itemChanged.connect(self._on_channel_changed)
        lay.addWidget(self._chan_list, 3)
        row = _all_none(self._chan_list, visible_only=True)
        self._ticked_only = QCheckBox("Ticked only")
        self._ticked_only.toggled.connect(self._apply_filter)
        row.addWidget(self._ticked_only)
        lay.addLayout(row)

        lay.addWidget(_section("STATISTICS"))
        self._stat_list = _CheckList()
        self._stat_list.fill(COUNT_STATS + STAT_NAMES, set())
        self._stat_list.itemChanged.connect(self._schedule)
        lay.addWidget(self._stat_list, 2)
        prow = QHBoxLayout()
        prow.addWidget(QLabel("Percentile:"))
        self._pct = QDoubleSpinBox()
        self._pct.setRange(0.0, 100.0)
        self._pct.setDecimals(1)
        self._pct.setValue(95.0)
        self._pct.setPrefix("P")
        self._pct.valueChanged.connect(self._schedule)
        prow.addWidget(self._pct)
        prow.addStretch()
        lay.addLayout(prow)
        return w

    def _populations(self) -> list[str]:
        """Population paths of all samples, current sample's order first."""
        order = ([self._current] if self._current else []) + \
            [s for s in self._samples if s is not self._current]
        out = ["All Events"]
        for s in order:
            for g in _bfs(s.gates):
                p = population_path(g)
                if p not in out:
                    out.append(p)
        return out

    def _channels(self) -> list[str]:
        out = []
        for s in self._samples:
            for ch in s.channels:
                if ch not in out:
                    out.append(ch)
        return out

    def _restore(self):
        st = self._settings
        pops = self._populations()
        saved_pops = set(st.value(_SETTINGS + "populations", [], list) or [])
        # default: the deepest population of the current sample
        default_pop = {pops[-1]} if pops else set()
        self._pop_list.fill(pops, (saved_pops & set(pops)) or default_pop)
        chans = self._channels()
        saved_ch = set(st.value(_SETTINGS + "channels", [], list) or [])
        self._chan_list.fill(chans, saved_ch & set(chans))
        stats = set(st.value(_SETTINGS + "stats", [], list) or []) or \
            {"Count", "Median"}
        self._stat_list.fill(COUNT_STATS + STAT_NAMES, stats)
        self._pct.setValue(float(st.value(_SETTINGS + "pct", 95.0)))
        self._layout.setCurrentIndex(int(st.value(_SETTINGS + "layout", 0)))
        self._apply_filter()

    def _remember(self):
        st = self._settings
        st.setValue(_SETTINGS + "populations", self._pop_list.checked())
        st.setValue(_SETTINGS + "channels", self._chan_list.checked())
        st.setValue(_SETTINGS + "stats", self._stat_list.checked())
        st.setValue(_SETTINGS + "pct", self._pct.value())
        st.setValue(_SETTINGS + "layout", self._layout.currentIndex())

    def _apply_filter(self, *_):
        text = self._filter.text().strip().lower()
        ticked = self._ticked_only.isChecked()
        for i in range(self._chan_list.count()):
            it = self._chan_list.item(i)
            hide = (text and text not in it.text().lower()) or \
                (ticked and it.checkState() != Qt.CheckState.Checked)
            it.setHidden(bool(hide))

    def _on_channel_changed(self, *_):
        if self._ticked_only.isChecked():
            self._apply_filter()
        self._schedule()

    # ------------------------------------------------------------------
    # Table
    # ------------------------------------------------------------------

    def _schedule(self, *_):
        self._timer.start()

    def _chosen(self):
        names = set(self._sample_list.checked())
        samples = [s for s in self._samples if s.name in names]
        return (samples, self._pop_list.checked(), self._chan_list.checked(),
                self._stat_list.checked())

    def _recompute(self):
        samples, pops, chans, stats = self._chosen()
        self._pct.setEnabled("Percentile" in stats)
        need_channel = any(s not in COUNT_STATS for s in stats)
        hint = []
        if not samples:
            hint.append("tick at least one sample")
        if not pops:
            hint.append("tick a population")
        if not stats:
            hint.append("tick a statistic")
        elif need_channel and not chans:
            hint.append("tick the channel(s) for "
                        + ", ".join(s for s in stats
                                    if s not in COUNT_STATS))
        if not samples or not pops or not stats:
            self._df = pd.DataFrame()
        else:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                self._df = export_table(samples, pops, chans, stats,
                                        self._pct.value(),
                                        wide=self._layout.currentIndex() == 0,
                                        meta_columns=self._meta if
                                        self._meta_cb.isChecked() else None)
            finally:
                QApplication.restoreOverrideCursor()
        self._fill_table()
        df = self._df
        missing = int(df.iloc[:, 1:].isna().sum().sum()) if not df.empty \
            else 0
        text = (f"{len(df)} rows × {len(df.columns)} columns."
                if not df.empty else "Nothing to show yet.")
        if hint:
            text += "  To continue: " + "; ".join(hint) + "."
        if missing:
            text += (f"  {missing} empty cell(s): population not gated in "
                     "that sample (use Apply Gates to Samples) or channel "
                     "missing.")
        self._summary.setText(text)
        self._remember()

    def _fill_table(self):
        df = self._df
        t = self._table
        t.clear()
        t.setRowCount(len(df))
        t.setColumnCount(len(df.columns))
        t.setHorizontalHeaderLabels([str(c) for c in df.columns])
        for r, row in enumerate(df.itertuples(index=False)):
            for c, (col, val) in enumerate(zip(df.columns, row)):
                if isinstance(val, str):
                    text = val
                elif pd.isna(val):
                    text = "—"
                elif str(col).endswith("Count"):
                    text = f"{int(val):,}"
                else:
                    text = (f"{val:,.0f}" if abs(val) >= 1000
                            else f"{val:,.2f}")
                it = QTableWidgetItem(text)
                if not isinstance(val, str):
                    it.setTextAlignment(Qt.AlignmentFlag.AlignRight |
                                        Qt.AlignmentFlag.AlignVCenter)
                t.setItem(r, c, it)
        t.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents)

    def _rounded(self) -> pd.DataFrame:
        df = self._df.copy()
        for c in df.columns:
            if df[c].dtype.kind == "f":
                df[c] = df[c].round(4)
        return df

    def _copy(self):
        if self._df.empty:
            return
        QApplication.clipboard().setText(
            self._rounded().to_csv(sep="\t", index=False))
        self._summary.setText(self._summary.text().split("  Copied")[0] +
                              "  Copied — paste into Excel / Prism.")

    def _export(self):
        if self._df.empty:
            QMessageBox.information(self, "Export Data",
                                    "Nothing to export yet.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Data", "flowpy_data.csv", "CSV files (*.csv)")
        if not path:
            return
        if not path.lower().endswith(".csv"):
            path += ".csv"
        try:
            # utf-8-sig so Excel shows µ, ×, accents… correctly
            self._rounded().to_csv(path, index=False, encoding="utf-8-sig")
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))


class ChannelPickerDialog(QDialog):
    """Tick channels from a searchable list."""

    def __init__(self, channels: list[str], chosen: list[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Choose Channels")
        self.resize(420, 560)
        lay = QVBoxLayout(self)
        self._filter = QLineEdit()
        self._filter.setPlaceholderText("Search channels, e.g. YG5")
        self._filter.setClearButtonEnabled(True)
        lay.addWidget(self._filter)
        self._list = _CheckList()
        self._list.fill(list(channels), set(chosen))
        lay.addWidget(self._list, 1)
        row = _all_none(self._list, visible_only=True)
        lay.addLayout(row)
        self._filter.textChanged.connect(self._apply_filter)
        buttons = QHBoxLayout()
        buttons.addStretch()
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        ok = QPushButton("Show")
        ok.setObjectName("Primary")
        ok.clicked.connect(self.accept)
        buttons.addWidget(cancel)
        buttons.addWidget(ok)
        lay.addLayout(buttons)

    def _apply_filter(self, text: str):
        t = text.strip().lower()
        for i in range(self._list.count()):
            it = self._list.item(i)
            it.setHidden(bool(t) and t not in it.text().lower())

    def selected(self) -> list[str]:
        return self._list.checked()
