"""Statistics table widget — shows per-gate stats, supports CSV export."""

from __future__ import annotations
import numpy as np
import pandas as pd

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QFileDialog,
    QComboBox, QDoubleSpinBox, QMessageBox,
)
from PyQt6.QtCore import Qt, QSettings
from PyQt6.QtGui import QFont

from ..core.statistics import (
    gates_in_order, sample_stats, stat_label, STAT_NAMES,
)
from ..core.gates import Gate

ALL_EVENTS = "__all__"
SHOW_MODES = ["Plot channels", "All fluorescence", "Chosen channels"]


class StatisticsWidget(QWidget):
    """Displays a table of statistics for all gates in the hierarchy."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._sample = None
        self._root_gates: list[Gate] = []
        self._masks: dict[str, np.ndarray] = {}
        self._channels: list[str] = []       # columns shown
        self._plot_channels: list[str] = []
        self._chosen: list[str] = list(
            QSettings("FlowPy", "FlowPy").value("stats/chosen", [], list)
            or [])
        # (gate_id, stat, pct, channel) → value. Masks never change for a
        # gate id unless the gate or the data changes (see invalidate).
        self._cache: dict[tuple, float] = {}
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        hdr = QHBoxLayout()
        title = QLabel("STATISTICS")
        title.setObjectName("SectionHeader")
        hdr.addWidget(title)
        hdr.addStretch()

        hdr.addWidget(QLabel("Show:"))
        self._show_combo = QComboBox()
        self._show_combo.addItems(SHOW_MODES)
        self._show_combo.setToolTip(
            "Plot channels: the channels on the plot above · All "
            "fluorescence: every fluorescence channel · Chosen channels: "
            "the ones you pick")
        self._show_combo.currentIndexChanged.connect(self._on_show_changed)
        hdr.addWidget(self._show_combo)
        self._choose_btn = QPushButton("Choose…")
        self._choose_btn.setToolTip("Pick the channels to show")
        self._choose_btn.clicked.connect(self._choose_channels)
        self._choose_btn.setVisible(False)
        hdr.addWidget(self._choose_btn)
        hdr.addSpacing(10)
        hdr.addWidget(QLabel("Stat:"))
        self._stat_combo = QComboBox()
        self._stat_combo.addItems(STAT_NAMES)
        self._stat_combo.setCurrentText("Median")
        self._stat_combo.currentIndexChanged.connect(self._on_stat_changed)
        hdr.addWidget(self._stat_combo)

        self._pct = QDoubleSpinBox()
        self._pct.setRange(0.0, 100.0)
        self._pct.setDecimals(1)
        self._pct.setValue(95.0)
        self._pct.setPrefix("P")
        self._pct.setToolTip("Which percentile")
        self._pct.valueChanged.connect(self._refresh_table)
        self._pct.setVisible(False)
        hdr.addWidget(self._pct)

        self._export_btn = QPushButton("Export CSV…")
        self._export_btn.setToolTip("Every statistic for every channel")
        self._export_btn.clicked.connect(self._export_csv)
        hdr.addWidget(self._export_btn)
        layout.addLayout(hdr)

        self._table = QTableWidget()
        self._table.setAlternatingRowColors(False)
        self._table.verticalHeader().setVisible(False)
        self._table.setShowGrid(False)
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows)
        layout.addWidget(self._table)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_sample(self, sample):
        self._sample = sample
        self._root_gates = []
        self._masks = {}
        self._cache = {}
        self._update_channels()
        self._refresh_table()

    def set_plot_channels(self, channels: list[str]):
        """Channels on the gating plot (shown in 'Plot channels' mode)."""
        self._plot_channels = list(channels)
        if self._show_combo.currentIndex() == 0:
            self._update_channels()
            self._refresh_table()

    def _update_channels(self):
        s = self._sample
        if s is None:
            self._channels = []
            return
        mode = self._show_combo.currentIndex()
        if mode == 0:
            chans = self._plot_channels
        elif mode == 1:
            chans = s.fluorescence
        else:
            chans = self._chosen
        self._channels = [c for c in chans if c in s.channels]

    def _on_show_changed(self):
        self._choose_btn.setVisible(self._show_combo.currentIndex() == 2)
        if self._show_combo.currentIndex() == 2 and not self._chosen:
            self._choose_channels()
        self._update_channels()
        self._refresh_table()

    def _choose_channels(self):
        if self._sample is None:
            return
        from .export_dialog import ChannelPickerDialog
        dlg = ChannelPickerDialog(self._sample.channels, self._chosen, self)
        if dlg.exec():
            self._chosen = dlg.selected()
            QSettings("FlowPy", "FlowPy").setValue("stats/chosen",
                                                   self._chosen)
            self._update_channels()
            self._refresh_table()

    def invalidate(self):
        """Drop cached stats (call after gates or sample data change)."""
        self._cache = {}

    def set_gates(self, root_gates: list[Gate],
                  masks: dict[str, np.ndarray]):
        self._root_gates = list(root_gates)
        self._masks = masks
        self._refresh_table()

    # ------------------------------------------------------------------
    # Computation
    # ------------------------------------------------------------------

    def _populations(self) -> list[tuple[str, str, np.ndarray | None,
                                         np.ndarray | None]]:
        """(key, label, mask, parent mask) for All Events + every gate."""
        rows = [(ALL_EVENTS, "All Events", None, None)]
        for g in gates_in_order(self._root_gates):
            mask = self._masks.get(g.gate_id)
            if mask is None:
                continue
            par = self._masks.get(g.parent.gate_id) if g.parent else None
            rows.append((g.gate_id, "   " * len(g.ancestors()) + g.name,
                         mask, par))
        return rows

    def _values(self, key: str, mask, stat: str, pct: float,
                channels: list[str] | None = None) -> np.ndarray:
        """Stat of *channels* (default: the shown ones); only channels not
        computed before are computed."""
        channels = self._channels if channels is None else channels
        p = pct if stat == "Percentile" else None
        todo = [c for c in channels if (key, stat, p, c) not in self._cache]
        if todo:
            for c, v in zip(todo, sample_stats(self._sample, todo, stat, pct,
                                               mask)):
                self._cache[(key, stat, p, c)] = v
        return np.array([self._cache[(key, stat, p, c)] for c in channels])

    def _frame(self, stats: list[str], pct: float,
               channels: list[str] | None = None) -> pd.DataFrame:
        channels = self._channels if channels is None else channels
        n_total = self._sample.n_events
        rows = []
        for key, label, mask, par in self._populations():
            n = int(mask.sum()) if mask is not None else n_total
            n_par = int(par.sum()) if par is not None else n_total
            row = {"Gate": label, "Count": n,
                   "% Total": 100.0 * n / n_total if n_total else 0.0,
                   "% Parent": 100.0 * n / n_par if n_par else 0.0}
            if channels:
                for st in stats:
                    vals = self._values(key, mask, st, pct, channels)
                    suffix = "" if len(stats) == 1 else " " + stat_label(st, pct)
                    for ch, v in zip(channels, vals):
                        row[ch + suffix] = v
            rows.append(row)
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Table refresh
    # ------------------------------------------------------------------

    def _on_stat_changed(self):
        self._pct.setVisible(self._stat_combo.currentText() == "Percentile")
        self._refresh_table()

    def _refresh_table(self):
        self._table.clear()
        if self._sample is None or not self._root_gates:
            self._table.setRowCount(0)
            self._table.setColumnCount(0)
            return
        stat = self._stat_combo.currentText()
        df = self._frame([stat], self._pct.value())

        self._table.setRowCount(len(df))
        self._table.setColumnCount(len(df.columns))
        self._table.setHorizontalHeaderLabels(list(df.columns))
        pct_cols = {"% Total", "% Parent"}
        for r, row in enumerate(df.itertuples(index=False)):
            for c, (col, val) in enumerate(zip(df.columns, row)):
                if c == 0:
                    item = QTableWidgetItem(str(val))
                    f = item.font()
                    f.setWeight(QFont.Weight.DemiBold)
                    item.setFont(f)
                else:
                    if col == "Count":
                        text = f"{int(val):,}"
                    elif pd.isna(val):
                        text = "—"
                    elif col in pct_cols:
                        text = f"{val:.1f}%"
                    else:
                        text = f"{val:,.1f}"
                    item = QTableWidgetItem(text)
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight |
                                          Qt.AlignmentFlag.AlignVCenter)
                self._table.setItem(r, c, item)

    # ------------------------------------------------------------------
    # CSV export
    # ------------------------------------------------------------------

    def _export_csv(self):
        if self._sample is None or not self._root_gates:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Statistics", "", "CSV files (*.csv)")
        if not path:
            return
        stats = [s for s in STAT_NAMES if s != "Percentile"]
        if self._stat_combo.currentText() == "Percentile":
            stats.append("Percentile")
        try:
            # the export has every fluorescence channel plus any shown
            chans = list(dict.fromkeys(self._sample.fluorescence +
                                       self._channels))
            df = self._frame(stats, self._pct.value(), chans)
            df["Gate"] = df["Gate"].str.strip()
            df.round(3).to_csv(path, index=False)
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))
