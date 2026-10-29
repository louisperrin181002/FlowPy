"""Dialogs for working on many samples at once: choosing samples to apply a
gating protocol to, and the batch statistics table."""

from __future__ import annotations
import pandas as pd

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QPushButton, QDialogButtonBox, QComboBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QFileDialog, QMessageBox, QDoubleSpinBox,
)
from PyQt6.QtCore import Qt

from ..core.protocol import batch_statistics, missing_channels
from ..core.statistics import STAT_NAMES


class ApplyProtocolDialog(QDialog):
    """Tick the samples that should receive the gating protocol."""

    def __init__(self, samples, roots, source=None, title="Apply Gates",
                 parent=None, formulas: dict | None = None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(460, 480)
        self._samples = samples
        lay = QVBoxLayout(self)

        n_gates = _count(roots)
        src = f" from <b>{source.name}</b>" if source is not None else ""
        info = QLabel(
            f"Copy the gating protocol ({n_gates} gates){src} to the ticked "
            "samples.<br><span style='color:#64748b;'>Their existing gates "
            "are replaced. Each sample gets its own copy, so you can still "
            "adjust a gate on one sample afterwards.</span>")
        info.setWordWrap(True)
        lay.addWidget(info)

        self._list = QListWidget()
        for i, s in enumerate(samples):
            missing = _really_missing(s, missing_channels(s, roots),
                                      formulas or {})
            if s is source:
                continue
            it = QListWidgetItem(s.name + (f"   — {len(_flat(s.gates))} gates "
                                           "(will be replaced)"
                                           if s.gates else ""))
            it.setData(Qt.ItemDataRole.UserRole, i)
            if missing:
                it.setText(f"{s.name}   — missing {', '.join(missing)}")
                it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEnabled)
                it.setCheckState(Qt.CheckState.Unchecked)
            else:
                it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                it.setCheckState(Qt.CheckState.Checked)
            self._list.addItem(it)
        lay.addWidget(self._list, 1)

        row = QHBoxLayout()
        for text, state in (("Select all", Qt.CheckState.Checked),
                            ("Select none", Qt.CheckState.Unchecked)):
            b = QPushButton(text)
            b.clicked.connect(lambda _c, st=state: self._set_all(st))
            row.addWidget(b)
        row.addStretch()
        lay.addLayout(row)

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                              QDialogButtonBox.StandardButton.Cancel)
        bb.button(QDialogButtonBox.StandardButton.Ok).setText("Apply")
        bb.button(QDialogButtonBox.StandardButton.Ok).setObjectName("Primary")
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def _set_all(self, state):
        for i in range(self._list.count()):
            it = self._list.item(i)
            if it.flags() & Qt.ItemFlag.ItemIsEnabled:
                it.setCheckState(state)

    def selected_samples(self):
        return [self._samples[self._list.item(i).data(Qt.ItemDataRole.UserRole)]
                for i in range(self._list.count())
                if self._list.item(i).checkState() == Qt.CheckState.Checked]


class BatchStatisticsDialog(QDialog):
    """Statistics for every population in every sample, in one table."""

    VIEWS = ["% Parent — samples × populations",
             "Count — samples × populations",
             "% Total — samples × populations",
             "Full table (all statistics)"]

    def __init__(self, samples, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Batch Statistics")
        self.resize(1000, 560)
        self._samples = samples
        self._df: pd.DataFrame | None = None
        lay = QVBoxLayout(self)

        top = QHBoxLayout()
        top.addWidget(QLabel("Show:"))
        self._view = QComboBox()
        self._view.addItems(self.VIEWS)
        self._view.currentIndexChanged.connect(self._refresh)
        top.addWidget(self._view)
        top.addSpacing(12)
        top.addWidget(QLabel("Channel statistic:"))
        self._stat = QComboBox()
        self._stat.addItems(STAT_NAMES)
        self._stat.setCurrentText("Median")
        self._stat.currentIndexChanged.connect(self._recompute)
        top.addWidget(self._stat)
        self._pct = QDoubleSpinBox()
        self._pct.setRange(0.0, 100.0)
        self._pct.setDecimals(1)
        self._pct.setValue(95.0)
        self._pct.setPrefix("P")
        self._pct.valueChanged.connect(self._recompute)
        top.addWidget(self._pct)
        top.addStretch()
        export = QPushButton("Export CSV…")
        export.setObjectName("Primary")
        export.clicked.connect(self._export)
        top.addWidget(export)
        lay.addLayout(top)

        self._table = QTableWidget()
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.verticalHeader().setVisible(False)
        self._table.setShowGrid(False)
        lay.addWidget(self._table, 1)
        self._recompute()

    def _recompute(self):
        stat = self._stat.currentText()
        self._pct.setVisible(stat == "Percentile")
        self._df = batch_statistics(self._samples, stat, self._pct.value())
        if self.sender() in (self._stat, self._pct):
            # channel statistics are only shown in the full table
            self._view.setCurrentIndex(3)
        self._refresh()

    def _shown(self) -> pd.DataFrame:
        df = self._df
        if df is None or df.empty:
            return pd.DataFrame()
        v = self._view.currentIndex()
        if v == 3:
            return df
        value = ["% Parent", "Count", "% Total"][v]
        order = list(dict.fromkeys(df["Population"]))
        wide = df.pivot_table(index="Sample", columns="Population",
                              values=value, aggfunc="first", sort=False)
        wide = wide[[c for c in order if c in wide.columns]]
        if v != 1:
            wide = wide.drop(columns=["All Events"], errors="ignore")
        return wide.reindex([s.name for s in self._samples]).reset_index()

    def _refresh(self):
        df = self._shown()
        self._table.clear()
        self._table.setRowCount(len(df))
        self._table.setColumnCount(len(df.columns))
        self._table.setHorizontalHeaderLabels([str(c) for c in df.columns])
        for r, row in enumerate(df.itertuples(index=False)):
            for c, val in enumerate(row):
                if isinstance(val, float):
                    text = "—" if pd.isna(val) else (
                        f"{val:,.0f}" if self._view.currentIndex() == 1
                        else f"{val:,.2f}" if abs(val) < 1000 else f"{val:,.0f}")
                elif isinstance(val, int):
                    text = f"{val:,}"
                else:
                    text = str(val)
                it = QTableWidgetItem(text)
                if c > 0 and not isinstance(val, str):
                    it.setTextAlignment(Qt.AlignmentFlag.AlignRight |
                                        Qt.AlignmentFlag.AlignVCenter)
                self._table.setItem(r, c, it)
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents)

    def _export(self):
        df = self._shown()
        if df.empty:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Batch Statistics", "batch_statistics.csv",
            "CSV files (*.csv)")
        if not path:
            return
        try:
            df.to_csv(path, index=False)
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))


def _really_missing(sample, missing: list[str], formulas: dict) -> list[str]:
    """Channels still missing once derived parameters in *formulas* are
    created (they need their own input channels)."""
    from ..core.formulas import channels_in

    def creatable(c, seen=()):
        spec = formulas.get(c)
        return spec is not None and c not in seen and all(
            ch in sample.channels or creatable(ch, seen + (c,))
            for ch in channels_in(spec["expr"]))
    return [c for c in missing if not creatable(c)]


def _flat(roots):
    out, q = [], list(roots)
    while q:
        g = q.pop(0)
        out.append(g)
        q.extend(g.children)
    return out


def _count(roots) -> int:
    return len(_flat(roots))


class ExchangeReportDialog(QDialog):
    """Result of a FlowJo / Gating-ML import or export, or a FlowKit
    cross-check: notes, plus FlowPy's counts next to FlowKit's."""

    def __init__(self, title: str, summary: str, report, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(820, 520)
        self._report = report
        lay = QVBoxLayout(self)
        n, bad = len(report.counts), report.mismatches
        if n:
            verdict = (f"<b style='color:#16a34a;'>✓ All {n} population "
                       "counts agree with FlowKit.</b>" if not bad else
                       f"<b style='color:#dc2626;'>{len(bad)} of {n} "
                       "population counts differ from FlowKit</b> "
                       "(highlighted).")
            summary += "<br>" + verdict
        info = QLabel(summary)
        info.setWordWrap(True)
        lay.addWidget(info)
        if report.notes:
            notes = QListWidget()
            notes.addItems(report.notes)
            notes.setMaximumHeight(140)
            lay.addWidget(notes)
        if n:
            t = QTableWidget(n, 5)
            t.setHorizontalHeaderLabels(["Sample", "Population", "FlowPy",
                                         "FlowKit", "Difference"])
            t.verticalHeader().setVisible(False)
            t.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
            for r, (s, pop, a, b) in enumerate(report.counts):
                vals = [s, pop, f"{a:,}", f"{b:,}",
                        "" if a == b else f"{a - b:+,}"]
                for c, v in enumerate(vals):
                    it = QTableWidgetItem(v)
                    if c >= 2:
                        it.setTextAlignment(Qt.AlignmentFlag.AlignRight |
                                            Qt.AlignmentFlag.AlignVCenter)
                    if a != b:
                        it.setBackground(Qt.GlobalColor.yellow)
                    t.setItem(r, c, it)
            t.horizontalHeader().setSectionResizeMode(
                1, QHeaderView.ResizeMode.Stretch)
            lay.addWidget(t, 1)
        row = QHBoxLayout()
        row.addStretch()
        if n:
            exp = QPushButton("Export CSV…")
            exp.clicked.connect(self._export)
            row.addWidget(exp)
        ok = QPushButton("Close")
        ok.setObjectName("Primary")
        ok.clicked.connect(self.accept)
        row.addWidget(ok)
        lay.addLayout(row)

    def _export(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Comparison", "flowkit_comparison.csv",
            "CSV files (*.csv)")
        if path:
            pd.DataFrame(self._report.counts, columns=[
                "Sample", "Population", "FlowPy", "FlowKit"]).to_csv(
                path, index=False)


class PasteGatesDialog(QDialog):
    """Where pasted gates go: which samples, and replace / next to / inside
    a population."""

    def __init__(self, samples, current, clip: dict, inside: str | None,
                 populations: list[str], parent=None):
        from PyQt6.QtWidgets import QRadioButton, QComboBox, QButtonGroup
        super().__init__(parent)
        self.setWindowTitle("Paste Gates")
        self.resize(500, 540)
        self._samples = samples
        roots = clip["roots"]
        lay = QVBoxLayout(self)
        names = ", ".join(g.name for g in roots[:4]) + (" …" if len(roots) > 4 else "")
        src = f" from <b>{clip['source']}</b>" if clip.get("source") else ""
        info = QLabel(f"Paste <b>{clip['n']} gate{'s' if clip['n'] != 1 else ''}</b> "
                      f"({names}){src}.")
        info.setWordWrap(True)
        lay.addWidget(info)

        lay.addWidget(QLabel("<b>Where</b>"))
        self._add = QRadioButton("Next to the existing gates")
        self._inside = QRadioButton("Inside the population:")
        self._replace = QRadioButton("Replace all gates")
        self._pop = QComboBox()
        self._pop.addItems(populations)
        group = QButtonGroup(self)
        for b in (self._add, self._inside, self._replace):
            group.addButton(b)
        lay.addWidget(self._add)
        row = QHBoxLayout()
        row.addWidget(self._inside)
        row.addWidget(self._pop, 1)
        lay.addLayout(row)
        lay.addWidget(self._replace)
        want = inside or clip.get("parent")
        if want and want != "All Events" and want in populations:
            self._pop.setCurrentText(want)
            self._inside.setChecked(True)
        else:
            self._add.setChecked(True)
        self._pop.currentIndexChanged.connect(lambda: self._inside.setChecked(True))

        lay.addWidget(QLabel("<b>Samples</b>"))
        self._list = QListWidget()
        for i, s in enumerate(samples):
            missing = _really_missing(s, missing_channels(s, roots),
                                      clip.get("formulas") or {})
            it = QListWidgetItem(s.name + ("   (this sample)" if s is current else ""))
            it.setData(Qt.ItemDataRole.UserRole, i)
            if missing:
                it.setText(f"{s.name}   — missing {', '.join(missing)}")
                it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEnabled)
                it.setCheckState(Qt.CheckState.Unchecked)
            else:
                it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                it.setCheckState(Qt.CheckState.Checked if s is current
                                 else Qt.CheckState.Unchecked)
            self._list.addItem(it)
        lay.addWidget(self._list, 1)
        row = QHBoxLayout()
        for text, state in (("Select all", Qt.CheckState.Checked),
                            ("Select none", Qt.CheckState.Unchecked)):
            b = QPushButton(text)
            b.clicked.connect(lambda _c, st=state: self._set_all(st))
            row.addWidget(b)
        row.addStretch()
        lay.addLayout(row)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                              QDialogButtonBox.StandardButton.Cancel)
        bb.button(QDialogButtonBox.StandardButton.Ok).setText("Paste")
        bb.button(QDialogButtonBox.StandardButton.Ok).setObjectName("Primary")
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def _set_all(self, state):
        for i in range(self._list.count()):
            it = self._list.item(i)
            if it.flags() & Qt.ItemFlag.ItemIsEnabled:
                it.setCheckState(state)

    def selected_samples(self):
        return [self._samples[self._list.item(i).data(Qt.ItemDataRole.UserRole)]
                for i in range(self._list.count())
                if self._list.item(i).checkState() == Qt.CheckState.Checked]

    def placement(self):
        """(mode, population path or None) — see protocol.paste_gates."""
        from ..core.protocol import PASTE_ADD, PASTE_INSIDE, PASTE_REPLACE
        if self._replace.isChecked():
            return PASTE_REPLACE, None
        if self._inside.isChecked():
            return PASTE_INSIDE, self._pop.currentText()
        return PASTE_ADD, None
