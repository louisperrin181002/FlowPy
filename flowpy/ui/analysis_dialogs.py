"""Dialogs for the Analysis menu: UMAP / t-SNE, clustering (+ heatmap),
automatic gating and spectral unmixing — and a helper that runs the heavy
computation off the UI thread."""

from __future__ import annotations
import traceback
from pathlib import Path
import numpy as np

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QListWidget,
    QListWidgetItem, QPushButton, QDialogButtonBox, QComboBox, QSpinBox,
    QDoubleSpinBox, QCheckBox, QProgressDialog, QTableWidget, QLineEdit,
    QHeaderView, QFileDialog, QMessageBox, QWidget, QStackedWidget,
)
from PyQt6.QtCore import Qt, QThread, QEventLoop

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from ..core.analysis import CLUSTER_METHODS, COVERAGE


# ---------------------------------------------------------------------------
# Background execution
# ---------------------------------------------------------------------------

class _Worker(QThread):
    def __init__(self, fn):
        super().__init__()
        self.fn, self.result, self.error, self.tb = fn, None, None, ""

    def run(self):
        try:
            self.result = self.fn()
        except Exception as exc:          # reported on the UI thread
            self.error, self.tb = exc, traceback.format_exc()


def run_busy(parent, text: str, fn):
    """Run *fn* in a worker thread behind a modal busy indicator; return its
    result or re-raise its exception."""
    dlg = QProgressDialog(text, None, 0, 0, parent)
    dlg.setWindowTitle("Working…")
    dlg.setWindowModality(Qt.WindowModality.WindowModal)
    dlg.setMinimumDuration(0)
    dlg.setCancelButton(None)
    worker = _Worker(fn)
    loop = QEventLoop()
    worker.finished.connect(loop.quit)
    worker.start()
    dlg.show()
    loop.exec()
    dlg.close()
    if worker.error is not None:
        raise worker.error
    return worker.result


# ---------------------------------------------------------------------------
# Shared widgets
# ---------------------------------------------------------------------------

class ChannelChecklist(QListWidget):
    def __init__(self, channels: list[str], checked: list[str]):
        super().__init__()
        for ch in channels:
            it = QListWidgetItem(ch)
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Checked if ch in checked
                             else Qt.CheckState.Unchecked)
            self.addItem(it)

    def checked(self) -> list[str]:
        return [self.item(i).text() for i in range(self.count())
                if self.item(i).checkState() == Qt.CheckState.Checked]


def _buttons(dlg, ok_text="Run"):
    bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                          QDialogButtonBox.StandardButton.Cancel)
    ok = bb.button(QDialogButtonBox.StandardButton.Ok)
    ok.setText(ok_text)
    ok.setObjectName("Primary")
    bb.accepted.connect(dlg._accept)
    bb.rejected.connect(dlg.reject)
    return bb


def _computable(sample) -> list[str]:
    """Measured channels only (not Time, not FlowPy-computed ones)."""
    return [c for c in sample.channels
            if c not in sample.derived and "TIME" not in c.upper()]


# ---------------------------------------------------------------------------
# UMAP / t-SNE
# ---------------------------------------------------------------------------

class DimRedDialog(QDialog):
    def __init__(self, sample, population: str, n_events: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("UMAP / t-SNE")
        self.setMinimumWidth(440)
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel(
            f"Embed <b>{population}</b> ({n_events:,} events) in 2-D. "
            "The result is added as two new channels you can plot and gate."))

        form = QFormLayout()
        self._method = QComboBox()
        self._method.addItems(["UMAP", "t-SNE"])
        self._method.currentIndexChanged.connect(self._on_method)
        form.addRow("Method:", self._method)
        self._n = QSpinBox()
        self._n.setRange(500, 500_000)
        self._n.setSingleStep(5000)
        self._n.setValue(min(20_000, n_events))
        self._n.setToolTip("Events are sampled at random; the others are "
                           "left out of the embedding")
        form.addRow("Events to embed:", self._n)
        self._p1 = QSpinBox()
        self._p1.setRange(2, 200)
        self._p1.setValue(15)
        form.addRow("Neighbours:", self._p1)
        self._p1_label = form.labelForField(self._p1)
        self._p2 = QDoubleSpinBox()
        self._p2.setRange(0.0, 1.0)
        self._p2.setSingleStep(0.05)
        self._p2.setValue(0.1)
        form.addRow("Min. distance:", self._p2)
        self._p2_label = form.labelForField(self._p2)
        lay.addLayout(form)

        lay.addWidget(QLabel("Channels to use:"))
        chans = _computable(sample)
        self._chans = ChannelChecklist(chans, sample.fluorescence or chans)
        lay.addWidget(self._chans)
        self._note = QLabel()
        self._note.setStyleSheet("color:#64748b;")
        lay.addWidget(self._note)
        lay.addWidget(_buttons(self))
        self._on_method()

    def _on_method(self):
        umap = self._method.currentText() == "UMAP"
        self._p1_label.setText("Neighbours:" if umap else "Perplexity:")
        self._p1.setValue(15 if umap else 30)
        self._p2.setVisible(umap)
        self._p2_label.setVisible(umap)
        self._note.setText(
            "UMAP: ~15 s for 20,000 events (the first run is slower)." if umap
            else "t-SNE is slow: keep it to ~10,000 events.")
        if not umap and self._n.value() > 10_000:
            self._n.setValue(10_000)

    def _accept(self):
        if len(self._chans.checked()) < 2:
            QMessageBox.warning(self, "UMAP / t-SNE", "Tick at least two "
                                "channels.")
            return
        self.accept()

    def params(self) -> dict:
        return dict(method=self._method.currentText(), n=self._n.value(),
                    channels=self._chans.checked(), p1=self._p1.value(),
                    p2=self._p2.value())


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

class ClusterDialog(QDialog):
    def __init__(self, sample, population: str, n_events: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Clustering")
        self.setMinimumWidth(440)
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel(
            f"Cluster the <b>{population}</b> events ({n_events:,}). Each "
            "event gets a cluster number (new <b>Cluster</b> channel)."))
        form = QFormLayout()
        self._method = QComboBox()
        self._method.addItems(CLUSTER_METHODS)
        self._method.setToolTip(
            "FlowSOM: the official implementation — 10×10 self-organising "
            "map merged into metaclusters (robust, the usual choice for "
            "cytometry).\nK-means: fast, round "
            "clusters.\nGaussian mixture: elliptical clusters.")
        form.addRow("Method:", self._method)
        self._k = QSpinBox()
        self._k.setRange(2, 40)
        self._k.setValue(8)
        form.addRow("Number of clusters:", self._k)
        lay.addLayout(form)
        lay.addWidget(QLabel("Channels to use:"))
        chans = _computable(sample)
        self._chans = ChannelChecklist(chans, sample.fluorescence or chans)
        lay.addWidget(self._chans)
        self._gates = QCheckBox("Create a gate for each cluster (shows "
                                "counts / % / statistics)")
        self._gates.setChecked(True)
        lay.addWidget(self._gates)
        lay.addWidget(_buttons(self))

    def _accept(self):
        if not self._chans.checked():
            QMessageBox.warning(self, "Clustering", "Tick at least one "
                                "channel.")
            return
        self.accept()

    def params(self) -> dict:
        return dict(method=self._method.currentText(), k=self._k.value(),
                    channels=self._chans.checked(),
                    make_gates=self._gates.isChecked())


class ClusterHeatmapDialog(QDialog):
    """Median expression of each channel per cluster (scaled 0–1 per
    channel), with cluster sizes."""

    def __init__(self, profile: np.ndarray, channels: list[str],
                 sizes: np.ndarray, colors: list[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Cluster Profiles")
        self.resize(760, 140 + 34 * len(sizes))
        lay = QVBoxLayout(self)
        lo, hi = np.nanmin(profile, axis=0), np.nanmax(profile, axis=0)
        scaled = (profile - lo) / np.where(hi > lo, hi - lo, 1.0)

        k = len(sizes)
        self._fig = Figure(figsize=(1.2 + 0.75 * len(channels),
                                    0.9 + 0.36 * k), dpi=100)
        ax = self._fig.add_subplot(111)
        im = ax.imshow(scaled, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(len(channels)))
        ax.set_xticklabels(channels, rotation=35, ha="right", fontsize=8)
        total = sizes.sum()
        ax.set_yticks(range(k))
        ax.set_yticklabels([f"Cluster {i + 1}  ({100 * n / total:.1f}%)"
                            for i, n in enumerate(sizes)], fontsize=8)
        for i, c in enumerate(colors):
            ax.get_yticklabels()[i].set_color(c)
        for sp in ax.spines.values():
            sp.set_visible(False)
        cb = self._fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cb.set_label("median, scaled per channel", fontsize=8)
        cb.ax.tick_params(labelsize=7)
        self._fig.tight_layout()
        lay.addWidget(FigureCanvas(self._fig))

        row = QHBoxLayout()
        row.addStretch()
        save = QPushButton("Export…")
        save.clicked.connect(self._export)
        row.addWidget(save)
        close = QPushButton("Close")
        close.setObjectName("Primary")
        close.clicked.connect(self.accept)
        row.addWidget(close)
        lay.addLayout(row)

    def _export(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Heatmap", "cluster_heatmap.png",
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)")
        if path:
            from .theme import export_dpi
            self._fig.savefig(path, dpi=export_dpi(), facecolor="white",
                              bbox_inches="tight")


# ---------------------------------------------------------------------------
# Automatic gating
# ---------------------------------------------------------------------------

class AutoGateDialog(QDialog):
    def __init__(self, x_ch: str, y_ch: str, population: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Auto Gate")
        self.setMinimumWidth(420)
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel(
            f"Find populations on <b>{x_ch} / {y_ch}</b> in <b>{population}"
            "</b> and draw an ellipse gate around each (Gaussian mixture "
            "model). Adjust them afterwards like any gate."))
        form = QFormLayout()
        self._n = QComboBox()
        self._n.addItems(["Automatic"] + [str(i) for i in range(1, 9)])
        form.addRow("Populations:", self._n)
        self._cov = QComboBox()
        self._cov.addItems(list(COVERAGE))
        self._cov.setCurrentText("95%")
        self._cov.setToolTip("How much of each population the ellipse "
                             "should contain")
        form.addRow("Gate size:", self._cov)
        self._min = QDoubleSpinBox()
        self._min.setRange(0.0, 50.0)
        self._min.setValue(2.0)
        self._min.setSuffix(" %")
        form.addRow("Ignore populations under:", self._min)
        self._name = QLineEdit("Auto")
        form.addRow("Gate names:", self._name)
        lay.addLayout(form)
        lay.addWidget(_buttons(self, "Find Populations"))

    def _accept(self):
        self.accept()

    def params(self) -> dict:
        n = self._n.currentText()
        return dict(n=None if n == "Automatic" else int(n),
                    coverage=self._cov.currentText(),
                    min_fraction=self._min.value() / 100.0,
                    name=self._name.text().strip() or "Auto")


# ---------------------------------------------------------------------------
# Spectral unmixing
# ---------------------------------------------------------------------------

ROLE_IGNORE, ROLE_UNSTAINED, ROLE_SINGLE, ROLE_UNMIX = \
    "—", "Unstained control", "Single-stain control", "Unmix"


class UnmixDialog(QDialog):
    """Assign controls, pick detectors, choose samples to unmix."""

    def __init__(self, samples, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Spectral Unmixing")
        self.resize(760, 560)
        self._samples = samples
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel(
            "Ordinary least-squares unmixing. Each single-stain control "
            "gives a fluorochrome signature (median spectrum of its "
            "brightest 5% of events, minus the unstained control); every "
            "sample marked <b>Unmix</b> gets one new channel per "
            "fluorochrome."))

        self._table = QTableWidget(len(samples), 3)
        self._table.setHorizontalHeaderLabels(["Sample", "Role",
                                               "Fluorochrome name"])
        self._table.verticalHeader().setVisible(False)
        self._roles, self._names = [], []
        for r, s in enumerate(samples):
            self._table.setCellWidget(r, 0, QLabel(" " + s.name))
            role = QComboBox()
            role.addItems([ROLE_IGNORE, ROLE_UNSTAINED, ROLE_SINGLE,
                           ROLE_UNMIX])
            low = s.name.lower()
            role.setCurrentText(ROLE_UNSTAINED if "unstain" in low
                                else ROLE_SINGLE if "single" in low
                                or "control" in low or "ctrl" in low
                                else ROLE_UNMIX)
            self._table.setCellWidget(r, 1, role)
            name = QLineEdit(Path(s.name).stem)
            self._table.setCellWidget(r, 2, name)
            self._roles.append(role)
            self._names.append(name)
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        lay.addWidget(self._table, 2)

        lay.addWidget(QLabel("Detectors (spectral channels):"))
        common = [c for c in samples[0].channels
                  if all(c in s.channels for s in samples)]
        dets = [c for c in common if c in samples[0].fluorescence]
        self._dets = ChannelChecklist(common, dets)
        lay.addWidget(self._dets, 1)
        lay.addWidget(_buttons(self, "Unmix"))

    def _accept(self):
        try:
            self._result = self._collect()
        except ValueError as exc:
            QMessageBox.warning(self, "Spectral Unmixing", str(exc))
            return
        self.accept()

    def _collect(self) -> dict:
        unstained, controls, targets = None, [], []
        for s, role, name in zip(self._samples, self._roles, self._names):
            r = role.currentText()
            if r == ROLE_UNSTAINED:
                if unstained is not None:
                    raise ValueError("Mark only one unstained control.")
                unstained = s
            elif r == ROLE_SINGLE:
                if not name.text().strip():
                    raise ValueError(f"Name the fluorochrome of {s.name}.")
                controls.append((s, name.text().strip()))
            elif r == ROLE_UNMIX:
                targets.append(s)
        dets = self._dets.checked()
        if not controls:
            raise ValueError("Mark at least one single-stain control.")
        if not targets:
            raise ValueError("Mark at least one sample to unmix.")
        if len(dets) < len(controls):
            raise ValueError(f"{len(controls)} fluorochromes need at least "
                             f"as many detectors (ticked: {len(dets)}).")
        names = [n for _, n in controls]
        if len(set(names)) != len(names):
            raise ValueError("Fluorochrome names must be different.")
        return dict(unstained=unstained, controls=controls, targets=targets,
                    detectors=dets)

    def params(self) -> dict:
        return self._result
