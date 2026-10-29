"""FlowPy — Main application window."""

from __future__ import annotations
import json
import numpy as np
from datetime import datetime
from pathlib import Path

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QSplitter, QVBoxLayout, QHBoxLayout,
    QListWidget, QListWidgetItem, QPushButton, QLabel, QToolBar,
    QFileDialog, QMessageBox, QStatusBar, QDockWidget, QTabWidget,
    QMenuBar, QMenu, QApplication, QDialog, QInputDialog, QToolButton,
)
from PyQt6.QtCore import Qt, QSettings, QTimer, QUrl, pyqtSignal
from PyQt6.QtGui import QAction, QKeySequence, QColor, QIcon, QDesktopServices

from .plot_widget import PlotWidget
from .gate_tree_widget import GateTreeWidget, _bfs
from .statistics_widget import StatisticsWidget
from .compensation_dialog import CompensationDialog
from .layout_editor import LayoutEditor
from .export_dialog import ExportDataDialog
from .derived_dialog import DerivedParameterDialog
from .sample_list import SampleList
from .sample_loader import SampleLoader
from .plate_dialog import PlateDialog
from .widgets import install_combo_fitter
from .metadata_dialog import MetadataDialog
from .group_plot import GroupPlotWidget
from ..core import groups as groups_core
from .batch_dialogs import (
    ApplyProtocolDialog, BatchStatisticsDialog, ExchangeReportDialog,
    PasteGatesDialog,
)
from .gate_editor import EditGateDialog, BooleanGateDialog
from . import analysis_dialogs as adlg
from . import theme

from ..core.sample import Sample
from ..core.gates import (
    Gate, EllipseGate, RangeGate, BooleanGate, next_gate_color,
    can_refit, refit, refit_error,
)
from ..core import protocol, analysis

UNDO_LIMIT = 50
WORKSPACE_FILTER = "FlowPy Workspace (*.fpw)"

# Until a workspace has been saved under a name, autosave keeps a recovery
# copy of the session here (a regular workspace file)
SESSIONS_DIR = Path.home() / "Documents" / "FlowPy" / "sessions"

# Every open window (each holds its own workspace); kept here so they
# aren't garbage-collected
_WINDOWS: list = []


def open_window(workspace: str | None = None, fcs: list | None = None,
                near=None):
    """A new FlowPy window, optionally opening a workspace or FCS files in
    it. A workspace already open in a window brings that window forward."""
    if workspace:
        target = Path(workspace).resolve()
        for w in _WINDOWS:
            if w._doc_path is not None and w._doc_path.resolve() == target:
                w.show()
                w.raise_()
                w.activateWindow()
                return w
    w = MainWindow()
    if near is not None:
        g = near.geometry()
        w.setGeometry(g.translated(30, 30))
    w.show()
    if workspace:
        w.load_workspace_file(workspace)
    if fcs:
        w.load_fcs_files(fcs)
    return w




def _fresh(sample) -> Sample:
    """A newly loaded copy of *sample*'s file (same compensation)."""
    s = Sample(sample.filepath)
    s.restore_comp_state(sample.comp_state())
    return s


def _cluster_colors(k: int) -> list[str]:
    import matplotlib
    cmap = matplotlib.colormaps["tab20" if k > 10 else "tab10"]
    return [matplotlib.colors.to_hex(cmap(i % cmap.N)) for i in range(k)]


SYNC_LOAD_BYTES = 16 * 2**20   # files this small open at once when clicked


def _file_size(path: str) -> int:
    try:
        return Path(path).stat().st_size
    except OSError:
        return 0


def _speed_note(done_bytes: int, total: int, seconds: float) -> str:
    """'  ·  4.1 of 26 GB  ·  9 MB/s  ·  about 40 min left' for big loads
    (a slow rate means the files are on a slow network share)."""
    if total < 2 * 2**30 or done_bytes == 0 or seconds < 2:
        return ""
    rate = done_bytes / seconds
    left = (total - done_bytes) / rate
    when = (f"about {left / 60:.0f} min left" if left >= 90
            else f"about {left:.0f} s left")
    note = (f"\n{done_bytes / 2**30:.1f} of {total / 2**30:.0f} GB  ·  "
            f"{rate / 2**20:.0f} MB/s  ·  {when}")
    if rate < 40 * 2**20:
        note += ("\nThe network drive is the limit here (a wired "
                 "connection on site is much faster than Wi-Fi or VPN).")
    return note


class MainWindow(QMainWindow):
    """Top-level application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("FlowPy — Flow Cytometry Analysis")
        self.resize(1400, 900)
        theme.apply_mpl_style()
        app = QApplication.instance()
        if app is not None and not hasattr(app, "_combo_fitter"):
            install_combo_fitter(app)
        self.setStyleSheet(theme.STYLESHEET)

        # Application state
        self._samples: list[Sample] = []
        # Sample metadata columns (values live in sample.annotations)
        self._meta_columns: list[str] = list(groups_core.DEFAULT_COLUMNS)
        self._current_sample: Sample | None = None
        self._selected_gate_id: str | None = None

        # The workspace file this window is editing (None = not saved yet).
        # Autosave writes to it; before it exists, to a recovery session.
        self._doc_path: Path | None = None
        self._dirty = False
        self._session_path: Path | None = None
        self._loading = False
        self._autosave_timer = QTimer(self)
        self._autosave_timer.setSingleShot(True)
        self._autosave_timer.setInterval(1500)
        self._autosave_timer.timeout.connect(self._autosave)

        # Samples open from their headers; their events are read in the
        # background (the one being looked at first)
        self._loader = SampleLoader(self)
        self._waiting_for: Sample | None = None

        self._build_menu()
        self._build_toolbar()
        self._build_central()
        self._build_statusbar()
        self._sample_list.set_busy_check(self._loader.is_busy)
        self._loader.started.connect(self._sample_started)
        self._loader.loaded.connect(self._sample_loaded)

        self._restore_geometry()
        self._update_title()
        _WINDOWS.append(self)

    # Each sample owns its gate tree and masks; these properties expose the
    # current sample's, so the rest of the window doesn't need to care.

    @property
    def _root_gates(self) -> list[Gate]:
        return self._current_sample.gates if self._current_sample else []

    @_root_gates.setter
    def _root_gates(self, roots: list[Gate]):
        if self._current_sample is not None:
            self._current_sample.gates = roots

    @property
    def _gate_masks(self) -> dict[str, np.ndarray]:
        return (self._current_sample.gate_masks if self._current_sample
                else {})

    @_gate_masks.setter
    def _gate_masks(self, masks: dict[str, np.ndarray]):
        if self._current_sample is not None:
            self._current_sample.gate_masks = masks

    # ------------------------------------------------------------------
    # Menu bar
    # ------------------------------------------------------------------

    def _build_menu(self):
        mb = self.menuBar()

        # File
        file_m = mb.addMenu("File")
        self._add_action(file_m, "New Window", lambda: open_window(near=self),
                         QKeySequence.StandardKey.New)
        self._add_action(file_m, "Open Workspace…", self._load_workspace,
                         QKeySequence("Ctrl+Shift+O"))
        self._add_action(file_m, "Clear This Window (new workspace here)",
                         self._new_workspace)
        file_m.addSeparator()
        self._add_action(file_m, "Open FCS…", self._open_fcs,
                         QKeySequence.StandardKey.Open)
        self._add_action(file_m, "Open FCS Folder…", self._open_folder)
        file_m.addSeparator()
        self._add_action(file_m, "Save", self._save_workspace,
                         QKeySequence.StandardKey.Save)
        self._add_action(file_m, "Save As…", self._save_workspace_as,
                         QKeySequence("Ctrl+Shift+S"))
        file_m.addSeparator()
        self._add_action(file_m, "Recover Last Unsaved Session",
                         self._restore_last_session)
        self._add_action(file_m, "Show Recovery Folder",
                         self._show_sessions_folder)
        file_m.addSeparator()
        file_m.addMenu(self._build_export_menu())
        file_m.addSeparator()
        self._add_action(file_m, "Quit", self.close,
                         QKeySequence.StandardKey.Quit)

        # Edit
        edit_m = mb.addMenu("Edit")
        self._undo_act = self._add_action(edit_m, "Undo", self._undo,
                                          QKeySequence.StandardKey.Undo)
        self._redo_act = self._add_action(edit_m, "Redo", self._redo,
                                          QKeySequence("Ctrl+Shift+Z"))
        edit_m.addSeparator()
        self._add_action(edit_m, "Copy Gates", lambda: self._copy_gates(None),
                         QKeySequence("Ctrl+Alt+C"))
        self._add_action(edit_m, "Paste Gates…", lambda: self._paste_gates(None),
                         QKeySequence("Ctrl+Alt+V"))
        edit_m.addSeparator()
        self._add_action(edit_m, "Compensation…", self._open_compensation)
        self._add_action(edit_m, "Remove Compensation",
                         self._remove_compensation)

        # Gating protocols / batch
        gate_m = mb.addMenu("Gating")
        self._add_action(gate_m, "Edit Selected Gate…",
                         self._plot_edit_selected)
        self._add_action(gate_m, "New Boolean Gate (AND / OR / NOT)…",
                         self._new_boolean_gate)
        gate_m.addSeparator()
        self._add_action(gate_m, "Apply Gates to Other Samples…",
                         self._apply_gates_to_samples)
        gate_m.addSeparator()
        self._add_action(gate_m, "Save Gating Protocol…",
                         self._save_protocol)
        self._add_action(gate_m, "Load Gating Protocol…",
                         self._load_protocol)
        gate_m.addSeparator()
        self._add_action(gate_m, "Import FlowJo Workspace (.wsp)…",
                         self._import_wsp)
        self._add_action(gate_m, "Import Gating-ML 2.0…",
                         self._import_gatingml)
        self._add_action(gate_m, "Export to FlowJo (.wsp)…",
                         self._export_wsp)
        self._add_action(gate_m, "Export Gating-ML 2.0…",
                         self._export_gatingml)
        self._add_action(gate_m, "Verify Counts with FlowKit…",
                         self._verify_flowkit)
        gate_m.addSeparator()
        self._add_action(gate_m, "Batch Statistics…",
                         self._show_batch_stats)
        self._add_action(gate_m, "Export Data…", self._show_export_data)

        # Analysis
        an_m = mb.addMenu("Analysis")
        self._add_action(an_m, "Sample Metadata (Groups)…",
                         self._edit_metadata, QKeySequence("Ctrl+Shift+M"))
        self._add_action(an_m, "Plate Layout…", self._show_plate,
                         QKeySequence("Ctrl+Shift+P"))
        self._add_action(an_m, "Compare Groups",
                         lambda: self._tabs.setCurrentWidget(self._groups))
        an_m.addSeparator()
        self._add_action(an_m, "Derived Parameters (Ratios, Formulas)…",
                         self._derived_parameters)
        an_m.addSeparator()
        self._add_action(an_m, "UMAP / t-SNE…", self._run_dimred)
        self._add_action(an_m, "Clustering…", self._run_clustering)
        self._add_action(an_m, "Auto Gate on Current Plot…",
                         self._run_auto_gate)
        an_m.addSeparator()
        self._add_action(an_m, "Spectral Unmixing…", self._run_unmixing)

        # View
        view_m = mb.addMenu("View")
        self._add_action(view_m, "Zoom In", lambda: self._plot.zoom(1.5))
        self._add_action(view_m, "Zoom Out", lambda: self._plot.zoom(1 / 1.5))
        self._add_action(view_m, "Fit Plot (Reset Zoom)", self._reset_zoom)

        # Help
        help_m = mb.addMenu("Help")
        self._add_action(help_m, "About FlowPy", self._show_about)

    def _add_action(self, menu: QMenu, label: str, slot,
                    shortcut=None) -> QAction:
        act = QAction(label, self)
        if shortcut:
            act.setShortcut(shortcut)
        act.triggered.connect(slot)
        menu.addAction(act)
        return act

    # ------------------------------------------------------------------
    # Toolbar
    # ------------------------------------------------------------------

    def _build_export_menu(self) -> QMenu:
        """One Export menu, shared by the File menu and the toolbar."""
        m = QMenu("Export", self)
        self._add_action(m, "Data — Statistics for All Samples…",
                         self._show_export_data, QKeySequence("Ctrl+E"))
        self._add_action(m, "Figure — Current Tab (PNG, PDF, SVG)…",
                         self._export_figure, QKeySequence("Ctrl+Shift+E"))
        res = m.addMenu("Figure Resolution (PNG)")
        from PyQt6.QtGui import QActionGroup
        grp = QActionGroup(res)
        for d in theme.DPI_CHOICES:
            a = res.addAction(f"{d} dpi" + {300: "", 600: "  (journals)",
                                            1200: "  (line art)"}[d])
            a.setCheckable(True)
            a.setChecked(d == theme.export_dpi())
            a.triggered.connect(lambda _c, d=d: theme.set_export_dpi(d))
            grp.addAction(a)
        res.addSeparator()
        res.addAction("PDF and SVG are vector: sharp at any size").setEnabled(False)
        m.addSeparator()
        self._add_action(m, "This Sample's Statistics Table (CSV)…",
                         self._export_stats)
        self._add_action(m, "FlowJo Workspace (.wsp)…", self._export_wsp)
        self._add_action(m, "Gating-ML 2.0…", self._export_gatingml)
        self._export_menu = m
        return m

    def _export_figure(self):
        """Export what the current tab shows."""
        w = self._tabs.currentWidget()
        if w is self._layout:
            self._layout._export()
        elif w is self._groups:
            self._groups._export_figure()
        else:
            self._plot.export_figure_dialog()

    def _build_toolbar(self):
        tb = QToolBar("Main")
        tb.setMovable(False)
        self.addToolBar(tb)

        brand = QLabel("◆ FlowPy")
        brand.setStyleSheet(f"color: {theme.ACCENT}; font-size: 15px; "
                            "font-weight: 700; padding-right: 14px;")
        tb.addWidget(brand)

        open_act = QAction("＋ Open FCS", self)
        open_act.triggered.connect(self._open_fcs)
        tb.addAction(open_act)
        tb.widgetForAction(open_act).setObjectName("Primary")

        for label, slot, tip in (
                ("Save", self._save_workspace, "Save the workspace (⌘S)"),
                ("Apply Gates to Samples", self._apply_gates_to_samples,
                 "Copy this sample's gates to other samples"),
                ("Metadata", self._edit_metadata,
                 "Groups, conditions… of your samples (⇧⌘M)")):
            act = QAction(label, self)
            act.setToolTip(tip)
            act.triggered.connect(slot)
            tb.addAction(act)
        export_btn = QToolButton()
        export_btn.setText("Export  ▾")
        export_btn.setToolTip("Export data tables, figures, FlowJo / "
                              "Gating-ML files")
        export_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        export_btn.setStyleSheet("QToolButton::menu-indicator { image: none; "
                                 "width: 0px; }")
        export_btn.setMenu(self._export_menu)
        tb.addWidget(export_btn)

    # ------------------------------------------------------------------
    # Central widget layout
    # ------------------------------------------------------------------

    def _build_central(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(10, 6, 10, 6)
        root_layout.setSpacing(0)

        # Main horizontal splitter: left panel | center plot | (future right)
        h_split = QSplitter(Qt.Orientation.Horizontal)

        # --- Left panel: samples above populations, resizable between ---
        left = QSplitter(Qt.Orientation.Vertical)
        left.setChildrenCollapsible(False)
        left.setMaximumWidth(720)
        left.setMinimumWidth(260)
        top = QWidget()
        left_layout = QVBoxLayout(top)
        left_layout.setContentsMargins(0, 0, 0, 4)
        left_layout.setSpacing(4)

        # Sample list
        sample_hdr = QLabel("SAMPLES")
        sample_hdr.setObjectName("SectionHeader")
        left_layout.addWidget(sample_hdr)

        self._sample_list = SampleList()
        self._sample_list.sample_selected.connect(self._select_sample)
        left_layout.addWidget(self._sample_list, 1)

        btn_row = QHBoxLayout()
        open_btn = QPushButton("＋ Open FCS")
        open_btn.clicked.connect(self._open_fcs)
        btn_row.addWidget(open_btn)
        remove_btn = QPushButton("Remove")
        remove_btn.clicked.connect(self._remove_sample)
        btn_row.addWidget(remove_btn)
        left_layout.addLayout(btn_row)
        left.addWidget(top)

        # Gate tree
        self._gate_tree = GateTreeWidget()
        self._gate_tree.gate_selected.connect(self._on_gate_tree_selected)
        self._gate_tree.gate_delete_requested.connect(self._delete_gate)
        self._gate_tree.gate_rename_requested.connect(self._rename_gate)
        self._gate_tree.gate_edit_requested.connect(self._edit_gate_dialog)
        self._gate_tree.refit_requested.connect(self._refit_gate)
        self._gate_tree.boolean_requested.connect(self._new_boolean_gate)
        self._gate_tree.copy_requested.connect(
            lambda gid: self._copy_gates(gid, whole=not gid))
        self._gate_tree.paste_requested.connect(
            lambda gid: self._paste_gates(gid or ""))
        left.addWidget(self._gate_tree)
        left.setStretchFactor(0, 1)
        left.setStretchFactor(1, 1)
        left.setSizes([380, 420])

        h_split.addWidget(left)

        # --- Center: plot + stats stacked ---
        center = QWidget()
        v_split = QSplitter(Qt.Orientation.Vertical)

        self._plot = PlotWidget()
        self._plot.gates_created.connect(self._on_gates_created)
        self._plot.gates_modified.connect(self._modify_gates)
        self._plot.gate_delete_requested.connect(self._delete_gate)
        self._plot.boolean_requested.connect(self._new_boolean_gate)
        self._plot.population_requested.connect(
            lambda gid: self._gate_tree.select_gate(gid))
        self._plot.add_to_layout.connect(self._on_add_to_layout)
        self._plot.axes_changed.connect(
            lambda ch: self._stats.set_plot_channels(ch))
        v_split.addWidget(self._plot)

        self._stats = StatisticsWidget()
        self._stats.setMaximumHeight(260)
        v_split.addWidget(self._stats)

        self._stats.setMinimumHeight(140)
        v_split.setStretchFactor(0, 3)
        v_split.setStretchFactor(1, 1)
        v_split.setSizes([620, 200])

        self._layout = LayoutEditor()
        self._layout.changed.connect(self._schedule_autosave)

        self._tabs = QTabWidget()
        self._tabs.addTab(v_split, "Gating")
        self._tabs.addTab(self._layout, "Layout")
        self._groups = GroupPlotWidget()
        self._groups.changed.connect(self._schedule_autosave)
        self._groups.edit_metadata.connect(self._edit_metadata)
        self._tabs.addTab(self._groups, "Compare Groups")

        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.addWidget(self._tabs)
        h_split.addWidget(center)

        h_split.setStretchFactor(0, 0)
        h_split.setStretchFactor(1, 1)
        h_split.setSizes([360, 1040])

        root_layout.addWidget(h_split)

    # ------------------------------------------------------------------
    # Status bar
    # ------------------------------------------------------------------

    def _build_statusbar(self):
        self._statusbar = QStatusBar()
        self.setStatusBar(self._statusbar)
        self._status_sample = QLabel("No sample loaded")
        self._status_gate   = QLabel("")
        self._status_loading = QLabel("")
        self._status_loading.setObjectName("Muted")
        self._statusbar.addWidget(self._status_sample)
        self._statusbar.addPermanentWidget(self._status_loading)
        self._statusbar.addPermanentWidget(self._status_gate)

    # ------------------------------------------------------------------
    # Sample management
    # ------------------------------------------------------------------

    def _add_or_new(self, what: str) -> str | None:
        """Opening FCS files while a workspace is open: add them to it, or
        open them in a new window (the open one stays as it is)? Returns
        "add", "new" or None (cancelled)."""
        if not self._samples:
            return "add"
        name = self._doc_path.stem if self._doc_path else "this workspace"
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle("Open FCS")
        box.setText(f"Add {what} to {name}, or open them in a new window?")
        box.setInformativeText("A new window starts a separate workspace; "
                               "this one stays open as it is.")
        new_btn = box.addButton("New Window",
                                QMessageBox.ButtonRole.AcceptRole)
        add_btn = box.addButton(f"Add to {name}",
                                QMessageBox.ButtonRole.ActionRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(new_btn)
        box.exec()
        if box.clickedButton() is add_btn:
            return "add"
        if box.clickedButton() is new_btn:
            return "new"
        return None

    def _open_paths(self, paths: list, what: str):
        choice = self._add_or_new(what) if paths else None
        if choice == "add":
            self.load_fcs_files(paths)
        elif choice == "new":
            open_window(fcs=paths, near=self)

    def _open_fcs(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Open FCS Files", "",
            "FCS Files (*.fcs);;All Files (*)")
        self._open_paths(paths, f"{len(paths)} file"
                         f"{'s' if len(paths) != 1 else ''}")

    def _open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Open FCS Folder")
        if folder:
            self._open_paths(sorted(str(p) for p in Path(folder).glob("*")
                                    if p.suffix.lower() == ".fcs"),
                             f"the files of {Path(folder).name}")

    def _read_samples(self, paths: list, title: str = "Opening FCS "
                      "files…", loader=Sample, show_speed: bool = True
                      ) -> tuple[list, list[str]]:
        """Read FCS files, several at a time, with a progress bar (and
        Cancel). No screen updates per file. *loader(item)* builds one
        sample (default: Sample(path)). Returns (sample or None per item,
        error messages)."""
        from concurrent.futures import ThreadPoolExecutor, wait, \
            FIRST_COMPLETED
        from PyQt6.QtWidgets import QProgressDialog
        n = len(paths)
        out: list = [None] * n
        errors: list[str] = []
        prog = QProgressDialog(title, "Cancel", 0, n, self)
        prog.setWindowModality(Qt.WindowModality.WindowModal)
        prog.setMinimumDuration(400)
        prog.setAutoClose(True)
        done = 0
        import os
        import time

        def size_of(item):
            try:
                return os.path.getsize(item["filepath"] if isinstance(
                    item, dict) else item)
            except (OSError, TypeError, KeyError):
                return 0
        sizes = [size_of(p) for p in paths] if show_speed else [0] * n
        total, read_bytes, t0 = sum(sizes), 0, time.perf_counter()
        with ThreadPoolExecutor(max_workers=min(4, n)) as ex:
            futs = {ex.submit(loader, p): i for i, p in enumerate(paths)}
            pending = set(futs)
            while pending:
                finished, pending = wait(pending, timeout=0.05,
                                         return_when=FIRST_COMPLETED)
                for f in finished:
                    i = futs[f]
                    try:
                        out[i] = f.result()
                    except Exception as exc:
                        item = paths[i]
                        name = Path(item["filepath"] if isinstance(item, dict)
                                    else item).name
                        errors.append(f"{name}: {exc}")
                    done += 1
                    read_bytes += sizes[i]
                prog.setValue(done)
                prog.setLabelText(f"{title}  {done} / {n}" + _speed_note(
                    read_bytes, total, time.perf_counter() - t0))
                QApplication.processEvents()
                if prog.wasCanceled():
                    for f in pending:
                        f.cancel()
                    errors.append(f"Cancelled — {n - done} file(s) not "
                                  "opened.")
                    break
        prog.close()
        return out, errors

    def load_fcs_files(self, paths: list[str]) -> list:
        """Open many FCS files at once: read them all (in parallel), then
        update the window once."""
        paths = [str(p) for p in paths]
        if not paths:
            return []
        import time
        t0 = time.perf_counter()
        # headers only (instant); events are read in the background
        samples, errors = self._read_samples(
            paths, "Opening FCS files…",
            loader=lambda p: Sample(p, lazy=True), show_speed=False)
        loaded = [s for s in samples if s is not None]
        self._samples.extend(loaded)
        self._update_sample_labels()
        if loaded:
            self._loader.enqueue(loaded, first=loaded[0])
            self._sample_list.select_sample(loaded[0])
            self._schedule_autosave()
        self._statusbar.showMessage(
            f"Opened {len(loaded)} file(s) in "
            f"{time.perf_counter() - t0:.1f} s"
            + (" — reading their events in the background"
               if self._loader.pending() else ""), 8000)
        if errors:
            QMessageBox.warning(self, "Some files were not opened",
                                "\n".join(errors[:30]) +
                                (f"\n… and {len(errors) - 30} more"
                                 if len(errors) > 30 else ""))
        return loaded

    def load_fcs_file(self, filepath: str):
        """Load one FCS file (public API for CLI usage)."""
        try:
            sample = Sample(filepath)
            self._samples.append(sample)
            self._update_sample_labels()
            self._sample_list.select_sample(sample)
            self._status_sample.setText(
                f"Loaded: {sample.name} | "
                f"{sample.n_events:,} events | "
                f"{len(sample.channels)} channels"
            )
            self._schedule_autosave()
            return sample
        except Exception as exc:
            QMessageBox.critical(self, "Error Loading FCS",
                                 f"Could not load:\n{filepath}\n\n{exc}")
            return None

    def _remove_sample(self):
        s = self._sample_list.current_sample()
        if s is None or s not in self._samples:
            return
        self._samples.remove(s)
        self._loader.discard([s])
        self._sample_list.select_sample(None, emit=False)
        self._update_sample_labels()
        self._current_sample = None
        self._selected_gate_id = None
        self._plot.set_sample(None)
        self._stats.set_sample(None)
        self._refresh_after_gate_change()
        self._schedule_autosave()

    def _select_sample(self, sample):
        if sample is None or sample not in self._samples:
            return
        if not sample.is_loaded and not sample.load_error \
                and not self._loader.is_busy(sample) \
                and _file_size(sample.filepath) < SYNC_LOAD_BYTES:
            try:                       # small file: just read it now
                sample.load()
            except IOError:
                pass
        if not sample.is_loaded:
            # show it as soon as its events are read (they go first)
            self._waiting_for = sample
            self._current_sample = None
            self._plot.show_waiting(self._waiting_text(sample))
            self._stats.set_sample(None)
            self._refresh_after_gate_change()
            self._status_sample.setText(
                f"{sample.name} | {sample.n_events:,} events | "
                f"{len(sample.channels)} ch | reading…")
            if sample.load_error:
                # clicking it again tries again (network drives fail
                # for a moment sometimes)
                if getattr(self, "_failed_shown", None) is sample:
                    sample.retry()
                    self._sample_list.refresh_states()
                    self._plot.show_waiting(self._waiting_text(sample))
                else:
                    self._failed_shown = sample
            if not sample.load_error:
                self._loader.prioritize([sample])
            return
        self._waiting_for = None
        self._current_sample = sample
        # Stay on the same population if this sample has it (it does when
        # both got their gates from the same protocol)
        if self._gate_by_id(self._selected_gate_id) is None:
            self._selected_gate_id = None
        self._status_gate.setText("")

        self._plot.set_sample(sample)
        self._stats.set_sample(sample)
        self._refresh_after_gate_change()

        comp_info = ("Compensated" if sample.is_compensated
                     else "No compensation")
        self._status_sample.setText(
            f"{sample.name} | {sample.n_events:,} events | "
            f"{len(sample.channels)} ch | {comp_info}")

    def _waiting_text(self, sample) -> str:
        if sample.load_error:
            return (f"{sample.name} can't be read:\n{sample.load_error}"
                    "\n\nClick it again to try again.")
        size = ""
        try:
            size = f" ({Path(sample.filepath).stat().st_size / 2**20:,.0f} MB)"
        except OSError:
            pass
        ahead = self._loader.pending() - 1
        return (f"Reading {sample.name}{size}…\n\n"
                "It goes first; the other samples are read in the "
                "background" + (f" ({ahead} to go)." if ahead > 0 else "."))

    def _sample_started(self, sample):
        self._sample_list.refresh_states()
        self._status_loading.setText(
            f"Reading samples… {self._loader.pending()} to go")

    def _sample_loaded(self, sample):
        """A sample's events were read in the background."""
        self._sample_list.refresh_states()
        left = self._loader.pending()
        if left:
            self._status_loading.setText(f"Reading samples… {left} to go")
        else:
            self._status_loading.setText("")
        if sample is self._waiting_for and sample in self._samples \
                and self._sample_list.current_sample() is sample:
            if sample.load_error:
                self._failed_shown = sample
                self._plot.show_waiting(self._waiting_text(sample))
                self._status_sample.setText(f"{sample.name} | can't be read")
            else:
                self._select_sample(sample)
        self._layout.sample_loaded(sample)
        self._groups.sample_loaded(sample)

    def _ensure_loaded(self, samples, title: str) -> bool:
        """Wait (with progress and Cancel) until *samples* have their
        events. Samples that can't be read are reported and left out."""
        if not self._loader.wait_for(samples, self, title):
            return False
        bad = [s for s in samples if s.load_error]
        if bad:
            QMessageBox.warning(
                self, "Some samples can't be read",
                "These are left out:\n\n" + "\n".join(
                    f"{s.name}: {s.load_error}" for s in bad[:20]))
        return True

    # ------------------------------------------------------------------
    # Gating
    # ------------------------------------------------------------------

    def _on_gates_created(self, gates: list[Gate]):
        """Receive newly drawn gate(s) from PlotWidget. They become children
        of the population currently shown."""
        if self._current_sample is None or not gates:
            return
        self._checkpoint()
        parent_gate = self._gate_by_id(self._selected_gate_id)
        parent_mask = (self._gate_masks.get(parent_gate.gate_id)
                       if parent_gate else None)
        for gate in gates:
            if parent_gate is not None:
                parent_gate.add_child(gate)
            else:
                self._root_gates.append(gate)
            self._gate_masks[gate.gate_id] = protocol.gate_mask(
                self._current_sample, gate, self._gate_masks)

        self._refresh_after_gate_change()
        self._schedule_autosave()

        n_par = (int(parent_mask.sum()) if parent_mask is not None
                 else self._current_sample.n_events)
        parts = [f"{g.name}: {int(self._gate_masks[g.gate_id].sum()):,} "
                 f"({100.0 * self._gate_masks[g.gate_id].sum() / max(n_par, 1):.1f}%)"
                 for g in gates]
        self._status_gate.setText(" · ".join(parts))

    def _on_gate_tree_selected(self, gate_id: str):
        """User picked a population in the tree → show its events and
        its child gates (new gates are drawn inside it)."""
        gate = self._gate_by_id(gate_id)
        self._selected_gate_id = gate.gate_id if gate else None
        # Show the axes of this population's own gates, if it has any
        if gate is not None and gate.children:
            first = gate.children[0]
            if first.y_channel is None:
                self._plot.set_plot_type("Histogram")
            elif self._plot.is_histogram():
                self._plot.set_plot_type("Pseudocolor")
            self._plot.set_axes(first.x_channel, first.y_channel,
                                first.x_transform, first.y_transform)
        self._refresh_plot()
        self._tabs.setCurrentIndex(0)

        if self._current_sample is not None:
            n_total = self._current_sample.n_events
            mask = self._gate_masks.get(gate.gate_id) if gate else None
            n = int(mask.sum()) if mask is not None else n_total
            self._status_gate.setText(
                f"{gate.name if gate else 'All Events'}: {n:,} / "
                f"{n_total:,} events ({100. * n / max(n_total, 1):.1f}%)")

    # ------------------------------------------------------------------
    # Changing gates (all through here, so they can be undone)
    # ------------------------------------------------------------------

    def _gates_changed(self, message: str = ""):
        """Recompute masks and refresh after any change to the gate tree."""
        live = self._all_gate_ids()
        if self._selected_gate_id not in live:
            self._selected_gate_id = None
        self._recompute_all_masks()
        self._schedule_autosave()
        if message:
            self._statusbar.showMessage(message, 4000)

    def _delete_gate(self, gate_id: str):
        gate = self._gate_by_id(gate_id)
        if gate is None:
            return
        self._checkpoint()
        if gate.parent is not None:
            gate.parent.remove_child(gate)
        else:
            self._root_gates.remove(gate)
        self._plot.select_gate(None)
        self._gates_changed(f"Deleted '{gate.name}' — Edit → Undo "
                            "(Ctrl/⌘+Z) brings it back")

    def _rename_gate(self, gate_id: str, new_name: str):
        gate = self._gate_by_id(gate_id)
        if gate is None or gate.name == new_name:
            return
        self._checkpoint()
        gate.name = new_name
        self._gates_changed()

    def _modify_gates(self, changes: list):
        """Apply [(gate, {attribute: value})] from dragging or a dialog."""
        if not changes:
            return
        self._checkpoint()
        for gate, attrs in changes:
            for k, v in attrs.items():
                setattr(gate, k, v)
        self._gates_changed()

    def _refit_gate(self, gate_id: str):
        """Re-describe a gate on the scales now shown, same events.

        A gate is evaluated on the scales it was drawn on, so one drawn on
        a linear plot cannot be edited on a log one. This moves it across
        rather than making the user switch back, and says first how many
        events the approximation moves — usually a handful out of tens of
        thousands, but the user decides.
        """
        gate = self._gate_by_id(gate_id)
        if gate is None:
            return
        x_tr, y_tr = self._plot.x_transform(), self._plot.y_transform()
        if isinstance(gate, BooleanGate):
            QMessageBox.information(
                self, "Re-fit Gate",
                "A boolean gate has no shape of its own — re-fit the gates "
                "it combines instead.")
            return
        if not can_refit(gate, x_tr, y_tr):
            QMessageBox.information(
                self, "Re-fit Gate",
                f"'{gate.name}' is already on {x_tr}"
                + (f" / {y_tr}" if gate.y_channel else "") + ".")
            return

        new = refit(gate, x_tr, y_tr)
        moved, total = 0, 0
        s = self._current_sample
        if s is not None and gate.x_channel in s.channels:
            try:
                xs = s.column(gate.x_channel)
                ys = (s.column(gate.y_channel)
                      if gate.y_channel in s.channels else None)
                moved = refit_error(gate, new, xs, ys)
                total = len(xs)
            except Exception:
                moved = total = 0
        changed = (f"{moved:,} of {total:,} events change "
                   f"({100 * moved / total:.2f}%)" if total else
                   "the number of events moved could not be checked here")
        became = ("  It becomes a polygon, because an ellipse on one scale "
                  "is not an ellipse on another." if
                  gate.gate_type != new.gate_type else "")
        if QMessageBox.question(
                self, "Re-fit Gate",
                f"Describe '{gate.name}' on {x_tr}"
                + (f" / {y_tr}" if gate.y_channel else "")
                + f", the scales now shown?\n\nIt was drawn on "
                f"{gate.x_transform}"
                + (f" / {gate.y_transform}" if gate.y_channel else "")
                + f". The population is kept: {changed}.{became}\n\n"
                "Sub-gates keep their own scales and are not touched.",
                QMessageBox.StandardButton.Ok
                | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Ok
        ) != QMessageBox.StandardButton.Ok:
            return

        self._checkpoint()
        self._replace_gate(gate, new)
        self._gates_changed()
        self._statusbar.showMessage(
            f"'{new.name}' re-fitted to {x_tr}"
            + (f" / {y_tr}" if new.y_channel else "")
            + (f" — {moved:,} event(s) changed" if moved else
               " — the same events"), 8000)

    def _replace_gate(self, old: Gate, new: Gate):
        """Swap *old* for *new* in the tree, keeping its place and family."""
        new.parent = old.parent
        new.children = old.children
        for c in new.children:
            c.parent = new
        siblings = old.parent.children if old.parent else self._root_gates
        try:
            siblings[siblings.index(old)] = new
        except ValueError:
            siblings.append(new)
        old.children = []

    def _edit_gate_dialog(self, gate_id: str):
        gate = self._gate_by_id(gate_id)
        if gate is None:
            return
        siblings = (gate.parent.children if gate.parent
                    else self._root_gates)
        dlg = EditGateDialog(gate, siblings, _bfs(self._root_gates), self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._modify_gates(dlg.changes())

    def _plot_edit_selected(self):
        gid = self._plot._edit_gate_id
        if gid is None:
            QMessageBox.information(self, "Edit Gate",
                                    "Click a gate on the plot first.")
            return
        self._edit_gate_dialog(gid)

    def _new_boolean_gate(self):
        if self._current_sample is None:
            return
        gates = _bfs(self._root_gates)
        if not gates:
            QMessageBox.information(self, "Boolean Gate",
                                    "Draw some gates first.")
            return
        pop = self._gate_by_id(self._selected_gate_id)
        dlg = BooleanGateDialog(gates, pop.name if pop else "All Events",
                                self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            dlg.gate._color = next_gate_color()
            self._on_gates_created([dlg.gate])

    # ------------------------------------------------------------------
    # Undo / redo (per sample: snapshots of the gate tree)
    # ------------------------------------------------------------------

    def _snapshot(self, sample) -> tuple:
        return (protocol.tree_to_dict(sample.gates), self._selected_gate_id)

    def _checkpoint(self, sample=None):
        """Remember the gate tree before a change."""
        sample = sample or self._current_sample
        if sample is None:
            return
        undo = sample.__dict__.setdefault("undo_stack", [])
        undo.append(self._snapshot(sample))
        del undo[:-UNDO_LIMIT]
        sample.__dict__["redo_stack"] = []
        self._mark_dirty()

    def _restore(self, snap: tuple):
        tree, selected = snap
        s = self._current_sample
        s.gates = protocol.tree_from_dict(tree)
        self._selected_gate_id = selected
        self._gates_changed()

    def _undo(self):
        s = self._current_sample
        undo = getattr(s, "undo_stack", None) if s else None
        if not undo:
            self._statusbar.showMessage("Nothing to undo", 2000)
            return
        s.redo_stack.append(self._snapshot(s))
        self._restore(undo.pop())
        self._statusbar.showMessage("Undone", 2000)

    def _redo(self):
        s = self._current_sample
        redo = getattr(s, "redo_stack", None) if s else None
        if not redo:
            self._statusbar.showMessage("Nothing to redo", 2000)
            return
        s.undo_stack.append(self._snapshot(s))
        self._restore(redo.pop())
        self._statusbar.showMessage("Redone", 2000)

    def _refresh_plot(self):
        self._plot.set_context(self._root_gates, self._gate_masks,
                               self._gate_by_id(self._selected_gate_id))

    def _refresh_after_gate_change(self):
        n_total = (self._current_sample.n_events
                   if self._current_sample else 0)
        self._gate_tree.set_gates(self._root_gates, self._gate_masks, n_total)
        self._gate_tree.select_gate(self._selected_gate_id, emit=False)
        self._refresh_plot()
        self._stats.set_gates(self._root_gates, self._gate_masks)
        self._layout.set_context(self._samples, self._current_sample)
        self._groups.set_context(self._samples, self._meta_columns)
        self._update_sample_labels()

    def _update_sample_labels(self):
        # grouped by the first metadata column that is filled in
        col = next((c for c in self._meta_columns
                    if any(s.annotations.get(c) for s in self._samples)),
                   None)
        self._sample_list.set_samples(self._samples, col)

    def _on_add_to_layout(self, spec: dict):
        self._layout.add_item(spec)
        self._statusbar.showMessage(
            f"Added to layout ({self._layout.count()} plots) — "
            "see the Layout tab", 4000)

    def _gate_by_id(self, gate_id: str | None) -> Gate | None:
        if gate_id is None:
            return None
        for g in _bfs(self._root_gates):
            if g.gate_id == gate_id:
                return g
        return None

    def _all_gate_ids(self) -> set[str]:
        return {g.gate_id for g in _bfs(self._root_gates)}

    # ------------------------------------------------------------------
    # Compensation
    # ------------------------------------------------------------------

    def _open_compensation(self):
        if self._current_sample is None:
            QMessageBox.information(self, "No Sample",
                                    "Please load an FCS file first.")
            return
        dlg = CompensationDialog(
            self._current_sample.comp_matrix,
            self._current_sample.channels,
            self
        )
        if dlg.exec() == QDialog.DialogCode.Accepted:
            matrix = dlg.get_matrix()
            self._current_sample.apply_compensation(matrix)
            self._recompute_all_masks()
            self._schedule_autosave()
            self._status_sample.setText(
                f"{self._current_sample.name} | "
                f"{self._current_sample.n_events:,} events | Compensated")

    def _remove_compensation(self):
        if self._current_sample is None:
            return
        self._current_sample.remove_compensation()
        self._recompute_all_masks()

    def _recompute_all_masks(self):
        """Recompute all gate masks after compensation change."""
        if self._current_sample is None:
            return
        self._gate_masks = protocol.compute_masks(self._current_sample,
                                                  self._root_gates)
        self._stats.invalidate()
        self._refresh_after_gate_change()

    # ------------------------------------------------------------------
    # Workspace save / load
    # ------------------------------------------------------------------

    def _workspace_dict(self) -> dict:
        return {
            "version": 2,
            "saved": datetime.now().isoformat(timespec="seconds"),
            # each sample carries its own gate tree
            "samples": [{**s.to_dict(), **protocol.tree_to_dict(s.gates),
                         "derived": s.derived_to_dict(),
                         "annotations": s.annotations}
                        for s in self._samples],
            "metadata_columns": self._meta_columns,
            "group_plot": self._groups.to_dict(),
            "current_sample": (self._samples.index(self._current_sample)
                               if self._current_sample in self._samples
                               else -1),
            "selected_gate": self._selected_gate_id,
            "layout": self._layout.to_dict(),
        }

    def _write_workspace(self, path: Path):
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "w") as f:
            json.dump(self._workspace_dict(), f)
        tmp.replace(path)

    def _save_workspace(self) -> bool:
        """Save to the current workspace file, or ask for one."""
        if self._doc_path is None:
            return self._save_workspace_as()
        try:
            self._write_workspace(self._doc_path)
        except Exception as exc:
            QMessageBox.critical(self, "Save Error", str(exc))
            return False
        self._drop_recovery_copy()
        self._set_clean()
        self._statusbar.showMessage(f"Saved {self._doc_path.name}", 3000)
        return True

    def _drop_recovery_copy(self):
        self._autosave_timer.stop()
        if self._session_path is not None:
            self._session_path.unlink(missing_ok=True)
            self._session_path = None

    def _save_workspace_as(self) -> bool:
        start = str(self._doc_path or (Path.home() / "Documents" /
                                       "FlowPy analysis.fpw"))
        path, _ = QFileDialog.getSaveFileName(self, "Save Workspace As",
                                              start, WORKSPACE_FILTER)
        if not path:
            return False
        path = Path(path if path.lower().endswith(".fpw") else path + ".fpw")
        try:
            self._write_workspace(path)
        except Exception as exc:
            QMessageBox.critical(self, "Save Error", str(exc))
            return False
        # Save (⌘S) now writes to this file; the recovery copy is no
        # longer needed
        self._drop_recovery_copy()
        self._doc_path = path
        self._set_clean()
        self._statusbar.showMessage(f"Saved {path.name}", 5000)
        return True

    def _load_workspace(self):
        """Opens a workspace here if this window is empty, otherwise in a
        new window — the one open here stays as it is."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Workspace", str(self._doc_path.parent)
            if self._doc_path else "", WORKSPACE_FILTER)
        if not path:
            return
        if self._samples or self._doc_path is not None:
            open_window(workspace=path, near=self)
        else:
            self.load_workspace_file(path)

    def _new_workspace(self):
        if not self._confirm_discard():
            return
        self._autosave_timer.stop()
        self._loader.clear()
        self._waiting_for = None
        self._samples.clear()
        self._sample_list.set_samples([])
        self._current_sample = None
        self._selected_gate_id = None
        self._plot.set_sample(None)
        self._stats.set_sample(None)
        self._layout.load_dict({})
        self._meta_columns = list(groups_core.DEFAULT_COLUMNS)
        self._groups.load_dict({})
        self._refresh_after_gate_change()
        self._doc_path = None
        self._session_path = None
        self._set_clean()

    # ------------------------------------------------------------------
    # Document state (title, unsaved changes)
    # ------------------------------------------------------------------

    def _update_title(self):
        name = self._doc_path.name if self._doc_path else "Untitled"
        self.setWindowTitle(f"{name}[*] — FlowPy")
        self.setWindowModified(self._dirty)

    def _mark_dirty(self):
        if not self._loading and not self._dirty:
            self._dirty = True
            self._update_title()

    def _set_clean(self):
        self._dirty = False
        self._update_title()

    def _confirm_discard(self) -> bool:
        """Before replacing the analysis: offer to save unsaved changes."""
        if not self._samples or not self._dirty:
            return True
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle("Save Workspace?")
        if self._doc_path is not None:
            box.setText(f"Save the changes to {self._doc_path.name}?")
            box.setInformativeText(
                "If you don't, the file stays as it was last saved; the "
                "changes are kept in a recovery copy (File → Recover Last "
                "Unsaved Session).")
        else:
            box.setText("Save this analysis as a workspace file?")
            box.setInformativeText("If you don't, a recovery copy stays in "
                                   "the recovery folder (File → Recover "
                                   "Last Unsaved Session).")
        box.setStandardButtons(QMessageBox.StandardButton.Save |
                               QMessageBox.StandardButton.Discard |
                               QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(QMessageBox.StandardButton.Save)
        answer = box.exec()
        if answer == QMessageBox.StandardButton.Save:
            return self._save_workspace()
        if answer == QMessageBox.StandardButton.Discard:
            self._flush_autosave()
            return True
        return False

    def load_workspace_file(self, path: str):
        try:
            with open(path) as f:
                ws = json.load(f)
        except Exception as exc:
            QMessageBox.critical(self, "Load Error", str(exc))
            return
        path = Path(path)
        # Opening a recovery copy starts a new unsaved analysis from it;
        # opening a workspace makes it the file being edited
        recovered = SESSIONS_DIR in path.parents
        self._doc_path = None if recovered else path
        self._session_path = None
        self._loading = True
        try:
            # Re-load samples, each with its own gates
            self._loader.clear()
            self._waiting_for = None
            self._samples.clear()
            self._sample_list.set_samples([])
            self._current_sample = None
            sds = ws.get("samples", [])
            notes: list[str] = []

            def restore(sd):
                """One sample with its compensation, computed channels,
                metadata and gates (runs in a worker thread)."""
                sample = Sample(sd["filepath"], lazy=True)
                if sd.get("name"):          # renamed in FlowPy
                    sample.name = sd["name"]
                try:
                    if "comp_matrix" in sd:
                        sample.restore_comp_state(sd["comp_matrix"])
                    sample.derived_from_dict(sd.get("derived", {}))
                    sample.annotations = dict(sd.get("annotations", {}))
                    if "gates" in sd:
                        # masks are computed once its events are read
                        sample.gates = protocol.tree_from_dict(sd)
                except Exception as exc:
                    notes.append(f"{sample.name}: gates not restored "
                                 f"({exc})")
                return sample

            loaded, errors = self._read_samples(
                sds, f"Opening {path.name}…", loader=restore,
                show_speed=False)
            self._samples.extend(s for s in loaded if s is not None)
            self._update_sample_labels()
            errors += notes
            if errors:
                QMessageBox.warning(
                    self, "Some samples were not restored",
                    "\n".join(errors[:30]) +
                    (f"\n… and {len(errors) - 30} more"
                     if len(errors) > 30 else ""))
            row = ws.get("current_sample", -1)
            # the saved row counts every sample, including any not found
            if 0 <= row < len(loaded) and loaded[row] is not None:
                row = self._samples.index(loaded[row])
            if not 0 <= row < len(self._samples):
                row = len(self._samples) - 1
            if self._samples:
                self._loader.enqueue(self._samples,
                                     first=self._samples[max(row, 0)])
            # Version-1 workspaces kept one gate tree, for the current sample
            if "gates" in ws and 0 <= row < len(self._samples):
                s0 = self._samples[row]
                s0.gates = protocol.tree_from_dict(ws)
                s0.gate_masks = protocol.compute_masks(s0, s0.gates)
            self._selected_gate_id = ws.get("selected_gate")
            if row >= 0:
                self._sample_list.select_sample(self._samples[row],
                                                emit=False)
                self._select_sample(self._samples[row])
            self._layout.load_dict(ws.get("layout", {}))
            self._meta_columns = list(ws.get(
                "metadata_columns", groups_core.DEFAULT_COLUMNS))
            self._groups.set_context(self._samples, self._meta_columns)
            self._groups.load_dict(ws.get("group_plot", {}))
            if self._current_sample is not None:
                self._plot.refresh_channels()
                self._stats.set_sample(self._current_sample)
            self._refresh_after_gate_change()
            self._statusbar.showMessage(
                f"Recovered session from {path.name} — use Save As to keep it"
                if recovered else f"Opened {path.name}", 5000)
        finally:
            self._loading = False
        if recovered:
            self._mark_dirty()
        else:
            self._set_clean()

    # ------------------------------------------------------------------
    # Gating protocols & batch analysis
    # ------------------------------------------------------------------

    def _apply_roots_to(self, roots: list[Gate], source=None,
                        title="Apply Gates", formulas: dict | None = None):
        """Ask which samples get *roots*, then apply them in one go.
        Derived parameters (*formulas*, default: the source sample's) are
        created in samples that lack them."""
        if formulas is None and source is not None:
            formulas = source.formula_specs()
        others = [s for s in self._samples if s is not source]
        if not others:
            QMessageBox.information(
                self, title, "Load more samples first (File → Open FCS "
                "Folder… loads a whole folder at once).")
            return
        dlg = ApplyProtocolDialog(self._samples, roots, source, title, self,
                                  formulas=formulas)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        targets = dlg.selected_samples()
        if not self._ensure_loaded(targets, "Reading the samples to gate…"):
            return
        targets = [s for s in targets if not s.load_error]
        failed = []
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            for i, s in enumerate(targets, 1):
                self._statusbar.showMessage(
                    f"Applying gates… {i}/{len(targets)}: {s.name}")
                QApplication.processEvents()
                try:
                    self._checkpoint(s)
                    protocol.apply_protocol(s, roots, formulas)
                except Exception as exc:
                    failed.append(f"{s.name}: {exc}")
        finally:
            QApplication.restoreOverrideCursor()
        if self._current_sample in targets:
            self._stats.invalidate()
            if self._gate_by_id(self._selected_gate_id) is None:
                self._selected_gate_id = None
        self._refresh_after_gate_change()
        self._schedule_autosave()
        done = len(targets) - len(failed)
        self._statusbar.showMessage(
            f"Gates applied to {done} sample{'s' if done != 1 else ''}", 5000)
        if failed:
            QMessageBox.warning(self, title, "Not applied to:\n\n"
                                + "\n".join(failed))

    def _apply_gates_to_samples(self):
        if not self._root_gates:
            QMessageBox.information(
                self, "Apply Gates",
                "Draw some gates on this sample first, or use "
                "Gating → Load Gating Protocol….")
            return
        self._apply_roots_to(self._root_gates, source=self._current_sample,
                             title="Apply Gates to Other Samples")

    # ------------------------------------------------------------------
    # Copy / paste gates (system clipboard: works between workspaces and
    # between FlowPy windows)
    # ------------------------------------------------------------------

    def _copy_gates(self, gate_id=None, whole=False):
        """Copy the selected gate with its sub-gates (or every gate)."""
        s = self._current_sample
        if s is None or not s.gates:
            QMessageBox.information(self, "Copy Gates",
                                    "This sample has no gates to copy.")
            return
        gate = None if whole else self._gate_by_id(
            gate_id if gate_id is not None else self._selected_gate_id)
        roots = [gate] if gate is not None else list(s.gates)
        parent = (protocol.population_path(gate.parent)
                  if gate is not None and gate.parent is not None
                  else "All Events")
        text = protocol.gates_to_clip(roots, s.formula_specs(), s.name, parent)
        from PyQt6.QtCore import QMimeData
        mime = QMimeData()
        mime.setText(text)
        mime.setData(protocol.CLIP_MIME, text.encode())
        QApplication.clipboard().setMimeData(mime)
        n = len(_bfs(roots))
        what = f"'{gate.name}'" + (f" and {n - 1} sub-gate"
                                   f"{'s' if n > 2 else ''}" if n > 1 else "") \
            if gate is not None else f"all {n} gates"
        self._statusbar.showMessage(
            f"Copied {what} — paste into any sample or workspace with "
            "Edit → Paste Gates (⌥⌘V)", 8000)

    def _paste_gates(self, inside_gate_id=None):
        title = "Paste Gates"
        if not self._samples:
            QMessageBox.information(self, title, "Open some FCS files first.")
            return
        mime = QApplication.clipboard().mimeData()
        text = (bytes(mime.data(protocol.CLIP_MIME)).decode()
                if mime is not None and mime.hasFormat(protocol.CLIP_MIME)
                else QApplication.clipboard().text())
        try:
            clip = protocol.gates_from_clip(text)
        except ValueError as exc:
            QMessageBox.information(self, title, str(exc))
            return
        cur = self._current_sample
        pops = ["All Events"] + ([protocol.population_path(g)
                                  for g in _bfs(cur.gates)] if cur else [])
        inside = None
        if inside_gate_id:
            g = self._gate_by_id(inside_gate_id)
            inside = protocol.population_path(g) if g else None
        dlg = PasteGatesDialog(self._samples, cur, clip, inside, pops, self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        targets = dlg.selected_samples()
        mode, where = dlg.placement()
        if not targets or not self._ensure_loaded(targets, "Reading the samples…"):
            return
        failed = []
        for t in [x for x in targets if not x.load_error]:
            self._checkpoint(t)
            try:
                protocol.paste_gates(t, clip["roots"], mode, where, clip["formulas"])
            except ValueError as exc:
                failed.append(f"{t.name}: {exc}")
        if self._current_sample in targets:
            self._stats.invalidate()
            self._plot.refresh_channels()
        self._refresh_after_gate_change()
        self._schedule_autosave()
        done = len(targets) - len(failed)
        self._statusbar.showMessage(
            f"Pasted {clip['n']} gate(s) into {done} sample(s) — Undo (⌘Z) "
            "takes them back", 8000)
        if failed:
            QMessageBox.warning(self, title, "Not pasted into:\n\n" + "\n".join(failed))

    def _save_protocol(self):
        if not self._root_gates:
            QMessageBox.information(self, "Save Gating Protocol",
                                    "There are no gates to save yet.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Gating Protocol", "gating_protocol.fpg",
            "FlowPy Gating Protocol (*.fpg)")
        if not path:
            return
        if not path.lower().endswith(".fpg"):
            path += ".fpg"
        try:
            protocol.save_protocol(path, self._root_gates,
                                   formulas=self._current_sample
                                   .formula_specs())
            self._statusbar.showMessage(f"Protocol saved: {path}", 4000)
        except Exception as exc:
            QMessageBox.critical(self, "Save Error", str(exc))

    def _load_protocol(self):
        if not self._samples:
            QMessageBox.information(self, "Load Gating Protocol",
                                    "Open your FCS files first.")
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Gating Protocol", "",
            "FlowPy Gating Protocol (*.fpg);;FlowPy Workspace (*.fpw)")
        if not path:
            return
        try:
            roots = protocol.load_protocol(path)
            formulas = protocol.load_protocol_formulas(path)
        except Exception as exc:
            QMessageBox.critical(self, "Load Error", str(exc))
            return
        self._apply_roots_to(roots, title=f"Apply {Path(path).name}",
                             formulas=formulas)

    def _show_plate(self):
        if not self._samples:
            QMessageBox.information(self, "Plate Layout",
                                    "Open some FCS files first.")
            return
        dlg = getattr(self, "_plate_dlg", None)
        if dlg is None:
            dlg = PlateDialog(self._samples, self._meta_columns, self)
            dlg.sample_requested.connect(
                lambda s: self._sample_list.select_sample(s))
            dlg.metadata_changed.connect(self._plate_metadata_changed)
            self._plate_dlg = dlg
        else:
            dlg.set_samples(self._samples, self._meta_columns)
        dlg.show()
        dlg.raise_()

    def _plate_metadata_changed(self):
        self._meta_columns = list(self._plate_dlg.columns)
        self._groups.set_context(self._samples, self._meta_columns)
        self._update_sample_labels()
        self._schedule_autosave()

    def _edit_metadata(self):
        if not self._samples:
            QMessageBox.information(self, "Sample Metadata",
                                    "Open some FCS files first.")
            return
        dlg = MetadataDialog(self._samples, self._meta_columns, self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        self._meta_columns = dlg.columns
        self._groups.set_context(self._samples, self._meta_columns)
        if getattr(self, "_plate_dlg", None) is not None:
            self._plate_dlg.set_samples(self._samples, self._meta_columns)
        self._update_sample_labels()
        self._schedule_autosave()

    def _show_export_data(self):
        if not self._samples:
            QMessageBox.information(self, "Export Data",
                                    "Open some FCS files first.")
            return
        if not self._ensure_loaded(self._samples,
                                   "Reading the samples to export…"):
            return
        ok = [s for s in self._samples if not s.load_error]
        ExportDataDialog(ok, self._current_sample, self,
                         meta_columns=self._meta_columns).exec()

    def _show_batch_stats(self):
        if not any(s.gates for s in self._samples):
            QMessageBox.information(
                self, "Batch Statistics",
                "No sample has gates yet. Gate one sample, then use "
                "Apply Gates to Samples.")
            return
        if not self._ensure_loaded([s for s in self._samples if s.gates],
                                   "Reading the gated samples…"):
            return
        BatchStatisticsDialog([s for s in self._samples
                               if not s.load_error], self).exec()

    # ------------------------------------------------------------------
    # FlowJo / Gating-ML exchange (via FlowKit)
    # ------------------------------------------------------------------

    def _gated_samples(self, title: str) -> list:
        gated = [s for s in self._samples if s.gates]
        if not gated:
            QMessageBox.information(self, title, "No sample has gates yet.")
            return []
        if not self._ensure_loaded(gated, "Reading the gated samples…"):
            return []
        return [s for s in gated if not s.load_error]

    def _import_wsp(self):
        title = "Import FlowJo Workspace"
        if not self._samples:
            QMessageBox.information(self, title, "Open the FCS files of "
                                    "that workspace first.")
            return
        path, _ = QFileDialog.getOpenFileName(self, title, "",
                                              "FlowJo Workspace (*.wsp)")
        if not path:
            return
        from ..core import flowjo_io
        try:
            result, report = adlg.run_busy(
                self, "Reading the FlowJo workspace…",
                lambda: flowjo_io.import_wsp(path, self._samples))
        except Exception as exc:
            QMessageBox.critical(self, title, str(exc))
            return
        if not self._ensure_loaded(list(result), "Reading the samples…"):
            return
        for s, roots in result.items():
            if s.load_error:
                continue
            if s.gates and QMessageBox.question(
                    self, title, f"Replace the gates of {s.name}?") != \
                    QMessageBox.StandardButton.Yes:
                continue
            self._checkpoint(s)
            s.gates = roots
            s.gate_masks = protocol.compute_masks(s, roots)
        self._stats.invalidate()
        if self._gate_by_id(self._selected_gate_id) is None:
            self._selected_gate_id = None
        self._refresh_after_gate_change()
        self._schedule_autosave()
        ExchangeReportDialog(
            title, f"Imported gates for {len(result)} sample(s) from "
            f"<b>{Path(path).name}</b>. FlowKit gated the same data the "
            "way FlowJo does, for comparison:", report, self).exec()

    def _import_gatingml(self):
        title = "Import Gating-ML"
        if not self._samples:
            QMessageBox.information(self, title, "Open your FCS files first.")
            return
        path, _ = QFileDialog.getOpenFileName(
            self, title, "", "Gating-ML (*.xml);;All files (*)")
        if not path:
            return
        from ..core import flowjo_io
        try:
            roots, report = flowjo_io.import_gatingml(
                path, self._current_sample)
        except Exception as exc:
            QMessageBox.critical(self, title, str(exc))
            return
        if report.notes:
            ExchangeReportDialog(title, "Notes on the conversion:", report,
                                 self).exec()
        self._apply_roots_to(roots, title=f"Apply {Path(path).name}")

    def _export_wsp(self):
        title = "Export to FlowJo"
        gated = self._gated_samples(title)
        if not gated:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, title, "FlowPy gates.wsp", "FlowJo Workspace (*.wsp)")
        if not path:
            return
        if not path.lower().endswith(".wsp"):
            path += ".wsp"
        from ..core import flowjo_io
        template = (self._current_sample if self._current_sample in gated
                    else gated[0])

        def work():
            report = flowjo_io.export_wsp(path, gated, template)
            # Read the file back the way FlowJo would; compare FlowKit's
            # counts with the ORIGINAL FlowPy gates (so a traced outline
            # shows up if it changes anything)
            check = flowjo_io.import_wsp(path, [_fresh(s) for s in gated])[1]
            fk_counts = {(c[0], c[1]): c[3] for c in check.counts}
            for s in gated:
                for g in _bfs(s.gates):
                    key = (s.name, " / ".join(
                        a.name for a in g.ancestors() + [g]))
                    if key in fk_counts and g.gate_id in s.gate_masks:
                        report.counts.append(
                            (*key, int(s.gate_masks[g.gate_id].sum()),
                             fk_counts[key]))
            report.notes += check.notes
            return report
        try:
            report = adlg.run_busy(self, "Writing the FlowJo workspace…",
                                   work)
        except Exception as exc:
            QMessageBox.critical(self, title, str(exc))
            return
        ExchangeReportDialog(
            title, f"Saved <b>{Path(path).name}</b> ({len(gated)} samples, "
            f"gates of {template.name}). It was then read back with FlowKit, "
            "which gates like FlowJo:", report, self).exec()

    def _export_gatingml(self):
        title = "Export Gating-ML"
        if not self._need_sample(title) or not self._root_gates:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, title, f"{Path(self._current_sample.name).stem} gates.xml",
            "Gating-ML (*.xml)")
        if not path:
            return
        from ..core import flowjo_io
        try:
            report = flowjo_io.export_gatingml(path, self._current_sample)
        except Exception as exc:
            QMessageBox.critical(self, title, str(exc))
            return
        ExchangeReportDialog(title, f"Saved <b>{Path(path).name}</b> "
                             "(Gating-ML 2.0).", report, self).exec()

    def _verify_flowkit(self):
        title = "Verify Counts with FlowKit"
        gated = self._gated_samples(title)
        if not gated:
            return
        from ..core import flowjo_io
        try:
            report = adlg.run_busy(self, "FlowKit is gating the samples…",
                                   lambda: flowjo_io.verify(gated))
        except Exception as exc:
            QMessageBox.critical(self, title, str(exc))
            return
        ExchangeReportDialog(
            title, "FlowKit (validated against the Gating-ML 2.0 compliance "
            "tests) recounted every population independently:", report,
            self).exec()

    # ------------------------------------------------------------------
    # Analysis: UMAP / t-SNE, clustering, auto gating, unmixing
    # ------------------------------------------------------------------

    def _population(self) -> tuple[Gate | None, np.ndarray, str]:
        """The population shown in the plot: (gate, event indices, name)."""
        s = self._current_sample
        pop = self._gate_by_id(self._selected_gate_id)
        mask = self._gate_masks.get(pop.gate_id) if pop else None
        idx = (np.flatnonzero(mask) if mask is not None
               else np.arange(s.n_events))
        return pop, idx, pop.name if pop else "All Events"

    def _need_sample(self, title: str) -> bool:
        if self._current_sample is None:
            QMessageBox.information(self, title, "Open an FCS file first.")
            return False
        return True

    def _derived_added(self):
        """Refresh everything after computed channels were added."""
        self._plot.refresh_channels()
        self._stats.set_sample(self._current_sample)
        self._recompute_all_masks()
        self._schedule_autosave()

    def _derived_parameters(self):
        if not self._need_sample("Derived Parameters"):
            return
        dlg = DerivedParameterDialog(self._samples, self._current_sample,
                                     self)
        dlg.exec()
        if dlg.changed:
            # a redefined formula moves events in every sample's gates
            for s in self._samples:
                # (unread samples get their masks once they are read)
                if s is not self._current_sample and s.gates \
                        and s.is_loaded:
                    s.gate_masks = protocol.compute_masks(s, s.gates)
            self._derived_added()
            self._layout.set_context(self._samples, self._current_sample)

    def _run_dimred(self):
        if not self._need_sample("UMAP / t-SNE"):
            return
        s = self._current_sample
        _pop, idx, name = self._population()
        if len(idx) < 50:
            QMessageBox.information(self, "UMAP / t-SNE",
                                    "This population has too few events.")
            return
        dlg = adlg.DimRedDialog(s, name, len(idx), self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        p = dlg.params()
        sub = analysis.subsample(idx, p["n"])
        umap = p["method"] == "UMAP"

        def work():
            x = analysis.feature_matrix(s, sub, p["channels"])
            return (analysis.run_umap(x, p["p1"], p["p2"]) if umap
                    else analysis.run_tsne(x, p["p1"]))
        try:
            emb = adlg.run_busy(self, f"Running {p['method']} on "
                                f"{len(sub):,} events…", work)
        except Exception as exc:
            QMessageBox.critical(self, p["method"], str(exc))
            return
        prefix = "UMAP" if umap else "tSNE"
        for j in (0, 1):
            col = np.full(s.n_events, np.nan)
            col[sub] = emb[:, j]
            s.add_derived(f"{prefix}-{j + 1}", col)
        self._derived_added()
        self._plot.set_plot_type("Pseudocolor")
        self._plot.set_axes(f"{prefix}-1", f"{prefix}-2", "Linear", "Linear")
        self._tabs.setCurrentIndex(0)
        self._statusbar.showMessage(
            f"{p['method']} done: channels {prefix}-1 / {prefix}-2 added "
            "(gate on them like any channel)", 6000)

    def _run_clustering(self):
        if not self._need_sample("Clustering"):
            return
        s = self._current_sample
        pop, idx, name = self._population()
        dlg = adlg.ClusterDialog(s, name, len(idx), self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        p = dlg.params()
        k, chans = p["k"], p["channels"]
        if len(idx) < k * 5:
            QMessageBox.information(self, "Clustering",
                                    "Too few events for that many clusters.")
            return

        def work():
            x = analysis.feature_matrix(s, idx, chans)
            labels = analysis.cluster(x, k, p["method"])
            return labels, analysis.cluster_profiles(s, idx, labels, chans, k)
        try:
            labels, profile = adlg.run_busy(
                self, f"Clustering {len(idx):,} events…", work)
        except Exception as exc:
            QMessageBox.critical(self, "Clustering", str(exc))
            return

        col = np.full(s.n_events, np.nan)
        col[idx] = labels + 1                       # clusters numbered 1…k
        s.add_derived("Cluster", col)
        colors = _cluster_colors(k)
        if p["make_gates"]:
            self._checkpoint()
            siblings = pop.children if pop else self._root_gates
            for g in [g for g in siblings if g.x_channel == "Cluster"]:
                siblings.remove(g)                  # replace an older run
                g.parent = None
            for c in range(k):
                g = RangeGate(f"Cluster {c + 1}", "Cluster", c + 0.5, c + 1.5)
                g._color = colors[c]
                if pop is not None:
                    pop.add_child(g)
                else:
                    self._root_gates.append(g)
            self._plot._color_gated_cb.setChecked(True)
        self._derived_added()
        for prefix in ("UMAP", "tSNE"):
            if f"{prefix}-1" in s.channels:
                self._plot.set_plot_type("Pseudocolor")
                self._plot.set_axes(f"{prefix}-1", f"{prefix}-2",
                                    "Linear", "Linear")
                break
        adlg.ClusterHeatmapDialog(profile, chans, np.bincount(labels,
                                  minlength=k), colors, self).exec()

    def _run_auto_gate(self):
        if not self._need_sample("Auto Gate"):
            return
        if self._plot.is_histogram():
            QMessageBox.information(self, "Auto Gate",
                                    "Auto gating works on a 2-D plot — "
                                    "switch the plot to Pseudocolor or Dot.")
            return
        s = self._current_sample
        _pop, idx, name = self._population()
        x_ch, y_ch = self._plot.x_channel(), self._plot.y_channel()
        x_tr, y_tr = self._plot.x_transform(), self._plot.y_transform()
        dlg = adlg.AutoGateDialog(x_ch, y_ch, name, self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        p = dlg.params()
        from ..core.transforms import apply_transform
        x = apply_transform(s.column(x_ch)[idx], x_tr)
        y = apply_transform(s.column(y_ch)[idx], y_tr)
        try:
            found = adlg.run_busy(
                self, "Finding populations…",
                lambda: analysis.auto_gate(x, y, p["n"], p["coverage"],
                                           p["min_fraction"]))
        except Exception as exc:
            QMessageBox.critical(self, "Auto Gate", str(exc))
            return
        gates = []
        for i, e in enumerate(found, 1):
            g = EllipseGate(f"{p['name']} {i}", x_ch, y_ch, e["cx"], e["cy"],
                            e["semi_x"], e["semi_y"], e["angle_deg"])
            g.x_transform, g.y_transform = x_tr, y_tr
            g._color = next_gate_color()
            gates.append(g)
        self._on_gates_created(gates)
        self._statusbar.showMessage(
            f"Found {len(gates)} population{'s' if len(gates) != 1 else ''}"
            " — click one to adjust it, or Undo", 6000)

    def _run_unmixing(self):
        if len(self._samples) < 2:
            QMessageBox.information(
                self, "Spectral Unmixing",
                "Load the single-stain controls (and an unstained control) "
                "together with the samples to unmix.")
            return
        dlg = adlg.UnmixDialog(self._samples, self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        p = dlg.params()
        dets = p["detectors"]
        used = [c for c, _ in p["controls"]] + list(p["targets"]) + \
            ([p["unstained"]] if p["unstained"] is not None else [])
        if not self._ensure_loaded(used, "Reading the samples to unmix…"):
            return
        if any(x.load_error for x in used):
            return

        def work():
            bg = (analysis.background_spectrum(p["unstained"], dets)
                  if p["unstained"] is not None else None)
            sigs = np.vstack([analysis.control_signature(c, dets, bg)
                              for c, _ in p["controls"]])
            return sigs, [(t, analysis.unmix(t, dets, sigs, bg))
                          for t in p["targets"]]
        try:
            sigs, results = adlg.run_busy(self, "Unmixing…", work)
        except Exception as exc:
            QMessageBox.critical(self, "Spectral Unmixing", str(exc))
            return
        names = [n for _, n in p["controls"]]
        for t, abund in results:
            for j, n in enumerate(names):
                t.add_derived(f"{n} (unmixed)", abund[:, j],
                              fluorescence=True)
        if self._current_sample is not None:
            self._derived_added()
        cond = np.linalg.cond(sigs)
        QMessageBox.information(
            self, "Spectral Unmixing",
            f"Added {len(names)} unmixed channels to {len(results)} "
            f"sample(s): {', '.join(n + ' (unmixed)' for n in names)}."
            + ("\n\nWarning: some signatures are very similar (condition "
               f"number {cond:.0f}); those fluorochromes will be hard to "
               "separate." if cond > 100 else ""))

    # ------------------------------------------------------------------
    # Session autosave
    # ------------------------------------------------------------------

    def _schedule_autosave(self):
        if not self._loading:
            self._mark_dirty()
            self._autosave_timer.start()

    def _flush_autosave(self):
        if self._autosave_timer.isActive():
            self._autosave_timer.stop()
            self._autosave()

    def _autosave(self):
        """Keep a recovery copy of unsaved changes. The workspace file
        itself only changes on Save, so opening an old analysis and
        starting something else can never overwrite it."""
        if not self._samples or not self._dirty:
            return
        try:
            SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
            if self._session_path is None:
                stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                n = 1                     # several windows: never the same file
                self._session_path = SESSIONS_DIR / f"session_{stamp}.fpw"
                while self._session_path.exists() or any(
                        w is not self and w._session_path == self._session_path
                        for w in _WINDOWS):
                    n += 1
                    self._session_path = SESSIONS_DIR / f"session_{stamp}_{n}.fpw"
            tmp = self._session_path.with_suffix(".tmp")
            with open(tmp, "w") as f:
                json.dump(self._workspace_dict(), f, indent=2)
            tmp.replace(self._session_path)
            self._statusbar.showMessage(
                f"Unsaved changes kept in a recovery copy — Save (⌘S) to "
                f"write them to {self._doc_path.name}" if self._doc_path
                else "Recovery copy saved — use File → Save As to name "
                "this workspace", 2500)
        except Exception as exc:
            self._statusbar.showMessage(f"Autosave failed: {exc}", 5000)

    def _restore_last_session(self):
        sessions = sorted(
            (p for p in SESSIONS_DIR.glob("session_*.fpw")
             if p != self._session_path),
            key=lambda p: p.stat().st_mtime) if SESSIONS_DIR.exists() else []
        if not sessions:
            QMessageBox.information(self, "Nothing to Recover",
                                    "There is no unsaved session to recover.")
            return
        if self._confirm_discard():
            self.load_workspace_file(str(sessions[-1]))

    def _show_sessions_folder(self):
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(SESSIONS_DIR)))

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def _export_stats(self):
        self._stats._export_csv()

    def _reset_zoom(self):
        self._plot.fit_view()

    def _show_about(self):
        QMessageBox.about(
            self, "About FlowPy",
            "<b>FlowPy 1.0</b> — Open-source Flow Cytometry Analysis<br><br>"
            "Features: FCS loading, interactive gating (polygon, rectangle, "
            "ellipse, quadrant, range), compensation matrix editor, "
            "hierarchical gate tree, statistics table with CSV export, "
            "pseudocolor / dot / histogram plots, logicle transform, "
            "layout editor with figure export, session autosave.<br><br>"
            "Built with Python · PyQt6 · matplotlib · numpy · fcsparser"
        )

    # ------------------------------------------------------------------
    # Window state persistence
    # ------------------------------------------------------------------

    def _restore_geometry(self):
        settings = QSettings("FlowPy", "FlowPy")
        geom = settings.value("geometry")
        if geom:
            self.restoreGeometry(geom)

    def closeEvent(self, event):
        if not self._confirm_discard():
            event.ignore()
            return
        settings = QSettings("FlowPy", "FlowPy")
        settings.setValue("geometry", self.saveGeometry())
        self._loader.stop()
        if self in _WINDOWS:
            _WINDOWS.remove(self)
        super().closeEvent(event)
