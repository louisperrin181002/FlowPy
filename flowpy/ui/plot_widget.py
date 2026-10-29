"""Interactive plot widget — pseudocolor / dot plots, histograms and
interactive gate drawing via matplotlib selectors.

The plot always shows one *population* (All Events, or the events inside a
gate) together with that population's child gates, FlowJo-style. Double-click
inside a gate to drill into it.
"""

from __future__ import annotations
import numpy as np
from typing import Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QLabel, QFrame,
    QPushButton, QSizePolicy, QButtonGroup, QCheckBox, QInputDialog, QDialog,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QKeySequence, QShortcut

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavToolbar
from matplotlib.figure import Figure
from matplotlib.widgets import PolygonSelector, RectangleSelector, EllipseSelector

from ..core.transforms import TRANSFORM_NAMES, inverse_transform
from ..core.gates import (
    Gate, PolygonGate, RectangleGate, EllipseGate,
    QuadrantGate, RangeGate, BooleanGate, next_gate_color,
)
from ..core.protocol import _bfs
from ..core.spectral import spectral_detectors, detector_kind
from ..core.statistics import sample_stats
from . import plotting, theme
from .gate_editor import (
    GateEditor, EditGateDialog, edit_blocker, quadrant_siblings,
)
from .plotting import PLOT_TYPES
from .widgets import ChannelCombo

# Drawing tool modes
MODE_NONE = "none"
MODE_POLYGON = "polygon"
MODE_RECTANGLE = "rectangle"
MODE_ELLIPSE = "ellipse"
MODE_QUADRANT = "quadrant"
MODE_RANGE = "range"

_MODES_2D = (MODE_POLYGON, MODE_RECTANGLE, MODE_ELLIPSE, MODE_QUADRANT)

_HINTS = {
    MODE_NONE: "Click a gate to select and edit it · double-click to "
               "open that population",
    MODE_POLYGON: "POLYGON — click to add points, click the first point to "
                  "close · Esc to cancel",
    MODE_RECTANGLE: "RECTANGLE — click and drag · Esc to cancel",
    MODE_ELLIPSE: "ELLIPSE — click and drag · Esc to cancel",
    MODE_QUADRANT: "QUADRANT — click where the quadrant lines should cross "
                   "· Esc to cancel",
    MODE_RANGE: "RANGE — click and drag across the X axis · Esc to cancel",
}

_DRAW_COLOR = theme.WARN


def _is_scatter_channel(ch: str) -> bool:
    """Channels shown on a linear scale by default (scatter, time and the
    channels FlowPy computes: UMAP, t-SNE, clusters, unmixed)."""
    return any(p in ch.upper() for p in ("FSC", "SSC", "TIME", "UMAP",
                                         "TSNE", "T-SNE", "CLUSTER"))


class PlotWidget(QWidget):
    """Embeds a matplotlib figure with interactive gating controls.

    Signals
    -------
    gates_created(list[Gate])  : user finished drawing (quadrant → 4 gates)
    gates_modified(list)       : [(gate, {attr: value})] from dragging or
                                 Edit Values — not applied yet
    gate_delete_requested(str) : Delete key on the selected gate
    boolean_requested()        : user wants a new AND / OR / NOT gate
    population_requested(str)  : user double-clicked inside a gate
    add_to_layout(dict)        : user wants the current view in the layout
    """

    gates_created = pyqtSignal(object)
    gates_modified = pyqtSignal(object)
    gate_delete_requested = pyqtSignal(str)
    boolean_requested = pyqtSignal()
    population_requested = pyqtSignal(str)
    add_to_layout = pyqtSignal(dict)
    axes_changed = pyqtSignal(list)      # channels shown on the plot

    def __init__(self, parent=None):
        super().__init__(parent)
        self._sample = None
        self._root_gates: list[Gate] = []
        self._masks: dict[str, np.ndarray] = {}
        self._population: Gate | None = None
        self._draw_mode = MODE_NONE
        self._selector = None
        self._quadrant_cid = None
        self._idx_cache: tuple | None = None   # (mask id, indices)
        # Zoomed view per (x channel, x scale, y channel, y scale)
        self._views: dict[tuple, tuple] = {}
        self._zoom_timer = QTimer(self)        # full redraw once zooming
        self._zoom_timer.setSingleShot(True)   # pauses (density, labels)
        self._zoom_timer.setInterval(250)
        self._zoom_timer.timeout.connect(self.refresh)
        self._edit_gate_id: str | None = None   # selected gate
        self._editor: GateEditor | None = None
        self._edit_block: str | None = None
        self._shown_channels: list[str] = []

        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(0)
        self._refresh_timer.timeout.connect(self.refresh)

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        card = QFrame()
        card.setObjectName("Card")
        outer.addWidget(card)
        root = QVBoxLayout(card)
        root.setContentsMargins(12, 10, 12, 6)
        root.setSpacing(6)

        # --- Population breadcrumb ---
        self._pop_label = QLabel()
        self._pop_label.setObjectName("Breadcrumb")
        self._pop_label.setTextFormat(Qt.TextFormat.RichText)
        root.addWidget(self._pop_label)

        # --- Top controls bar ---
        ctrl = QHBoxLayout()
        ctrl.setContentsMargins(0, 0, 0, 0)
        ctrl.setSpacing(6)

        ctrl.addWidget(self._muted("PLOT"))
        self._plot_type = QComboBox()
        self._plot_type.addItems(PLOT_TYPES)
        self._plot_type.currentIndexChanged.connect(self._on_plot_type)
        ctrl.addWidget(self._plot_type)
        ctrl.addSpacing(12)

        self._x_label = self._muted("X")
        ctrl.addWidget(self._x_label)
        self._x_combo = ChannelCombo()
        self._x_combo.setToolTip("X channel — type to search")
        self._x_combo.currentIndexChanged.connect(
            lambda: self._auto_scale(self._x_combo, self._x_scale))
        ctrl.addWidget(self._x_combo)

        self._x_scale = QComboBox()
        self._x_scale.addItems(TRANSFORM_NAMES)
        self._x_scale.setCurrentText("Logicle")
        self._x_scale.currentIndexChanged.connect(self.schedule_refresh)
        ctrl.addWidget(self._x_scale)
        ctrl.addSpacing(12)

        self._y_label = self._muted("Y")
        ctrl.addWidget(self._y_label)
        self._y_combo = ChannelCombo()
        self._y_combo.setToolTip("Y channel — type to search")
        self._y_combo.currentIndexChanged.connect(
            lambda: self._auto_scale(self._y_combo, self._y_scale))
        ctrl.addWidget(self._y_combo)

        self._y_scale = QComboBox()
        self._y_scale.addItems(TRANSFORM_NAMES)
        self._y_scale.setCurrentText("Logicle")
        self._y_scale.currentIndexChanged.connect(self.schedule_refresh)
        ctrl.addWidget(self._y_scale)

        self._color_gated_cb = QCheckBox("Colour gated events")
        self._color_gated_cb.setToolTip(
            "Paint events inside each gate in the gate's colour (Spectrum: "
            "add each gated population's median spectrum)")
        self._color_gated_cb.stateChanged.connect(self.schedule_refresh)

        self._zoom_box = QWidget()
        zl = QHBoxLayout(self._zoom_box)
        zl.setContentsMargins(0, 0, 0, 0)
        zl.setSpacing(4)
        zl.addWidget(self._muted("ZOOM"))
        for text, tip, slot in (
                ("−", "Zoom out (⌘−)", lambda: self.zoom(1 / 1.5)),
                ("+", "Zoom in (⌘+) — or scroll / pinch on the plot",
                 lambda: self.zoom(1.5)),
                ("Fit", "Show the whole range (⌘0)", self.fit_view)):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.setMinimumWidth(52 if text == "Fit" else 38)
            if text != "Fit":
                f = b.font()
                f.setPointSizeF(f.pointSizeF() + 3)
                f.setBold(True)
                b.setFont(f)
            b.clicked.connect(slot)
            zl.addWidget(b)

        ctrl.addStretch()
        self._layout_btn = QPushButton("＋ Add to Layout")
        self._layout_btn.setObjectName("Primary")
        self._layout_btn.setToolTip(
            "Copy this plot into the Layout tab for figure building")
        self._layout_btn.clicked.connect(self._emit_add_to_layout)
        ctrl.addWidget(self._layout_btn)
        root.addLayout(ctrl)

        # --- Gate drawing toolbar ---
        self._gate_bar = QWidget()
        gtb = QHBoxLayout(self._gate_bar)
        gtb.setContentsMargins(0, 0, 0, 0)
        gtb.setSpacing(0)
        gtb.addWidget(self._muted("GATE"))
        gtb.addSpacing(6)

        self._gate_btn_group = QButtonGroup(self)
        self._gate_btn_group.setExclusive(True)
        self._mode_buttons: dict[str, QPushButton] = {}

        btn_specs = [
            ("Select",      MODE_NONE,      "Select and edit gates (Esc)"),
            ("⬡ Polygon",   MODE_POLYGON,   "Draw a polygon gate (P)"),
            ("▭ Rectangle", MODE_RECTANGLE, "Draw a rectangle gate (R)"),
            ("◯ Ellipse",   MODE_ELLIPSE,   "Draw an ellipse gate (E)"),
            ("⊞ Quadrant",  MODE_QUADRANT,  "Draw a quadrant gate (Q)"),
            ("↔ Range",     MODE_RANGE,     "Draw a 1-D range gate (G)"),
        ]
        for label, mode, tip in btn_specs:
            btn = QPushButton(label)
            btn.setObjectName("Segment")
            btn.setProperty("first", mode == MODE_NONE)
            btn.setProperty("last", mode == MODE_RANGE)
            btn.setToolTip(tip)
            btn.setCheckable(True)
            if mode == MODE_NONE:
                btn.setChecked(True)
            btn.clicked.connect(lambda _c, m=mode: self._set_draw_mode(m))
            self._gate_btn_group.addButton(btn)
            self._mode_buttons[mode] = btn
            gtb.addWidget(btn)

        gtb.addSpacing(10)
        self._edit_btn = QPushButton("Edit Values…")
        self._edit_btn.setToolTip("Exact values, name and colour of the "
                                  "selected gate")
        self._edit_btn.setEnabled(False)
        self._edit_btn.clicked.connect(self._open_edit_dialog)
        gtb.addWidget(self._edit_btn)
        gtb.addSpacing(6)
        bool_btn = QPushButton("AND / OR / NOT…")
        bool_btn.setToolTip("Combine existing gates into a Boolean gate")
        bool_btn.clicked.connect(self.boolean_requested.emit)
        gtb.addWidget(bool_btn)
        gtb.addStretch()
        root.addWidget(self._gate_bar)
        # what to do next — its own line, wrapping, so it is never cut off
        self._hint = QLabel()
        self._hint.setObjectName("Hint")
        self._hint.setWordWrap(True)
        self._hint.setSizePolicy(QSizePolicy.Policy.Ignored,
                                 QSizePolicy.Policy.Preferred)
        root.addWidget(self._hint)
        self._update_hint()

        # --- Matplotlib canvas ---
        self._fig = Figure(figsize=(6, 5))
        self._ax = self._fig.add_subplot(111)
        self._canvas = FigureCanvas(self._fig)
        self._canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                   QSizePolicy.Policy.Expanding)
        root.addWidget(self._canvas)

        # Gates of this population that live on other axes
        self._other_gates = QLabel()
        self._other_gates.setObjectName("Muted")
        self._other_gates.setWordWrap(True)
        self._other_gates.linkActivated.connect(self._show_gate_axes)
        root.addWidget(self._other_gates)

        # under the plot: display options, zoom and matplotlib's toolbar
        bottom = QHBoxLayout()
        bottom.setContentsMargins(0, 0, 0, 0)
        bottom.setSpacing(10)
        bottom.addWidget(self._color_gated_cb)
        bottom.addSpacing(8)
        bottom.addWidget(self._zoom_box)
        bottom.addStretch()
        self._nav = NavToolbar(self._canvas, self)
        self._nav.setMaximumHeight(30)
        bottom.addWidget(self._nav)
        root.addLayout(bottom)

        self._canvas.mpl_connect("button_press_event", self._on_click)
        self._canvas.mpl_connect("resize_event", self._fit_margins)
        self._canvas.mpl_connect("scroll_event", self._on_scroll)
        self._canvas.mpl_connect("button_release_event",
                                 self._on_nav_release)
        QShortcut(QKeySequence(QKeySequence.StandardKey.ZoomIn), self,
                  activated=lambda: self.zoom(1.5))
        QShortcut(QKeySequence("Ctrl+="), self,
                  activated=lambda: self.zoom(1.5))
        QShortcut(QKeySequence(QKeySequence.StandardKey.ZoomOut), self,
                  activated=lambda: self.zoom(1 / 1.5))
        QShortcut(QKeySequence("Ctrl+0"), self, activated=self.fit_view)
        QShortcut(QKeySequence(Qt.Key.Key_Escape), self,
                  activated=self._on_escape)
        for key in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            QShortcut(QKeySequence(key), self, activated=self._delete_selected)
        # one-key gate tools (typing in the channel boxes still works:
        # text fields take printable keys first)
        for key, mode in (("P", MODE_POLYGON), ("R", MODE_RECTANGLE),
                          ("E", MODE_ELLIPSE), ("Q", MODE_QUADRANT),
                          ("G", MODE_RANGE)):
            QShortcut(QKeySequence(key), self,
                      activated=lambda m=mode: self._tool_shortcut(m))

    def _fit_margins(self, _event=None):
        """Fixed margins in pixels, so tick labels and axis titles never get
        clipped whatever the window size."""
        w, h = self._canvas.get_width_height()
        if w < 50 or h < 50:
            return
        spec = self.is_spectrum()
        # the spectrum has vertical detector names below and laser names
        # above the plot
        self._fig.subplots_adjust(left=72 / w, right=1 - 18 / w,
                                  bottom=(84 if spec else 58) / h,
                                  top=1 - (34 if spec else 14) / h)

    @staticmethod
    def _muted(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setObjectName("SectionHeader")
        lbl.setStyleSheet("padding: 0 2px;")
        return lbl

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def show_waiting(self, text: str):
        """No sample, just *text* (e.g. while one is being read); the
        channel boxes keep their choice for when it arrives."""
        self._sample = None
        self._root_gates = []
        self._masks = {}
        self._population = None
        self._idx_cache = None
        self._set_draw_mode(MODE_NONE)
        self._message = text
        self.refresh()

    def set_sample(self, sample):
        """Load a new sample into the widget."""
        self._message = None
        self._sample = sample
        self._root_gates = []
        self._masks = {}
        self._population = None
        self._idx_cache = None
        self._set_draw_mode(MODE_NONE)

        # Keep the current view when the new sample has the same channels,
        # so flipping between samples compares like with like
        prev = (self.x_channel(), self.y_channel(),
                self.x_transform(), self.y_transform())

        channels = sample.channels if sample else []
        for combo in (self._x_combo, self._y_combo):
            combo.blockSignals(True)
            combo.set_channels(channels)
            combo.blockSignals(False)

        if prev[0] in channels and prev[1] in channels:
            self.set_axes(*prev)
        elif channels:
            # Sensible defaults: FSC-A vs SSC-A
            x_ch = next((c for c in channels if "FSC" in c.upper()),
                        channels[0])
            y_ch = next((c for c in channels if "SSC" in c.upper()),
                        channels[min(1, len(channels) - 1)])
            self.set_axes(x_ch, y_ch)
        self.schedule_refresh()

    def refresh_channels(self):
        """Re-read the sample's channel list (after computed channels were
        added), keeping the current selection."""
        channels = self._sample.channels if self._sample else []
        for combo in (self._x_combo, self._y_combo):
            combo.blockSignals(True)
            combo.set_channels(channels)
            combo.blockSignals(False)
        self._idx_cache = None
        self.schedule_refresh()

    def set_context(self, root_gates: list[Gate],
                    masks: dict[str, np.ndarray],
                    population: Gate | None):
        """Show *population* (None = All Events) and its child gates."""
        self._root_gates = list(root_gates)
        self._masks = masks
        self._population = population
        if self._selected_gate() is None:
            self._edit_gate_id = None
        self.schedule_refresh()

    def set_axes(self, x_ch: str, y_ch: str | None,
                 x_tr: str | None = None, y_tr: str | None = None):
        """Switch channels; scales default to Linear for scatter channels
        and Logicle for fluorescence."""
        for combo, scale, ch, tr in ((self._x_combo, self._x_scale, x_ch, x_tr),
                                     (self._y_combo, self._y_scale, y_ch, y_tr)):
            if ch is None:
                continue
            for w in (combo, scale):
                w.blockSignals(True)
            combo.setCurrentText(ch)
            scale.setCurrentText(tr or ("Linear" if _is_scatter_channel(ch)
                                        else "Logicle"))
            for w in (combo, scale):
                w.blockSignals(False)
        self.schedule_refresh()

    def set_plot_type(self, plot_type: str):
        self._plot_type.setCurrentText(plot_type)

    def x_channel(self) -> str:
        return self._x_combo.currentText()

    def y_channel(self) -> str:
        return self._y_combo.currentText()

    def x_transform(self) -> str:
        return self._x_scale.currentText()

    def y_transform(self) -> str:
        return self._y_scale.currentText()

    def is_histogram(self) -> bool:
        return self._plot_type.currentText() == "Histogram"

    def is_spectrum(self) -> bool:
        return self._plot_type.currentText() == "Spectrum"

    def schedule_refresh(self, *_args):
        """Coalesce several changes into a single redraw."""
        self._refresh_timer.start()

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def _visible_gates(self) -> list[Gate]:
        return (self._population.children if self._population
                else self._root_gates)

    def _population_mask(self) -> np.ndarray | None:
        return (self._masks.get(self._population.gate_id)
                if self._population else None)

    def _indices(self, mask) -> np.ndarray:
        key = (id(mask), self._sample.n_events, id(self._sample.data))
        if self._idx_cache is None or self._idx_cache[0] != key:
            self._idx_cache = (key, plotting.population_indices(
                self._sample.n_events, mask))
        return self._idx_cache[1]

    def refresh(self):
        """Re-draw the plot."""
        self._refresh_timer.stop()
        if self._editor is not None:
            self._editor.disconnect()
            self._editor = None
        ax = self._ax
        ax.cla()
        self._update_breadcrumb()

        if self._sample is None:
            msg = getattr(self, "_message", None)
            if msg:
                ax.axis("off")
            ax.text(0.5, 0.5, msg or "Open an FCS file to start",
                    ha="center", va="center", color="#94a3b8",
                    transform=ax.transAxes)
            self._other_gates.setText("")
            self._canvas.draw_idle()
            return

        sample = self._sample
        hist = self.is_histogram()
        x_ch, x_tr = self.x_channel(), self.x_transform()
        y_ch = None if hist else self.y_channel()
        y_tr = None if hist else self.y_transform()

        pop_mask = self._population_mask()
        idx = self._indices(pop_mask)
        if len(idx) == 0:
            ax.text(0.5, 0.5, "No events in this population", ha="center",
                    va="center", color="gray", transform=ax.transAxes)
            self._canvas.draw_idle()
            return

        self._fit_margins()
        if self.is_spectrum():
            dets = spectral_detectors(sample.channels,
                                      detector_kind(sample.channels))
            self._emit_axes([d.channel for d in dets])
            self._draw_spectrum(pop_mask, idx)
            return
        self._emit_axes([x_ch] if hist else [x_ch, y_ch])

        # Computed channels (UMAP, clusters…) are NaN for events that were
        # not part of the computation — leave those out of the picture
        finite = np.isfinite(sample.column(x_ch)[idx])
        if not hist:
            finite &= np.isfinite(sample.column(y_ch)[idx])
        if not finite.all():
            idx = idx[finite]
        xlim = plotting.axis_limits(sample, x_ch, x_tr)
        x = plotting.display_values(sample, idx, x_ch, x_tr, xlim)
        gates = self._visible_gates()

        view = self._views.get(self._view_key())
        if hist:
            plotting.draw_histogram(ax, x, view[0] if view else xlim)
            plotting.style_axes(ax, x_ch, x_tr, xlim)
            if view:
                ax.set_xlim(view[0])
        else:
            ylim = plotting.axis_limits(sample, y_ch, y_tr)
            y = plotting.display_values(sample, idx, y_ch, y_tr, ylim)
            if self._plot_type.currentText() == "Dot":
                plotting.draw_dots(ax, x, y)
            else:
                # density is binned over what is on screen, so zooming in
                # reveals finer structure
                plotting.draw_pseudocolor(ax, x, y, *(view or (xlim, ylim)))
            if self._color_gated_cb.isChecked():
                # every child population, whatever axes its gate is on
                for g in gates:
                    m = self._masks.get(g.gate_id)
                    # boolean gates overlap their operands by definition
                    if m is not None and not isinstance(g, BooleanGate):
                        sel = m[idx]
                        plotting.draw_dots(ax, x[sel], y[sel],
                                           color=g._color or "#E74C3C",
                                           size=1.5, alpha=0.8)
            plotting.style_axes(ax, x_ch, x_tr, xlim, y_ch, y_tr, ylim)
            if view:
                ax.set_xlim(view[0])
                ax.set_ylim(view[1])

        hidden = []
        selected = self._selected_gate()
        sel_group = ([selected] if not isinstance(selected, QuadrantGate)
                     else quadrant_siblings(selected, gates))
        for g in gates:
            if plotting.gate_fits_plot(g, x_ch, y_ch):
                pct = plotting.percent_of(self._masks.get(g.gate_id),
                                          pop_mask, sample.n_events)
                plotting.draw_gate(ax, g, x_tr, y_tr, pct,
                                   selected=g in sel_group)
            elif not isinstance(g, BooleanGate):
                hidden.append(g)
        self._set_other_gates(hidden)

        # Drag handles for the selected gate
        self._edit_block = None
        if selected is not None and plotting.gate_fits_plot(selected, x_ch,
                                                            y_ch):
            self._edit_block = edit_blocker(selected, x_tr, y_tr)
            if self._edit_block is None:
                self._editor = GateEditor(ax, selected, gates, x_tr, y_tr,
                                          self.gates_modified.emit)
        self._edit_btn.setEnabled(selected is not None)
        self._update_hint()

        self._canvas.draw_idle()

    def _draw_spectrum(self, pop_mask, idx):
        """Full spectral profile of the population; with 'Colour gated
        events', the median spectrum of each child population too."""
        ax, sample = self._ax, self._sample
        dets = spectral_detectors(sample.channels,
                                  detector_kind(sample.channels))
        self._set_other_gates([])
        if not dets:
            ax.text(0.5, 0.5, "No spectral detectors in this file.\n\n"
                    "The full spectrum needs a spectral cytometer\n"
                    "(e.g. BD FACSDiscover, Cytek Aurora) whose detectors\n"
                    "are named like UV1-A … R8-A.", ha="center",
                    va="center", color="#64748b", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            self._canvas.draw_idle()
            return
        tr = self.y_transform()
        ylim = plotting.spectrum_limits(sample, dets, tr)
        chans = [d.channel for d in dets]
        medians = [(sample_stats(sample, chans, "Median", mask=pop_mask),
                    "#0f172a", "Median")]
        if self._color_gated_cb.isChecked():
            for g in self._visible_gates():
                m = self._masks.get(g.gate_id)
                if m is not None and m.any() and \
                        not isinstance(g, BooleanGate):
                    medians.append((sample_stats(sample, chans, "Median",
                                                 mask=m),
                                    g._color or "#E74C3C", g.name))
        plotting.draw_spectrum(ax, sample, idx, dets, tr, ylim,
                               medians=medians, fontsize=8)
        if len(medians) > 1:
            ax.legend(fontsize=8, frameon=True, loc="upper right",
                      framealpha=0.9)
        self._canvas.draw_idle()

    # ------------------------------------------------------------------
    # Zoom
    # ------------------------------------------------------------------

    def _view_key(self) -> tuple:
        hist = self.is_histogram()
        return (self.x_channel(), self.x_transform(),
                None if hist else self.y_channel(),
                None if hist else self.y_transform())

    def _full_limits(self):
        hist = self.is_histogram()
        xl = plotting.axis_limits(self._sample, self.x_channel(),
                                  self.x_transform())
        yl = None if hist else plotting.axis_limits(
            self._sample, self.y_channel(), self.y_transform())
        return xl, yl

    def zoom(self, factor: float, center: tuple | None = None):
        """Zoom by *factor* (>1 = in) around *center* (data coordinates,
        default: the middle of the plot). Zooming out stops at the full
        range."""
        if self._sample is None:
            return
        ax, hist = self._ax, self.is_histogram()
        (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
        cx, cy = center or ((x0 + x1) / 2, (y0 + y1) / 2)
        nx = (cx - (cx - x0) / factor, cx + (x1 - cx) / factor)
        ny = (cy - (cy - y0) / factor, cy + (y1 - cy) / factor)
        fx, fy = self._full_limits()
        if nx[1] - nx[0] >= fx[1] - fx[0] and \
                (hist or ny[1] - ny[0] >= fy[1] - fy[0]):
            return self.fit_view()
        self._views[self._view_key()] = (nx, None if hist else ny)
        ax.set_xlim(nx)
        if not hist:
            ax.set_ylim(ny)
        self._canvas.draw_idle()
        self._zoom_timer.start()

    def fit_view(self):
        """Back to the whole range of the current axes."""
        self._views.pop(self._view_key(), None)
        self.schedule_refresh()

    def _on_scroll(self, event):
        if event.inaxes != self._ax or event.xdata is None or \
                self.is_spectrum():
            return
        step = getattr(event, "step", 1) or (1 if event.button == "up"
                                              else -1)
        self.zoom(1.15 ** step, (event.xdata, event.ydata))

    def _on_nav_release(self, _event):
        """Keep zooms / pans made with the toolbar under the plot."""
        if self._nav.mode and self._sample is not None:
            hist = self.is_histogram()
            self._views[self._view_key()] = (
                self._ax.get_xlim(), None if hist else self._ax.get_ylim())
            self._zoom_timer.start()

    def _update_breadcrumb(self):
        if self._sample is None:
            self._pop_label.setText(
                f'<span style="color:{theme.MUTED};">No sample loaded</span>')
            return
        chain = (self._population.ancestors() + [self._population]
                 if self._population else [])
        names = ["All Events"] + [g.name for g in chain]
        sep = f' <span style="color:{theme.BORDER_2};">/</span> '
        path = sep.join(f'<span style="color:{theme.MUTED};">{n}</span>'
                        for n in names[:-1])
        path = (path + sep if path else "") + names[-1]
        mask = self._population_mask()
        n = int(mask.sum()) if mask is not None else self._sample.n_events
        self._pop_label.setText(
            f'{path}&nbsp;&nbsp;<span style="color:{theme.MUTED}; '
            f'font-weight:400; font-size:12px;">{n:,} events · '
            f'{self._sample.name}</span>')

    def _set_other_gates(self, hidden: list[Gate]):
        if not hidden:
            self._other_gates.setText("")
            return
        MAX_LINKS = 8
        more = len(hidden) - MAX_LINKS
        hidden = hidden[:MAX_LINKS]
        links = ", ".join(
            f'<a href="{g.gate_id}" style="color:{g._color or "#E74C3C"}; '
            f'text-decoration:none;"><b>{g.name}</b></a> '
            f'<span style="color:{theme.MUTED};">({g.x_channel}'
            f'{" / " + g.y_channel if g.y_channel else ""})</span>'
            for g in hidden)
        if more > 0:
            links += f' <span style="color:{theme.MUTED};">+{more} more '\
                     '(see the population tree)</span>'
        self._other_gates.setText(
            f'<span style="color:{theme.MUTED};">Also gated here, on other '
            f'axes (click to show):</span> {links}')

    def _show_gate_axes(self, gate_id: str):
        gate = next((g for g in self._visible_gates()
                     if g.gate_id == gate_id), None)
        if gate is None:
            return
        if gate.y_channel is None:
            self.set_plot_type("Histogram")
        elif self.is_histogram():
            self.set_plot_type("Pseudocolor")
        self.set_axes(gate.x_channel, gate.y_channel,
                      gate.x_transform, gate.y_transform)

    def _auto_scale(self, combo: QComboBox, scale: QComboBox):
        scale.blockSignals(True)
        scale.setCurrentText("Linear" if _is_scatter_channel(
            combo.currentText()) else "Logicle")
        scale.blockSignals(False)
        self.schedule_refresh()

    def _on_plot_type(self):
        """Show only the controls that apply to this kind of plot."""
        hist, spec = self.is_histogram(), self.is_spectrum()
        # histogram: no Y channel, no 2-D gates. Spectrum: every detector,
        # so only its intensity scale applies, and nothing to gate on
        for w in (self._y_combo,):
            w.setVisible(not hist and not spec)
        self._y_scale.setVisible(not hist)
        self._y_label.setVisible(not hist)
        self._y_label.setText("INTENSITY" if spec else "Y")
        for w in (self._x_label, self._x_combo, self._x_scale,
                  self._zoom_box):
            w.setVisible(not spec)
        self._gate_bar.setVisible(not spec)
        self._hint.setVisible(not spec)
        for m in _MODES_2D:
            self._mode_buttons[m].setVisible(not hist)
        self._mode_buttons[MODE_RANGE].setProperty("last", True)
        self._mode_buttons[MODE_QUADRANT].setProperty("last", False)
        if spec:
            self._y_scale.blockSignals(True)
            if self._y_scale.currentText() == "Linear":
                self._y_scale.setCurrentText("Logicle")
            self._y_scale.blockSignals(False)
        if (hist and self._draw_mode in _MODES_2D) or \
                (spec and self._draw_mode != MODE_NONE):
            self._set_draw_mode(MODE_NONE)
        self.schedule_refresh()

    def export_figure(self, path: str, dpi: int | None = None):
        """Save the plot as it is shown (without edit handles)."""
        from .theme import export_dpi
        dpi = dpi or export_dpi()
        sel = self._edit_gate_id
        self._edit_gate_id = None
        try:
            self.refresh()
            self._fig.savefig(path, dpi=dpi, facecolor="white",
                              bbox_inches="tight")
        finally:
            self._edit_gate_id = sel
            self.refresh()

    def export_figure_dialog(self):
        if self._sample is None:
            return
        from PyQt6.QtWidgets import QFileDialog, QMessageBox
        path, flt = QFileDialog.getSaveFileName(
            self, "Export Plot", "plot.png",
            f"PNG image, {theme.export_dpi()} dpi (*.png);;PDF document (*.pdf);;"
            "SVG vector (*.svg)")
        if not path:
            return
        ext = {"PNG": ".png", "PDF": ".pdf", "SVG": ".svg"}[flt.split()[0]]
        if not path.lower().endswith(ext):
            path += ext
        try:
            self.export_figure(path)
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))

    def _tool_shortcut(self, mode: str):
        btn = self._mode_buttons[mode]
        if self._sample is not None and btn.isVisible() and btn.isEnabled():
            self._set_draw_mode(mode)

    def _emit_axes(self, channels: list[str]):
        if channels != self._shown_channels:
            self._shown_channels = list(channels)
            self.axes_changed.emit(list(channels))

    def shown_channels(self) -> list[str]:
        return list(self._shown_channels)

    def _emit_add_to_layout(self):
        if self._sample is None:
            return
        self.add_to_layout.emit({
            "population_ids": [self._population.gate_id
                               if self._population else ""],
            "plot_type": self._plot_type.currentText(),
            "x_channel": self.x_channel(),
            "y_channel": self.y_channel(),
            "x_transform": self.x_transform(),
            "y_transform": self.y_transform(),
        })

    # ------------------------------------------------------------------
    # Interactive gate drawing
    # ------------------------------------------------------------------

    def _update_hint(self):
        text = _HINTS[self._draw_mode]
        sel = self._selected_gate()
        if self._draw_mode == MODE_NONE and sel is not None:
            text = (f"<b>{sel.name}</b> selected — " +
                    (self._edit_block or "drag the gate or its handles · "
                     "Delete removes it · Esc deselects"))
        self._hint.setText(text)
        self._hint.setProperty("active", self._draw_mode != MODE_NONE)
        self._hint.style().unpolish(self._hint)
        self._hint.style().polish(self._hint)

    def _set_draw_mode(self, mode: str):
        self._cancel_selector()
        self._draw_mode = mode if self._sample is not None else MODE_NONE
        self._mode_buttons[self._draw_mode].setChecked(True)
        self._update_hint()
        if self._draw_mode == MODE_NONE:
            self._canvas.draw_idle()
            return

        ax = self._ax
        # Leave pan/zoom so clicks go to the gate tool
        if self._nav.mode:
            (self._nav.pan if "pan" in str(self._nav.mode).lower()
             else self._nav.zoom)()
        line = dict(color=_DRAW_COLOR, linewidth=2.5)
        box = dict(edgecolor=_DRAW_COLOR, facecolor=_DRAW_COLOR, alpha=0.15,
                   fill=True, linewidth=2.5)

        if mode == MODE_POLYGON:
            self._selector = PolygonSelector(
                ax, self._on_polygon_done, useblit=True, props=line,
                handle_props=dict(markersize=7, markerfacecolor=_DRAW_COLOR,
                                  markeredgecolor="white"))
        elif mode == MODE_RECTANGLE:
            self._selector = RectangleSelector(
                ax, self._on_rect_done, useblit=True, props=box)
        elif mode == MODE_ELLIPSE:
            self._selector = EllipseSelector(
                ax, self._on_ellipse_done, useblit=True, props=box)
        elif mode == MODE_QUADRANT:
            self._quadrant_cid = self._canvas.mpl_connect(
                "button_press_event", self._on_quadrant_click)
        elif mode == MODE_RANGE:
            self._selector = RectangleSelector(
                ax, self._on_range_done, useblit=True, props=box)

    def _cancel_selector(self):
        if self._selector is not None:
            try:
                self._selector.set_active(False)
                self._selector.set_visible(False)
            except Exception:
                pass
            self._selector = None
        if self._quadrant_cid is not None:
            self._canvas.mpl_disconnect(self._quadrant_cid)
            self._quadrant_cid = None

    def _finish(self, gates: list[Gate]):
        """Common tail of every drawing callback."""
        x_tr, y_tr = self.x_transform(), self.y_transform()
        for g in gates:
            g.x_transform = x_tr
            if g.y_channel is not None:
                g.y_transform = y_tr
        # Deferred: selectors must not be torn down inside their own callback
        QTimer.singleShot(0, lambda: self._set_draw_mode(MODE_NONE))
        if gates:
            self.gates_created.emit(gates)

    def _ask_gate_name(self, kind: str) -> str | None:
        default = f"{kind} {len(self._visible_gates()) + 1}"
        name, ok = QInputDialog.getText(self, "Gate Name", "Name:",
                                        text=default)
        return name.strip() if ok and name.strip() else None

    def _raw(self, value: float, axis: str) -> float:
        tr = self.x_transform() if axis == "x" else self.y_transform()
        return float(inverse_transform(np.array([value]), tr)[0])

    # --- Callbacks for each selector ---

    def _on_polygon_done(self, vertices):
        if len(vertices) < 3:
            return self._finish([])
        name = self._ask_gate_name("Polygon")
        if not name:
            return self._finish([])
        raw_verts = [(self._raw(vx, "x"), self._raw(vy, "y"))
                     for vx, vy in vertices]
        gate = PolygonGate(name, self.x_channel(), self.y_channel(), raw_verts)
        gate._color = next_gate_color()
        self._finish([gate])

    def _on_rect_done(self, eclick, erelease):
        if eclick.xdata is None or erelease.xdata is None:
            return self._finish([])
        x0, x1 = sorted([eclick.xdata, erelease.xdata])
        y0, y1 = sorted([eclick.ydata, erelease.ydata])
        if abs(x1 - x0) < 1e-10 or abs(y1 - y0) < 1e-10:
            return self._finish([])
        name = self._ask_gate_name("Rectangle")
        if not name:
            return self._finish([])
        gate = RectangleGate(name, self.x_channel(), self.y_channel(),
                             self._raw(x0, "x"), self._raw(x1, "x"),
                             self._raw(y0, "y"), self._raw(y1, "y"))
        gate._color = next_gate_color()
        self._finish([gate])

    def _on_ellipse_done(self, eclick, erelease):
        if eclick.xdata is None or erelease.xdata is None:
            return self._finish([])
        # Ellipse parameters are stored in display space (see EllipseGate)
        cx = (eclick.xdata + erelease.xdata) / 2
        cy = (eclick.ydata + erelease.ydata) / 2
        sx = abs(erelease.xdata - eclick.xdata) / 2
        sy = abs(erelease.ydata - eclick.ydata) / 2
        if sx < 1e-10 or sy < 1e-10:
            return self._finish([])
        name = self._ask_gate_name("Ellipse")
        if not name:
            return self._finish([])
        gate = EllipseGate(name, self.x_channel(), self.y_channel(),
                           cx, cy, sx, sy, angle_deg=0.0)
        gate._color = next_gate_color()
        self._finish([gate])

    def _on_quadrant_click(self, event):
        if event.inaxes != self._ax or event.xdata is None:
            return
        cx, cy = self._raw(event.xdata, "x"), self._raw(event.ydata, "y")
        self._cancel_selector()
        name_base = self._ask_gate_name("Quadrant")
        if not name_base:
            return self._finish([])
        x_ch, y_ch = self.x_channel(), self.y_channel()
        color = next_gate_color()
        gates = []
        for q in ("Q1", "Q2", "Q3", "Q4"):
            gate = QuadrantGate(f"{name_base} {q}", x_ch, y_ch, cx, cy, q)
            gate._color = color
            gates.append(gate)
        self._finish(gates)

    def _on_range_done(self, eclick, erelease):
        if eclick.xdata is None or erelease.xdata is None:
            return self._finish([])
        x0, x1 = sorted([eclick.xdata, erelease.xdata])
        if abs(x1 - x0) < 1e-10:
            return self._finish([])
        name = self._ask_gate_name("Range")
        if not name:
            return self._finish([])
        gate = RangeGate(name, self.x_channel(),
                         self._raw(x0, "x"), self._raw(x1, "x"))
        gate._color = next_gate_color()
        self._finish([gate])

    # ------------------------------------------------------------------
    # Double-click: drill into a gate
    # ------------------------------------------------------------------

    def _selected_gate(self) -> Gate | None:
        return next((g for g in self._visible_gates()
                     if g.gate_id == self._edit_gate_id), None)

    def select_gate(self, gate_id: str | None):
        self._edit_gate_id = gate_id
        self.schedule_refresh()

    def _on_escape(self):
        if self._draw_mode != MODE_NONE:
            self._set_draw_mode(MODE_NONE)
        elif self._edit_gate_id is not None:
            self.select_gate(None)

    def _delete_selected(self):
        if self._edit_gate_id is not None and self._draw_mode == MODE_NONE:
            self.gate_delete_requested.emit(self._edit_gate_id)

    def _open_edit_dialog(self):
        gate = self._selected_gate()
        if gate is None:
            return
        dlg = EditGateDialog(gate, self._visible_gates(),
                             _bfs(self._root_gates), self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self.gates_modified.emit(dlg.changes())

    def _gate_at(self, event) -> Gate | None:
        hist = self.is_histogram()
        x_ch, y_ch = self.x_channel(), None if hist else self.y_channel()
        rx = np.array([self._raw(event.xdata, "x")])
        ry = None if hist else np.array([self._raw(event.ydata, "y")])
        for g in reversed(self._visible_gates()):
            if isinstance(g, BooleanGate) or \
                    not plotting.gate_fits_plot(g, x_ch, y_ch):
                continue
            if bool(g.apply(rx, ry if g.y_channel else None)[0]):
                return g
        return None

    def _on_click(self, event):
        if (self._draw_mode != MODE_NONE or event.inaxes != self._ax
                or self.is_spectrum()
                or event.xdata is None or self._nav.mode):
            return
        if event.dblclick:                 # drill into the population
            g = self._gate_at(event)
            if g is not None:
                self.population_requested.emit(g.gate_id)
            return
        if self._editor is not None and self._editor.press(event):
            return
        g = self._gate_at(event)
        new_id = g.gate_id if g is not None else None
        if new_id != self._edit_gate_id:
            self.select_gate(new_id)
