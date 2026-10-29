"""Layout editor — a simple FlowJo-style graph builder.

Collect plots of any population into a grid, overlay several populations
and/or several samples (histogram overlays, optionally staggered, or dot-plot
overlays), style them (colours, colour maps, line widths, axis ranges and
labels, statistics such as the median marked on the plot, legends) and export
the page as a high-resolution PNG, PDF or SVG figure.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
import numpy as np
import matplotlib

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QListWidget,
    QListWidgetItem, QPushButton, QLabel, QFormLayout, QLineEdit, QComboBox,
    QCheckBox, QSpinBox, QDoubleSpinBox, QScrollArea, QFileDialog,
    QMessageBox, QGroupBox, QTabWidget, QColorDialog, QFrame,
)
from PyQt6.QtCore import Qt, QTimer, QSize, pyqtSignal
from PyQt6.QtGui import QColor, QIcon, QPixmap, QImage

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from ..core.transforms import TRANSFORM_NAMES, apply_transform
from ..core.gates import Gate, BooleanGate
from ..core.statistics import STAT_NAMES, stat_1d, stat_label
from . import plotting
from .plotting import PLOT_TYPES, PALETTES, COLORMAPS
from .gate_tree_widget import _bfs
from .widgets import ChannelCombo
from ..core.spectral import spectral_detectors, detector_kind
from ..core.statistics import sample_stats

MAX_LAYOUT_EVENTS = 20_000              # per layer per panel
ALL_EVENTS = ""                         # population id for ungated data
CURRENT = ""                            # sample id meaning "current sample"
STAGGER_STEP = 70.0                     # % of max between staggered layers
NO_STAT = "None"
# Statistics that are a position on the axis, so they can be marked
LOCATION_STATS = {"Mean", "Median", "Geometric Mean", "Min", "Max",
                  "Percentile"}
SINGLE_COLORS = {"Histogram": "#2563eb", "Dot": "#334155"}
POSITIONS = ["Upper right", "Upper left", "Lower right", "Lower left"]
LEGEND_POSITIONS = POSITIONS + ["Best", "Outside right", "Hidden"]
FONTS = ["Helvetica / Arial", "Arial", "Helvetica", "Times New Roman",
         "Georgia", "Courier New", "DejaVu Sans"]
EXPORT_DPI = [150, 300, 600, 1200]

# Fields copied by "Apply style to all plots" (everything that is not data)
STYLE_FIELDS = ("colormap", "palette", "dot_size", "opacity", "line_width",
                "fill_alpha", "smoothing", "box", "grid", "show_title",
                "show_gates", "gate_names", "gate_pct", "gate_lw", "stat",
                "stat_pct", "stat_mark", "show_count", "stat_loc", "legend",
                "normalize", "stagger")


@dataclass
class LayoutItem:
    population_ids: list[str] = field(default_factory=lambda: [ALL_EVENTS])
    # Sample file paths; [""] = whichever sample is selected
    sample_ids: list[str] = field(default_factory=lambda: [CURRENT])
    plot_type: str = "Pseudocolor"
    x_channel: str = ""
    y_channel: str = ""
    x_transform: str = "Logicle"
    y_transform: str = "Logicle"
    title: str = ""
    show_gates: bool = True
    normalize: bool = True             # histogram overlays as % of max
    stagger: bool = False              # offset overlaid histograms
    # --- style ---
    colormap: str = "Turbo"            # pseudocolour density colours
    palette: str = "FlowPy"            # overlay colours
    colors: dict = field(default_factory=dict)   # layer key -> "#rrggbb"
    dot_size: float = 1.5
    opacity: float = 60.0              # % (dot plots / overlays)
    line_width: float = 1.4
    fill_alpha: float = 35.0           # % (histograms)
    smoothing: float = 1.0             # histogram smoothing (0 = raw bins)
    box: bool = False                  # frame on all four sides
    grid: bool = False
    # --- axes (ranges in raw channel units; None = automatic) ---
    x_label: str = ""
    y_label: str = ""
    x_min: float | None = None
    x_max: float | None = None
    y_min: float | None = None
    y_max: float | None = None
    # --- annotations ---
    show_title: bool = True
    gate_names: bool = True
    gate_pct: bool = True
    gate_lw: float = 2.0
    stat: str = NO_STAT                # e.g. "Median"
    stat_pct: float = 50.0
    stat_mark: bool = True             # line (histogram) / cross (2-D)
    show_count: bool = False
    stat_loc: str = "Upper left"
    legend: str = "Upper right"

    @classmethod
    def from_dict(cls, d: dict) -> "LayoutItem":
        known = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class _Layer:
    key: str                           # "<sample id>|<population id>"
    sample: object
    gate: Gate | None
    mask: np.ndarray | None
    label: str


def _fmt(v: float) -> str:
    """Compact number for stat labels: 21,126 · 845 · 3.27 · –12.5"""
    if not np.isfinite(v):
        return "–"
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    if abs(v) >= 100:
        return f"{v:.0f}"
    return f"{v:.3g}"


def _swatch(color: str) -> QIcon:
    pm = QPixmap(14, 14)
    pm.fill(QColor(color))
    return QIcon(pm)


def _scrolled(widget: QWidget) -> QScrollArea:
    area = QScrollArea()
    area.setWidget(widget)
    area.setWidgetResizable(True)
    area.setFrameShape(QFrame.Shape.NoFrame)
    return area


class _RangeEdit(QLineEdit):
    """Number box where empty means automatic."""

    def __init__(self, placeholder: str):
        super().__init__()
        self.setPlaceholderText(placeholder)

    def value(self) -> float | None:
        try:
            v = float(self.text().replace(",", "").strip())
        except ValueError:
            return None
        return v if np.isfinite(v) else None

    def set_value(self, v: float | None):
        self.setText("" if v is None else f"{v:g}")


class LayoutEditor(QWidget):
    """Grid of publication-style plots built from gated populations."""

    changed = pyqtSignal()   # layout content edited (for autosave)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._samples: list = []
        self._current = None
        self._items: list[LayoutItem] = []
        self._dirty = True

        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.setInterval(120)
        self._render_timer.timeout.connect(self.render)

        self._build_ui()
        self._load_item_into_form(None)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        split = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(split)

        # --- Left: item list + properties + page ---
        left = QWidget()
        left.setMinimumWidth(340)
        left.setMaximumWidth(600)
        lv = QVBoxLayout(left)
        lv.setContentsMargins(6, 6, 6, 6)

        hdr = QLabel("PLOTS IN LAYOUT")
        hdr.setObjectName("SectionHeader")
        lv.addWidget(hdr)
        self._list = QListWidget()
        self._list.setIconSize(QSize(72, 56))
        self._list.setMinimumHeight(150)
        self._list.setMaximumHeight(230)
        self._list.setWordWrap(True)
        self._list.currentRowChanged.connect(self._on_row_changed)
        lv.addWidget(self._list)

        row = QHBoxLayout()
        for text, tip, slot in (("Add", "Add a plot", self._add_blank),
                                ("Duplicate", "Copy selected plot",
                                 self._duplicate),
                                ("Remove", "Remove selected plot",
                                 self._remove),
                                ("▲", "Move up", lambda: self._move(-1)),
                                ("▼", "Move down", lambda: self._move(1))):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            row.addWidget(b)
        lv.addLayout(row)

        self._form_widgets = []
        self._props = QTabWidget()
        self._props.addTab(_scrolled(self._build_data_tab()), "Data")
        self._props.addTab(_scrolled(self._build_style_tab()), "Style")
        self._props.addTab(_scrolled(self._build_axes_tab()), "Axes")
        self._props.addTab(_scrolled(self._build_annot_tab()), "Labels")
        lv.addWidget(self._props, 1)

        self._style_all = QPushButton("Apply this style to all plots")
        self._style_all.setToolTip(
            "Copy colours, sizes, statistics, legend and gate display of "
            "the selected plot to every plot (data and axes are kept)")
        self._style_all.clicked.connect(self._apply_style_to_all)
        lv.addWidget(self._style_all)

        # page settings live in a small window, to leave room for the plot
        # options
        from PyQt6.QtWidgets import QDialog, QDialogButtonBox
        self._page_dialog = QDialog(self)
        self._page_dialog.setWindowTitle("Page Setup")
        pdl = QVBoxLayout(self._page_dialog)
        pdl.addWidget(self._build_page_group())
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        bb.rejected.connect(self._page_dialog.hide)
        pdl.addWidget(bb)
        bottom = QHBoxLayout()
        page_btn = QPushButton("Page Setup…")
        page_btn.setToolTip("Columns, plot size, font, export resolution")
        page_btn.clicked.connect(self._page_dialog.show)
        bottom.addWidget(page_btn)
        export = QPushButton("Export Figure…")
        export.setObjectName("Primary")
        export.setToolTip("Save the whole layout as PNG, PDF or SVG")
        export.clicked.connect(self._export)
        bottom.addWidget(export, 1)
        lv.addLayout(bottom)

        split.addWidget(left)

        # --- Right: the page ---
        self._fig = Figure(figsize=(3.6 * 3, 3.3), dpi=100)
        self._canvas = FigureCanvas(self._fig)
        scroll = QScrollArea()
        scroll.setWidget(self._canvas)
        scroll.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        scroll.setStyleSheet("QScrollArea { background: #e9edf2; "
                             "border-radius: 10px; }")
        split.addWidget(scroll)
        split.setStretchFactor(1, 1)

    def _connect(self, w):
        """Hook a form widget to _form_changed whatever its type."""
        if isinstance(w, QLineEdit):
            w.textEdited.connect(self._form_changed)
        elif isinstance(w, QComboBox):
            w.currentIndexChanged.connect(self._form_changed)
        elif isinstance(w, QCheckBox):
            w.toggled.connect(self._form_changed)
        elif isinstance(w, (QSpinBox, QDoubleSpinBox)):
            w.valueChanged.connect(self._form_changed)
        elif isinstance(w, QListWidget):
            w.itemChanged.connect(self._form_changed)
        self._form_widgets.append(w)
        return w

    @staticmethod
    def _dspin(lo, hi, step, decimals=1, suffix=""):
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setSingleStep(step)
        s.setDecimals(decimals)
        s.setSuffix(suffix)
        return s

    def _build_data_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        self._title_edit = self._connect(QLineEdit())
        self._title_edit.setPlaceholderText("(automatic)")
        form.addRow("Title:", self._title_edit)

        self._type_combo = self._connect(QComboBox())
        self._type_combo.addItems(PLOT_TYPES)
        form.addRow("Type:", self._type_combo)

        self._sample_list = self._connect(QListWidget())
        self._sample_list.setMinimumHeight(90)
        self._sample_list.setToolTip(
            "Tick several samples to overlay the same population from each")
        form.addRow("Samples:", self._sample_list)

        self._pop_list = self._connect(QListWidget())
        self._pop_list.setMinimumHeight(90)
        self._pop_list.setToolTip(
            "Tick several populations to overlay them in different colours")
        form.addRow("Populations:", self._pop_list)

        self._norm_cb = self._connect(
            QCheckBox("Normalise histograms (% of max)"))
        form.addRow(self._norm_cb)
        self._stagger_cb = self._connect(
            QCheckBox("Stagger overlaid histograms (offset)"))
        form.addRow(self._stagger_cb)
        return w

    def _build_style_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        self._cmap_combo = self._connect(QComboBox())
        self._cmap_combo.addItems(COLORMAPS)
        self._cmap_combo.setToolTip("Density colours of pseudocolour plots")
        form.addRow("Colour map:", self._cmap_combo)

        self._palette_combo = self._connect(QComboBox())
        self._palette_combo.addItems(PALETTES)
        self._palette_combo.setToolTip("Colours of overlaid populations / "
                                       "samples")
        form.addRow("Palette:", self._palette_combo)

        self._layer_list = QListWidget()
        self._layer_list.setMinimumHeight(70)
        self._layer_list.setMaximumHeight(130)
        self._layer_list.setToolTip("Click a layer to choose its colour")
        self._layer_list.itemClicked.connect(self._pick_layer_color)
        form.addRow("Colours:", self._layer_list)
        reset = QPushButton("Reset colours to palette")
        reset.clicked.connect(self._reset_colors)
        form.addRow("", reset)

        self._dot_size = self._connect(self._dspin(0.2, 20, 0.5))
        form.addRow("Dot size:", self._dot_size)
        self._opacity = self._connect(self._dspin(5, 100, 5, 0, " %"))
        self._opacity.setToolTip("Dot opacity (dot plots and overlays)")
        form.addRow("Dot opacity:", self._opacity)
        self._line_width = self._connect(self._dspin(0.3, 6, 0.2))
        form.addRow("Line width:", self._line_width)
        self._fill_alpha = self._connect(self._dspin(0, 100, 5, 0, " %"))
        self._fill_alpha.setToolTip("Histogram fill opacity (0 = outline "
                                    "only)")
        form.addRow("Fill opacity:", self._fill_alpha)
        self._smoothing = self._connect(self._dspin(0, 6, 0.5))
        self._smoothing.setToolTip("Histogram smoothing (0 = raw bins)")
        form.addRow("Smoothing:", self._smoothing)
        self._box_cb = self._connect(QCheckBox("Frame on all sides"))
        form.addRow(self._box_cb)
        self._grid_cb = self._connect(QCheckBox("Grid lines"))
        form.addRow(self._grid_cb)
        return w

    def _build_axes_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        self._x_combo, self._x_scale = ChannelCombo(), QComboBox()
        self._y_combo, self._y_scale = ChannelCombo(), QComboBox()
        self._x_min, self._x_max = _RangeEdit("auto"), _RangeEdit("auto")
        self._y_min, self._y_max = _RangeEdit("auto"), _RangeEdit("auto")
        self._x_label = QLineEdit()
        self._y_label = QLineEdit()
        for axis, ch, sc, lo, hi, lab in (
                ("X", self._x_combo, self._x_scale, self._x_min,
                 self._x_max, self._x_label),
                ("Y", self._y_combo, self._y_scale, self._y_min,
                 self._y_max, self._y_label)):
            sc.addItems(TRANSFORM_NAMES)
            for wid in (ch, sc, lo, hi, lab):
                self._connect(wid)
            form.addRow(f"{axis} channel:", ch)
            form.addRow(f"{axis} scale:", sc)
            lab.setPlaceholderText("(channel name)")
            form.addRow(f"{axis} label:", lab)
            h = QHBoxLayout()
            h.addWidget(lo)
            h.addWidget(QLabel("to"))
            h.addWidget(hi)
            form.addRow(f"{axis} range:", h)
        note = QLabel("Ranges are in channel units (e.g. 100 to 100000); "
                      "leave empty for automatic. For histograms, the Y "
                      "range is the count / % of max.")
        note.setWordWrap(True)
        note.setObjectName("Hint")
        form.addRow(note)
        return w

    def _build_annot_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        self._show_title_cb = self._connect(QCheckBox("Show title"))
        form.addRow(self._show_title_cb)

        self._stat_combo = self._connect(QComboBox())
        self._stat_combo.addItems([NO_STAT] + STAT_NAMES)
        self._stat_combo.setToolTip("Statistic of each population shown on "
                                    "the plot (all events, not just the "
                                    "drawn ones)")
        form.addRow("Statistic:", self._stat_combo)
        self._stat_pct = self._connect(self._dspin(0, 100, 1, 1))
        form.addRow("Percentile:", self._stat_pct)
        self._stat_mark_cb = self._connect(
            QCheckBox("Mark on plot (line / cross)"))
        form.addRow(self._stat_mark_cb)
        self._count_cb = self._connect(QCheckBox("Show event count"))
        form.addRow(self._count_cb)
        self._stat_loc = self._connect(QComboBox())
        self._stat_loc.addItems(POSITIONS)
        form.addRow("Stats box:", self._stat_loc)
        self._legend_combo = self._connect(QComboBox())
        self._legend_combo.addItems(LEGEND_POSITIONS)
        form.addRow("Legend:", self._legend_combo)

        sep = QLabel("GATES")
        sep.setObjectName("SectionHeader")
        form.addRow(sep)
        self._gates_cb = self._connect(QCheckBox("Show gates"))
        form.addRow(self._gates_cb)
        self._gate_names_cb = self._connect(QCheckBox("Gate names"))
        form.addRow(self._gate_names_cb)
        self._gate_pct_cb = self._connect(QCheckBox("Gate percentages"))
        form.addRow(self._gate_pct_cb)
        self._gate_lw = self._connect(self._dspin(0.5, 6, 0.25, 2))
        form.addRow("Gate line width:", self._gate_lw)
        return w

    def _build_page_group(self) -> QGroupBox:
        box = QGroupBox("PAGE")
        form = QFormLayout(box)

        h = QHBoxLayout()
        self._cols = QSpinBox()
        self._cols.setRange(1, 8)
        self._cols.setValue(3)
        h.addWidget(self._cols)
        h.addWidget(QLabel("  Size:"))
        self._panel_w = self._dspin(1.5, 10, 0.1, 1, " in")
        self._panel_w.setValue(3.6)
        self._panel_h = self._dspin(1.5, 10, 0.1, 1, " in")
        self._panel_h.setValue(3.3)
        h.addWidget(self._panel_w)
        h.addWidget(QLabel("×"))
        h.addWidget(self._panel_h)
        form.addRow("Columns:", h)

        h = QHBoxLayout()
        self._font_combo = QComboBox()
        self._font_combo.addItems(FONTS)
        self._font_size = QSpinBox()
        self._font_size.setRange(5, 20)
        self._font_size.setValue(8)
        self._font_size.setSuffix(" pt")
        h.addWidget(self._font_combo, 1)
        h.addWidget(self._font_size)
        form.addRow("Font:", h)

        h = QHBoxLayout()
        self._dpi_combo = QComboBox()
        self._dpi_combo.addItems([f"{d} dpi" for d in EXPORT_DPI])
        from .theme import export_dpi
        self._dpi_combo.setCurrentIndex(EXPORT_DPI.index(export_dpi()))
        self._dpi_combo.setToolTip("Resolution of PNG export")
        self._transparent_cb = QCheckBox("Transparent")
        self._transparent_cb.setToolTip("Export without a white background")
        h.addWidget(self._dpi_combo)
        h.addWidget(self._transparent_cb)
        form.addRow("Export:", h)

        for w in (self._cols, self._panel_w, self._panel_h, self._font_size):
            w.valueChanged.connect(self._layout_changed)
        for w in (self._font_combo, self._dpi_combo):
            w.currentIndexChanged.connect(self._layout_changed)
        self._transparent_cb.toggled.connect(self._layout_changed)

        return box

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_context(self, samples: list, current):
        """All loaded samples and the one selected in the gating window."""
        old_channels = self._channels()
        self._samples = list(samples)
        self._current = current
        if self._channels() != old_channels:
            for combo in (self._x_combo, self._y_combo):
                combo.blockSignals(True)
                combo.set_channels(self._channels())
                combo.blockSignals(False)
        self._refresh_list_labels()
        self._load_item_into_form(self._current_item())
        self._layout_changed(emit=False)

    def sample_loaded(self, sample):
        """Redraw when a sample used by a plot has been read."""
        if any(sample.filepath in it.sample_ids for it in self._items):
            self._layout_changed(emit=False)

    def add_item(self, spec: dict):
        self._items.append(LayoutItem.from_dict(spec))
        self._list.addItem(QListWidgetItem())
        self._refresh_list_labels()
        self._list.setCurrentRow(len(self._items) - 1)
        self._layout_changed()

    def count(self) -> int:
        return len(self._items)

    def _page(self) -> dict:
        return {"columns": self._cols.value(),
                "panel_w": self._panel_w.value(),
                "panel_h": self._panel_h.value(),
                "font": self._font_combo.currentText(),
                "font_size": self._font_size.value(),
                "dpi": EXPORT_DPI[self._dpi_combo.currentIndex()],
                "transparent": self._transparent_cb.isChecked()}

    def to_dict(self) -> dict:
        return {**self._page(), "items": [asdict(it) for it in self._items]}

    def load_dict(self, d: dict):
        self._items = [LayoutItem.from_dict(x) for x in d.get("items", [])]
        page = (self._cols, self._panel_w, self._panel_h, self._font_combo,
                self._font_size, self._dpi_combo, self._transparent_cb)
        for w in page:
            w.blockSignals(True)
        self._cols.setValue(int(d.get("columns", 3)))
        self._panel_w.setValue(float(d.get("panel_w", 3.6)))
        self._panel_h.setValue(float(d.get("panel_h", 3.3)))
        self._font_combo.setCurrentText(d.get("font", FONTS[0]))
        self._font_size.setValue(int(d.get("font_size", 8)))
        from .theme import export_dpi
        dpi = int(d.get("dpi", export_dpi()))
        self._dpi_combo.setCurrentIndex(
            EXPORT_DPI.index(dpi) if dpi in EXPORT_DPI
            else EXPORT_DPI.index(export_dpi()))
        self._transparent_cb.setChecked(bool(d.get("transparent", False)))
        for w in page:
            w.blockSignals(False)
        self._list.blockSignals(True)
        self._list.clear()
        for _ in self._items:
            self._list.addItem(QListWidgetItem())
        self._list.blockSignals(False)
        self._refresh_list_labels()
        self._list.setCurrentRow(0 if self._items else -1)
        self._load_item_into_form(self._current_item())
        self._layout_changed(emit=False)

    # ------------------------------------------------------------------
    # Samples / populations
    # ------------------------------------------------------------------

    def _channels(self) -> list[str]:
        ref = self._current or (self._samples[0] if self._samples else None)
        return list(ref.channels) if ref is not None else []

    def _resolve_samples(self, item: LayoutItem) -> list[tuple[str, object]]:
        """(sample id, sample) for each sample of *item* that is open and
        whose events have been read."""
        out, seen = [], set()
        for sid in item.sample_ids:
            s = (self._current if sid == CURRENT else
                 next((x for x in self._samples if x.filepath == sid), None))
            if s is not None and id(s) not in seen \
                    and getattr(s, "is_loaded", True):
                seen.add(id(s))
                out.append((sid, s))
        return out

    @staticmethod
    def _gate_in(sample, gid: str) -> Gate | None:
        return next((g for g in _bfs(sample.gates) if g.gate_id == gid), None)

    def _known_populations(self) -> list[tuple[str, str]]:
        """(gate_id, path name) of every population in any loaded sample,
        current sample first."""
        seen, out = set(), [(ALL_EVENTS, "All Events")]
        order = ([self._current] if self._current else []) + \
            [s for s in self._samples if s is not self._current]
        for s in order:
            for g in _bfs(s.gates):
                if g.gate_id not in seen:
                    seen.add(g.gate_id)
                    out.append((g.gate_id, " / ".join(
                        a.name for a in g.ancestors() + [g])))
        return out

    def _pop_name(self, gid: str) -> str | None:
        if gid == ALL_EVENTS:
            return "All Events"
        for s in ([self._current] if self._current else []) + self._samples:
            g = self._gate_in(s, gid)
            if g is not None:
                return g.name
        return None

    def _layers(self, item: LayoutItem) -> list[_Layer]:
        samples = self._resolve_samples(item)
        multi_s = len(samples) > 1
        multi_p = len(item.population_ids) > 1
        layers = []
        for sid, s in samples:
            for gid in item.population_ids:
                if gid == ALL_EVENTS:
                    gate, mask = None, None
                else:
                    gate = self._gate_in(s, gid)
                    mask = s.gate_masks.get(gid)
                    if gate is None or mask is None:
                        continue
                pname = gate.name if gate else "All Events"
                label = (f"{s.name} · {pname}" if multi_s and multi_p
                         else s.name if multi_s else pname)
                layers.append(_Layer(f"{sid}|{gid}", s, gate, mask, label))
        return layers

    @staticmethod
    def _layer_colors(item: LayoutItem, layers: list[_Layer]) -> list[str]:
        if len(layers) == 1:
            default = [SINGLE_COLORS.get(item.plot_type, "#2563eb")]
        else:
            pal = PALETTES.get(item.palette, PALETTES["FlowPy"])
            default = [pal[k % len(pal)] for k in range(len(layers))]
        return [item.colors.get(lay.key, d) for lay, d in zip(layers, default)]

    def _auto_title(self, item: LayoutItem) -> str:
        names = [n for n in (self._pop_name(p) for p in item.population_ids)
                 if n]
        title = " vs ".join(names) if names else "(missing population)"
        samples = self._resolve_samples(item)
        if len(samples) > 1:
            title += f" · {len(samples)} samples"
        elif samples and item.sample_ids != [CURRENT]:
            title += f" · {samples[0][1].name}"
        return title

    # ------------------------------------------------------------------
    # List / form
    # ------------------------------------------------------------------

    def _current_item(self) -> LayoutItem | None:
        r = self._list.currentRow()
        return self._items[r] if 0 <= r < len(self._items) else None

    def _refresh_list_labels(self):
        for i, it in enumerate(self._items):
            what = (it.x_channel if it.plot_type == "Histogram"
                    else "full spectrum" if it.plot_type == "Spectrum"
                    else f"{it.x_channel} / {it.y_channel}")
            self._list.item(i).setText(
                f"{i + 1}. {it.title or self._auto_title(it)}  —  {what}")

    def _on_row_changed(self, _row):
        self._load_item_into_form(self._current_item())

    @staticmethod
    def _fill_checklist(lst: QListWidget, entries, checked):
        lst.clear()
        for key, text in entries:
            li = QListWidgetItem(text)
            li.setData(Qt.ItemDataRole.UserRole, key)
            li.setFlags(li.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            li.setCheckState(Qt.CheckState.Checked if key in checked
                             else Qt.CheckState.Unchecked)
            lst.addItem(li)

    @staticmethod
    def _checked(lst: QListWidget) -> list[str]:
        return [lst.item(i).data(Qt.ItemDataRole.UserRole)
                for i in range(lst.count())
                if lst.item(i).checkState() == Qt.CheckState.Checked]

    def _load_item_into_form(self, item: LayoutItem | None):
        self._props.setEnabled(item is not None)
        self._style_all.setEnabled(item is not None and len(self._items) > 1)
        for w in self._form_widgets:
            w.blockSignals(True)
        self._pop_list.clear()
        self._sample_list.clear()
        if item is not None:
            self._title_edit.setText(item.title)
            self._type_combo.setCurrentText(item.plot_type)
            self._fill_checklist(
                self._sample_list,
                [(CURRENT, "Selected sample (follows the Gating tab)")] +
                [(s.filepath, s.name) for s in self._samples],
                item.sample_ids)
            self._fill_checklist(self._pop_list, self._known_populations(),
                                 item.population_ids)
            self._norm_cb.setChecked(item.normalize)
            self._stagger_cb.setChecked(item.stagger)
            # style
            self._cmap_combo.setCurrentText(item.colormap)
            self._palette_combo.setCurrentText(item.palette)
            self._dot_size.setValue(item.dot_size)
            self._opacity.setValue(item.opacity)
            self._line_width.setValue(item.line_width)
            self._fill_alpha.setValue(item.fill_alpha)
            self._smoothing.setValue(item.smoothing)
            self._box_cb.setChecked(item.box)
            self._grid_cb.setChecked(item.grid)
            # axes
            self._x_combo.setCurrentText(item.x_channel)
            self._x_scale.setCurrentText(item.x_transform)
            self._y_combo.setCurrentText(item.y_channel)
            self._y_scale.setCurrentText(item.y_transform)
            self._x_label.setText(item.x_label)
            self._y_label.setText(item.y_label)
            self._x_min.set_value(item.x_min)
            self._x_max.set_value(item.x_max)
            self._y_min.set_value(item.y_min)
            self._y_max.set_value(item.y_max)
            # annotations
            self._show_title_cb.setChecked(item.show_title)
            self._stat_combo.setCurrentText(item.stat)
            self._stat_pct.setValue(item.stat_pct)
            self._stat_mark_cb.setChecked(item.stat_mark)
            self._count_cb.setChecked(item.show_count)
            self._stat_loc.setCurrentText(item.stat_loc)
            self._legend_combo.setCurrentText(item.legend)
            self._gates_cb.setChecked(item.show_gates)
            self._gate_names_cb.setChecked(item.gate_names)
            self._gate_pct_cb.setChecked(item.gate_pct)
            self._gate_lw.setValue(item.gate_lw)
            self._sync_enabled(item)
        self._refresh_layer_list(item)
        for w in self._form_widgets:
            w.blockSignals(False)

    def _sync_enabled(self, item: LayoutItem):
        hist = item.plot_type == "Histogram"
        spec = item.plot_type == "Spectrum"
        pseudo = item.plot_type == "Pseudocolor"
        n = len(self._layers(item))
        for w in (self._y_combo, self._y_scale):
            w.setEnabled(not hist)
        for w in (self._x_combo, self._x_scale, self._y_combo, self._x_min,
                  self._x_max):
            w.setEnabled(not spec)
        self._norm_cb.setEnabled(hist or (spec and n > 1))
        for w in (self._fill_alpha, self._smoothing):
            w.setEnabled(hist)
        self._stagger_cb.setEnabled(hist and n > 1)
        # pseudocolour only applies to a single population
        self._cmap_combo.setEnabled(pseudo and n <= 1)
        self._palette_combo.setEnabled(n > 1)
        self._line_width.setEnabled(hist)
        self._opacity.setEnabled(not hist)
        self._dot_size.setEnabled(not hist)
        has_stat = item.stat != NO_STAT
        self._stat_pct.setEnabled(item.stat == "Percentile")
        self._stat_mark_cb.setEnabled(item.stat in LOCATION_STATS)
        self._stat_loc.setEnabled((has_stat or item.show_count) and n <= 1)
        self._legend_combo.setEnabled(n > 1)
        for w in (self._gate_names_cb, self._gate_pct_cb, self._gate_lw):
            w.setEnabled(item.show_gates)

    def _refresh_layer_list(self, item: LayoutItem | None):
        self._layer_list.clear()
        if item is None:
            return
        layers = self._layers(item)
        pseudo_single = item.plot_type == "Pseudocolor" and len(layers) == 1
        for lay, color in zip(layers, self._layer_colors(item, layers)):
            li = QListWidgetItem(_swatch(color), lay.label)
            li.setData(Qt.ItemDataRole.UserRole, lay.key)
            if pseudo_single:
                li.setFlags(Qt.ItemFlag.NoItemFlags)
                li.setToolTip("Pseudocolour uses the colour map — switch "
                              "to Dot to colour a single population")
            self._layer_list.addItem(li)

    def _pick_layer_color(self, li: QListWidgetItem):
        item = self._current_item()
        if item is None:
            return
        if not li.flags() & Qt.ItemFlag.ItemIsEnabled:
            return
        key = li.data(Qt.ItemDataRole.UserRole)
        layers = self._layers(item)
        current = dict(zip((l.key for l in layers),
                           self._layer_colors(item, layers))).get(key)
        color = QColorDialog.getColor(QColor(current or "#2563eb"), self,
                                      f"Colour of {li.text()}")
        if color.isValid():
            item.colors[key] = color.name()
            self._refresh_layer_list(item)
            self._layout_changed()

    def _reset_colors(self):
        item = self._current_item()
        if item is not None and item.colors:
            item.colors.clear()
            self._refresh_layer_list(item)
            self._layout_changed()

    def _apply_style_to_all(self):
        src = self._current_item()
        if src is None:
            return
        for it in self._items:
            if it is not src:
                for f in STYLE_FIELDS:
                    setattr(it, f, getattr(src, f))
        self._layout_changed()

    def _form_changed(self, *_):
        item = self._current_item()
        if item is None:
            return
        item.title = self._title_edit.text().strip()
        item.plot_type = self._type_combo.currentText()
        item.sample_ids = self._checked(self._sample_list) or [CURRENT]
        item.population_ids = self._checked(self._pop_list) or [ALL_EVENTS]
        item.normalize = self._norm_cb.isChecked()
        item.stagger = self._stagger_cb.isChecked()
        item.colormap = self._cmap_combo.currentText()
        item.palette = self._palette_combo.currentText()
        item.dot_size = self._dot_size.value()
        item.opacity = self._opacity.value()
        item.line_width = self._line_width.value()
        item.fill_alpha = self._fill_alpha.value()
        item.smoothing = self._smoothing.value()
        item.box = self._box_cb.isChecked()
        item.grid = self._grid_cb.isChecked()
        item.x_channel = self._x_combo.currentText()
        item.x_transform = self._x_scale.currentText()
        item.y_channel = self._y_combo.currentText()
        item.y_transform = self._y_scale.currentText()
        item.x_label = self._x_label.text().strip()
        item.y_label = self._y_label.text().strip()
        item.x_min, item.x_max = self._x_min.value(), self._x_max.value()
        item.y_min, item.y_max = self._y_min.value(), self._y_max.value()
        item.show_title = self._show_title_cb.isChecked()
        item.stat = self._stat_combo.currentText()
        item.stat_pct = self._stat_pct.value()
        item.stat_mark = self._stat_mark_cb.isChecked()
        item.show_count = self._count_cb.isChecked()
        item.stat_loc = self._stat_loc.currentText()
        item.legend = self._legend_combo.currentText()
        item.show_gates = self._gates_cb.isChecked()
        item.gate_names = self._gate_names_cb.isChecked()
        item.gate_pct = self._gate_pct_cb.isChecked()
        item.gate_lw = self._gate_lw.value()
        self._sync_enabled(item)
        self._refresh_layer_list(item)
        self._refresh_list_labels()
        self._layout_changed()

    def _add_blank(self):
        ref = self._current or (self._samples[0] if self._samples else None)
        if ref is None:
            return
        fl = ref.fluorescence or ref.channels
        self.add_item({"x_channel": fl[0],
                       "y_channel": fl[min(1, len(fl) - 1)]})

    def _duplicate(self):
        item = self._current_item()
        if item is not None:
            self.add_item(asdict(item))

    def _remove(self):
        r = self._list.currentRow()
        if 0 <= r < len(self._items):
            del self._items[r]
            self._list.takeItem(r)
            self._refresh_list_labels()
            self._layout_changed()

    def _move(self, step: int):
        r = self._list.currentRow()
        n = r + step
        if 0 <= r < len(self._items) and 0 <= n < len(self._items):
            self._items[r], self._items[n] = self._items[n], self._items[r]
            self._refresh_list_labels()
            self._list.setCurrentRow(n)
            self._layout_changed()

    def _layout_changed(self, *_args, emit: bool = True):
        self._dirty = True
        self._style_all.setEnabled(self._current_item() is not None
                                   and len(self._items) > 1)
        if self.isVisible():
            self._render_timer.start()
        if emit:
            self.changed.emit()

    def showEvent(self, event):
        super().showEvent(event)
        if self._dirty:
            self._render_timer.start()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _rc(self) -> dict:
        page = self._page()
        font = page["font"]
        family = (["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]
                  if font == FONTS[0] else [font, "DejaVu Sans"])
        return {"font.family": family}

    def render(self):
        self._dirty = False
        fig = self._fig
        fig.clear()
        page = self._page()
        pw, ph = page["panel_w"], page["panel_h"]
        n = len(self._items)
        cols = min(page["columns"], max(n, 1))
        rows = max(1, -(-n // cols))
        fig.set_size_inches(pw * cols, ph * rows)
        dpi = fig.get_dpi()
        self._canvas.setFixedSize(int(pw * cols * dpi), int(ph * rows * dpi))

        if n == 0 or not self._samples:
            fig.text(0.5, 0.5,
                     "Empty layout\n\nOn the Gating tab, set up a plot and "
                     "press '＋ Add to Layout',\nor press 'Add' here.",
                     ha="center", va="center", color="#888888", fontsize=11)
            self._canvas.draw_idle()
            return

        with matplotlib.rc_context(self._rc()):
            for i, item in enumerate(self._items):
                ax = fig.add_subplot(rows, cols, i + 1)
                try:
                    self._render_item(ax, item, page["font_size"])
                except Exception as exc:  # one bad panel shouldn't kill all
                    ax.cla()
                    ax.text(0.5, 0.5, f"Cannot draw:\n{exc}", ha="center",
                            va="center", color="#c0392b", fontsize=8,
                            transform=ax.transAxes)
            try:
                fig.tight_layout(pad=1.2)
            except Exception:
                pass
        self._canvas.draw()
        self._update_thumbnails()

    def _update_thumbnails(self):
        """Small picture of each plot in the list, cut from the page."""
        try:
            w, h = self._canvas.get_width_height(physical=True)
            buf = bytes(self._canvas.buffer_rgba())
        except Exception:
            return
        img = QImage(buf, w, h, QImage.Format.Format_RGBA8888).copy()
        ratio = w / max(self._canvas.get_width_height()[0], 1)
        for i, ax in enumerate(self._fig.axes[:self._list.count()]):
            bb = ax.get_tightbbox(self._canvas.get_renderer())
            if bb is None:
                continue
            x0, x1 = max(0, int(bb.x0 * ratio)), min(w, int(bb.x1 * ratio))
            y0 = max(0, int(h - bb.y1 * ratio))
            y1 = min(h, int(h - bb.y0 * ratio))
            if x1 - x0 < 4 or y1 - y0 < 4:
                continue
            thumb = img.copy(x0, y0, x1 - x0, y1 - y0).scaled(
                self._list.iconSize(), Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation)
            self._list.item(i).setIcon(QIcon(QPixmap.fromImage(thumb)))

    @staticmethod
    def _limits(ref, ch, tr, lo, hi):
        """Display-space axis limits, honouring raw-unit overrides."""
        auto = plotting.axis_limits(ref, ch, tr)

        def disp(v):
            d = float(apply_transform(np.array([v], dtype=float), tr)[0])
            return d if np.isfinite(d) else None

        a = disp(lo) if lo is not None else None
        b = disp(hi) if hi is not None else None
        a = auto[0] if a is None else a
        b = auto[1] if b is None else b
        return (a, b) if b > a else auto

    def _stats(self, item, lay, channels) -> list[float]:
        """The item's statistic of each channel over the whole population."""
        if item.stat == NO_STAT:
            return []
        s = lay.sample
        return [stat_1d(s.column(ch), item.stat, item.stat_pct, lay.mask)
                for ch in channels]

    def _render_item(self, ax, item: LayoutItem, fs: float = 8):
        layers = self._layers(item)
        if item.show_title:
            ax.set_title(item.title or self._auto_title(item),
                         fontsize=fs + 1, weight="bold")
        if not layers:
            ax.text(0.5, 0.5, "Population not found in the chosen "
                    "sample(s)", ha="center", va="center", color="#888888",
                    fontsize=fs, transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            return

        if item.plot_type == "Spectrum":
            return self._render_spectrum(ax, item, layers, fs)
        ref = layers[0].sample
        overlay = len(layers) > 1
        hist = item.plot_type == "Histogram"
        x_ch, x_tr = item.x_channel, item.x_transform
        y_ch = None if hist else item.y_channel
        y_tr = None if hist else item.y_transform
        for lay in layers:
            missing = [c for c in (x_ch, y_ch) if c and
                       c not in lay.sample.channels]
            if missing:
                raise ValueError(f"{lay.sample.name} has no "
                                 f"{', '.join(missing)}")
        xlim = self._limits(ref, x_ch, x_tr, item.x_min, item.x_max)
        ylim = None if hist else self._limits(ref, y_ch, y_tr, item.y_min,
                                              item.y_max)
        stagger = hist and overlay and item.stagger
        normalize = item.normalize or stagger
        colors = self._layer_colors(item, layers)
        alpha = item.opacity / 100.0
        channels = [x_ch] if hist else [x_ch, y_ch]
        stat_name = stat_label(item.stat, item.stat_pct)
        mark = item.stat_mark and item.stat in LOCATION_STATS
        stat_lines = []     # text for the single-population stats box

        for k, lay in enumerate(layers):
            s = lay.sample
            idx = plotting.population_indices(s.n_events, lay.mask,
                                              MAX_LAYOUT_EVENTS)
            keep = np.isfinite(s.column(x_ch)[idx])
            if not hist:
                keep &= np.isfinite(s.column(y_ch)[idx])
            idx = idx[keep]
            color = colors[k]
            vals = self._stats(item, lay, channels)
            n_pop = (int(lay.mask.sum()) if lay.mask is not None
                     else s.n_events)
            label = lay.label
            if overlay:
                extra = []
                if vals:
                    extra.append(", ".join(_fmt(v) for v in vals))
                if item.show_count:
                    extra.append(f"n={n_pop:,}")
                if extra:
                    label += " — " + " · ".join(extra)
            else:
                if vals:
                    stat_lines.append(stat_name)
                    stat_lines += [f"{ch}: {_fmt(v)}"
                                   for ch, v in zip(channels, vals)]
                if item.show_count:
                    stat_lines.append(f"n = {n_pop:,}")

            x = plotting.display_values(s, idx, x_ch, x_tr, xlim)
            disp = [float(apply_transform(np.array([v]), tr)[0])
                    for v, tr in zip(vals, (x_tr, y_tr))] if mark else []
            if hist:
                base = (len(layers) - 1 - k) * STAGGER_STEP if stagger else 0
                plotting.draw_histogram(
                    ax, x, xlim, color=color, label=label,
                    normalize=normalize,
                    filled=(not overlay or stagger) and item.fill_alpha > 0,
                    baseline=base, linewidth=item.line_width,
                    fill_alpha=item.fill_alpha / 100.0,
                    smoothing=item.smoothing)
                if disp and np.isfinite(disp[0]):
                    if stagger:
                        ax.plot([disp[0]] * 2, [base, base + 100],
                                color=color, ls="--", lw=1.1, zorder=6)
                    else:
                        ax.axvline(disp[0], color=color if overlay
                                   else "#0f172a", ls="--", lw=1.1, zorder=6)
                if stagger:
                    ax.text(xlim[1], base + 8, label + " ", color=color,
                            fontsize=fs - 1, weight="bold", va="bottom",
                            ha="right")
                continue
            y = plotting.display_values(s, idx, y_ch, y_tr, ylim)
            if overlay:
                plotting.draw_dots(ax, x, y, color=color, size=item.dot_size,
                                   alpha=alpha, label=label)
            elif item.plot_type == "Dot":
                plotting.draw_dots(ax, x, y, color=color, size=item.dot_size,
                                   alpha=alpha)
            else:
                plotting.draw_pseudocolor(
                    ax, x, y, xlim, ylim, size=item.dot_size,
                    cmap=COLORMAPS.get(item.colormap, "turbo"))
            if len(disp) == 2 and all(np.isfinite(disp)):
                # white halo so the cross shows on any colour
                for c, w in (("white", 4.0), (color if overlay else "#0f172a",
                                              1.8)):
                    ax.plot(*disp, marker="+", ms=14, mew=w, color=c,
                            zorder=7)

        plotting.style_axes(ax, x_ch, x_tr, xlim, y_ch, y_tr, ylim,
                            y_label=item.y_label or (
                                "% of Max" if hist and normalize
                                else "Count" if hist else None),
                            fontsize=fs, x_label=item.x_label or None,
                            grid=item.grid, box=item.box)
        if hist and not stagger and (item.y_min is not None
                                     or item.y_max is not None):
            lo, hi = ax.get_ylim()
            ax.set_ylim(item.y_min if item.y_min is not None else lo,
                        item.y_max if item.y_max is not None else hi)
        if stagger:
            ax.set_yticks([])
            ax.set_ylabel("")
            ax.spines["left"].set_visible(False)
            ax.set_ylim(0, (len(layers) - 1) * STAGGER_STEP + 110)
        elif overlay:
            self._legend(ax, item, fs, stat_name if item.stat != NO_STAT
                         else None)
        elif item.show_gates:
            lay = layers[0]
            children = (lay.gate.children if lay.gate
                        else lay.sample.gates)
            for g in children:
                if not isinstance(g, BooleanGate) and \
                        plotting.gate_fits_plot(g, x_ch, y_ch):
                    pct = plotting.percent_of(
                        lay.sample.gate_masks.get(g.gate_id), lay.mask,
                        lay.sample.n_events)
                    plotting.draw_gate(ax, g, x_tr, y_tr, pct, fontsize=fs - 1,
                                       show_name=item.gate_names,
                                       show_pct=item.gate_pct,
                                       linewidth=item.gate_lw)
        if stat_lines:
            self._stats_box(ax, stat_lines, item.stat_loc, fs)

    def _render_spectrum(self, ax, item: LayoutItem, layers, fs: float):
        """Full spectral profile. One population: density + median; several:
        their median spectra, optionally normalised to peak = 1."""
        ref = layers[0].sample
        dets = spectral_detectors(ref.channels, detector_kind(ref.channels))
        if not dets:
            raise ValueError(f"{ref.name} has no spectral detectors")
        chans = [d.channel for d in dets]
        for lay in layers:
            if any(c not in lay.sample.channels for c in chans):
                raise ValueError(f"{lay.sample.name} has other detectors")
        overlay = len(layers) > 1
        colors = self._layer_colors(item, layers)
        tr = item.y_transform
        norm = overlay and item.normalize
        medians = []
        for lay, c in zip(layers, colors):
            med = sample_stats(lay.sample, chans, "Median", mask=lay.mask)
            if norm:
                peak = np.nanmax(np.abs(med))
                med = med / peak if peak else med
            medians.append((med, c if overlay else "#0f172a",
                            lay.label if overlay else "Median"))
        if norm:
            tr = "Linear"
            ylim = (min(-0.05, float(np.nanmin([m.min() for m, *_ in
                                               medians])) - 0.05), 1.08)
        else:
            ylim = self._limits(ref, chans[0], tr, item.y_min, item.y_max)
            if item.y_min is None and item.y_max is None:
                ylim = plotting.spectrum_limits(ref, dets, tr)
        idx = plotting.population_indices(ref.n_events, layers[0].mask,
                                          MAX_LAYOUT_EVENTS)
        plotting.draw_spectrum(
            ax, ref, idx, dets, tr, ylim,
            cmap=COLORMAPS.get(item.colormap, "turbo"),
            density=not overlay, medians=medians, fontsize=fs - 1)
        if norm:
            ax.set_ylabel(item.y_label or "Normalised intensity",
                          fontsize=fs)
        elif item.y_label:
            ax.set_ylabel(item.y_label, fontsize=fs)
        if item.x_label:
            ax.set_xlabel(item.x_label, fontsize=fs)
        if item.grid:
            ax.grid(True, axis="y", color="#e2e8f0", linewidth=0.6)
        if overlay:
            self._legend(ax, item, fs, None, markerscale=1.0)
        if item.show_title:     # above the laser names
            ax.set_title(item.title or self._auto_title(item),
                         fontsize=fs + 1, weight="bold", pad=fs * 2.6)

    @staticmethod
    def _legend(ax, item: LayoutItem, fs: float, title: str | None,
                markerscale: float | None = None):
        if item.legend == "Hidden":
            return
        # legend dots about the same size whatever the plotted dot size
        kw = dict(fontsize=fs - 1, frameon=False,
                  markerscale=markerscale or
                  6 * np.sqrt(1.5 / max(item.dot_size, 0.1)),
                  title=title, title_fontsize=fs - 1)
        if item.legend == "Outside right":
            kw.update(loc="upper left", bbox_to_anchor=(1.02, 1.0),
                      borderaxespad=0)
        else:
            kw["loc"] = item.legend.lower()
        leg = ax.legend(**kw)
        for h in leg.legend_handles:
            h.set_alpha(1)

    @staticmethod
    def _stats_box(ax, lines: list[str], loc: str, fs: float):
        top = loc.startswith("Upper")
        right = loc.endswith("right")
        ax.text(0.97 if right else 0.03, 0.97 if top else 0.03,
                "\n".join(lines), transform=ax.transAxes,
                ha="right" if right else "left", va="top" if top else "bottom",
                fontsize=fs - 1, color="#0f172a", zorder=8, linespacing=1.3,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#cbd5e1",
                          lw=0.6, alpha=0.9))

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _export(self):
        if not self._items:
            QMessageBox.information(self, "Empty layout",
                                    "Add some plots to the layout first.")
            return
        page = self._page()
        path, flt = QFileDialog.getSaveFileName(
            self, "Export Layout", "layout.png",
            f"PNG image, {page['dpi']} dpi (*.png);;PDF document (*.pdf);;"
            "SVG vector (*.svg)")
        if not path:
            return
        ext = {"PNG": ".png", "PDF": ".pdf", "SVG": ".svg"}[flt.split()[0]]
        if not path.lower().endswith(ext):
            path += ext
        self.export_to(path)

    def export_to(self, path: str):
        page = self._page()
        self.render()
        try:
            with matplotlib.rc_context(self._rc()):
                if page["transparent"]:
                    self._fig.savefig(path, dpi=page["dpi"], transparent=True)
                else:
                    self._fig.savefig(path, dpi=page["dpi"],
                                      facecolor="white")
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))
