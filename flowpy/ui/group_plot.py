"""Compare Groups — plot one statistic per sample (e.g. the median of YG5 in
the gated cells) grouped by sample metadata: bars, boxes or points with
error bars, lines over a numeric variable (time, dose), optional colouring
by a second variable, a summary table and a t-test / ANOVA (or their
non-parametric equivalents)."""

from __future__ import annotations
from dataclasses import dataclass, asdict, fields
import numpy as np
import pandas as pd

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QComboBox,
    QDoubleSpinBox, QCheckBox, QLineEdit, QPushButton, QSplitter,
    QTableWidget, QTableWidgetItem, QHeaderView, QFileDialog, QMessageBox,
    QScrollArea, QFrame, QApplication, QCompleter,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from ..core import groups
from ..core.groups import NO_GROUP
from ..core.protocol import COUNT_STATS, population_path, _bfs
from ..core.statistics import STAT_NAMES
from .plotting import PALETTES
from .theme import export_dpi

PLOT_KINDS = ["Bar + points", "Box + points", "Points + mean",
              "Line (numeric X)"]
NONE = "None"


@dataclass
class GroupPlotSettings:
    population: str = "All Events"
    stat: str = "Median"
    channel: str = ""
    pct: float = 95.0
    group_col: str = "Group"
    hue_col: str = NONE
    kind: str = PLOT_KINDS[0]
    error: str = "SD"
    test: str = groups.TESTS[1]
    log_y: bool = False
    include_ungrouped: bool = False
    show_points: bool = True
    show_p: bool = True
    palette: str = "FlowPy"
    title: str = ""
    y_label: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "GroupPlotSettings":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (d or {}).items() if k in known})


# ---------------------------------------------------------------------------
# Drawing (pure matplotlib)
# ---------------------------------------------------------------------------

def _jitter(n: int, width: float, seed: int) -> np.ndarray:
    if n <= 1:
        return np.zeros(n)
    rng = np.random.default_rng(seed)
    return rng.uniform(-width, width, n)


def draw_group_plot(ax, df: pd.DataFrame, st: GroupPlotSettings,
                    y_label: str) -> list[str]:
    """Draw *df* (Sample, Group, [Hue], Value) on *ax*; returns lines
    describing the statistical tests."""
    hue = "Hue" in df.columns
    order = groups.group_order(list(df["Group"]))
    hues = groups.group_order(list(df["Hue"])) if hue else [None]
    pal = PALETTES.get(st.palette, PALETTES["FlowPy"])
    colors = {h: pal[i % len(pal)] for i, h in enumerate(hues)}
    numeric_x = st.kind == PLOT_KINDS[3] and groups.is_numeric(order)
    xpos = ({g: float(str(g).replace(",", ".")) for g in order}
            if numeric_x else {g: float(i) for i, g in enumerate(order)})
    nh = len(hues)
    span = (min(np.diff(sorted(xpos.values()))) if numeric_x and
            len(xpos) > 1 else 1.0)
    w = 0.8 * span / (nh if st.kind != PLOT_KINDS[3] else 1)
    test_lines = []

    vals_all = df["Value"].to_numpy(float)
    if st.log_y:
        df = df.assign(Value=np.where(vals_all > 0, vals_all, np.nan))

    for j, h in enumerate(hues):
        sub = df if h is None else df[df["Hue"] == h]
        c = colors[h]
        off = 0.0 if st.kind == PLOT_KINDS[3] else (j - (nh - 1) / 2) * w
        xs, means, errs, per_group = [], [], [], {}
        for g in order:
            v = sub.loc[sub["Group"] == g, "Value"].to_numpy(float)
            v = v[np.isfinite(v)]
            per_group[g] = v
            if not len(v):
                continue
            x = xpos[g] + off
            m = float(np.mean(v))
            e = groups.spread(v, st.error)
            xs.append(x)
            means.append(m)
            errs.append(e if np.isfinite(e) else 0.0)
            if st.kind == PLOT_KINDS[0]:
                ax.bar(x, m, width=w * 0.85, color=c, alpha=0.35,
                       edgecolor=c, linewidth=1.2, zorder=2)
            elif st.kind == PLOT_KINDS[1]:
                bp = ax.boxplot([v], positions=[x], widths=w * 0.7,
                                patch_artist=True, showfliers=False,
                                manage_ticks=False, zorder=2)
                for b in bp["boxes"]:
                    b.set(facecolor=c, alpha=0.35, edgecolor=c)
                for part in ("whiskers", "caps", "medians"):
                    for ln in bp[part]:
                        ln.set(color=c, linewidth=1.3)
            elif st.kind == PLOT_KINDS[2]:
                ax.plot([x - w * 0.3, x + w * 0.3], [m, m], color=c,
                        linewidth=2.2, zorder=4)
            if st.show_points and st.kind != PLOT_KINDS[3]:
                jit = _jitter(len(v), w * 0.18, seed=len(v) + j)
                ax.scatter(x + jit, v, s=28, color=c, edgecolor="white",
                           linewidth=0.7, zorder=5)
        if st.kind in (PLOT_KINDS[0], PLOT_KINDS[2]) and \
                st.error != "None" and xs:
            ax.errorbar(xs, means, yerr=errs, fmt="none", ecolor="#334155",
                        elinewidth=1.2, capsize=4, zorder=6)
        if st.kind == PLOT_KINDS[3] and xs:
            ax.errorbar(xs, means, yerr=errs if st.error != "None" else None,
                        color=c, marker="o", markersize=5, linewidth=1.8,
                        capsize=3, zorder=4)
            if st.show_points:
                for g in order:
                    v = per_group.get(g, np.array([]))
                    ax.scatter(np.full(len(v), xpos[g]), v, s=14, color=c,
                               alpha=0.45, linewidth=0, zorder=3)
        name, p = groups.compare(per_group, st.test)
        if name:
            n_groups = sum(len(v) >= 2 for v in per_group.values())
            what = (" vs ".join(g for g, v in per_group.items()
                                if len(v) >= 2) if n_groups == 2
                    else f"{n_groups} groups")
            prefix = f"{h}: " if h is not None else ""
            test_lines.append(f"{prefix}{what} — {name}, "
                              f"{groups.format_p(p)}")

    if numeric_x:
        ax.set_xlabel(st.group_col)
    else:
        ax.set_xticks([xpos[g] for g in order])
        long = max((len(str(g)) for g in order), default=0)
        ax.set_xticklabels(order, rotation=30 if long * len(order) > 60
                           else 0, ha="right" if long * len(order) > 60
                           else "center")
        ax.set_xlim(-0.6, len(order) - 0.4)
        ax.set_xlabel(st.group_col if st.group_col != "Sample" else "")
    ax.set_ylabel(st.y_label or y_label)
    if st.log_y:
        ax.set_yscale("log")
    elif st.kind == PLOT_KINDS[0]:
        ax.set_ylim(bottom=min(0, np.nanmin(vals_all)
                               if np.isfinite(vals_all).any() else 0))
    if hue:
        handles = [ax.scatter([], [], s=40, color=colors[h]) for h in hues]
        ax.legend(handles, hues, title=st.hue_col, frameon=False,
                  fontsize=8, title_fontsize=8, loc="upper left",
                  bbox_to_anchor=(1.01, 1.0))
    if st.title:
        ax.set_title(st.title, weight="bold")
    if st.show_p and test_lines:
        ax.text(0.01, 0.99, "\n".join(test_lines), transform=ax.transAxes,
                ha="left", va="top", fontsize=7.5, color="#334155",
                zorder=8, bbox=dict(boxstyle="round,pad=0.3", fc="white",
                                    ec="#cbd5e1", lw=0.6, alpha=0.9))
    return test_lines


# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------

def _search_combo() -> QComboBox:
    c = QComboBox()
    c.setEditable(True)
    c.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
    c.completer().setFilterMode(Qt.MatchFlag.MatchContains)
    c.completer().setCompletionMode(
        QCompleter.CompletionMode.PopupCompletion)
    return c


class GroupPlotWidget(QWidget):
    """The 'Compare Groups' tab."""

    changed = pyqtSignal()               # settings edited (autosave)
    edit_metadata = pyqtSignal()         # "Edit Sample Metadata…" pressed

    def __init__(self, parent=None):
        super().__init__(parent)
        self._samples: list = []
        self._meta_cols: list[str] = []
        self._st = GroupPlotSettings()
        self._df = pd.DataFrame()
        self._summary = pd.DataFrame()
        self._dirty = True
        self._ready = False              # ignore signals while building
        self._group_chosen = False       # user picked "Group by" themself
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(150)
        self._timer.timeout.connect(self.render)
        self._build_ui()

    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        split = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(split)

        panel = QWidget()
        form = QFormLayout(panel)
        self._w = {}

        def add(key, widget, label):
            self._w[key] = widget
            if isinstance(widget, QComboBox):
                widget.currentTextChanged.connect(self._form_changed)
            elif isinstance(widget, QCheckBox):
                widget.toggled.connect(self._form_changed)
            elif isinstance(widget, QDoubleSpinBox):
                widget.valueChanged.connect(self._form_changed)
            elif isinstance(widget, QLineEdit):
                widget.textEdited.connect(self._form_changed)
            if label is None:
                form.addRow(widget)
            else:
                form.addRow(label, widget)
            return widget

        hdr = QLabel("WHAT TO COMPARE")
        hdr.setObjectName("SectionHeader")
        form.addRow(hdr)
        add("population", _search_combo(), "Population:")
        st = add("stat", QComboBox(), "Statistic:")
        st.addItems(COUNT_STATS + STAT_NAMES)
        add("channel", _search_combo(), "Channel:")
        pct = add("pct", QDoubleSpinBox(), "Percentile:")
        pct.setRange(0, 100)
        pct.setDecimals(1)
        pct.setPrefix("P")

        hdr = QLabel("GROUPS")
        hdr.setObjectName("SectionHeader")
        form.addRow(hdr)
        add("group_col", QComboBox(), "Group by:")
        add("hue_col", QComboBox(), "Colour by:")
        add("include_ungrouped",
            QCheckBox("Include samples without a group"), None)
        meta_btn = QPushButton("Edit Sample Metadata…")
        meta_btn.clicked.connect(self.edit_metadata.emit)
        form.addRow(meta_btn)

        hdr = QLabel("PLOT")
        hdr.setObjectName("SectionHeader")
        form.addRow(hdr)
        add("kind", QComboBox(), "Type:").addItems(PLOT_KINDS)
        add("error", QComboBox(), "Error bars:").addItems(groups.ERRORS)
        add("test", QComboBox(), "Test:").addItems(groups.TESTS)
        add("show_p", QCheckBox("Show test result on plot"), None)
        add("show_points", QCheckBox("Show each sample as a dot"), None)
        add("log_y", QCheckBox("Log Y axis"), None)
        add("palette", QComboBox(), "Colours:").addItems(PALETTES)
        add("title", QLineEdit(), "Title:")
        add("y_label", QLineEdit(), "Y label:").setPlaceholderText(
            "(automatic)")

        row = QHBoxLayout()
        b = QPushButton("Export Figure…")
        b.setObjectName("Primary")
        b.clicked.connect(self._export_figure)
        row.addWidget(b)
        b = QPushButton("Export Values…")
        b.setToolTip("One row per sample (with all metadata) plus the group "
                     "summary, as CSV")
        b.clicked.connect(self._export_values)
        row.addWidget(b)
        b = QPushButton("Copy")
        b.setToolTip("Copy the per-sample values (paste into Excel/Prism)")
        b.clicked.connect(self._copy_values)
        row.addWidget(b)
        form.addRow(row)

        scroll = QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setMinimumWidth(340)
        scroll.setMaximumWidth(600)
        split.addWidget(scroll)

        right = QSplitter(Qt.Orientation.Vertical)
        self._fig = Figure(figsize=(6, 4), dpi=100)
        self._canvas = FigureCanvas(self._fig)
        right.addWidget(self._canvas)
        bottom = QWidget()
        bv = QVBoxLayout(bottom)
        bv.setContentsMargins(0, 4, 0, 0)
        self._test_lbl = QLabel("")
        self._test_lbl.setWordWrap(True)
        bv.addWidget(self._test_lbl)
        self._table = QTableWidget()
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.verticalHeader().setVisible(False)
        self._table.setShowGrid(False)
        bv.addWidget(self._table)
        right.addWidget(bottom)
        right.setSizes([520, 200])
        split.addWidget(right)
        split.setStretchFactor(1, 1)
        self._ready = True
        self._load_form()

    # ------------------------------------------------------------------
    # Context / settings
    # ------------------------------------------------------------------

    def sample_loaded(self, sample):
        if sample in self._samples:
            self._schedule()

    def set_context(self, samples: list, meta_columns: list[str]):
        self._samples = list(samples)
        self._meta_cols = list(meta_columns)
        self._load_form()
        self._schedule()

    def to_dict(self) -> dict:
        return asdict(self._st)

    def load_dict(self, d: dict):
        self._st = GroupPlotSettings.from_dict(d)
        self._group_chosen = bool(d)
        self._load_form()
        self._schedule()

    def _populations(self) -> list[str]:
        out = ["All Events"]
        for s in self._samples:
            for g in _bfs(s.gates):
                p = population_path(g)
                if p not in out:
                    out.append(p)
        return out

    def _channels(self) -> list[str]:
        out = []
        for s in self._samples:
            for c in s.fluorescence + [c for c in s.channels
                                       if c not in s.fluorescence]:
                if c not in out:
                    out.append(c)
        return out

    def _load_form(self):
        st, w = self._st, self._w
        for x in w.values():
            x.blockSignals(True)

        def fill(combo, items, current):
            combo.clear()
            combo.addItems(items)
            if current in items:
                combo.setCurrentText(current)
            elif items:
                combo.setCurrentIndex(0)

        pops = self._populations()
        fill(w["population"], pops, st.population if st.population in pops
             else pops[-1])
        chans = self._channels()
        fill(w["channel"], chans, st.channel)
        w["stat"].setCurrentText(st.stat)
        w["pct"].setValue(st.pct)
        group_opts = self._meta_cols + ["Sample"]
        if (st.group_col not in self._meta_cols and self._meta_cols
                and not self._group_chosen):
            st.group_col = self._meta_cols[0]   # metadata arrived later
        fill(w["group_col"], group_opts, st.group_col)
        fill(w["hue_col"], [NONE] + self._meta_cols, st.hue_col)
        for key in ("kind", "error", "test", "palette"):
            w[key].setCurrentText(getattr(st, key))
        for key in ("include_ungrouped", "show_points", "show_p", "log_y"):
            w[key].setChecked(getattr(st, key))
        w["title"].setText(st.title)
        w["y_label"].setText(st.y_label)
        for x in w.values():
            x.blockSignals(False)
        self._read_form()
        self._sync_enabled()

    def _read_form(self):
        st, w = self._st, self._w
        st.population = w["population"].currentText()
        st.stat = w["stat"].currentText()
        st.channel = w["channel"].currentText()
        st.pct = w["pct"].value()
        st.group_col = w["group_col"].currentText() or "Sample"
        st.hue_col = w["hue_col"].currentText() or NONE
        for key in ("kind", "error", "test", "palette"):
            setattr(st, key, w[key].currentText())
        for key in ("include_ungrouped", "show_points", "show_p", "log_y"):
            setattr(st, key, w[key].isChecked())
        st.title = w["title"].text().strip()
        st.y_label = w["y_label"].text().strip()

    def _sync_enabled(self):
        st, w = self._st, self._w
        w["channel"].setEnabled(st.stat not in COUNT_STATS)
        w["pct"].setEnabled(st.stat == "Percentile")
        w["error"].setEnabled(st.kind != PLOT_KINDS[1])

    def _form_changed(self, *_):
        if not self._ready:
            return
        if self.sender() is self._w.get("group_col"):
            self._group_chosen = True
        self._read_form()
        self._sync_enabled()
        self._schedule()
        self.changed.emit()

    def _schedule(self):
        self._dirty = True
        if self.isVisible():
            self._timer.start()

    def showEvent(self, event):
        super().showEvent(event)
        if self._dirty:
            self._timer.start()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _data(self) -> pd.DataFrame:
        # (samples that can't be read are left out)
        samples = [x for x in self._samples
                   if not getattr(x, "load_error", None)]
        st = self._st
        hue = st.hue_col if st.hue_col not in (NONE, "", st.group_col) \
            else None
        return groups.group_table(
            samples, st.group_col, hue, st.population,
            st.channel if st.stat not in COUNT_STATS else None, st.stat,
            st.pct, st.include_ungrouped)

    def render(self):
        self._dirty = False
        fig = self._fig
        fig.clear()
        ax = fig.add_subplot(111)
        st = self._st
        msg = None
        df = pd.DataFrame()
        unread = [x for x in self._samples
                  if not getattr(x, "is_loaded", True) and not x.load_error]
        if not self._samples:
            msg = "Open some FCS files first."
        elif unread:
            n = len(self._samples)
            msg = (f"Reading samples in the background… {n - len(unread)} "
                   f"of {n} ready.\n\nThe chart appears when all are read.")
        elif st.stat not in COUNT_STATS and not st.channel:
            msg = "Choose a channel."
        else:
            try:
                df = self._data()
            except Exception as exc:
                msg = f"Cannot compute: {exc}"
            if msg is None and df.empty:
                msg = ("No sample has a value in '"
                       f"{st.group_col}'.\n\nGive your samples groups with "
                       "'Edit Sample Metadata…'\n(or tick 'Include samples "
                       "without a group').")
            elif msg is None and not np.isfinite(
                    df["Value"].to_numpy(float)).any():
                msg = ("No values — is the population gated in these "
                       "samples?\n(Gating → Apply Gates to Samples)")
        self._df = df
        tests = []
        if msg:
            ax.axis("off")
            ax.text(0.5, 0.5, msg, ha="center", va="center",
                    color="#64748b", fontsize=10, transform=ax.transAxes)
            self._summary = pd.DataFrame()
        else:
            label = groups.value_label(st.population, st.channel, st.stat,
                                       st.pct)
            tests = draw_group_plot(ax, df, st, label)
            by = (["Hue", "Group"] if "Hue" in df.columns else ["Group"])
            self._summary = groups.summarize(df, by).rename(
                columns={"Hue": st.hue_col, "Group": st.group_col})
            missing = int((~np.isfinite(df["Value"].to_numpy(float))).sum())
            if missing:
                tests.append(f"{missing} sample(s) have no value (population "
                             "not gated or channel missing) and are left out.")
        try:
            fig.tight_layout()
        except Exception:
            pass
        self._canvas.draw_idle()
        self._test_lbl.setText("<br>".join(tests))
        self._fill_table(self._summary)

    def _fill_table(self, df: pd.DataFrame):
        t = self._table
        t.clear()
        t.setRowCount(len(df))
        t.setColumnCount(len(df.columns))
        t.setHorizontalHeaderLabels([str(c) for c in df.columns])
        for r, row in enumerate(df.itertuples(index=False)):
            for c, val in enumerate(row):
                if isinstance(val, str):
                    text = val
                elif isinstance(val, (int, np.integer)):
                    text = f"{val:,}"
                elif pd.isna(val):
                    text = "—"
                else:
                    text = f"{val:,.0f}" if abs(val) >= 1000 else f"{val:.4g}"
                it = QTableWidgetItem(text)
                if not isinstance(val, str):
                    it.setTextAlignment(Qt.AlignmentFlag.AlignRight |
                                        Qt.AlignmentFlag.AlignVCenter)
                t.setItem(r, c, it)
        t.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _values_frame(self) -> pd.DataFrame:
        df = self._df.copy()
        if df.empty:
            return df
        st = self._st
        df = df.rename(columns={"Group": st.group_col, "Hue": st.hue_col,
                                "Value": groups.value_label(
                                    st.population, st.channel, st.stat,
                                    st.pct)})
        by_name = {s.name: s for s in self._samples}
        for c in self._meta_cols:
            if c not in df.columns:
                df[c] = [by_name[n].annotations.get(c, "")
                         if n in by_name else "" for n in df["Sample"]]
        return df

    def export_figure(self, path: str, dpi: int | None = None):
        dpi = dpi or export_dpi()
        self.render()
        self._fig.savefig(path, dpi=dpi, facecolor="white",
                          bbox_inches="tight")

    def _export_figure(self):
        if self._df.empty:
            QMessageBox.information(self, "Compare Groups",
                                    "Nothing to export yet.")
            return
        path, flt = QFileDialog.getSaveFileName(
            self, "Export Figure", "groups.png",
            f"PNG image, {export_dpi()} dpi (*.png);;PDF document (*.pdf);;"
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

    def _export_values(self):
        df = self._values_frame()
        if df.empty:
            QMessageBox.information(self, "Compare Groups",
                                    "Nothing to export yet.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Values", "group_values.csv", "CSV (*.csv)")
        if not path:
            return
        if not path.lower().endswith(".csv"):
            path += ".csv"
        try:
            with open(path, "w", encoding="utf-8-sig", newline="") as f:
                df.to_csv(f, index=False)
                f.write("\n")
                self._summary.to_csv(f, index=False)
                if self._test_lbl.text():
                    f.write("\n" + self._test_lbl.text().replace(
                        "<br>", "\n") + "\n")
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))

    def _copy_values(self):
        df = self._values_frame()
        if not df.empty:
            QApplication.clipboard().setText(df.to_csv(sep="\t",
                                                       index=False))
