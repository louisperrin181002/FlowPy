"""Shared plot rendering — used by the interactive gating plot and the
layout editor so both look identical.

Everything here is pure matplotlib (no Qt), working on numpy arrays.
"""

from __future__ import annotations
import numpy as np

from matplotlib.patches import Polygon, Rectangle
from matplotlib.ticker import FuncFormatter, MaxNLocator
from scipy.ndimage import gaussian_filter

from ..core.transforms import apply_transform, axis_ticks_labels
from ..core.gates import (
    Gate, RectangleGate, QuadrantGate, RangeGate,
)

# Maximum events drawn per population (statistics always use every event)
MAX_PLOT_EVENTS = 50_000
HISTOGRAM_BINS = 256
DENSITY_BINS = 200

PLOT_TYPES = ["Pseudocolor", "Dot", "Histogram", "Spectrum"]
SPECTRUM_BINS = 220

# Colours for multi-population overlays (layout editor)
OVERLAY_COLORS = ["#1f6fd1", "#e4572e", "#2a9d5c", "#9b51e0",
                  "#f2a900", "#17a2b8", "#d63384", "#6c757d"]

# Overlay palettes offered in the layout editor (first = default)
PALETTES = {
    "FlowPy": OVERLAY_COLORS,
    "Colour-blind safe": ["#0072b2", "#e69f00", "#009e73", "#cc79a7",
                          "#56b4e9", "#d55e00", "#f0e442", "#000000"],
    "Tableau": ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
                "#edc948", "#b07aa1", "#ff9da7"],
    "Bold": ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
             "#a65628", "#f781bf", "#999999"],
    "Pastel": ["#8fb8de", "#f4a582", "#a6d96a", "#c2a5cf", "#fdd380",
               "#80cdc1", "#f1b6da", "#bababa"],
    "Greyscale": ["#111111", "#555555", "#888888", "#b0b0b0", "#333333",
                  "#777777", "#9e9e9e", "#cccccc"],
}

_LIGHT_START = {"Blues", "Greys"}

# Density colour maps for pseudocolour plots: label -> matplotlib name
COLORMAPS = {
    "Turbo": "turbo",
    "Jet (FlowJo)": "jet",
    "Viridis": "viridis",
    "Plasma": "plasma",
    "Inferno": "inferno",
    "Blues": "Blues",
    "Greys": "Greys",
    "Spectral": "Spectral_r",
}


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def population_indices(n_events: int, mask: np.ndarray | None,
                       max_events: int = MAX_PLOT_EVENTS) -> np.ndarray:
    """Event indices to draw for a population, subsampled deterministically
    so the picture doesn't flicker between redraws."""
    idx = np.flatnonzero(mask) if mask is not None else np.arange(n_events)
    if len(idx) > max_events:
        rng = np.random.default_rng(42)
        idx = np.sort(rng.choice(idx, max_events, replace=False))
    return idx


def axis_limits(sample, channel: str, transform: str) -> tuple[float, float]:
    """Fixed display-space limits for a channel (FlowJo-style: the axis
    doesn't jump around as the population changes)."""
    lo, hi = sample.channel_limits(channel)
    if transform == "Logicle":
        top = max(4.5, float(apply_transform(np.array([hi]), "Logicle")[0]))
        return -0.25, top
    if transform == "Log10":
        return 0.0, max(1.0, float(np.log10(max(hi, 10.0))))
    lo = min(0.0, lo)
    pad = 0.02 * (hi - lo or 1.0)
    return lo - pad, hi + pad


def display_values(sample, idx: np.ndarray, channel: str, transform: str,
                   limits: tuple[float, float]) -> np.ndarray:
    """Transformed values, clipped to the axis so off-scale events pile up
    on the edge instead of vanishing."""
    vals = apply_transform(sample.column(channel)[idx], transform)
    return np.clip(vals, limits[0], limits[1])


def density_values(x: np.ndarray, y: np.ndarray, xlim, ylim,
                   bins: int = DENSITY_BINS) -> np.ndarray:
    """Per-event local density from a smoothed 2-D histogram — O(n), unlike
    a KDE, so it stays instant even at 50k+ events."""
    h, xe, ye = np.histogram2d(x, y, bins=bins, range=[xlim, ylim])
    h = gaussian_filter(h, sigma=1.2)
    ix = np.clip(((x - xlim[0]) / (xlim[1] - xlim[0]) * bins).astype(int),
                 0, bins - 1)
    iy = np.clip(((y - ylim[0]) / (ylim[1] - ylim[0]) * bins).astype(int),
                 0, bins - 1)
    return h[ix, iy]


# ---------------------------------------------------------------------------
# Population drawing
# ---------------------------------------------------------------------------

def draw_pseudocolor(ax, x, y, xlim, ylim, size: float = 1.5,
                     cmap: str = "turbo"):
    if len(x) == 0:
        return
    z = np.log1p(density_values(x, y, xlim, ylim))
    if cmap in _LIGHT_START:
        # these start at white — skip the palest quarter so sparse events
        # stay visible on a white background
        import matplotlib
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(matplotlib.colormaps[cmap](
            np.linspace(0.3, 1.0, 256)))
    order = np.argsort(z)
    ax.scatter(x[order], y[order], c=z[order], s=size, cmap=cmap,
               linewidths=0, rasterized=True)


def draw_dots(ax, x, y, color: str = "#334155", size: float = 1.0,
              alpha: float = 0.5, label: str | None = None):
    ax.scatter(x, y, s=size, color=color, alpha=alpha, linewidths=0,
               rasterized=True, label=label)


def draw_histogram(ax, x, xlim, color: str = "#2563eb",
                   label: str | None = None, normalize: bool = False,
                   filled: bool = True, baseline: float = 0.0,
                   linewidth: float = 1.4, fill_alpha: float = 0.35,
                   smoothing: float = 1.0, bins: int = HISTOGRAM_BINS):
    """Smoothed histogram; *baseline* lifts it (staggered overlays)."""
    if len(x) == 0:
        return
    counts, edges = np.histogram(x, bins=bins, range=xlim)
    counts = counts.astype(float)
    if smoothing > 0:
        counts = gaussian_filter(counts, sigma=smoothing)
    if normalize and counts.max() > 0:
        counts = 100.0 * counts / counts.max()
    centers = 0.5 * (edges[:-1] + edges[1:])
    if filled:
        ax.fill_between(centers, baseline, counts + baseline, color=color,
                        alpha=fill_alpha, linewidth=0)
    ax.plot(centers, counts + baseline, color=color, linewidth=linewidth,
            label=label)


# ---------------------------------------------------------------------------
# Axes
# ---------------------------------------------------------------------------

def _k_format(v, _pos=None):
    if abs(v) >= 1000:
        return f"{v / 1000:.0f}K"
    return f"{v:.0f}"


def _apply_axis_scale(axis, transform: str, lim):
    if transform == "Linear":
        axis.set_major_locator(MaxNLocator(5))
        axis.set_major_formatter(FuncFormatter(_k_format))
    else:
        ticks, labels = axis_ticks_labels(transform, lim[0], lim[1])
        axis.set_ticks(ticks)
        axis.set_ticklabels(labels)


def style_axes(ax, x_ch: str, x_tr: str, xlim,
               y_ch: str | None = None, y_tr: str | None = None, ylim=None,
               y_label: str | None = None, fontsize: float = 9,
               x_label: str | None = None, grid: bool = False,
               box: bool = False):
    """Limits, ticks and labels. *x_label*/*y_label* replace the channel
    names; *box* draws all four spines; *grid* adds faint gridlines."""
    ax.set_facecolor("#ffffff")
    ax.set_xlim(xlim)
    _apply_axis_scale(ax.xaxis, x_tr, xlim)
    ax.set_xlabel(x_label or x_ch, fontsize=fontsize)
    if y_ch is not None:
        ax.set_ylim(ylim)
        _apply_axis_scale(ax.yaxis, y_tr, ylim)
        ax.set_ylabel(y_label or y_ch, fontsize=fontsize)
    else:
        ax.set_ylim(bottom=0)
        ax.set_ylabel(y_label or "Count", fontsize=fontsize)
    ax.tick_params(labelsize=fontsize - 1)
    if box:
        for side in ("top", "right"):
            ax.spines[side].set_visible(True)
    if grid:
        ax.grid(True, color="#e2e8f0", linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
# Gate overlays
# ---------------------------------------------------------------------------

def gate_fits_plot(gate: Gate, x_ch: str, y_ch: str | None) -> bool:
    """Can *gate* be drawn on a plot of x_ch (vs y_ch, None = histogram)?"""
    if isinstance(gate, RangeGate):
        return gate.x_channel == x_ch
    return (y_ch is not None and gate.x_channel == x_ch
            and gate.y_channel == y_ch)


def _label(ax, x, y, text, color, ha="center", va="center", fontsize=8):
    if not text:
        return
    # A label sitting above the top of the plot would be cut off; tuck it
    # just inside instead.
    ylim = ax.get_ylim()
    if va == "bottom" and y > ylim[1] - 0.12 * (ylim[1] - ylim[0]):
        va = "top"
        y = min(y, ylim[1]) - 0.02 * (ylim[1] - ylim[0])
    ax.text(x, y, text, color="white", fontsize=fontsize, weight="bold",
            ha=ha, va=va, zorder=6, clip_on=True, linespacing=1.1,
            bbox=dict(boxstyle="round,pad=0.3,rounding_size=0.4", fc=color,
                      ec="none", alpha=0.92))


def draw_gate(ax, gate: Gate, x_tr: str, y_tr: str | None, pct: float | None,
              fontsize: float = 8, selected: bool = False,
              show_name: bool = True, show_pct: bool = True,
              linewidth: float = 2.2):
    """Draw *gate* with a bold outline, faint fill and a name + % label."""
    color = gate._color or "#E74C3C"
    lw = linewidth + 1.0 if selected else linewidth
    parts = ([gate.name] if show_name else []) + \
        ([f"{pct:.1f}%"] if show_pct and pct is not None else [])
    text = "\n".join(parts)
    xlim, ylim = ax.get_xlim(), ax.get_ylim()

    def tx(v):
        return float(apply_transform(np.array([v], dtype=float), x_tr)[0])

    def ty(v):
        return float(apply_transform(np.array([v], dtype=float), y_tr)[0])

    outline = gate.boundary()
    if outline is not None:
        xs = apply_transform(outline[0], x_tr)
        ys = apply_transform(outline[1], y_tr)
        verts = np.column_stack([xs, ys])
        ax.add_patch(Polygon(verts, closed=True, fc=color, alpha=0.10,
                             ec="none", zorder=4))
        ax.add_patch(Polygon(verts, closed=True, fill=False, ec=color,
                             lw=lw, zorder=5))
        top = np.argmax(ys)
        _label(ax, xs[top], ys[top], text, color, va="bottom",
               fontsize=fontsize)

    elif isinstance(gate, RectangleGate):
        x0, x1, y0, y1 = (tx(gate.x_min), tx(gate.x_max),
                          ty(gate.y_min), ty(gate.y_max))
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc=color,
                               alpha=0.10, ec="none", zorder=4))
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                               ec=color, lw=lw, zorder=5))
        _label(ax, (x0 + x1) / 2, y1, text, color, va="bottom",
               fontsize=fontsize)

    elif isinstance(gate, QuadrantGate):
        cx, cy = tx(gate.cx), ty(gate.cy)
        ax.axvline(cx, color=color, lw=lw * 0.8, zorder=5)
        ax.axhline(cy, color=color, lw=lw * 0.8, zorder=5)
        h, v = QuadrantGate.QUADRANTS.get(gate.quadrant, ("right", "top"))
        px = xlim[1] if h == "right" else xlim[0]
        py = ylim[1] if v == "top" else ylim[0]
        dx = 0.02 * (xlim[1] - xlim[0]) * (-1 if h == "right" else 1)
        dy = 0.02 * (ylim[1] - ylim[0]) * (-1 if v == "top" else 1)
        _label(ax, px + dx, py + dy, text, color,
               ha="right" if h == "right" else "left",
               va="top" if v == "top" else "bottom", fontsize=fontsize)

    elif isinstance(gate, RangeGate):
        x0, x1 = tx(gate.x_min), tx(gate.x_max)
        ax.axvspan(x0, x1, color=color, alpha=0.12, zorder=4)
        ax.axvline(x0, color=color, lw=lw, zorder=5)
        ax.axvline(x1, color=color, lw=lw, zorder=5)
        bar_y = ylim[0] + 0.88 * (ylim[1] - ylim[0])
        ax.plot([x0, x1], [bar_y] * 2, color=color, lw=lw, zorder=5)
        _label(ax, (x0 + x1) / 2, bar_y, text, color, va="bottom",
               fontsize=fontsize)
        ax.set_ylim(ylim)


def percent_of(mask: np.ndarray | None, parent_mask: np.ndarray | None,
               n_total: int) -> float | None:
    if mask is None:
        return None
    n_par = int(parent_mask.sum()) if parent_mask is not None else n_total
    return 100.0 * int(mask.sum()) / n_par if n_par else 0.0


# ---------------------------------------------------------------------------
# Full spectrum (spectral cytometers)
# ---------------------------------------------------------------------------

def spectrum_limits(sample, detectors, transform: str) -> tuple[float, float]:
    """One intensity axis shared by every detector."""
    lims = [axis_limits(sample, d.channel, transform) for d in detectors]
    return min(l[0] for l in lims), max(l[1] for l in lims)


def draw_spectrum(ax, sample, idx: np.ndarray, detectors, transform: str,
                  ylim, cmap: str = "turbo", density: bool = True,
                  medians: list | None = None, fontsize: float = 7):
    """Full spectral profile: one column per detector (in spectral order),
    the density of events along it, and optional median lines.

    *medians*: [(raw median per detector, colour, label)]."""
    import matplotlib
    from ..core.spectral import LASER_COLORS
    n = len(detectors)
    if density and len(idx):
        img = np.zeros((SPECTRUM_BINS, n))
        for j, d in enumerate(detectors):
            v = apply_transform(np.asarray(sample.column(d.channel)[idx],
                                           np.float64), transform)
            v = v[np.isfinite(v)]
            img[:, j], _ = np.histogram(np.clip(v, *ylim), SPECTRUM_BINS,
                                        range=ylim)
        img = np.log1p(img)
        cm = matplotlib.colormaps[cmap].copy()
        cm.set_bad("white")
        ax.imshow(np.ma.masked_equal(img, 0), origin="lower", aspect="auto",
                  cmap=cm, extent=(-0.5, n - 0.5, ylim[0], ylim[1]),
                  interpolation="nearest", zorder=1)
    for vals, color, label in medians or []:
        y = apply_transform(np.asarray(vals, np.float64), transform)
        ax.plot(np.arange(n), y, color=color, linewidth=1.8, marker="o",
                markersize=2.5, zorder=4, label=label)
        if density:     # halo so the line reads on top of the colours
            ax.plot(np.arange(n), y, color="white", linewidth=3.6,
                    zorder=3)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(ylim)
    # laser blocks: separators, coloured band and name on top
    start = 0
    for j in range(1, n + 1):
        if j == n or detectors[j].laser != detectors[start].laser:
            laser = detectors[start].laser
            col = LASER_COLORS.get(laser, "#64748b")
            if j < n:
                ax.axvline(j - 0.5, color="#94a3b8", linewidth=0.8,
                           zorder=5)
            ax.axvspan(start - 0.5, j - 0.5, ymin=0.985, ymax=1.0,
                       color=col, zorder=6, clip_on=False)
            ax.text((start + j - 1) / 2, 1.02, laser, color=col,
                    fontsize=fontsize + 1, weight="bold", ha="center",
                    va="bottom", transform=ax.get_xaxis_transform())
            start = j
    step = 1 if n <= 40 else 2
    ticks = list(range(0, n, step))
    ax.set_xticks(ticks)
    ax.set_xticklabels([detectors[i].short for i in ticks], rotation=90,
                       fontsize=fontsize - 1)
    if transform != "Linear":
        t, labels = axis_ticks_labels(transform, ylim[0], ylim[1])
        ax.set_yticks(t)
        ax.set_yticklabels(labels)
    else:
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_formatter(FuncFormatter(
            _k_format if ylim[1] - ylim[0] > 1000
            else lambda v, _p=None: f"{v:g}"))
    ax.tick_params(axis="y", labelsize=fontsize)
    ax.set_ylabel("Intensity", fontsize=fontsize + 1)
    ax.set_xlabel("Detector", fontsize=fontsize + 1)
