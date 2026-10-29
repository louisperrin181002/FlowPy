"""Visual theme for FlowPy.

The look itself lives in ``flowpy/ui/labstyle.py``, shared with BioStat
Explorer and the Expression Calculator so the three apps match. Edit the
canonical copy at ``_shared/labstyle.py`` and run
``_shared/sync_style.sh``; this module re-exports it under the names
FlowPy already used, and keeps FlowPy's own export settings.
"""

from __future__ import annotations
import matplotlib

from . import labstyle

# --- Design tokens, from the shared palette -------------------------------
_T = labstyle.LIGHT

BG        = _T["bg"]          # window background
SURFACE   = _T["surface"]     # cards, inputs, tables
SURFACE_2 = _T["surface2"]    # subtle fills (headers, hover)
BORDER    = _T["border"]
BORDER_2  = _T["border2"]     # stronger borders (inputs on hover)
TEXT      = _T["text"]
MUTED     = _T["muted"]
ACCENT    = _T["accent"]
ACCENT_HI = _T["accent_hi"]
ACCENT_BG = _T["accent_bg"]   # selection / checked tint
WARN      = _T["flag"]        # drawing-mode highlight

FONT = labstyle.FONT
MONO = labstyle.MONO

STYLESHEET = labstyle.stylesheet(False)


# Resolution of exported PNG figures (Export menu → Resolution)
DPI_CHOICES = (300, 600, 1200)


def export_dpi() -> int:
    from PyQt6.QtCore import QSettings
    try:
        v = int(QSettings("FlowPy", "FlowPy").value("export/dpi", 600))
    except (TypeError, ValueError):
        v = 600
    return v if v in DPI_CHOICES else 600


def set_export_dpi(v: int):
    from PyQt6.QtCore import QSettings
    QSettings("FlowPy", "FlowPy").setValue("export/dpi", int(v))


def apply_mpl_style():
    """Matplotlib defaults matching the UI (and clean for publication)."""
    matplotlib.rcParams.update({
        "font.family": ["Helvetica Neue", "Helvetica", "Arial",
                        "DejaVu Sans"],
        "font.size": 9,
        "axes.edgecolor": "#94a3b8",
        "axes.labelcolor": TEXT,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.titlecolor": TEXT,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.9,
        "xtick.color": "#475569",
        "ytick.color": "#475569",
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        # Exported SVG / PDF keep text as real, editable text (Inkscape,
        # Illustrator) instead of outlines
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
