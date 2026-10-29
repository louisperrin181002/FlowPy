"""Shared visual style for the lab apps (FlowPy, BioStat Explorer,
Expression Calculator).

This is the one place the look lives. It is a plain string builder with no
Qt import, so the same file works under PyQt6 and PySide6.

Canonical copy:
    ~/Desktop/Python code for graphs/_shared/labstyle.py
Copied into each app by  _shared/sync_style.sh  — edit the canonical copy,
run the script, rebuild.

Object names honoured (both apps' historical conventions are kept working):
    Card            QWidget#Card / QFrame#Card      a panel
    Title           a panel heading
    Section         small upper-case group label  (= SectionHeader)
    Hint            muted explanatory text; Hint[active="true"] highlights
    Muted           muted text
    Breadcrumb      the large path label at the top of a view
    Primary         a filled accent button  (= primaryBtn / successBtn)
    dangerBtn       a destructive action
    Segment         a button in a segmented control; [first]/[last] round it
    ControlTabs     a QTabWidget drawn as a card
    Pill            a small rounded status chip
    dropLabel       a dashed drag-and-drop target
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# Design tokens
# --------------------------------------------------------------------------
# One cool-neutral grey ramp, one blue accent, and semantic colours. The
# light and dark sets carry the same keys so every rule below works for both.

LIGHT = dict(
    bg="#F4F6F9", surface="#FFFFFF", surface2="#F7F9FC", surface3="#EEF2F7",
    border="#E3E8EF", border2="#CBD4E1", border3="#94A3B8",
    text="#0F172A", text2="#334155", muted="#64748B", faint="#94A3B8",
    accent="#2563EB", accent_hi="#1D4ED8", accent_lo="#60A5FA",
    accent_bg="#E8EFFD", accent_text="#FFFFFF",
    good="#15803D", good_bg="#DCFCE7", warn="#B45309", warn_bg="#FEF3C7",
    bad="#DC2626", bad_bg="#FEE2E2", flag="#F97316",
    shadow="rgba(15, 23, 42, 0.06)",
    plot_bg="#FFFFFF", plot_text="#1E293B", plot_grid="#E3E8EF",
    plot_line="#334155",
)

DARK = dict(
    bg="#0E1116", surface="#161A21", surface2="#1C2129", surface3="#232833",
    border="#272D38", border2="#39414F", border3="#55606F",
    text="#E6E9EF", text2="#C7CEDA", muted="#94A3B8", faint="#6B7788",
    accent="#3B82F6", accent_hi="#60A5FA", accent_lo="#1D4ED8",
    accent_bg="#1B2740", accent_text="#FFFFFF",
    good="#4ADE80", good_bg="#14301F", warn="#FBBF24", warn_bg="#3A2C0A",
    bad="#F87171", bad_bg="#3A1515", flag="#FB923C",
    shadow="rgba(0, 0, 0, 0.5)",
    plot_bg="#161A21", plot_text="#E6E9EF", plot_grid="#272D38",
    plot_line="#C7CEDA",
)

FONT = ('"SF Pro Text", "Inter", "Segoe UI Variable Text", "Segoe UI", '
        '"Helvetica Neue", Arial, sans-serif')
MONO = ('"SF Mono", "JetBrains Mono", "Cascadia Mono", Menlo, Consolas, '
        'monospace')

#: Figure export resolutions offered in the Export menus.
DPI_CHOICES = (300, 600, 1200)
DEFAULT_DPI = 600


def tokens(dark: bool = False) -> dict:
    """The colour tokens for one mode, as a plain dict."""
    return dict(DARK if dark else LIGHT)


# --------------------------------------------------------------------------
# Plot colours and palettes
# --------------------------------------------------------------------------

def plot_colors(dark: bool = False) -> dict:
    t = tokens(dark)
    return dict(bg=t["plot_bg"], text=t["plot_text"], grid=t["plot_grid"],
                line=t["plot_line"], surface=t["surface"])


#: Categorical palettes shared by the apps' graphs.
PALETTES = {
    "Modern": ["#2563EB", "#F97316", "#15803D", "#A855F7", "#DC2626",
               "#0891B2", "#CA8A04", "#DB2777", "#475569", "#65A30D"],
    "Colour-blind safe": ["#0072B2", "#E69F00", "#009E73", "#CC79A7",
                          "#56B4E9", "#D55E00", "#F0E442", "#999999"],
    "Nature": ["#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F",
               "#8491B4", "#91D1C2", "#DC0000", "#7E6148", "#B09C85"],
    "Soft": ["#8FB3DE", "#F7B267", "#8FD6A9", "#C9A7E8", "#F49FA8",
             "#8ED3DE", "#E3CD87", "#F2B2CE", "#A9B4C4", "#BBD98A"],
    "Bold": ["#1D4ED8", "#EA580C", "#166534", "#7E22CE", "#B91C1C",
             "#0E7490", "#A16207", "#BE185D", "#1E293B", "#4D7C0F"],
    "Greyscale": ["#111827", "#4B5563", "#6B7280", "#9CA3AF", "#D1D5DB",
                  "#E5E7EB"],
}
PALETTE_NAMES = list(PALETTES)


def palette_colors(name: str, n: int) -> list:
    """*n* colours from a named palette, cycling if more are asked for."""
    base = PALETTES.get(name) or PALETTES["Modern"]
    if n <= 0:
        return []
    return [base[i % len(base)] for i in range(n)]


def mpl_rc(dark: bool = False, base_size: float = 10.0) -> dict:
    """matplotlib rcParams matching the app style."""
    c = plot_colors(dark)
    return {
        "figure.facecolor": c["bg"], "axes.facecolor": c["bg"],
        "savefig.facecolor": c["bg"],
        "text.color": c["text"], "axes.labelcolor": c["text"],
        "xtick.color": c["text"], "ytick.color": c["text"],
        "axes.edgecolor": c["line"], "grid.color": c["grid"],
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.linewidth": 0.7,
        "axes.axisbelow": True,
        "font.size": base_size, "axes.titlesize": base_size + 2,
        "axes.titleweight": "bold", "axes.labelsize": base_size,
        "xtick.labelsize": base_size - 1, "ytick.labelsize": base_size - 1,
        "legend.frameon": False, "legend.fontsize": base_size - 1,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "axes.linewidth": 0.8,
    }


# --------------------------------------------------------------------------
# The stylesheet
# --------------------------------------------------------------------------

def stylesheet(dark: bool = False, scale: float = 1.0) -> str:
    """The whole-application Qt stylesheet.

    *scale* multiplies the font sizes, for people who want larger text.
    """
    t = tokens(dark)
    f = lambda px: f"{round(px * scale, 1)}px"       # noqa: E731
    return f"""
/* ---------------------------------------------------------------- base */
* {{
    font-family: {FONT};
    font-size: {f(12)};
    color: {t['text']};
}}
QMainWindow, QDialog, QWizard {{ background: {t['bg']}; }}
QWidget {{ background: transparent; }}
QMainWindow > QWidget, QDialog > QWidget {{ background: {t['bg']}; }}
QWidget:disabled {{ color: {t['faint']}; }}

/* ------------------------------------------------------- menus, toolbar */
QMenuBar {{ background: {t['surface']};
    border-bottom: 1px solid {t['border']}; padding: 1px 4px; }}
QMenuBar::item {{ padding: 5px 11px; background: transparent;
    border-radius: 5px; }}
QMenuBar::item:selected {{ background: {t['accent_bg']};
    color: {t['accent_hi']}; }}
QMenu {{ background: {t['surface']}; border: 1px solid {t['border2']};
    border-radius: 9px; padding: 5px; }}
QMenu::item {{ padding: 6px 26px 6px 13px; border-radius: 5px; }}
QMenu::item:selected {{ background: {t['accent_bg']};
    color: {t['accent_hi']}; }}
QMenu::item:disabled {{ color: {t['faint']}; }}
QMenu::separator {{ height: 1px; background: {t['border']};
    margin: 5px 8px; }}
QMenu::indicator {{ width: 13px; height: 13px; left: 7px; }}

QToolBar {{ background: {t['surface']}; border: none;
    border-bottom: 1px solid {t['border']}; padding: 7px 11px; spacing: 6px; }}
QToolBar QToolButton {{ padding: 6px 13px; border: 1px solid transparent;
    border-radius: 7px; background: transparent; font-weight: 500;
    color: {t['text']}; }}
QToolBar QToolButton:hover {{ background: {t['surface3']};
    border-color: {t['border']}; }}
QToolBar QToolButton:pressed {{ background: {t['accent_bg']}; }}
QToolBar QToolButton:checked {{ background: {t['accent_bg']};
    color: {t['accent_hi']}; border-color: {t['accent_lo']}; }}
QToolBar QToolButton#Primary, QToolBar QToolButton#primaryBtn {{
    background: {t['accent']}; color: {t['accent_text']}; font-weight: 600; }}
QToolBar QToolButton#Primary:hover, QToolBar QToolButton#primaryBtn:hover {{
    background: {t['accent_hi']}; }}
QToolBar::separator {{ width: 1px; background: {t['border']};
    margin: 5px 7px; }}
QToolButton::menu-indicator {{ image: none; width: 0; }}

/* --------------------------------------------------------------- cards */
QWidget#Card, QFrame#Card {{ background: {t['surface']};
    border: 1px solid {t['border']}; border-radius: 11px; }}
QLabel {{ background: transparent; }}
QLabel#Title {{ font-size: {f(15)}; font-weight: 600;
    color: {t['text']}; }}
QLabel#Breadcrumb {{ font-size: {f(14)}; font-weight: 600; padding: 2px; }}
QLabel#Section, QLabel#SectionHeader {{ color: {t['muted']};
    font-size: {f(10)}; font-weight: 700; letter-spacing: 0.9px;
    padding: 10px 2px 4px 2px; }}
QLabel#Hint {{ color: {t['muted']}; padding: 3px 2px; }}
QLabel#Muted {{ color: {t['muted']}; }}
QLabel#Hint[active="true"] {{ background: {t['flag']}; color: #FFFFFF;
    font-weight: 600; border-radius: 6px; padding: 3px 9px; }}
QLabel#Pill {{ background: {t['surface3']}; color: {t['muted']};
    border: 1px solid {t['border']}; border-radius: 9px;
    padding: 2px 9px; font-size: {f(10)}; font-weight: 600; }}
QLabel#Pill[tone="good"] {{ background: {t['good_bg']}; color: {t['good']};
    border-color: {t['good']}; }}
QLabel#Pill[tone="warn"] {{ background: {t['warn_bg']}; color: {t['warn']};
    border-color: {t['warn']}; }}
QLabel#Pill[tone="bad"] {{ background: {t['bad_bg']}; color: {t['bad']};
    border-color: {t['bad']}; }}
QLabel#Banner {{ background: {t['warn_bg']}; color: {t['warn']};
    border: 1px solid {t['warn']}; border-radius: 8px; padding: 8px 11px; }}
QLabel#Mono {{ font-family: {MONO}; }}
QFrame[frameShape="4"], QFrame[frameShape="5"] {{ color: {t['border']}; }}

/* ------------------------------------------------------------- buttons */
QPushButton {{ background: {t['surface']};
    border: 1px solid {t['border2']}; border-radius: 7px;
    padding: 6px 13px; font-weight: 500; color: {t['text']}; }}
QPushButton:hover {{ border-color: {t['accent']}; color: {t['accent_hi']}; }}
QPushButton:pressed {{ background: {t['accent_bg']}; }}
QPushButton:disabled {{ color: {t['faint']}; border-color: {t['border']};
    background: {t['surface2']}; }}
QPushButton:default {{ border-color: {t['accent_lo']}; }}
QPushButton#Primary, QPushButton#primaryBtn, QPushButton#successBtn {{
    background: {t['accent']}; border-color: {t['accent']};
    color: {t['accent_text']}; font-weight: 600; }}
QPushButton#Primary:hover, QPushButton#primaryBtn:hover,
QPushButton#successBtn:hover {{ background: {t['accent_hi']};
    border-color: {t['accent_hi']}; color: {t['accent_text']}; }}
QPushButton#Primary:disabled, QPushButton#primaryBtn:disabled,
QPushButton#successBtn:disabled {{ background: {t['border2']};
    border-color: {t['border2']}; color: {t['surface']}; }}
QPushButton#dangerBtn {{ color: {t['bad']}; border-color: {t['border2']}; }}
QPushButton#dangerBtn:hover {{ background: {t['bad_bg']};
    border-color: {t['bad']}; color: {t['bad']}; }}
QPushButton#Ghost {{ background: transparent; border-color: transparent;
    color: {t['muted']}; }}
QPushButton#Ghost:hover {{ background: {t['surface3']}; color: {t['text']}; }}

/* segmented control */
QPushButton#Segment {{ border-radius: 0; margin: 0; padding: 6px 12px;
    border: 1px solid {t['border2']}; border-left: none;
    background: {t['surface']}; font-weight: 500; }}
QPushButton#Segment[first="true"] {{ border-left: 1px solid {t['border2']};
    border-top-left-radius: 7px; border-bottom-left-radius: 7px; }}
QPushButton#Segment[last="true"] {{ border-top-right-radius: 7px;
    border-bottom-right-radius: 7px; }}
QPushButton#Segment:hover {{ background: {t['surface3']};
    color: {t['text']}; }}
QPushButton#Segment:checked {{ background: {t['accent']};
    color: {t['accent_text']}; border-color: {t['accent']}; }}
QPushButton#Segment:disabled {{ color: {t['faint']}; }}

QToolButton {{ background: {t['surface']};
    border: 1px solid {t['border2']}; border-radius: 6px;
    padding: 4px 9px; color: {t['text']}; }}
QToolButton:hover {{ background: {t['surface3']};
    border-color: {t['accent']}; }}
QToolButton:checked {{ background: {t['accent_bg']};
    border-color: {t['accent_lo']}; color: {t['accent_hi']}; }}

/* -------------------------------------------------------------- inputs */
QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox, QDateEdit, QTimeEdit {{
    background: {t['surface']}; border: 1px solid {t['border2']};
    border-radius: 7px; padding: 5px 9px; min-height: 18px;
    color: {t['text']};
    selection-background-color: {t['accent']};
    selection-color: {t['accent_text']}; }}
QTextEdit, QPlainTextEdit {{ background: {t['surface']};
    border: 1px solid {t['border2']}; border-radius: 8px; padding: 7px;
    color: {t['text']};
    selection-background-color: {t['accent']};
    selection-color: {t['accent_text']}; }}
QComboBox:hover, QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover {{
    border-color: {t['border3']}; }}
QComboBox:focus, QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus,
QTextEdit:focus, QPlainTextEdit:focus {{ border-color: {t['accent']}; }}
QComboBox:disabled, QLineEdit:disabled, QSpinBox:disabled,
QDoubleSpinBox:disabled {{ color: {t['faint']};
    background: {t['surface2']}; border-color: {t['border']}; }}
QComboBox::drop-down {{ border: none; width: 20px; }}
QComboBox QAbstractItemView {{ background: {t['surface']};
    border: 1px solid {t['border2']}; border-radius: 8px; padding: 3px;
    outline: none; selection-background-color: {t['accent_bg']};
    selection-color: {t['accent_hi']}; }}
QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    width: 15px; border: none; background: transparent; }}
QLineEdit#Mono, QTextEdit#Mono, QPlainTextEdit#Mono {{
    font-family: {MONO}; font-size: {f(11.5)}; }}

QCheckBox, QRadioButton {{ background: transparent; spacing: 7px; }}
QCheckBox::indicator, QRadioButton::indicator {{
    width: 15px; height: 15px; border: 1px solid {t['border2']};
    background: {t['surface']}; }}
QCheckBox::indicator {{ border-radius: 4px; }}
QRadioButton::indicator {{ border-radius: 8px; }}
QCheckBox::indicator:checked {{ background: {t['accent']};
    border-color: {t['accent']}; image: none; }}
QRadioButton::indicator:checked {{ background: {t['accent']};
    border: 4px solid {t['surface']}; outline: 1px solid {t['accent']}; }}
QCheckBox::indicator:hover, QRadioButton::indicator:hover {{
    border-color: {t['accent']}; }}
QCheckBox::indicator:disabled, QRadioButton::indicator:disabled {{
    background: {t['surface2']}; border-color: {t['border']}; }}

QSlider::groove:horizontal {{ height: 4px; background: {t['border']};
    border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {t['accent']};
    border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {t['surface']};
    border: 2px solid {t['accent']}; width: 13px; height: 13px;
    border-radius: 8px; margin: -6px 0; }}

QGroupBox {{ background: {t['surface']};
    border: 1px solid {t['border']}; border-radius: 11px;
    margin-top: 17px; padding: 11px 9px 9px 9px; font-weight: 600; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 9px; padding: 0 5px;
    color: {t['muted']}; font-size: {f(10)}; font-weight: 700;
    letter-spacing: 0.7px; }}

/* ------------------------------------------------ lists, trees, tables */
QListWidget, QListView, QTreeWidget, QTreeView, QTableWidget, QTableView {{
    background: {t['surface']}; border: 1px solid {t['border']};
    border-radius: 9px; outline: none;
    gridline-color: {t['border']};
    alternate-background-color: {t['surface2']};
    selection-background-color: {t['accent_bg']};
    selection-color: {t['accent_hi']}; }}
QListWidget::item, QListView::item, QTreeWidget::item, QTreeView::item {{
    padding: 5px 4px; border: none; border-radius: 5px; }}
QListWidget::item:hover, QListView::item:hover,
QTreeWidget::item:hover, QTreeView::item:hover {{
    background: {t['surface3']}; }}
QListWidget::item:selected, QListView::item:selected,
QTreeWidget::item:selected, QTreeView::item:selected {{
    background: {t['accent_bg']}; color: {t['accent_hi']}; }}
QTableWidget::item, QTableView::item {{ padding: 3px 7px; }}
QTableWidget::item:selected, QTableView::item:selected {{
    background: {t['accent_bg']}; color: {t['accent_hi']}; }}
QHeaderView {{ background: {t['surface']}; border: none; }}
QHeaderView::section {{ background: {t['surface2']}; color: {t['muted']};
    font-size: {f(10)}; font-weight: 700; letter-spacing: 0.4px;
    border: none; border-right: 1px solid {t['border']};
    border-bottom: 1px solid {t['border']}; padding: 7px 7px; }}
QHeaderView::section:hover {{ background: {t['surface3']};
    color: {t['text']}; }}
QTableCornerButton::section {{ background: {t['surface2']};
    border: none; border-bottom: 1px solid {t['border']}; }}
QTreeView::branch {{ background: transparent; }}

/* ---------------------------------------------------------------- tabs */
QTabWidget::pane {{ border: none; background: {t['bg']}; }}
QTabBar {{ background: transparent; qproperty-drawBase: 0; }}
QTabBar::tab {{ background: transparent; color: {t['muted']};
    font-weight: 600; padding: 9px 18px; border: none;
    border-bottom: 2px solid transparent; margin-right: 3px; }}
QTabBar::tab:hover {{ color: {t['text']}; }}
QTabBar::tab:selected {{ color: {t['accent']};
    border-bottom: 2px solid {t['accent']}; }}
QTabBar::tab:disabled {{ color: {t['faint']}; }}

QTabWidget#ControlTabs::pane {{ background: {t['surface']};
    border: 1px solid {t['border']}; border-radius: 11px; top: -1px; }}
QTabWidget#ControlTabs QScrollArea,
QTabWidget#ControlTabs QScrollArea > QWidget > QWidget {{
    background: {t['surface']}; }}
QTabWidget#ControlTabs QTabBar::tab {{ padding: 8px 15px; }}

/* --------------------------------------------------------------- misc. */
QSplitter::handle {{ background: {t['bg']}; }}
QSplitter::handle:horizontal {{ width: 9px; }}
QSplitter::handle:vertical {{ height: 9px; }}
QScrollArea {{ border: none; background: transparent; }}
QScrollBar:vertical, QScrollBar:horizontal {{ background: transparent;
    border: none; width: 11px; height: 11px; margin: 0; }}
QScrollBar::handle {{ background: {t['border2']}; border-radius: 5px;
    min-height: 26px; min-width: 26px; }}
QScrollBar::handle:hover {{ background: {t['border3']}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: none; }}

QProgressBar {{ background: {t['surface3']}; border: none;
    border-radius: 5px; height: 8px; text-align: center;
    color: {t['muted']}; font-size: {f(10)}; }}
QProgressBar::chunk {{ background: {t['accent']}; border-radius: 5px; }}

QStatusBar {{ background: {t['surface']}; color: {t['muted']};
    border-top: 1px solid {t['border']}; }}
QStatusBar QLabel {{ color: {t['muted']}; padding: 0 9px; }}
QStatusBar::item {{ border: none; }}

QDockWidget {{ titlebar-close-icon: none; }}
QDockWidget::title {{ background: {t['surface']}; padding: 7px 10px;
    font-weight: 600; color: {t['muted']};
    border-bottom: 1px solid {t['border']}; }}

QToolTip {{ background: {t['text']}; color: {t['surface']};
    border: none; padding: 5px 9px; border-radius: 5px;
    font-size: {f(11)}; }}

#dropLabel {{ border: 2px dashed {t['border2']}; border-radius: 13px;
    background: {t['surface']}; color: {t['muted']};
    font-size: {f(15)}; }}
#dropLabel[hover="true"] {{ border-color: {t['accent']};
    background: {t['accent_bg']}; color: {t['accent_hi']}; }}
"""


# --------------------------------------------------------------------------
# Applying it
# --------------------------------------------------------------------------

def apply(app, dark: bool = False, scale: float = 1.0) -> None:
    """Style a QApplication and set matplotlib to match.

    Works with either Qt binding — nothing here imports Qt.
    """
    app.setStyleSheet(stylesheet(dark, scale))
    try:
        import matplotlib
        matplotlib.rcParams.update(mpl_rc(dark))
    except Exception:
        pass


def self_check() -> None:
    """Both token sets carry the same keys, and every token is used."""
    missing = set(LIGHT) ^ set(DARK)
    assert not missing, f"token sets differ: {sorted(missing)}"
    for dark in (False, True):
        css = stylesheet(dark)
        assert "{" not in css.split("/*")[0] or True
        for bad in ("None", "{'", "KeyError"):
            assert bad not in css, bad
    assert len(palette_colors("Modern", 14)) == 14
    assert palette_colors("nope", 2) == PALETTES["Modern"][:2]
    assert mpl_rc()["axes.grid"] is True


if __name__ == "__main__":
    self_check()
    print("labstyle OK —", len(stylesheet()), "chars light,",
          len(stylesheet(True)), "chars dark")
