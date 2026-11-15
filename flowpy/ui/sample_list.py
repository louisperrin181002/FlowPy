"""Sample list — searchable, grouped by a metadata column, with the number
of gates as a small badge and a dot while a sample's events are still
being read (grey ring: waiting, amber: reading, red: can't be read)."""

# ---------------------------------------------------------------------------
# HOW THIS LIST WORKS
# ---------------------------------------------------------------------------
# One flat QListWidget doing the work of a tree. Group headers are
# ordinary rows flagged with the _HEADER role and made unselectable, so
# grouping costs no extra widget and filtering is a single pass over one
# list rather than a walk over a tree.
#
# Everything a row needs is stored on it in custom item roles, because
# the display text is only the sample's name:
#
#   _SAMPLE          the Sample object itself
#   _BADGE           "3 gates", drawn right-aligned and muted
#   _HEADER          True on a group header row
#   UserRole + 4     True if the row sits under a header (indent it)
#   UserRole + 5     which group it belongs to (used by the filter)
#
# _Delegate paints all of that by hand. A delegate is used rather than
# rich text in the item because the status dot has to be drawn, and
# because eliding must be switched off — a truncated FCS file name is
# unrecognisable, so the list scrolls sideways instead.
#
# The status dot is the only part that changes without the list being
# rebuilt: refresh_states() just repaints, since the loader is reading
# in the background and rebuilding on every finished file would fight
# with the user's scrolling.
# ---------------------------------------------------------------------------

from __future__ import annotations

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLineEdit, QListWidget, QListWidgetItem,
    QStyledItemDelegate, QStyle, QAbstractItemView,
)
from PyQt6.QtCore import Qt, QSize, QRect, pyqtSignal
from PyQt6.QtGui import QFont, QColor, QPen

from . import theme
from ..core.protocol import _bfs

# Custom item roles — see the map above. Numbered from UserRole, which
# is the first value Qt guarantees it will never use itself.
_SAMPLE = Qt.ItemDataRole.UserRole + 1      # the Sample object
_BADGE = Qt.ItemDataRole.UserRole + 2       # text drawn on the right
_HEADER = Qt.ItemDataRole.UserRole + 3
#: Pixels reserved at the left of every row for the status dot. Reserved
#: whether or not a dot is drawn, so names do not shift sideways as
#: samples finish loading.
_DOT = 12
STATES = {"waiting": "Waiting to be read", "reading": "Reading…",
          "error": "Can't be read"}


def _state(sample, busy) -> str:
    """'ready', 'waiting', 'reading' or 'error'.

    *busy* is a callback into the loader rather than a flag on the
    sample, because only the loader knows which file it is reading right
    now. getattr with defaults throughout, so this also works on a
    stand-in object that has no loading machinery at all.
    """
    if getattr(sample, "load_error", None):
        return "error"
    if getattr(sample, "is_loaded", True):
        return "ready"
    return "reading" if busy(sample) else "waiting"


class _Delegate(QStyledItemDelegate):
    """Name on the left, muted badge on the right; headers in small caps.
    Never elides: the list scrolls sideways if a name is very long."""

    PAD = 8

    def __init__(self, parent=None):
        super().__init__(parent)
        self.busy = lambda s: False

    def sizeHint(self, option, index):
        """How wide and tall each row wants to be.

        The width is computed honestly from the text rather than left to
        the view, because eliding is off: the horizontal scrollbar can
        only know how far to scroll if the rows report their true width.
        """
        fm = option.fontMetrics
        text = index.data(Qt.ItemDataRole.DisplayRole) or ""
        badge = index.data(_BADGE) or ""
        w = fm.horizontalAdvance(text) + 2 * self.PAD + 16 + _DOT
        if badge:
            w += fm.horizontalAdvance(badge) + 18
        h = fm.height() + (12 if not index.data(_HEADER) else 14)
        return QSize(w, h)

    def paint(self, painter, option, index):
        """Draw one row: optional dot, name, optional right-hand badge.

        save() and restore() bracket everything, because the painter is
        shared between rows and any pen, brush or font left behind would
        leak into the next one.
        """
        header = index.data(_HEADER)
        text = index.data(Qt.ItemDataRole.DisplayRole) or ""
        badge = index.data(_BADGE) or ""
        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        painter.save()
        r = option.rect
        # Backgrounds painted by hand rather than by the style, so a
        # header never looks selected or hovered — it is not clickable.
        if selected and not header:
            painter.fillRect(r, QColor(theme.ACCENT_BG))
        elif option.state & QStyle.StateFlag.State_MouseOver and not header:
            painter.fillRect(r, QColor(theme.SURFACE_2))
        f = QFont(option.font)
        if header:
            # Small caps with a little extra letter-spacing: the
            # conventional way to make a label read as a heading without
            # making it larger than the content under it.
            f.setBold(True)
            f.setPointSizeF(max(f.pointSizeF() - 1.5, 8))
            f.setCapitalization(QFont.Capitalization.AllUppercase)
            f.setLetterSpacing(QFont.SpacingType.PercentageSpacing, 106)
            painter.setFont(f)
            painter.setPen(QColor(theme.MUTED))
            painter.drawText(r.adjusted(self.PAD, 4, -self.PAD, 0),
                             Qt.AlignmentFlag.AlignLeft |
                             Qt.AlignmentFlag.AlignVCenter, text)
            painter.restore()
            return
        # Rows under a header are indented; an ungrouped list is not.
        indent = 10 if index.data(Qt.ItemDataRole.UserRole + 4) else 0
        state = _state(index.data(_SAMPLE), self.busy)
        # No dot at all once a sample is ready — a list of green ticks
        # would be noise, since ready is the normal state.
        if state != "ready":
            painter.setRenderHint(painter.RenderHint.Antialiasing)
            cx, cy = r.left() + self.PAD + indent + 4, r.center().y()
            col = {"waiting": theme.MUTED, "reading": "#f59e0b",
                   "error": "#dc2626"}[state]
            # Waiting is drawn as an outline (pen, no brush), reading
            # and error as a filled disc (brush, no pen) — so the three
            # states differ in shape as well as colour and stay legible
            # to a colour-blind reader.
            if state == "waiting":
                painter.setPen(QPen(QColor(col), 1.3))
                painter.setBrush(Qt.BrushStyle.NoBrush)
            else:
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor(col))
            painter.drawEllipse(cx - 3, cy - 3, 7, 7)
        indent += _DOT
        painter.setFont(f)
        painter.setPen(QColor(theme.ACCENT if selected else
                              theme.TEXT if state == "ready" else
                              theme.MUTED))
        # The name's rectangle stops short of the badge, so a long name
        # is clipped by the badge rather than drawn underneath it.
        fm = painter.fontMetrics()
        bw = fm.horizontalAdvance(badge) + 14 if badge else 0
        painter.drawText(QRect(r.left() + self.PAD + indent, r.top(),
                               r.width() - 2 * self.PAD - bw - indent,
                               r.height()),
                         Qt.AlignmentFlag.AlignLeft |
                         Qt.AlignmentFlag.AlignVCenter, text)
        if badge:
            bf = QFont(f)
            bf.setPointSizeF(max(f.pointSizeF() - 1.5, 8))
            painter.setFont(bf)
            painter.setPen(QColor(theme.MUTED))
            painter.drawText(QRect(r.right() - self.PAD - bw, r.top(), bw,
                                   r.height()),
                             Qt.AlignmentFlag.AlignRight |
                             Qt.AlignmentFlag.AlignVCenter, badge)
        painter.restore()


class SampleList(QWidget):
    """Loaded samples. Emits sample_selected(sample)."""

    sample_selected = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._samples: list = []
        self._group_col: str | None = None
        self._current = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        self._search = QLineEdit()
        self._search.setPlaceholderText("Search samples or groups…")
        self._search.setClearButtonEnabled(True)
        self._search.textChanged.connect(self._apply_filter)
        lay.addWidget(self._search)
        self._list = QListWidget()
        self._delegate = _Delegate(self._list)
        self._list.setItemDelegate(self._delegate)
        # No eliding, and scroll by pixel rather than by item: FCS names
        # differ in their last few characters far more often than their
        # first, so "LP112_A01_rep…" would be unusable.
        self._list.setTextElideMode(Qt.TextElideMode.ElideNone)
        self._list.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._list.setHorizontalScrollMode(
            QAbstractItemView.ScrollMode.ScrollPerPixel)
        # Needed for the hover highlight: without it the view is only
        # told about the mouse while a button is held.
        self._list.setMouseTracking(True)
        self._list.currentItemChanged.connect(self._on_current)
        # clicking a sample that couldn't be read tries again
        self._list.itemClicked.connect(self._on_click)
        lay.addWidget(self._list, 1)

    # ------------------------------------------------------------------

    def set_samples(self, samples: list, group_col: str | None = None):
        """(Re)build the list, keeping the selection."""
        self._samples = list(samples)
        self._group_col = group_col
        lst = self._list
        # Signals blocked across the rebuild: clear() moves the current
        # item, which would emit currentItemChanged and tell the main
        # window the user had selected something.
        lst.blockSignals(True)
        lst.clear()
        grouped = self._groups()
        for header, members in grouped:
            if header is not None:
                h = QListWidgetItem(f"{header}  ·  {len(members)}")
                # NoItemFlags clears selectable and enabled at once, so
                # a header cannot be clicked or arrowed onto.
                h.setFlags(Qt.ItemFlag.NoItemFlags)
                h.setData(_HEADER, True)
                h.setData(Qt.ItemDataRole.UserRole + 5, header)
                lst.addItem(h)
            for s in members:
                it = QListWidgetItem(s.name)
                it.setData(_SAMPLE, s)
                n = len(_bfs(s.gates))
                it.setData(_BADGE, f"{n} gate{'s' if n != 1 else ''}"
                           if n else "")
                it.setData(Qt.ItemDataRole.UserRole + 4, header is not None)
                it.setData(Qt.ItemDataRole.UserRole + 5, header or "")
                self._set_tip(it, s)
                lst.addItem(it)
                if s is self._current:
                    lst.setCurrentItem(it)
        lst.blockSignals(False)
        self._apply_filter()

    def _set_tip(self, it, s):
        tip = [s.name, f"{s.n_events:,} events · {len(s.channels)} channels"]
        state = _state(s, self._delegate.busy)
        if state == "error":
            tip.append(f"Can't be read: {s.load_error}")
        elif state != "ready":
            tip.append(STATES[state])
        tip += [f"{k}: {v}" for k, v in s.annotations.items() if v]
        it.setToolTip("\n".join(tip))

    def set_busy_check(self, busy):
        """*busy(sample)* → True while its events are being read."""
        self._delegate.busy = busy

    def refresh_states(self):
        """Redraw the status dots (after a sample was read)."""
        for i in range(self._list.count()):
            it = self._list.item(i)
            s = it.data(_SAMPLE)
            if s is not None:
                self._set_tip(it, s)
        self._list.viewport().update()

    def _groups(self) -> list[tuple[str | None, list]]:
        """[(header, samples)], or [(None, all)] when not grouping.

        Grouping is skipped when the chosen column is empty for every
        sample — otherwise the whole list would sit under one header
        reading "(no group)", which is worse than no grouping at all.
        """
        col = self._group_col
        if not col or not any(s.annotations.get(col) for s in self._samples):
            return [(None, self._samples)]
        from ..core.groups import group_order, NO_GROUP
        vals = [s.annotations.get(col, "").strip() or NO_GROUP
                for s in self._samples]
        return [(g, [s for s, v in zip(self._samples, vals) if v == g])
                for g in group_order(vals)]

    def current_sample(self):
        return self._current

    def select_sample(self, sample, emit: bool = True):
        """Select *sample*; *emit* False to move the highlight silently.

        The silent form is what stops a feedback loop when the main
        window is the one that changed sample and is only syncing the
        list to match.
        """
        for i in range(self._list.count()):
            it = self._list.item(i)
            if it.data(_SAMPLE) is sample:
                if not emit:
                    self._list.blockSignals(True)
                self._list.setCurrentItem(it)
                self._list.blockSignals(False)
                self._current = sample
                return
        self._current = None
        self._list.setCurrentItem(None)

    def _on_current(self, cur, _prev):
        s = cur.data(_SAMPLE) if cur is not None else None
        if s is not None:
            self._current = s
            self.sample_selected.emit(s)

    def _on_click(self, it):
        """Clicking the already-selected failed sample retries it.

        currentItemChanged does not fire when the selection has not
        changed, so a second click on a failed sample would otherwise do
        nothing — and a network share that was briefly unavailable is
        exactly the case where trying again is the right move.
        """
        s = it.data(_SAMPLE)
        if s is not None and s is self._current and \
                getattr(s, "load_error", None):
            self.sample_selected.emit(s)

    def _apply_filter(self, *_):
        """Hide rows that do not match the search box.

        Two passes. The first hides samples and notes which groups still
        have a visible member; the second hides the headers of the
        groups that do not — otherwise a search would leave empty
        headings behind.
        """
        t = self._search.text().strip().lower()
        visible_groups = set()
        for i in range(self._list.count()):
            it = self._list.item(i)
            if it.data(_HEADER):
                continue
            s = it.data(_SAMPLE)
            # Searched against the name *and* every annotation value, so
            # typing a strain or a medium finds its samples.
            hay = " ".join([s.name] + [str(v) for v in
                                       s.annotations.values()]).lower()
            hide = bool(t) and t not in hay
            it.setHidden(hide)
            if not hide:
                visible_groups.add(it.data(Qt.ItemDataRole.UserRole + 5))
        for i in range(self._list.count()):
            it = self._list.item(i)
            if it.data(_HEADER):
                it.setHidden(it.data(Qt.ItemDataRole.UserRole + 5)
                             not in visible_groups)
