"""Shared widgets: a searchable channel picker grouped by laser, and an
application-wide rule that drop-downs are wide enough for their text."""

from __future__ import annotations

import numpy as np

from PyQt6.QtWidgets import (QComboBox, QCompleter, QApplication,
                             QLineEdit)
from PyQt6.QtCore import Qt, QObject, QEvent
from PyQt6.QtGui import QStandardItemModel, QStandardItem, QFont

from ..core.spectral import spectral_detectors, LASER_ORDER

LASER_NAMES = {"UV": "UV laser", "V": "Violet laser", "B": "Blue laser",
               "YG": "Yellow-green laser", "Y": "Yellow laser",
               "G": "Green laser", "R": "Red laser", "IR": "Infrared laser"}
# ---------------------------------------------------------------------------
# WHAT THIS MODULE SOLVES
# ---------------------------------------------------------------------------
# A spectral cytometer reports 180 detectors with names like
# "B14-A (525/50)". Two problems follow, and each has an answer here:
#
#   picking one     ChannelCombo — type any part of the name to search,
#                   detectors grouped under their laser
#   seeing it       fit_combo / _ComboFitter — Qt sizes a combo box to
#                   its CURRENT entry, so a long name is cut off; the
#                   fitter applies the rule to every combo in the app
#                   without each one having to ask
# ---------------------------------------------------------------------------

#: A custom item role marking a laser heading, so the headings can be
#: told from real channels. UserRole + 7 only has to avoid colliding
#: with the other custom roles in the same model.
_HEADER = Qt.ItemDataRole.UserRole + 7
MAX_COMBO_CHARS = 40          # wider names show in full in the popup
# A ceiling on the BOX, not on the popup: without it one 90-character
# channel name would stretch the combo across the whole window.


def _popup_width(combo: QComboBox) -> int:
    """Pixels needed to show the widest entry whole."""
    # The VIEW's font metrics, not the combo's: the popup list may be
    # drawn in a different font from the box, and measuring the wrong
    # one gives a list that still clips.
    fm = combo.view().fontMetrics()
    widest = max((fm.horizontalAdvance(combo.itemText(i))
                  for i in range(combo.count())), default=0)
    return widest + 48        # padding + scrollbar


def fit_combo(combo: QComboBox):
    """Make *combo* show its whole text: the box sizes to its longest entry
    (up to MAX_COMBO_CHARS) and the popup list to the widest entry."""
    # Marked on the widget itself rather than in a set here: the event
    # filter below fires on every Polish event, and a widget that is
    # hidden and reshown would otherwise be reconnected each time,
    # accumulating duplicate signal connections. A Qt property also
    # disappears with the widget, so nothing leaks.
    if combo.property("_fitted"):
        return
    combo.setProperty("_fitted", True)
    combo.setSizeAdjustPolicy(
        QComboBox.SizeAdjustPolicy.AdjustToContents)

    # No ellipsis in the popup: "B14-A (525/50)" and "B14-A (530/30)"
    # elide to the same thing, so the list would offer two entries that
    # read identically.
    combo.view().setTextElideMode(Qt.TextElideMode.ElideNone)

    def update(*_):
        combo.view().setMinimumWidth(_popup_width(combo))
        longest = max((len(combo.itemText(i)) for i in
                       range(combo.count())), default=0)
        if longest > MAX_COMBO_CHARS:
            combo.setSizeAdjustPolicy(
                QComboBox.SizeAdjustPolicy
                .AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(MAX_COMBO_CHARS)
        combo.updateGeometry()

    # Re-measured whenever the contents change, because the widths
    # depend on the entries and a combo is usually empty when it is
    # built and filled later.
    m = combo.model()
    m.rowsInserted.connect(update)
    m.rowsRemoved.connect(update)
    m.modelReset.connect(update)
    update()


class _ComboFitter(QObject):
    """Applies fit_combo to every combo box when it is first shown."""

    def eventFilter(self, obj, event):
        # Polish is Qt's "about to be shown for the first time, styling
        # applied" moment — the earliest point at which the font
        # metrics are correct, and so the earliest the widths can be
        # measured.
        if event.type() == QEvent.Type.Polish and isinstance(obj, QComboBox):
            fit_combo(obj)
        # False, never True: returning True would swallow the event and
        # the widget would never finish being polished.
        return False


def install_combo_fitter(app: QApplication):
    """Apply the sizing rule to every combo box in the application."""
    # Stored on the app because installEventFilter does not take
    # ownership: a filter that is garbage collected stops working, and
    # silently — the combos simply go back to being too narrow.
    app._combo_fitter = _ComboFitter(app)     # keep a reference
    app.installEventFilter(app._combo_fitter)


class ChannelCombo(QComboBox):
    """Channel picker for 100+ channels: type to search (any part of the
    name), spectral detectors grouped under their laser."""

    def __init__(self, parent=None):
        super().__init__(parent)
        # Editable so it can be typed into and searched; NoInsert so
        # that typing a name which is not a channel does not add it to
        # the list as though the instrument had such a detector.
        self.setEditable(True)
        self.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self._channels: list[str] = []
        self._last = ""
        self.setModel(QStandardItemModel(self))
        # Two models on purpose. The combo's own model holds the laser
        # headings as unselectable rows; the completer gets a flat model
        # of channels only, so searching never offers a heading as a
        # match.
        comp = QCompleter(self)
        self._flat = QStandardItemModel(self)     # completer: channels only
        comp.setModel(self._flat)
        # Matches any part of the name, not just the start: a user
        # looking for the FITC detector types "fitc" or "525", neither
        # of which begins a channel name like "B2-A (525/50) FITC".
        comp.setFilterMode(Qt.MatchFlag.MatchContains)
        comp.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        comp.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        comp.activated[str].connect(self.setCurrentText)
        self.setCompleter(comp)
        self.lineEdit().editingFinished.connect(self._revert_invalid)
        self.lineEdit().setPlaceholderText("type to search…")
        self.currentIndexChanged.connect(self._remember)
        fit_combo(self)

    # --- items ---

    def set_channels(self, channels: list[str]):
        """Fill with *channels*; on a spectral instrument, detectors are
        grouped by laser under non-selectable headers."""
        cur = self.currentText()
        self.blockSignals(True)
        model: QStandardItemModel = self.model()
        model.clear()
        self._flat.clear()
        self._channels = list(channels)
        groups = _group_by_laser(channels)
        for header, chans in groups:
            # No headings on a conventional cytometer: with one group
            # they would be a heading over the entire list, which says
            # nothing.
            if header and len(groups) > 1:
                h = QStandardItem(header.upper())
                # No flags at all: not selectable, not enabled — the
                # heading is a label in the list, and choosing "BLUE
                # LASER" as a channel is meaningless.
                h.setFlags(Qt.ItemFlag.NoItemFlags)
                h.setData(True, _HEADER)
                f = QFont()
                f.setBold(True)
                f.setPointSizeF(max(f.pointSizeF() - 2, 8))
                h.setFont(f)
                model.appendRow(h)
            for c in chans:
                model.appendRow(QStandardItem(c))
                self._flat.appendRow(QStandardItem(c))
        # The previous choice is kept if the new instrument also has
        # that channel, so switching between samples from the same
        # machine does not reset every axis.
        if cur in self._channels:
            self.setCurrentIndex(self.findText(cur,
                                               Qt.MatchFlag.MatchExactly))
        elif self._channels:
            self.setCurrentIndex(self.findText(self._channels[0],
                                               Qt.MatchFlag.MatchExactly))
        self._last = self.currentText()
        self.blockSignals(False)
        # never squeezed below the width of its longest channel name
        self.setMinimumWidth(min(self.sizeHint().width(),
                                 self.fontMetrics().horizontalAdvance(
                                     "M" * MAX_COMBO_CHARS)))

    def showPopup(self):
        self.view().setMinimumWidth(_popup_width(self))
        super().showPopup()

    def channels(self) -> list[str]:
        return list(self._channels)

    # an editable combo's setCurrentText only edits the text; select the
    # matching entry so currentIndex / signals follow
    def setCurrentText(self, text: str):
        i = self.findText(text, Qt.MatchFlag.MatchExactly)
        # The heading check matters: a laser name could in principle
        # match, and selecting that row would leave the widget reporting
        # a "channel" no sample has.
        if i >= 0 and not self.model().item(i).data(_HEADER):
            self.setCurrentIndex(i)
            self.setEditText(text)
            self._last = text

    def currentText(self) -> str:
        """The channel that is selected, not the text being typed.

        Overridden because an editable combo's currentText() returns the
        contents of the line edit, so a half-typed search string would
        be read as the chosen channel while the popup is still open.
        """
        i = self.currentIndex()
        return self.itemText(i) if i >= 0 else super().currentText()

    def _remember(self, _i):
        t = self.currentText()
        if t in self._channels:
            self._last = t

    def _revert_invalid(self):
        typed = self.lineEdit().text().strip()
        if typed in self._channels:
            self.setCurrentText(typed)
            return
        # unique partial match → take it; otherwise go back
        # — typing "fitc" and tabbing away should land on the FITC
        # detector, but only when there is exactly one; two matches
        # would be a guess, so the previous valid choice is restored.
        hits = [c for c in self._channels if typed.lower() in c.lower()]
        if typed and len(hits) == 1:
            self.setCurrentText(hits[0])
        else:
            self.setEditText(self._last)


def _group_by_laser(channels: list[str]) -> list[tuple[str, list[str]]]:
    """[(header, channels)]: scatter & other channels first, then one group
    per laser (area, height, width of each detector together)."""
    dets = []
    for kind in ("A", "H", "W"):
        dets += spectral_detectors(channels, kind)
    # No detectors recognised means a conventional cytometer, or names
    # this parser does not know: the channels are returned as one
    # unheaded group rather than being dropped.
    if not dets:
        return [("", list(channels))]
    by_laser: dict[str, list] = {}
    for d in dets:
        by_laser.setdefault(d.laser, []).append(d)
    det_names = {d.channel for d in dets}
    # Scatter first, because FSC and SSC are where gating starts.
    # Anything the detector parser did not claim lands here too, so no
    # channel can disappear from the list.
    out = [("Scatter & other", [c for c in channels if c not in det_names])]
    # Lasers in physical order (UV, violet, blue, …) rather than
    # alphabetical, which is how the detector names are numbered and how
    # a cytometrist reads a panel. 99 puts an unknown laser last instead
    # of raising on the index lookup.
    for laser in sorted(by_laser, key=lambda l: (LASER_ORDER.index(l)
                                                 if l in LASER_ORDER else 99,
                                                 l)):
        # Detector number first, then the last character of the name —
        # which is A, H or W, so area, height and width of the same
        # detector stay together instead of forming three separate runs.
        ds = sorted(by_laser[laser], key=lambda d: (d.number,
                                                    d.channel[-1]))
        out.append((LASER_NAMES.get(laser, f"{laser} laser"),
                    [d.channel for d in ds]))
    # Empty groups dropped: an instrument with no scatter channels
    # should not show an empty "Scatter & other" heading.
    return [g for g in out if g[1]]


def format_axis_value(v: float | None) -> str:
    """An axis limit as these boxes show it, or "" for automatic.

    Plain digits with separators where a cytometer's numbers live, rather
    than 1.21601e+06: the boxes are read off the axis and typed back in,
    and RangeEdit.value() strips the separators again.

    A shared function and not just formatting inside set_value, because
    the rounding has to be known elsewhere: a box showing -251,450 is
    reporting a limit of -251,450.3, and anything comparing what is in a
    box with a computed limit has to compare them at the same precision
    or they never agree.
    """
    if v is None:
        return ""
    a = abs(float(v))
    return (f"{v:,.0f}" if a >= 1000
            else f"{v:.4g}" if a >= 1 else f"{v:.3g}")


class RangeEdit(QLineEdit):
    """Axis limit box where empty means automatic.

    In the channel's own units, not the transformed position: on a
    Logicle axis labelled 10², 10³, 10⁵ you type 100000, because that is
    the number the axis is telling you about. Shared by the Gating tab
    and the layout editor so the two parse, display and complain alike.
    """

    def __init__(self, placeholder: str = "auto", parent=None):
        super().__init__(parent)
        self.setPlaceholderText(placeholder)
        self._base_tip = ""

    def value(self) -> float | None:
        """The number typed, or None for automatic."""
        try:
            # Thousands separators removed, since someone reading an axis
            # may well type 10,000.
            v = float(self.text().replace(",", "").strip())
        except ValueError:
            return None
        # "inf" and "nan" parse as floats but are not usable axis limits;
        # both become automatic.
        return v if np.isfinite(v) else None

    def set_value(self, v: float | None):
        # Never while it is being typed into: these boxes double as a
        # readout of the current axis, and a redraw mid-keystroke would
        # otherwise replace what someone was halfway through entering.
        if self.hasFocus():
            return
        self.setText(format_axis_value(v))

    def setToolTip(self, text: str):
        super().setToolTip(text)
        if not getattr(self, "_complaining", False):
            self._base_tip = text

    def complain(self, reason: str):
        """Mark the box as holding something unusable, and say why.

        An inverted pair used to be dropped in silence: you typed two
        numbers, the axis did not move, and nothing said the pair was the
        wrong way round.
        """
        self._complaining = bool(reason)
        self.setStyleSheet(
            "QLineEdit { border: 1px solid #DC2626; }" if reason else "")
        super().setToolTip(reason or self._base_tip)


def check_range_pair(lo: RangeEdit, hi: RangeEdit) -> bool:
    """True when the pair makes a usable range; marks both when it does not."""
    a, b = lo.value(), hi.value()
    bad = a is not None and b is not None and b <= a
    reason = ("The maximum has to be greater than the minimum — "
              "this range is being ignored, and the axis left automatic."
              if bad else "")
    lo.complain(reason)
    hi.complain(reason)
    return not bad
