"""Shared widgets: a searchable channel picker grouped by laser, and an
application-wide rule that drop-downs are wide enough for their text."""

from __future__ import annotations

from PyQt6.QtWidgets import QComboBox, QCompleter, QApplication
from PyQt6.QtCore import Qt, QObject, QEvent
from PyQt6.QtGui import QStandardItemModel, QStandardItem, QFont

from ..core.spectral import spectral_detectors, LASER_ORDER

LASER_NAMES = {"UV": "UV laser", "V": "Violet laser", "B": "Blue laser",
               "YG": "Yellow-green laser", "Y": "Yellow laser",
               "G": "Green laser", "R": "Red laser", "IR": "Infrared laser"}
_HEADER = Qt.ItemDataRole.UserRole + 7
MAX_COMBO_CHARS = 40          # wider names show in full in the popup


def _popup_width(combo: QComboBox) -> int:
    fm = combo.view().fontMetrics()
    widest = max((fm.horizontalAdvance(combo.itemText(i))
                  for i in range(combo.count())), default=0)
    return widest + 48        # padding + scrollbar


def fit_combo(combo: QComboBox):
    """Make *combo* show its whole text: the box sizes to its longest entry
    (up to MAX_COMBO_CHARS) and the popup list to the widest entry."""
    if combo.property("_fitted"):
        return
    combo.setProperty("_fitted", True)
    combo.setSizeAdjustPolicy(
        QComboBox.SizeAdjustPolicy.AdjustToContents)

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

    m = combo.model()
    m.rowsInserted.connect(update)
    m.rowsRemoved.connect(update)
    m.modelReset.connect(update)
    update()


class _ComboFitter(QObject):
    """Applies fit_combo to every combo box when it is first shown."""

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.Polish and isinstance(obj, QComboBox):
            fit_combo(obj)
        return False


def install_combo_fitter(app: QApplication):
    app._combo_fitter = _ComboFitter(app)     # keep a reference
    app.installEventFilter(app._combo_fitter)


class ChannelCombo(QComboBox):
    """Channel picker for 100+ channels: type to search (any part of the
    name), spectral detectors grouped under their laser."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setEditable(True)
        self.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self._channels: list[str] = []
        self._last = ""
        self.setModel(QStandardItemModel(self))
        comp = QCompleter(self)
        self._flat = QStandardItemModel(self)     # completer: channels only
        comp.setModel(self._flat)
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
            if header and len(groups) > 1:
                h = QStandardItem(header.upper())
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
        if i >= 0 and not self.model().item(i).data(_HEADER):
            self.setCurrentIndex(i)
            self.setEditText(text)
            self._last = text

    def currentText(self) -> str:
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
    if not dets:
        return [("", list(channels))]
    by_laser: dict[str, list] = {}
    for d in dets:
        by_laser.setdefault(d.laser, []).append(d)
    det_names = {d.channel for d in dets}
    out = [("Scatter & other", [c for c in channels if c not in det_names])]
    for laser in sorted(by_laser, key=lambda l: (LASER_ORDER.index(l)
                                                 if l in LASER_ORDER else 99,
                                                 l)):
        ds = sorted(by_laser[laser], key=lambda d: (d.number,
                                                    d.channel[-1]))
        out.append((LASER_NAMES.get(laser, f"{laser} laser"),
                    [d.channel for d in ds]))
    return [g for g in out if g[1]]
