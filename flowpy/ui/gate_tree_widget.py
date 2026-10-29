"""Gate hierarchy tree widget with context menu for rename/delete."""

from __future__ import annotations
from typing import Callable
import numpy as np

from PyQt6.QtWidgets import (
    QTreeWidget, QTreeWidgetItem, QTreeWidgetItemIterator, QMenu, QInputDialog,
    QMessageBox, QWidget, QVBoxLayout, QLabel, QHBoxLayout, QPushButton,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QBrush, QFont, QIcon, QPixmap

from ..core.gates import Gate, BooleanGate


# ---------------------------------------------------------------------------
# Tree item columns
# ---------------------------------------------------------------------------
COL_NAME    = 0
COL_COUNT   = 1
COL_PCT_PAR = 2
COL_PCT_TOT = 3

ROOT_ID = ""   # gate_id used for the "All Events" row


def _swatch(color: str) -> QIcon:
    from PyQt6.QtGui import QPainter
    pm = QPixmap(12, 12)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setBrush(QColor(color))
    p.setPen(Qt.PenStyle.NoPen)
    p.drawRoundedRect(1, 1, 10, 10, 3, 3)
    p.end()
    return QIcon(pm)


class GateTreeWidget(QWidget):
    """Panel containing the gate hierarchy tree and action buttons.

    Signals
    -------
    gate_selected(gate_id)          : user picked a population ("" = All Events)
    gate_delete_requested(gate_id)  : user wants a gate (and children) deleted
    gate_rename_requested(id, name) : user typed a new name
    gate_edit_requested(gate_id)    : user wants the Edit Values dialog
    boolean_requested()             : user wants a new AND / OR / NOT gate

    The tree never changes gates itself; the main window does, so every
    change can be undone.
    """

    gate_selected = pyqtSignal(str)
    gate_delete_requested = pyqtSignal(str)
    gate_rename_requested = pyqtSignal(str, str)
    gate_edit_requested = pyqtSignal(str)
    boolean_requested = pyqtSignal()
    # "" = all gates / All Events
    copy_requested = pyqtSignal(str)
    paste_requested = pyqtSignal(str)
    refit_requested = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._gates: list[Gate] = []         # top-level roots
        self._all_gates: list[Gate] = []     # flat list (BFS order)
        self._masks: dict[str, np.ndarray] = {}
        self._n_total: int = 0
        self._current_id: str = ROOT_ID
        self._root_item: QTreeWidgetItem | None = None
        self._build_ui()
        self._rebuild()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        header = QLabel("POPULATIONS")
        header.setObjectName("SectionHeader")
        layout.addWidget(header)

        self._tree = QTreeWidget()
        self._tree.setColumnCount(4)
        self._tree.setHeaderLabels(["POPULATION", "COUNT", "%", "% TOTAL"])
        from PyQt6.QtWidgets import QHeaderView
        hdr = self._tree.header()
        hdr.setStretchLastSection(False)
        # names are never shortened: the column is as wide as the longest
        # name (the tree scrolls sideways if the panel is narrower)
        hdr.setSectionResizeMode(COL_NAME,
                                 QHeaderView.ResizeMode.Interactive)
        for col in (COL_COUNT, COL_PCT_PAR, COL_PCT_TOT):
            hdr.setSectionResizeMode(
                col, QHeaderView.ResizeMode.ResizeToContents)
        self._tree.setColumnHidden(COL_PCT_TOT, True)   # in Statistics too
        self._tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._tree.customContextMenuRequested.connect(self._show_context_menu)
        self._tree.currentItemChanged.connect(self._on_current_changed)
        self._tree.setAlternatingRowColors(False)
        self._tree.setIndentation(12)
        self._tree.setTextElideMode(Qt.TextElideMode.ElideNone)
        self._tree.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        layout.addWidget(self._tree)

        # Button row
        btn_row = QHBoxLayout()
        self._del_btn = QPushButton("Delete Gate")
        self._del_btn.setEnabled(False)
        self._del_btn.clicked.connect(self._delete_selected)
        btn_row.addWidget(self._del_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_gates(self, root_gates: list[Gate],
                  masks: dict[str, np.ndarray],
                  n_total: int):
        """Rebuild the tree from the given gate forest."""
        self._gates = list(root_gates)
        self._all_gates = _bfs(root_gates)
        self._masks = masks
        self._n_total = n_total
        self._rebuild()

    def add_gate(self, gate: Gate, masks: dict[str, np.ndarray], n_total: int):
        """Convenience: add one gate and refresh."""
        self._masks.update(masks)
        self._n_total = n_total
        if gate.parent is None:
            if gate not in self._gates:
                self._gates.append(gate)
        self._all_gates = _bfs(self._gates)
        self._rebuild()

    def selected_gate_id(self) -> str | None:
        items = self._tree.selectedItems()
        if not items:
            return None
        return items[0].data(COL_NAME, Qt.ItemDataRole.UserRole)

    # ------------------------------------------------------------------
    # Internal rebuild
    # ------------------------------------------------------------------

    def select_gate(self, gate_id: str | None, emit: bool = True):
        """Highlight the population row for *gate_id* (None/"" = All Events)."""
        gate_id = gate_id or ROOT_ID
        target = self._root_item
        it = QTreeWidgetItemIterator(self._tree)
        while it.value():
            if it.value().data(COL_NAME, Qt.ItemDataRole.UserRole) == gate_id:
                target = it.value()
                break
            it += 1
        self._tree.blockSignals(not emit)
        self._tree.setCurrentItem(target)
        self._tree.blockSignals(False)
        self._mark_current(target)

    def _mark_current(self, current: QTreeWidgetItem | None):
        it = QTreeWidgetItemIterator(self._tree)
        while it.value():
            f = it.value().font(COL_NAME)
            f.setBold(it.value() is current)
            it.value().setFont(COL_NAME, f)
            it += 1
        gid = (current.data(COL_NAME, Qt.ItemDataRole.UserRole)
               if current else ROOT_ID)
        self._current_id = gid
        self._del_btn.setEnabled(bool(gid))

    def _rebuild(self):
        self._tree.blockSignals(True)
        self._tree.clear()

        self._root_item = QTreeWidgetItem()
        self._root_item.setData(COL_NAME, Qt.ItemDataRole.UserRole, ROOT_ID)
        self._root_item.setText(COL_NAME, "All Events")
        self._root_item.setText(COL_COUNT, f"{self._n_total:,}")
        self._root_item.setText(COL_PCT_PAR, "")
        self._root_item.setText(COL_PCT_TOT, "100.0%")
        for col in (COL_COUNT, COL_PCT_TOT):
            self._root_item.setTextAlignment(
                col, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._tree.addTopLevelItem(self._root_item)

        for gate in self._gates:
            item = self._make_item(gate)
            self._root_item.addChild(item)
            self._add_children(item, gate)

        self._tree.expandAll()
        self._tree.blockSignals(False)
        self.select_gate(self._current_id, emit=False)
        self._fit_name_column()

    def _fit_name_column(self):
        """Name column: the whole longest name, or all the spare width."""
        t = self._tree
        t.resizeColumnToContents(COL_NAME)
        need = t.columnWidth(COL_NAME) + 8
        others = sum(t.columnWidth(c) for c in range(1, t.columnCount())
                     if not t.isColumnHidden(c))
        t.setColumnWidth(COL_NAME, max(need, t.viewport().width() - others))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit_name_column()

    def _add_children(self, parent_item: QTreeWidgetItem, gate: Gate):
        for child in gate.children:
            item = self._make_item(child)
            parent_item.addChild(item)
            self._add_children(item, child)

    def _make_item(self, gate: Gate) -> QTreeWidgetItem:
        item = QTreeWidgetItem()
        item.setData(COL_NAME, Qt.ItemDataRole.UserRole, gate.gate_id)

        mask = self._masks.get(gate.gate_id)
        n = int(mask.sum()) if mask is not None else 0
        pct_total = 100.0 * n / self._n_total if self._n_total > 0 else 0.0

        if gate.parent is not None:
            par_mask = self._masks.get(gate.parent.gate_id)
            n_par = int(par_mask.sum()) if par_mask is not None else self._n_total
        else:
            n_par = self._n_total
        pct_par = 100.0 * n / n_par if n_par > 0 else 0.0

        item.setText(COL_NAME, gate.name)
        item.setText(COL_COUNT, f"{n:,}")
        item.setText(COL_PCT_PAR, f"{pct_par:.1f}%")
        item.setText(COL_PCT_TOT, f"{pct_total:.1f}%")

        item.setIcon(COL_NAME, _swatch(gate._color or "#AAAAAA"))
        if isinstance(gate, BooleanGate):
            names = {g.gate_id: g.name for g in self._all_gates}
            item.setToolTip(COL_NAME, f"{gate.name} = "
                            f"{gate.describe(names)}")
        else:
            item.setToolTip(COL_NAME, f"{gate.name} — {gate.x_channel}"
                            + (f" / {gate.y_channel}" if gate.y_channel
                               else ""))

        for col in (COL_COUNT, COL_PCT_PAR, COL_PCT_TOT):
            item.setTextAlignment(col, Qt.AlignmentFlag.AlignRight |
                                       Qt.AlignmentFlag.AlignVCenter)
        return item

    # ------------------------------------------------------------------
    # Context menu & actions
    # ------------------------------------------------------------------

    def _show_context_menu(self, pos):
        item = self._tree.itemAt(pos)
        if item is None:
            return
        gate = self._gate_by_id(item.data(COL_NAME, Qt.ItemDataRole.UserRole))

        menu = QMenu(self)
        edit_act = rename_act = delete_act = copy_act = refit_act = None
        if gate is not None:
            edit_act = menu.addAction("Edit Values…")
            refit_act = menu.addAction("Re-fit Gate to These Axes…")
            refit_act.setToolTip(
                "Describe this gate on the scales now shown, keeping the "
                "same events, so it can be seen and edited here")
            rename_act = menu.addAction("Rename…")
            delete_act = menu.addAction("Delete")
            menu.addSeparator()
            copy_act = menu.addAction("Copy Gate (with its sub-gates)")
        copy_all_act = menu.addAction("Copy All Gates")
        paste_act = menu.addAction(
            "Paste Gates Here…" if gate is not None else "Paste Gates…")
        menu.addSeparator()
        bool_act = menu.addAction("New Boolean Gate (AND / OR / NOT)…")
        action = menu.exec(self._tree.viewport().mapToGlobal(pos))

        if action is None:
            return
        if action == edit_act:
            self.gate_edit_requested.emit(gate.gate_id)
        elif action == refit_act:
            self.refit_requested.emit(gate.gate_id)
        elif action == rename_act:
            self._rename_gate(gate)
        elif action == delete_act:
            self.gate_delete_requested.emit(gate.gate_id)
        elif action == bool_act:
            self.boolean_requested.emit()
        elif action == copy_act:
            self.copy_requested.emit(gate.gate_id)
        elif action == copy_all_act:
            self.copy_requested.emit("")
        elif action == paste_act:
            self.paste_requested.emit(gate.gate_id if gate is not None else "")

    def _on_current_changed(self, item: QTreeWidgetItem | None, _prev):
        if item is None:
            return
        self._mark_current(item)
        self.gate_selected.emit(item.data(COL_NAME, Qt.ItemDataRole.UserRole))

    def _delete_selected(self):
        gate_id = self.selected_gate_id()
        if gate_id:
            self.gate_delete_requested.emit(gate_id)

    def _rename_gate(self, gate: Gate):
        name, ok = QInputDialog.getText(self, "Rename Gate", "New name:",
                                        text=gate.name)
        if ok and name.strip():
            self.gate_rename_requested.emit(gate.gate_id, name.strip())

    def _gate_by_id(self, gate_id: str) -> Gate | None:
        for g in self._all_gates:
            if g.gate_id == gate_id:
                return g
        return None


# ---------------------------------------------------------------------------
# BFS helper
# ---------------------------------------------------------------------------

def _bfs(roots: list[Gate]) -> list[Gate]:
    result, queue = [], list(roots)
    while queue:
        g = queue.pop(0)
        result.append(g)
        queue.extend(g.children)
    return result


# Patch so statistics.py can import this symbol from core.gates
# (gates_in_order_list is defined here to avoid circular import)
def gates_in_order_list(roots: list[Gate]) -> list[Gate]:
    return _bfs(roots)
