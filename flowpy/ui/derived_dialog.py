"""Derived parameters dialog — create ratios and other formula channels
(e.g. YG5-A / BL1-A) that can be plotted, gated and exported like any
measured channel."""

# ---------------------------------------------------------------------------
# HOW THIS DIALOG WORKS
# ---------------------------------------------------------------------------
# A list of existing formula channels on the left, an editor on the
# right. The editor is in one of two states, held in self._editing:
#
#     None          -> creating; the button reads "Create Parameter"
#     "<name>"      -> editing that one; the button reads "Update"
#
# _new() and _on_select() are the two ways in and out of those states,
# and every other method reads _editing to know which it is in.
#
# Typing in the formula box does not evaluate immediately. It restarts a
# 200 ms timer, and only when typing stops does _preview() run the
# formula over the whole current sample. Without that, every keystroke
# would evaluate a formula across millions of events.
#
# _save() is the only method that writes anything, and it writes to
# every selected sample. The current sample is deliberately done first,
# so that a mistake in the formula is reported against the sample the
# user was looking at rather than whichever happened to be first.
# ---------------------------------------------------------------------------

from __future__ import annotations
import numpy as np

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QLineEdit,
    QComboBox, QPushButton, QCheckBox, QListWidget, QListWidgetItem,
    QMessageBox, QCompleter, QWidget, QGroupBox,
)
from PyQt6.QtCore import Qt, QTimer

from ..core import formulas as fx
from ..core.protocol import _bfs


def _channel_combo(channels: list[str]) -> QComboBox:
    """Combo you can type into to search 100+ channels."""
    c = QComboBox()
    c.setEditable(True)
    c.addItems(channels)
    # Editable so it can be typed into, but NoInsert so that typing a
    # name that is not a channel does not silently add it to the list.
    c.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
    # MatchContains rather than the default MatchStartsWith: on a
    # spectral panel the useful part of a name is in the middle, so
    # typing "mKate" should find "YG5-A (mKate)".
    comp = c.completer()
    comp.setFilterMode(Qt.MatchFlag.MatchContains)
    comp.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
    c.setMinimumContentsLength(14)
    return c


def gates_using(samples, name: str) -> list[str]:
    """'sample: gate' for every gate drawn on channel *name*.

    Used before renaming: a gate stores the channel it was drawn on by
    name, so renaming out from under one would leave it pointing at a
    channel that no longer exists.
    """
    out = []
    for s in samples:
        for g in _bfs(s.gates):
            # getattr with a default because a range gate is
            # one-dimensional and has no y_channel at all.
            if name in (g.x_channel, getattr(g, "y_channel", None)):
                out.append(f"{s.name}: {g.name}")
    return out


class DerivedParameterDialog(QDialog):
    """List, create, edit and delete formula channels."""

    def __init__(self, samples, current, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Derived Parameters")
        self.resize(860, 520)
        self._samples = samples
        self._s = current
        self.changed = False          # read by the main window
        self._editing: str | None = None

        # The debounce behind the live preview — see the note at the top.
        # setSingleShot means a restart replaces the pending shot rather
        # than queueing another.
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(200)
        self._timer.timeout.connect(self._preview)

        root = QHBoxLayout(self)

        # --- existing parameters ---
        left = QVBoxLayout()
        hdr = QLabel("DERIVED PARAMETERS")
        hdr.setObjectName("SectionHeader")
        left.addWidget(hdr)
        self._list = QListWidget()
        self._list.currentItemChanged.connect(self._on_select)
        left.addWidget(self._list, 1)
        row = QHBoxLayout()
        new = QPushButton("New")
        new.clicked.connect(self._new)
        self._delete_btn = QPushButton("Delete")
        self._delete_btn.clicked.connect(self._delete)
        row.addWidget(new)
        row.addWidget(self._delete_btn)
        left.addLayout(row)
        lw = QWidget()
        lw.setLayout(left)
        lw.setMinimumWidth(260)
        root.addWidget(lw)

        # --- editor ---
        right = QVBoxLayout()
        chans = self._channels()

        quick = QGroupBox("QUICK RATIO")
        q = QHBoxLayout(quick)
        self._ratio_a = _channel_combo(chans)
        self._ratio_b = _channel_combo(chans)
        # Default the quick ratio to two fluorescence channels, since a
        # ratio of scatter channels is rarely what anyone wants. min()
        # guards the one-channel case, where both boxes point at it.
        if len(chans) > 1:
            fl = [c for c in current.fluorescence if c in chans] or chans
            self._ratio_a.setCurrentText(fl[0])
            self._ratio_b.setCurrentText(fl[min(1, len(fl) - 1)])
        q.addWidget(self._ratio_a, 1)
        q.addWidget(QLabel("÷"))
        q.addWidget(self._ratio_b, 1)
        use = QPushButton("Use")
        use.setToolTip("Fill in the formula and name for this ratio")
        use.clicked.connect(self._use_ratio)
        q.addWidget(use)
        right.addWidget(quick)

        box = QGroupBox("PARAMETER")
        form = QFormLayout(box)
        self._name = QLineEdit()
        self._name.setPlaceholderText("e.g. YG5/BL1 ratio")
        form.addRow("Name:", self._name)
        self._expr = QLineEdit()
        self._expr.setPlaceholderText("[YG5-A] / [BL1-A]")
        self._expr.textChanged.connect(lambda: self._timer.start())
        form.addRow("Formula:", self._expr)

        ins = QHBoxLayout()
        self._ins_combo = _channel_combo(chans)
        ins.addWidget(self._ins_combo, 1)
        b = QPushButton("Insert channel")
        b.clicked.connect(lambda: self._insert(
            f"[{self._ins_combo.currentText()}]"))
        ins.addWidget(b)
        form.addRow("", ins)
        ops = QHBoxLayout()
        ops.setSpacing(3)
        for text, put in (("+", " + "), ("−", " - "), ("×", " * "),
                          ("÷", " / "), ("(", "("), (")", ")"),
                          ("log10", "log10("), ("ln", "ln("),
                          ("sqrt", "sqrt("), ("asinh", "asinh(")):
            ob = QPushButton(text)
            ob.setMinimumWidth(64 if len(text) > 1 else 38)
            # p=put binds the current value into the lambda's default.
            # Without it every button would close over the same variable
            # and insert the last operator in the loop.
            ob.clicked.connect(lambda _=False, p=put: self._insert(p))
            ops.addWidget(ob)
        ops.addStretch()
        form.addRow("", ops)

        self._fluor_cb = QCheckBox("Fluorescence-like (Logicle scale by "
                                   "default, listed with fluorescence "
                                   "statistics)")
        form.addRow(self._fluor_cb)
        self._all_cb = QCheckBox(f"Create in all {len(samples)} samples"
                                 if len(samples) > 1 else
                                 "Create in all samples")
        self._all_cb.setChecked(True)
        self._all_cb.setEnabled(len(samples) > 1)
        form.addRow(self._all_cb)
        right.addWidget(box)

        self._preview_lbl = QLabel("")
        self._preview_lbl.setWordWrap(True)
        self._preview_lbl.setTextFormat(Qt.TextFormat.RichText)
        right.addWidget(self._preview_lbl)

        help_lbl = QLabel(
            "Put channel names in <b>[square brackets]</b>. Allowed: "
            "+ − × ÷, ** (power), brackets, numbers and "
            + ", ".join(fx.FUNCTIONS) + ".<br>Examples: "
            "<code>[YG5-A] / [BL1-A]</code> · "
            "<code>log10([mKate-A]) - log10([GFP-A])</code> · "
            "<code>([B-A] - [A-A]) / ([B-A] + [A-A])</code><br>"
            "Events where the result is undefined (e.g. ÷ 0) are left out "
            "of plots and gates. Formulas are recomputed when compensation "
            "changes and are saved in workspaces and gating protocols.")
        help_lbl.setWordWrap(True)
        help_lbl.setObjectName("Hint")
        right.addWidget(help_lbl)
        right.addStretch()

        btns = QHBoxLayout()
        btns.addStretch()
        self._save_btn = QPushButton("Create Parameter")
        self._save_btn.setObjectName("Primary")
        self._save_btn.clicked.connect(self._save)
        btns.addWidget(self._save_btn)
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        btns.addWidget(close)
        right.addLayout(btns)
        root.addLayout(right, 1)

        self._refresh_list()
        self._new()

    # ------------------------------------------------------------------

    def _channels(self) -> list[str]:
        """Channels a formula may refer to.

        The parameter being edited is excluded: a formula that mentions
        itself would be circular, and the validator would accept it
        because by then the channel does exist.
        """
        return [c for c in self._s.channels if c != self._editing]

    def _refresh_list(self, select: str | None = None):
        """Rebuild the list on the left, optionally selecting one entry."""
        # Signals blocked so that clearing and refilling does not fire
        # currentItemChanged -> _on_select, which would overwrite the
        # editor the user is in the middle of typing into.
        self._list.blockSignals(True)
        self._list.clear()
        for name, expr in self._s.formulas.items():
            # Two lines per row: the name, then its formula indented
            # underneath. The name alone goes in the item's data, since
            # the display text is not a usable key.
            it = QListWidgetItem(f"{name}\n   = {expr}")
            it.setData(Qt.ItemDataRole.UserRole, name)
            self._list.addItem(it)
            if name == select:
                self._list.setCurrentItem(it)
        self._list.blockSignals(False)
        self._delete_btn.setEnabled(self._list.currentItem() is not None)

    def _on_select(self, cur, _prev):
        if cur is None:
            return
        name = cur.data(Qt.ItemDataRole.UserRole)
        self._editing = name
        self._name.setText(name)
        self._expr.setText(self._s.formulas[name])
        self._fluor_cb.setChecked(name in self._s.fluorescence)
        self._save_btn.setText("Update Parameter")
        self._delete_btn.setEnabled(True)

    def _new(self):
        self._editing = None
        self._list.clearSelection()
        self._list.setCurrentItem(None)
        self._name.clear()
        self._expr.clear()
        self._fluor_cb.setChecked(False)
        self._save_btn.setText("Create Parameter")
        self._delete_btn.setEnabled(False)
        self._preview()

    def _insert(self, text: str):
        self._expr.insert(text)
        self._expr.setFocus()

    def _use_ratio(self):
        """Fill the formula and a sensible name from the two combos."""
        a, b = self._ratio_a.currentText(), self._ratio_b.currentText()
        self._expr.setText(fx.ratio_formula(a, b))
        # Trim the -A/-H/-W suffix for the *name* only: "YG5/BL1 ratio"
        # reads better than "YG5-A/BL1-A ratio", while the formula keeps
        # the full channel names it needs.
        short = lambda c: c[:-2] if c[-2:] in ("-A", "-H", "-W") else c
        self._name.setText(f"{short(a)}/{short(b)} ratio")

    def _preview(self):
        expr = self._expr.text().strip()
        if not expr:
            self._preview_lbl.setText("")
            return
        try:
            fx.validate(expr, self._channels())
            v = fx.evaluate(expr, self._s.column)
        except fx.FormulaError as exc:
            self._preview_lbl.setText(f"<span style='color:#dc2626;'>"
                                      f"{exc}</span>")
            return
        # Finite values only. A formula frequently produces some NaN —
        # dividing by zero on a dim event — and those must not reach
        # np.percentile, which would return NaN for everything.
        ok = v[np.isfinite(v)]
        if not len(ok):
            self._preview_lbl.setText("<span style='color:#dc2626;'>The "
                                      "result is undefined for every "
                                      "event.</span>")
            return
        # 1st to 99th percentile rather than min to max: one extreme
        # event would otherwise set the whole range and say nothing about
        # where the data actually sits.
        lo, med, hi = np.percentile(ok, [1, 50, 99])
        nan = len(v) - len(ok)
        txt = (f"<b>Preview on {self._s.name}:</b> median {med:.4g} · "
               f"1st–99th percentile {lo:.4g} to {hi:.4g}")
        if nan:
            txt += (f" · <span style='color:#b45309;'>{nan:,} events "
                    f"undefined ({100 * nan / len(v):.1f}%)</span>")
        self._preview_lbl.setText(txt)

    def _save(self):
        name = self._name.text().strip()
        expr = self._expr.text().strip()
        fluor = self._fluor_cb.isChecked()
        if not name:
            QMessageBox.information(self, "Derived Parameters",
                                    "Give the parameter a name.")
            return
        # Renaming is refused while gates depend on the old name — see
        # gates_using(). Creating and editing in place are both fine.
        old = self._editing
        if old is not None and name != old:
            used = gates_using(self._samples, old)
            if used:
                QMessageBox.warning(
                    self, "Derived Parameters",
                    f"'{old}' is used by gates, so it can't be renamed:\n\n"
                    + "\n".join(used[:10]))
                return
        targets = self._samples if self._all_cb.isChecked() else [self._s]
        # Current sample first, then the rest with it removed so it is
        # not done twice. Its error is the one worth interrupting for:
        # the others may legitimately lack the channel, and those are
        # collected into *failed* and reported together at the end.
        targets = [self._s] + [t for t in targets if t is not self._s]
        failed = []
        for i, t in enumerate(targets):
            try:
                if old is not None and name != old and old in t.formulas:
                    t.remove_derived(old)
                # Creating something that collides with a *measured*
                # channel is refused; colliding with an existing formula
                # is just an edit of it, which is allowed.
                if old is None and name in t.channels \
                        and name not in t.formulas:
                    raise fx.FormulaError(f"'{name}' is already a channel.")
                t.add_formula(name, expr, fluor)
            except fx.FormulaError as exc:
                if i == 0:
                    QMessageBox.warning(self, "Derived Parameters", str(exc))
                    return
                failed.append(f"{t.name}: {exc}")
        self.changed = True
        self._editing = name
        self._refresh_list(select=name)
        self._save_btn.setText("Update Parameter")
        if failed:
            QMessageBox.warning(self, "Derived Parameters",
                                "Not created in:\n\n" + "\n".join(failed))

    def _delete(self):
        it = self._list.currentItem()
        if it is None:
            return
        name = it.data(Qt.ItemDataRole.UserRole)
        used = gates_using(self._samples, name)
        if used:
            QMessageBox.warning(
                self, "Derived Parameters",
                f"'{name}' is used by gates — delete those gates first:\n\n"
                + "\n".join(used[:10]))
            return
        for t in self._samples:
            if name in t.formulas:
                t.remove_derived(name)
        self.changed = True
        self._refresh_list()
        self._new()
