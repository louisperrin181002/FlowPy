"""Editing existing gates.

* GateEditor      — drag handles on the plot: move a whole gate, move polygon
                    vertices / rectangle corners / range edges / quadrant
                    centre, resize and rotate ellipses.
* EditGateDialog  — exact numeric values (plus name and colour).
* BooleanGateDialog — build an AND / OR / NOT gate from existing gates.

Edits are reported as ``[(gate, {attribute: new value})]`` so the main window
can take an undo snapshot before applying them.
"""

from __future__ import annotations
import numpy as np

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QLineEdit, QDialogButtonBox, QLabel,
    QTableWidget, QTableWidgetItem, QPushButton, QHBoxLayout, QColorDialog,
    QComboBox, QListWidget, QListWidgetItem, QMessageBox, QHeaderView,
    QWidget,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from matplotlib.lines import Line2D
from matplotlib.path import Path as MplPath

from ..core.transforms import apply_transform, inverse_transform
from ..core.gates import (
    Gate, PolygonGate, RectangleGate, EllipseGate, QuadrantGate, RangeGate,
    BooleanGate,
)

PICK_PX = 9          # handle hit radius, in screen pixels


def _fwd(v, tr):
    return float(apply_transform(np.array([v], dtype=float), tr)[0])


def _inv(v, tr):
    return float(inverse_transform(np.array([v], dtype=float), tr)[0])


def quadrant_siblings(gate: QuadrantGate, siblings: list[Gate]) -> list[Gate]:
    """The four Q gates drawn together share channels and centre."""
    return [g for g in siblings if isinstance(g, QuadrantGate)
            and g.x_channel == gate.x_channel and g.y_channel == gate.y_channel
            and g.cx == gate.cx and g.cy == gate.cy]


def edit_blocker(gate: Gate, x_tr: str, y_tr: str | None) -> str | None:
    """Why *gate* can't be edited on this plot (None = it can)."""
    if isinstance(gate, BooleanGate):
        return "Boolean gates are edited with Edit Values…"
    if isinstance(gate, (PolygonGate, EllipseGate)) and \
            (gate.x_transform != x_tr or gate.y_transform != y_tr):
        return (f"drawn on {gate.x_transform} / {gate.y_transform} — "
                "switch the axes back to edit it, or right-click it in the "
                "gate list and choose Re-fit Gate to These Axes")
    return None


class GateEditor:
    """Drag handles for one gate on a matplotlib Axes (display space)."""

    def __init__(self, ax, gate: Gate, siblings: list[Gate], x_tr: str,
                 y_tr: str | None, on_done):
        self.ax, self.canvas = ax, ax.figure.canvas
        self.gate, self.x_tr, self.y_tr = gate, x_tr, y_tr
        self.siblings = siblings
        self.on_done = on_done
        self.color = gate._color or "#E74C3C"
        self._load_geometry()

        self._drag = None             # (kind, index, start_xy, start_geom)
        self._background = None
        self._handles = Line2D([], [], ls="none", marker="s", ms=7,
                               mfc="white", mec=self.color, mew=1.8, zorder=8,
                               animated=False)
        self._preview = Line2D([], [], color=self.color, lw=1.6, ls="--",
                               zorder=7, animated=True)
        ax.add_line(self._handles)
        ax.add_line(self._preview)
        self._update_handles()
        self._cids = [self.canvas.mpl_connect("motion_notify_event",
                                              self._on_motion),
                      self.canvas.mpl_connect("button_release_event",
                                              self._on_release)]

    # ------------------------------------------------------------------
    # Geometry in display coordinates
    # ------------------------------------------------------------------

    def _load_geometry(self):
        g, xt, yt = self.gate, self.x_tr, self.y_tr
        if isinstance(g, PolygonGate):
            self.geom = {"verts": [(_fwd(x, xt), _fwd(y, yt))
                                   for x, y in g.vertices]}
        elif isinstance(g, RectangleGate):
            self.geom = {"x0": _fwd(g.x_min, xt), "x1": _fwd(g.x_max, xt),
                         "y0": _fwd(g.y_min, yt), "y1": _fwd(g.y_max, yt)}
        elif isinstance(g, EllipseGate):
            self.geom = {"cx": g.cx, "cy": g.cy, "sx": g.semi_x,
                         "sy": g.semi_y, "ang": g.angle_deg}
        elif isinstance(g, RangeGate):
            self.geom = {"x0": _fwd(g.x_min, xt), "x1": _fwd(g.x_max, xt)}
        elif isinstance(g, QuadrantGate):
            self.geom = {"cx": _fwd(g.cx, xt), "cy": _fwd(g.cy, yt)}

    def _handle_points(self, geom) -> list[tuple[float, float]]:
        g = self.gate
        if isinstance(g, PolygonGate):
            return list(geom["verts"])
        if isinstance(g, RectangleGate):
            return [(geom["x0"], geom["y0"]), (geom["x1"], geom["y0"]),
                    (geom["x1"], geom["y1"]), (geom["x0"], geom["y1"])]
        if isinstance(g, EllipseGate):
            t = np.deg2rad(geom["ang"])
            return [(geom["cx"], geom["cy"]),
                    (geom["cx"] + geom["sx"] * np.cos(t),
                     geom["cy"] + geom["sx"] * np.sin(t)),
                    (geom["cx"] - geom["sy"] * np.sin(t),
                     geom["cy"] + geom["sy"] * np.cos(t))]
        if isinstance(g, RangeGate):
            ylo, yhi = self.ax.get_ylim()
            mid = ylo + 0.5 * (yhi - ylo)
            return [(geom["x0"], mid), (geom["x1"], mid)]
        if isinstance(g, QuadrantGate):
            return [(geom["cx"], geom["cy"])]
        return []

    def _outline(self, geom) -> np.ndarray:
        g = self.gate
        if isinstance(g, PolygonGate):
            v = np.array(geom["verts"])
            return np.vstack([v, v[:1]])
        if isinstance(g, RectangleGate):
            p = self._handle_points(geom)
            return np.array(p + p[:1])
        if isinstance(g, EllipseGate):
            t = np.linspace(0, 2 * np.pi, 120)
            a = np.deg2rad(geom["ang"])
            ex, ey = geom["sx"] * np.cos(t), geom["sy"] * np.sin(t)
            return np.column_stack([geom["cx"] + ex * np.cos(a) - ey * np.sin(a),
                                    geom["cy"] + ex * np.sin(a) + ey * np.cos(a)])
        ylo, yhi = self.ax.get_ylim()
        xlo, xhi = self.ax.get_xlim()
        if isinstance(g, RangeGate):
            return np.array([(geom["x0"], ylo), (geom["x0"], yhi),
                             (geom["x1"], yhi), (geom["x1"], ylo),
                             (geom["x0"], ylo)])
        if isinstance(g, QuadrantGate):     # crosshair drawn as one path
            return np.array([(xlo, geom["cy"]), (xhi, geom["cy"]),
                             (np.nan, np.nan), (geom["cx"], ylo),
                             (geom["cx"], yhi)])
        return np.empty((0, 2))

    def _inside(self, x, y) -> bool:
        if isinstance(self.gate, QuadrantGate):
            return False
        return MplPath(self._outline(self.geom)).contains_point((x, y))

    # ------------------------------------------------------------------
    # Mouse handling (press is forwarded by the plot widget)
    # ------------------------------------------------------------------

    def press(self, event) -> bool:
        """Start a drag if the press hits a handle or the gate body.
        Returns True when the press was consumed."""
        if event.inaxes != self.ax or event.xdata is None or event.button != 1:
            return False
        pts = self._handle_points(self.geom)
        if pts:
            scr = self.ax.transData.transform(np.array(pts))
            d = np.hypot(scr[:, 0] - event.x, scr[:, 1] - event.y)
            i = int(np.argmin(d))
            if d[i] <= PICK_PX:
                self._start("handle", i, event)
                return True
        if self._inside(event.xdata, event.ydata):
            self._start("move", -1, event)
            return True
        return False

    def _start(self, kind, index, event):
        import copy
        self._drag = (kind, index, (event.xdata, event.ydata),
                      copy.deepcopy(self.geom))
        self._moved = False
        self._handles.set_visible(False)
        self.canvas.draw()
        self._background = self.canvas.copy_from_bbox(self.ax.bbox)
        self._handles.set_visible(True)
        self._handles.set_animated(True)

    def _on_motion(self, event):
        if self._drag is None or event.inaxes != self.ax or event.xdata is None:
            return
        kind, i, (x0, y0), start = self._drag
        dx, dy = event.xdata - x0, event.ydata - y0
        self._moved = True
        g, geom = self.gate, {k: (list(v) if isinstance(v, list) else v)
                              for k, v in start.items()}
        x, y = event.xdata, event.ydata

        if kind == "move":
            if "verts" in geom:
                geom["verts"] = [(vx + dx, vy + dy) for vx, vy in geom["verts"]]
            for k in ("x0", "x1", "cx"):
                if k in geom:
                    geom[k] += dx
            for k in ("y0", "y1", "cy"):
                if k in geom:
                    geom[k] += dy
        elif isinstance(g, PolygonGate):
            geom["verts"][i] = (x, y)
        elif isinstance(g, RectangleGate):
            geom["x0" if i in (0, 3) else "x1"] = x
            geom["y0" if i in (0, 1) else "y1"] = y
        elif isinstance(g, EllipseGate):
            if i == 0:
                geom["cx"], geom["cy"] = start["cx"] + dx, start["cy"] + dy
            elif i == 1:                       # major axis: size + rotation
                vx, vy = x - geom["cx"], y - geom["cy"]
                geom["sx"] = max(np.hypot(vx, vy), 1e-9)
                geom["ang"] = float(np.degrees(np.arctan2(vy, vx)))
            else:                              # minor axis: size only
                a = np.deg2rad(geom["ang"])
                geom["sy"] = max(abs(-(x - geom["cx"]) * np.sin(a)
                                     + (y - geom["cy"]) * np.cos(a)), 1e-9)
        elif isinstance(g, RangeGate):
            geom["x0" if i == 0 else "x1"] = x
        elif isinstance(g, QuadrantGate):
            geom["cx"], geom["cy"] = x, y

        self.geom = geom
        self._draw_preview()

    def _draw_preview(self):
        self.canvas.restore_region(self._background)
        out = self._outline(self.geom)
        self._preview.set_data(out[:, 0], out[:, 1])
        pts = self._handle_points(self.geom)
        self._handles.set_data([p[0] for p in pts], [p[1] for p in pts])
        self.ax.draw_artist(self._preview)
        self.ax.draw_artist(self._handles)
        self.canvas.blit(self.ax.bbox)

    def _on_release(self, event):
        if self._drag is None:
            return
        self._drag = None
        self._handles.set_animated(False)
        if self._moved:
            self.on_done(self._changes())
        else:
            self._update_handles()

    def _update_handles(self):
        pts = self._handle_points(self.geom)
        self._handles.set_data([p[0] for p in pts], [p[1] for p in pts])

    def disconnect(self):
        for cid in self._cids:
            self.canvas.mpl_disconnect(cid)
        for art in (self._handles, self._preview):
            try:
                art.remove()
            except (ValueError, NotImplementedError):
                pass

    # ------------------------------------------------------------------
    # Result → gate attribute changes
    # ------------------------------------------------------------------

    def _changes(self) -> list[tuple[Gate, dict]]:
        g, geom, xt, yt = self.gate, self.geom, self.x_tr, self.y_tr
        if isinstance(g, PolygonGate):
            return [(g, {"vertices": [(_inv(x, xt), _inv(y, yt))
                                      for x, y in geom["verts"]]})]
        if isinstance(g, RectangleGate):
            xs = sorted([_inv(geom["x0"], xt), _inv(geom["x1"], xt)])
            ys = sorted([_inv(geom["y0"], yt), _inv(geom["y1"], yt)])
            return [(g, {"x_min": xs[0], "x_max": xs[1],
                         "y_min": ys[0], "y_max": ys[1]})]
        if isinstance(g, EllipseGate):
            return [(g, {"cx": geom["cx"], "cy": geom["cy"],
                         "semi_x": geom["sx"], "semi_y": geom["sy"],
                         "angle_deg": geom["ang"] % 360.0})]
        if isinstance(g, RangeGate):
            xs = sorted([_inv(geom["x0"], xt), _inv(geom["x1"], xt)])
            return [(g, {"x_min": xs[0], "x_max": xs[1]})]
        if isinstance(g, QuadrantGate):
            new = {"cx": _inv(geom["cx"], xt), "cy": _inv(geom["cy"], yt)}
            return [(q, dict(new))
                    for q in quadrant_siblings(g, self.siblings)]
        return []


# ---------------------------------------------------------------------------
# Numeric editing
# ---------------------------------------------------------------------------

def _num(text: str) -> float:
    return float(text.replace(",", "").strip())


class EditGateDialog(QDialog):
    """Exact values for a gate. ``changes()`` gives the edits to apply."""

    def __init__(self, gate: Gate, siblings: list[Gate],
                 all_gates: list[Gate], parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Edit Gate — {gate.name}")
        self.setMinimumWidth(420)
        self.gate, self.siblings, self.all_gates = gate, siblings, all_gates
        self._color = gate._color or "#E74C3C"
        self._fields: dict[str, QLineEdit] = {}

        lay = QVBoxLayout(self)
        form = QFormLayout()
        lay.addLayout(form)
        self._name = QLineEdit(gate.name)
        form.addRow("Name:", self._name)
        self._color_btn = QPushButton()
        self._color_btn.clicked.connect(self._pick_color)
        self._show_color()
        form.addRow("Colour:", self._color_btn)

        def field(key, label, value):
            e = QLineEdit(f"{value:.6g}")
            self._fields[key] = e
            form.addRow(label, e)

        g = gate
        if isinstance(g, PolygonGate):
            lay.addWidget(QLabel(f"Vertices (raw values of {g.x_channel} / "
                                 f"{g.y_channel}):"))
            self._table = QTableWidget(len(g.vertices), 2)
            self._table.setHorizontalHeaderLabels([g.x_channel, g.y_channel])
            self._table.horizontalHeader().setSectionResizeMode(
                QHeaderView.ResizeMode.Stretch)
            for r, (x, y) in enumerate(g.vertices):
                self._table.setItem(r, 0, QTableWidgetItem(f"{x:.6g}"))
                self._table.setItem(r, 1, QTableWidgetItem(f"{y:.6g}"))
            lay.addWidget(self._table)
        elif isinstance(g, RectangleGate):
            field("x_min", f"{g.x_channel} min:", g.x_min)
            field("x_max", f"{g.x_channel} max:", g.x_max)
            field("y_min", f"{g.y_channel} min:", g.y_min)
            field("y_max", f"{g.y_channel} max:", g.y_max)
        elif isinstance(g, EllipseGate):
            form.addRow(QLabel(
                f"<span style='color:#64748b;'>In plot units of the scales "
                f"it was drawn on ({g.x_transform} / {g.y_transform}).</span>"))
            field("cx", "Centre X:", g.cx)
            field("cy", "Centre Y:", g.cy)
            field("semi_x", "Half-width:", g.semi_x)
            field("semi_y", "Half-height:", g.semi_y)
            field("angle_deg", "Rotation (°):", g.angle_deg)
        elif isinstance(g, RangeGate):
            field("x_min", f"{g.x_channel} from:", g.x_min)
            field("x_max", f"{g.x_channel} to:", g.x_max)
        elif isinstance(g, QuadrantGate):
            form.addRow(QLabel("<span style='color:#64748b;'>Moves all four "
                               "quadrants together.</span>"))
            field("cx", f"{g.x_channel} split:", g.cx)
            field("cy", f"{g.y_channel} split:", g.cy)
        elif isinstance(g, BooleanGate):
            self._bool = _BooleanPicker(all_gates, g)
            lay.addWidget(self._bool)

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                              QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self._accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def _show_color(self):
        self._color_btn.setText(self._color)
        self._color_btn.setStyleSheet(
            f"background:{self._color}; color:white; font-weight:600;")

    def _pick_color(self):
        c = QColorDialog.getColor(QColor(self._color), self, "Gate colour")
        if c.isValid():
            self._color = c.name()
            self._show_color()

    def _accept(self):
        try:
            self._result = self._collect()
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid value", str(exc))
            return
        self.accept()

    def _collect(self) -> list[tuple[Gate, dict]]:
        g = self.gate
        name = self._name.text().strip()
        if not name:
            raise ValueError("The gate needs a name.")
        common = {"name": name, "_color": self._color}
        vals = {k: _num(e.text()) for k, e in self._fields.items()}
        if isinstance(g, PolygonGate):
            verts = []
            for r in range(self._table.rowCount()):
                verts.append((_num(self._table.item(r, 0).text()),
                              _num(self._table.item(r, 1).text())))
            return [(g, {**common, "vertices": verts})]
        if isinstance(g, (RectangleGate, RangeGate)):
            if vals["x_min"] >= vals["x_max"] or \
                    vals.get("y_min", 0) >= vals.get("y_max", 1):
                raise ValueError("Each minimum must be below its maximum.")
            return [(g, {**common, **vals})]
        if isinstance(g, EllipseGate):
            if vals["semi_x"] <= 0 or vals["semi_y"] <= 0:
                raise ValueError("Half-width and half-height must be > 0.")
            return [(g, {**common, **vals})]
        if isinstance(g, QuadrantGate):
            sibs = quadrant_siblings(g, self.siblings)
            return [(q, ({**common, **vals} if q is g
                         else {**vals, "_color": self._color}))
                    for q in sibs]
        if isinstance(g, BooleanGate):
            op, ids = self._bool.value()
            return [(g, {**common, "operator": op, "operand_ids": ids})]
        return [(g, common)]

    def changes(self) -> list[tuple[Gate, dict]]:
        return self._result


class _BooleanPicker(QWidget):
    """Operator + operand checklist (used inside the dialogs)."""

    def __init__(self, all_gates: list[Gate], current: BooleanGate | None):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        row.addWidget(QLabel("Events that are in"))
        self._op = QComboBox()
        self._op.addItems(["ALL of (AND)", "ANY of (OR)", "NONE of (NOT)"])
        if current is not None:
            self._op.setCurrentIndex(BooleanGate.OPERATORS.index(
                current.operator))
        row.addWidget(self._op)
        row.addStretch()
        lay.addLayout(row)
        self._list = QListWidget()
        for g in all_gates:
            if current is not None and g is current:
                continue
            path = " / ".join(a.name for a in g.ancestors() + [g])
            it = QListWidgetItem(path)
            it.setData(Qt.ItemDataRole.UserRole, g.gate_id)
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Checked
                             if current and g.gate_id in current.operand_ids
                             else Qt.CheckState.Unchecked)
            self._list.addItem(it)
        lay.addWidget(self._list)

    def value(self) -> tuple[str, list[str]]:
        ids = [self._list.item(i).data(Qt.ItemDataRole.UserRole)
               for i in range(self._list.count())
               if self._list.item(i).checkState() == Qt.CheckState.Checked]
        op = BooleanGate.OPERATORS[self._op.currentIndex()]
        if not ids:
            raise ValueError("Tick at least one gate.")
        if op == "AND" and len(ids) < 2:
            raise ValueError("AND needs at least two gates.")
        return op, ids


class BooleanGateDialog(QDialog):
    """Create a new AND / OR / NOT gate."""

    def __init__(self, all_gates: list[Gate], population_name: str,
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle("New Boolean Gate")
        self.setMinimumWidth(420)
        lay = QVBoxLayout(self)
        form = QFormLayout()
        self._name = QLineEdit("Boolean")
        form.addRow("Name:", self._name)
        lay.addLayout(form)
        self._picker = _BooleanPicker(all_gates, None)
        lay.addWidget(self._picker)
        lay.addWidget(QLabel(f"<span style='color:#64748b;'>Created inside "
                             f"<b>{population_name}</b> (the population "
                             f"shown in the plot).</span>"))
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                              QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self._accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def _accept(self):
        try:
            op, ids = self._picker.value()
            name = self._name.text().strip()
            if not name:
                raise ValueError("The gate needs a name.")
        except ValueError as exc:
            QMessageBox.warning(self, "Boolean Gate", str(exc))
            return
        self.gate = BooleanGate(name, op, ids)
        self.accept()
