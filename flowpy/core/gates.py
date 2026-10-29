"""Gate classes: Polygon, Rectangle, Ellipse, Quadrant, Range (1-D)."""

from __future__ import annotations
import uuid
import json
import numpy as np
from dataclasses import dataclass, field
from typing import Any

from .transforms import apply_transform, inverse_transform


# ---------------------------------------------------------------------------
# Base gate
# ---------------------------------------------------------------------------

class Gate:
    """Abstract base class for all gate types."""

    gate_type: str = "base"

    def __init__(self, name: str, x_channel: str, y_channel: str | None = None):
        self.gate_id: str = str(uuid.uuid4())[:8]
        self.name: str = name
        self.x_channel: str = x_channel
        self.y_channel: str | None = y_channel
        self.parent: "Gate | None" = None
        self.children: list["Gate"] = []
        self._color: str | None = None  # overridden per-instance for display
        # Display scales the gate was drawn on. Shape-based gates (polygon,
        # ellipse) are evaluated in this space so the drawn outline matches
        # exactly the events counted.
        self.x_transform: str = "Linear"
        self.y_transform: str = "Linear"

    # ------------------------------------------------------------------
    # Tree helpers
    # ------------------------------------------------------------------

    def add_child(self, child: "Gate"):
        child.parent = self
        self.children.append(child)

    def remove_child(self, child: "Gate"):
        if child in self.children:
            child.parent = None
            self.children.remove(child)

    def ancestors(self) -> list["Gate"]:
        chain = []
        node = self.parent
        while node is not None:
            chain.insert(0, node)
            node = node.parent
        return chain

    # ------------------------------------------------------------------
    # Gate application
    # ------------------------------------------------------------------

    def _candidates(self, x, y, xlo, xhi, ylo, yhi) -> np.ndarray:
        """Indices of events whose raw values fall in the display-space box
        [xlo, xhi] × [ylo, yhi] (transforms are monotonic, so the box maps
        to a raw-space box). Lets shape gates transform and test only the
        events near them — on 10 million events that is most of the cost."""
        def raw(lo, hi, tr):
            r = inverse_transform(np.array([lo, hi], dtype=np.float64), tr)
            a, b = float(r[0]), float(r[1])
            if tr == "Log10" and lo <= 0:      # log maps x <= 0 to 0
                a = -np.inf
            pad = 1e-9 * max(abs(a), abs(b), 1.0)
            return a - pad, b + pad
        ax, bx = raw(xlo, xhi, self.x_transform)
        ay, by = raw(ylo, yhi, self.y_transform)
        return np.flatnonzero((x >= ax) & (x <= bx) & (y >= ay) & (y <= by))

    def apply(self, x: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        """Return boolean mask for events inside this gate.

        Parameters
        ----------
        x : values for x_channel (1-D array)
        y : values for y_channel (1-D array or None for 1-D gates)
        """
        raise NotImplementedError

    def apply_to_dataframe(self, df) -> np.ndarray:
        """Convenience: apply using a DataFrame."""
        x = df[self.x_channel].to_numpy(dtype=float)
        y = df[self.y_channel].to_numpy(dtype=float) if self.y_channel else None
        return self.apply(x, y)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "gate_type": self.gate_type,
            "gate_id": self.gate_id,
            "name": self.name,
            "x_channel": self.x_channel,
            "y_channel": self.y_channel,
            "color": self._color,
            "x_transform": self.x_transform,
            "y_transform": self.y_transform,
        }

    def _restore_common(self, d: dict) -> "Gate":
        self.gate_id = d.get("gate_id", self.gate_id)
        self._color = d.get("color")
        self.x_transform = d.get("x_transform", "Linear")
        self.y_transform = d.get("y_transform", "Linear")
        return self

    @classmethod
    def from_dict(cls, d: dict) -> "Gate":
        raise NotImplementedError

    def boundary(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Outline of a shape gate as raw-space (xs, ys), densified so it
        can be re-plotted on any axis scale. None for non-shape gates."""
        return None

    def __repr__(self):
        return f"<{self.gate_type} '{self.name}' [{self.x_channel}]>"


# ---------------------------------------------------------------------------
# Polygon gate
# ---------------------------------------------------------------------------

class PolygonGate(Gate):
    """2-D polygon gate defined by a list of (x, y) vertex tuples."""

    gate_type = "polygon"

    def __init__(self, name: str, x_channel: str, y_channel: str,
                 vertices: list[tuple[float, float]]):
        super().__init__(name, x_channel, y_channel)
        self.vertices: list[tuple[float, float]] = list(vertices)

    def _display_vertices(self) -> np.ndarray:
        v = np.asarray(self.vertices, dtype=np.float64)
        return np.column_stack([apply_transform(v[:, 0], self.x_transform),
                                apply_transform(v[:, 1], self.y_transform)])

    def apply(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        from matplotlib.path import Path
        if len(self.vertices) < 3:
            return np.zeros(len(x), dtype=bool)
        verts = self._display_vertices()
        path = Path(verts)
        idx = self._candidates(x, y, verts[:, 0].min(),
                               verts[:, 0].max(), verts[:, 1].min(),
                               verts[:, 1].max())
        out = np.zeros(len(x), dtype=bool)
        if len(idx):
            points = np.column_stack([
                apply_transform(np.asarray(x[idx], np.float64),
                                self.x_transform),
                apply_transform(np.asarray(y[idx], np.float64),
                                self.y_transform)])
            out[idx] = path.contains_points(points)
        return out

    def boundary(self):
        v = self._display_vertices()
        closed = np.vstack([v, v[:1]])
        t = np.linspace(0.0, 1.0, 24, endpoint=False)
        pts = np.vstack([a + (b - a) * t[:, None]
                         for a, b in zip(closed[:-1], closed[1:])] + [v[:1]])
        return (inverse_transform(pts[:, 0], self.x_transform),
                inverse_transform(pts[:, 1], self.y_transform))

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["vertices"] = self.vertices
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "PolygonGate":
        g = cls(d["name"], d["x_channel"], d["y_channel"],
                [(float(p[0]), float(p[1])) for p in d["vertices"]])
        return g._restore_common(d)


# ---------------------------------------------------------------------------
# Rectangle gate
# ---------------------------------------------------------------------------

class RectangleGate(Gate):
    """2-D rectangle gate (axis-aligned)."""

    gate_type = "rectangle"

    def __init__(self, name: str, x_channel: str, y_channel: str,
                 x_min: float, x_max: float, y_min: float, y_max: float):
        super().__init__(name, x_channel, y_channel)
        self.x_min, self.x_max = float(x_min), float(x_max)
        self.y_min, self.y_max = float(y_min), float(y_max)

    def apply(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return ((x >= self.x_min) & (x <= self.x_max) &
                (y >= self.y_min) & (y <= self.y_max))

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update(x_min=self.x_min, x_max=self.x_max,
                 y_min=self.y_min, y_max=self.y_max)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "RectangleGate":
        g = cls(d["name"], d["x_channel"], d["y_channel"],
                d["x_min"], d["x_max"], d["y_min"], d["y_max"])
        return g._restore_common(d)


# ---------------------------------------------------------------------------
# Ellipse gate
# ---------------------------------------------------------------------------

class EllipseGate(Gate):
    """2-D ellipse gate (possibly rotated).

    Centre, semi-axes and angle are expressed in the gate's display space
    (``x_transform`` / ``y_transform``), which is plain raw space for Linear.
    """

    gate_type = "ellipse"

    def __init__(self, name: str, x_channel: str, y_channel: str,
                 cx: float, cy: float,
                 semi_x: float, semi_y: float, angle_deg: float = 0.0):
        super().__init__(name, x_channel, y_channel)
        self.cx, self.cy = float(cx), float(cy)
        self.semi_x, self.semi_y = float(semi_x), float(semi_y)
        self.angle_deg = float(angle_deg)

    def apply(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        theta = np.deg2rad(self.angle_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        hx = np.hypot(self.semi_x * cos_t, self.semi_y * sin_t)
        hy = np.hypot(self.semi_x * sin_t, self.semi_y * cos_t)
        idx = self._candidates(x, y, self.cx - hx, self.cx + hx,
                               self.cy - hy, self.cy + hy)
        out = np.zeros(len(x), dtype=bool)
        if not len(idx):
            return out
        dx = apply_transform(np.asarray(x[idx], np.float64),
                             self.x_transform) - self.cx
        dy = apply_transform(np.asarray(y[idx], np.float64),
                             self.y_transform) - self.cy
        x_rot = cos_t * dx + sin_t * dy
        y_rot = -sin_t * dx + cos_t * dy
        dist = (x_rot / self.semi_x) ** 2 + (y_rot / self.semi_y) ** 2
        out[idx] = dist <= 1.0
        return out

    def boundary(self):
        t = np.linspace(0.0, 2.0 * np.pi, 181)
        theta = np.deg2rad(self.angle_deg)
        ex, ey = self.semi_x * np.cos(t), self.semi_y * np.sin(t)
        xs = self.cx + ex * np.cos(theta) - ey * np.sin(theta)
        ys = self.cy + ex * np.sin(theta) + ey * np.cos(theta)
        return (inverse_transform(xs, self.x_transform),
                inverse_transform(ys, self.y_transform))

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update(cx=self.cx, cy=self.cy,
                 semi_x=self.semi_x, semi_y=self.semi_y,
                 angle_deg=self.angle_deg)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "EllipseGate":
        g = cls(d["name"], d["x_channel"], d["y_channel"],
                d["cx"], d["cy"], d["semi_x"], d["semi_y"],
                d.get("angle_deg", 0.0))
        return g._restore_common(d)


# ---------------------------------------------------------------------------
# Quadrant gate
# ---------------------------------------------------------------------------

class QuadrantGate(Gate):
    """Divides a 2-D plot into four quadrants at a crosshair (cx, cy).

    Yields four child-like masks named Q1–Q4 (but stored as a single gate
    object with a *quadrant* attribute selecting which quarter to keep).
    """

    gate_type = "quadrant"

    # Quadrant labels (FlowJo convention)
    QUADRANTS = {
        "Q1": ("left",  "top"),    # x < cx, y >= cy
        "Q2": ("right", "top"),    # x >= cx, y >= cy
        "Q3": ("left",  "bottom"), # x < cx, y < cy
        "Q4": ("right", "bottom"), # x >= cx, y < cy
    }

    def __init__(self, name: str, x_channel: str, y_channel: str,
                 cx: float, cy: float, quadrant: str = "Q1"):
        super().__init__(name, x_channel, y_channel)
        self.cx, self.cy = float(cx), float(cy)
        self.quadrant = quadrant  # "Q1" | "Q2" | "Q3" | "Q4"

    def apply(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        h, v = self.QUADRANTS.get(self.quadrant, ("right", "top"))
        x_mask = (x >= self.cx) if h == "right" else (x < self.cx)
        y_mask = (y >= self.cy) if v == "top" else (y < self.cy)
        return x_mask & y_mask

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update(cx=self.cx, cy=self.cy, quadrant=self.quadrant)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "QuadrantGate":
        g = cls(d["name"], d["x_channel"], d["y_channel"],
                d["cx"], d["cy"], d.get("quadrant", "Q1"))
        return g._restore_common(d)


# ---------------------------------------------------------------------------
# Range gate (1-D histogram gate)
# ---------------------------------------------------------------------------

class RangeGate(Gate):
    """1-D gate selecting events within [x_min, x_max] on one channel."""

    gate_type = "range"

    def __init__(self, name: str, x_channel: str, x_min: float, x_max: float):
        super().__init__(name, x_channel, y_channel=None)
        self.x_min, self.x_max = float(x_min), float(x_max)

    def apply(self, x: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        return (x >= self.x_min) & (x <= self.x_max)

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update(x_min=self.x_min, x_max=self.x_max)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "RangeGate":
        g = cls(d["name"], d["x_channel"], d["x_min"], d["x_max"])
        return g._restore_common(d)


# ---------------------------------------------------------------------------
# Boolean gate (AND / OR / NOT of other gates)
# ---------------------------------------------------------------------------

class BooleanGate(Gate):
    """Logical combination of other gates (FlowJo "Boolean gate").

    operator    : "AND" (in all), "OR" (in any), "NOT" (in none)
    operand_ids : gate_ids of the operand gates

    It has no shape: its mask is computed from the operands' masks (see
    ``combine``), then restricted to its parent population.
    """

    gate_type = "boolean"
    OPERATORS = ("AND", "OR", "NOT")

    def __init__(self, name: str, operator: str, operand_ids: list[str]):
        super().__init__(name, x_channel="")
        self.operator = operator.upper()
        if self.operator not in self.OPERATORS:
            raise ValueError(f"Unknown boolean operator: {operator}")
        self.operand_ids = list(operand_ids)

    def combine(self, masks: list[np.ndarray]) -> np.ndarray:
        if not masks:
            raise ValueError("Boolean gate has no operands")
        if self.operator == "AND":
            return np.logical_and.reduce(masks)
        any_ = np.logical_or.reduce(masks)
        return any_ if self.operator == "OR" else ~any_

    def apply(self, x: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        raise NotImplementedError(
            "Boolean gates are evaluated from their operands' masks")

    def describe(self, names: dict[str, str]) -> str:
        ops = [names.get(i, "?") for i in self.operand_ids]
        if self.operator == "NOT":
            return "NOT " + (ops[0] if len(ops) == 1
                             else "(" + " OR ".join(ops) + ")")
        return f" {self.operator} ".join(ops)

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update(operator=self.operator, operand_ids=self.operand_ids)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "BooleanGate":
        g = cls(d["name"], d["operator"], d["operand_ids"])
        return g._restore_common(d)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_GATE_CLASSES = {
    "polygon": PolygonGate,
    "rectangle": RectangleGate,
    "ellipse": EllipseGate,
    "quadrant": QuadrantGate,
    "range": RangeGate,
    "boolean": BooleanGate,
}


def gate_from_dict(d: dict) -> Gate:
    cls = _GATE_CLASSES.get(d.get("gate_type", ""))
    if cls is None:
        raise ValueError(f"Unknown gate type: {d.get('gate_type')}")
    return cls.from_dict(d)


# ---------------------------------------------------------------------------
# Colour cycling for new gates
# ---------------------------------------------------------------------------

_GATE_COLORS = [
    "#E74C3C", "#2ECC71", "#3498DB", "#F39C12", "#9B59B6",
    "#1ABC9C", "#E67E22", "#27AE60", "#8E44AD", "#2980B9",
]
_color_idx = 0


def next_gate_color() -> str:
    global _color_idx
    c = _GATE_COLORS[_color_idx % len(_GATE_COLORS)]
    _color_idx += 1
    return c


# ---------------------------------------------------------------------------
# Moving a gate onto different axis scales
# ---------------------------------------------------------------------------

#: Gates whose region does not depend on the scale it was drawn on: an
#: axis-aligned box stays an axis-aligned box under any monotonic
#: per-axis transform, so only the recorded scales need updating.
SCALE_FREE = ("rectangle", "range", "quadrant", "boolean")


def can_refit(gate: "Gate", x_tr: str, y_tr: str | None) -> bool:
    """True when *gate* is not already on these scales."""
    if gate.gate_type == "boolean":
        return False
    same_x = gate.x_transform == x_tr
    same_y = gate.y_channel is None or gate.y_transform == y_tr
    return not (same_x and same_y)


def refit(gate: "Gate", x_tr: str, y_tr: str | None) -> "Gate":
    """A copy of *gate* expressed on the given scales, holding the same
    events.

    A gate is evaluated in the space it was drawn on, so a polygon drawn
    on a linear plot is a straight-edged shape there and a curved one on
    a log plot. Re-fitting keeps the region — and therefore the
    population — and re-describes it in the new space so it can be seen
    and edited there.

    The curve is approximated by the densified outline, so membership is
    preserved to within that resolution rather than exactly; refit_error()
    reports how many events actually move. An ellipse becomes a polygon,
    because an ellipse on one scale is not an ellipse on another.
    """
    new = gate_from_dict(gate.to_dict())     # independent copy
    new.gate_id = gate.gate_id
    new.name = gate.name
    new._color = gate._color
    if gate.gate_type in SCALE_FREE or not can_refit(gate, x_tr, y_tr):
        new.x_transform = x_tr
        if new.y_channel is not None:
            new.y_transform = y_tr or new.y_transform
        return new

    outline = gate.boundary()
    if outline is None:                      # nothing geometric to move
        return new
    xs, ys = outline
    poly = PolygonGate(gate.name, gate.x_channel, gate.y_channel,
                       [(float(a), float(b)) for a, b in zip(xs, ys)])
    poly.gate_id = gate.gate_id
    poly._color = gate._color
    poly.x_transform = x_tr
    poly.y_transform = y_tr or gate.y_transform
    return poly


def refit_error(gate: "Gate", new: "Gate", x: np.ndarray,
                y: np.ndarray | None) -> int:
    """How many events change hands between *gate* and its re-fit — the
    honest cost of the approximation, for the interface to show."""
    try:
        return int(np.count_nonzero(gate.apply(x, y) != new.apply(x, y)))
    except Exception:
        return 0
