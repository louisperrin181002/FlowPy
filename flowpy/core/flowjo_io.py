"""Exchange gates with FlowJo (.wsp) and Gating-ML 2.0 through FlowKit, and
cross-check FlowPy's counts with FlowKit's gating engine (which is validated
against the Gating-ML 2.0 compliance tests).

Translation is exact for FlowPy's scales (Linear, Log10 and Logicle with
T=262144, W=0.5, M=4.5, A=0). Gates drawn in FlowJo on other scales (its
default "biex", hyperlog, arcsinh, other Logicle widths) are converted by
following their outline closely; `verify` reports any count difference so
you can see whether it matters.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

from .gates import (
    Gate, PolygonGate, RectangleGate, EllipseGate, QuadrantGate, RangeGate,
    BooleanGate,
)
from .transforms import apply_transform
from .protocol import _bfs, compute_masks

LOGICLE = dict(param_t=262144.0, param_w=0.5, param_m=4.5, param_a=0.0)
LOGICLE_M = 4.5
TID = {"Logicle": "flowpy_logicle", "Log10": "flowpy_log10"}
_BIG = 1e12          # stands in for an open-ended rectangle edge
COMP_ID = "FlowPy_compensation"


def _xml_id(name: str) -> str:
    """Gating-ML gate IDs must be XML IDs: no spaces, not starting with a
    digit."""
    safe = re.sub(r"[^A-Za-z0-9_.\-]", "_", name)
    return safe if re.match(r"[A-Za-z_]", safe) else "_" + safe


@dataclass
class Report:
    """What happened during an import / export / verification."""
    notes: list[str] = field(default_factory=list)
    # (sample, population path, FlowPy count, FlowKit count)
    counts: list[tuple[str, str, int, int]] = field(default_factory=list)

    @property
    def mismatches(self):
        return [c for c in self.counts if c[2] != c[3]]


def _fk():
    try:
        import flowkit
    except ImportError as exc:
        raise RuntimeError("This needs the 'flowkit' package "
                           "(pip install flowkit)") from exc
    return flowkit


# ---------------------------------------------------------------------------
# FlowPy → FlowKit
# ---------------------------------------------------------------------------

def _fk_transforms():
    fk = _fk()
    return {TID["Logicle"]: fk.transforms.LogicleTransform(**LOGICLE),
            # Gating-ML flog(x) = log10(x/T)/M + 1, which with T=10, M=1 is
            # exactly FlowPy's log10(x)
            TID["Log10"]: fk.transforms.LogTransform(param_t=10.0,
                                                     param_m=1.0)}


def _to_fk_units(values, tr: str) -> np.ndarray:
    """FlowPy display coordinates → FlowKit transformed coordinates."""
    v = np.asarray(values, dtype=float)
    return v / LOGICLE_M if tr == "Logicle" else v


def _display(raw, tr: str) -> np.ndarray:
    return apply_transform(np.asarray(raw, dtype=float), tr)


class _Exporter:
    """FlowPy gates → FlowKit gates.

    *channel_scales* (FlowJo mode) forces one scale per channel, because a
    FlowJo workspace has a single scale per parameter: gates on another
    scale are then re-expressed on it (exactly for rectangles, ranges and
    quadrants; by tracing the outline for polygons and ellipses)."""

    def __init__(self, sample, report: Report,
                 channel_scales: dict[str, str] | None = None,
                 comp_id: str | None = None):
        self.s, self.report = sample, report
        self.scales = channel_scales
        self.comp_id = comp_id
        self.paths: dict[str, tuple] = {}      # gate_id → FlowKit gate path
        self.names: dict[str, str] = {}        # gate_id → FlowKit gate name
        self.comp_channels = (set(sample.comp_matrix.channels)
                              if sample.comp_matrix else set())

    def scale(self, channel: str, tr: str | None) -> str | None:
        return self.scales.get(channel, "Linear") if self.scales is not None \
            else tr

    def dim(self, channel: str, tr: str | None, **kw):
        fk = _fk()
        comp = (self.comp_id if channel in self.comp_channels and
                self.comp_id else "uncompensated")
        tr = self.scale(channel, tr)
        return fk.Dimension(self.s.pnn.get(channel, channel),
                            compensation_ref=comp,
                            transformation_ref=TID.get(tr) if tr else None,
                            **kw)

    def bound(self, channel: str, raw: float | None, **_) -> dict:
        """A rectangle edge (raw value) in the units of the channel's
        exported scale."""
        tr = self.scale(channel, None)
        if raw is None or tr in (None, "Linear"):
            return raw
        return float(_to_fk_units(_display([raw], tr), tr)[0])

    def rect_dim(self, channel: str, lo=None, hi=None):
        kw = {}
        if lo is not None:
            kw["range_min"] = self.bound(channel, lo)
        if hi is not None:
            kw["range_max"] = self.bound(channel, hi)
        return self.dim(channel, None, **kw)

    def outline_polygon(self, g: Gate, name: str):
        """Trace a shape gate's outline on the exported scales."""
        fk = _fk()
        xs, ys = g.boundary()
        tx, ty = self.scale(g.x_channel, None), self.scale(g.y_channel, None)
        if (tx, ty) != (g.x_transform, g.y_transform):
            self.report.notes.append(
                f"'{g.name}' was drawn on {g.x_transform}/{g.y_transform}; "
                f"FlowJo shows these channels on {tx}/{ty}, so its outline "
                "was traced onto that scale")
        else:
            self.report.notes.append(
                f"'{g.name}' exported as a polygon following the ellipse "
                "(FlowKit cannot write FlowJo ellipse gates yet)")
        fx = _to_fk_units(_display(xs, tx), tx)
        fy = _to_fk_units(_display(ys, ty), ty)
        return fk.gates.PolygonGate(
            name, [self.dim(g.x_channel, None), self.dim(g.y_channel, None)],
            [[float(a), float(b)] for a, b in zip(fx, fy)])

    def exportable(self, g: Gate) -> bool:
        if isinstance(g, BooleanGate) and self.scales is not None:
            self.report.notes.append(
                f"'{g.name}' (AND/OR/NOT) not included: FlowKit cannot write "
                "FlowJo Boolean gates yet — use Gating-ML export, or recreate "
                "it in FlowJo")
            return False
        chans = [] if isinstance(g, BooleanGate) else \
            [c for c in (g.x_channel, g.y_channel) if c]
        bad = [c for c in chans if c in self.s.derived or c not in self.s.pnn]
        if bad:
            self.report.notes.append(
                f"'{g.name}' skipped (with its sub-gates): {', '.join(bad)} "
                "is computed by FlowPy and unknown to FlowJo")
            return False
        if isinstance(g, BooleanGate) and \
                not all(i in self.paths for i in g.operand_ids):
            self.report.notes.append(f"'{g.name}' skipped: one of its "
                                     "operand gates was not exported")
            return False
        return True

    def convert(self, g: Gate):
        fk = _fk()
        G = fk.gates
        name = self.names[g.gate_id]
        if self.scales is not None and (isinstance(g, EllipseGate) or (
                isinstance(g, PolygonGate) and
                (self.scale(g.x_channel, None) != g.x_transform or
                 self.scale(g.y_channel, None) != g.y_transform))):
            return self.outline_polygon(g, name)
        if isinstance(g, PolygonGate):
            v = np.asarray(g.vertices, dtype=float)
            xs = _to_fk_units(_display(v[:, 0], g.x_transform), g.x_transform)
            ys = _to_fk_units(_display(v[:, 1], g.y_transform), g.y_transform)
            return G.PolygonGate(name, [self.dim(g.x_channel, g.x_transform),
                                        self.dim(g.y_channel, g.y_transform)],
                                 [[float(a), float(b)] for a, b in zip(xs, ys)])
        if isinstance(g, EllipseGate):
            sx = LOGICLE_M if g.x_transform == "Logicle" else 1.0
            sy = LOGICLE_M if g.y_transform == "Logicle" else 1.0
            t = np.deg2rad(g.angle_deg)
            u, w = np.array([np.cos(t), np.sin(t)]), np.array([-np.sin(t), np.cos(t)])
            cov = g.semi_x ** 2 * np.outer(u, u) + g.semi_y ** 2 * np.outer(w, w)
            scale = np.diag([1 / sx, 1 / sy])
            cov = scale @ cov @ scale
            return G.EllipsoidGate(
                name, [self.dim(g.x_channel, g.x_transform),
                       self.dim(g.y_channel, g.y_transform)],
                [g.cx / sx, g.cy / sy], cov.tolist(), 1.0)
        if isinstance(g, RectangleGate):
            return G.RectangleGate(name, [
                self.rect_dim(g.x_channel, g.x_min, g.x_max),
                self.rect_dim(g.y_channel, g.y_min, g.y_max)])
        if isinstance(g, RangeGate):
            return G.RectangleGate(name, [self.rect_dim(g.x_channel,
                                                        g.x_min, g.x_max)])
        if isinstance(g, QuadrantGate):
            # one FlowPy quadrant = a half-open rectangle (exact)
            h, v = QuadrantGate.QUADRANTS[g.quadrant]
            xd = (dict(lo=g.cx) if h == "right" else dict(hi=g.cx))
            yd = (dict(lo=g.cy) if v == "top" else dict(hi=g.cy))
            return G.RectangleGate(name, [self.rect_dim(g.x_channel, **xd),
                                          self.rect_dim(g.y_channel, **yd)])
        if isinstance(g, BooleanGate):
            refs = [{"ref": self.names[i], "path": self.paths[i],
                     "complement": g.operator == "NOT"}
                    for i in g.operand_ids]
            if g.operator == "NOT" and len(refs) == 1:
                for r in refs:
                    r["complement"] = False
                return G.BooleanGate(name, "not", refs)
            # NOT of several = in none of them = AND of the complements
            return G.BooleanGate(name, "and" if g.operator != "OR" else "or",
                                 refs)
        raise ValueError(f"Cannot export gate type {type(g).__name__}")


def flowjo_scales(sample) -> dict[str, str]:
    """One scale per channel (FlowJo's rule): the scale most shape gates on
    that channel were drawn on; Logicle for other fluorescence channels."""
    votes: dict[str, dict[str, int]] = {}
    for g in _bfs(sample.gates):
        if isinstance(g, (PolygonGate, EllipseGate)):
            for ch, tr in ((g.x_channel, g.x_transform),
                           (g.y_channel, g.y_transform)):
                votes.setdefault(ch, {}).setdefault(tr, 0)
                votes[ch][tr] += 1
    scales = {ch: max(v, key=v.get) for ch, v in votes.items()}
    for ch in sample.fluorescence:
        scales.setdefault(ch, "Logicle")
    return scales


def build_session(samples, template, report: Report, flowjo: bool = False,
                  xml_ids: bool = False):
    """A FlowKit Session holding *template*'s gates for all *samples*;
    samples whose own gates differ in shape get custom sample gates.
    *flowjo* applies FlowJo's one-scale-per-channel rule; *xml_ids* makes
    gate names valid Gating-ML IDs. The template's compensation matrix is
    written out explicitly. Returns (session, {gate_id: (name, path)})."""
    fk = _fk()
    session = fk.Session()
    for tid, tr in _fk_transforms().items():
        session.add_transform(tid, tr)

    comp_id = None
    cm = template.comp_matrix
    if cm is not None:
        comp_id = COMP_ID
        session.add_comp_matrix(comp_id, fk.Matrix(
            np.asarray(cm.matrix, float),
            [template.pnn.get(c, c) for c in cm.channels],
            fluorochromes=list(cm.channels)))
        for s in samples:
            if s.comp_matrix is None:
                report.notes.append(
                    f"{s.name} is not compensated but the gates use "
                    f"compensated channels: it gets {template.name}'s "
                    "matrix in the export, so its counts will differ")

    scales = flowjo_scales(template) if flowjo else None
    exp = _Exporter(template, report, scales, comp_id)
    used: dict[tuple, set] = {}
    for g in _bfs(template.gates):
        if g.parent is not None and g.parent.gate_id not in exp.paths:
            continue                                  # parent was skipped
        if not exp.exportable(g):
            continue
        path = (("root",) if g.parent is None else
                exp.paths[g.parent.gate_id] + (exp.names[g.parent.gate_id],))
        name = _xml_id(g.name) if xml_ids else g.name.replace("/", "-")
        if name != g.name:
            report.notes.append(f"'{g.name}' written as '{name}'")
        taken = used.setdefault(path, set())
        if name in taken:
            n = 2
            while f"{name} ({n})" in taken:
                n += 1
            report.notes.append(f"'{name}' renamed '{name} ({n})' "
                                "(gate names must be unique per level)")
            name = f"{name} ({n})"
        taken.add(name)
        exp.paths[g.gate_id], exp.names[g.gate_id] = path, name
        session.add_gate(exp.convert(g), path)

    ids = {gid: (exp.names[gid], exp.paths[gid]) for gid in exp.paths}
    for s in samples:
        session.add_samples(fk.Sample(s.filepath, sample_id=s.name))
        if s is template:
            continue
        mine = {g.gate_id: g for g in _bfs(s.gates)}
        if set(mine) != set(g.gate_id for g in _bfs(template.gates)):
            report.notes.append(f"{s.name}: its gate tree differs from "
                                f"{template.name}'s; exported with "
                                f"{template.name}'s gates")
            continue
        e = _Exporter(s, report, scales, comp_id)
        e.paths, e.names = exp.paths, exp.names
        for gid in ids:
            a, b = mine[gid], next(g for g in _bfs(template.gates)
                                   if g.gate_id == gid)
            if a.to_dict() != {**b.to_dict(), "name": a.name}:
                session.add_gate(e.convert(a), ids[gid][1],
                                 sample_id=s.name)
    return session, ids


def verify(samples) -> Report:
    """Recount every gate of every sample (each with its own gates and
    compensation) with FlowKit, and compare with FlowPy's counts."""
    report = Report()
    for s in samples:
        if not s.gates:
            continue
        session, ids = build_session([s], s, report)
        session.analyze_samples(use_mp=False)
        res = session.get_gating_results(s.name)
        for gid, (name, path) in ids.items():
            mask = s.gate_masks.get(gid)
            if mask is None:
                continue
            fk_n = int(res.get_gate_count(name, gate_path=path))
            pop = " / ".join(path[1:] + (name,))
            report.counts.append((s.name, pop, int(mask.sum()), fk_n))
    return report


def export_wsp(path: str | Path, samples, template=None,
               group: str = "FlowPy") -> Report:
    report = Report()
    template = template or next(s for s in samples if s.gates)
    session, _ = build_session(samples, template, report, flowjo=True)
    with open(path, "wb") as fh:
        session.export_wsp(fh, group)
    _finish_wsp(path, samples)
    return report


def _finish_wsp(path, samples):
    """Make FlowKit's .wsp look like FlowJo's own: each <Sample> carries its
    spillover matrix (FlowJo applies compensation per sample, and FlowKit
    only writes the shared <Matrices> list), and points at its FCS file."""
    from lxml import etree
    tree = etree.parse(str(path))
    root = tree.getroot()
    matrices = root.find("Matrices")
    matrix = matrices[0] if matrices is not None and len(matrices) else None
    by_name = {s.name: s for s in samples}
    for sample_el in root.iter("Sample"):
        ds = sample_el.find("DataSet")
        node = sample_el.find("SampleNode")
        s = by_name.get(node.get("name")) if node is not None else None
        if ds is not None and s is not None:
            ds.set("uri", Path(s.filepath).as_uri())
        if matrix is not None and ds is not None and not any(
                c.tag.endswith("spilloverMatrix") for c in sample_el):
            own = etree.fromstring(etree.tostring(matrix))
            if s is not None and s.comp_matrix is not None:
                _set_coefficients(own, s)
            ds.addnext(own)
    tree.write(str(path), xml_declaration=True, encoding="UTF-8")


def export_gatingml(path: str | Path, sample) -> Report:
    report = Report()
    session, _ = build_session([sample], sample, report, xml_ids=True)
    with open(path, "wb") as fh:
        session.export_gml(fh, sample_id=sample.name)
    return report


# ---------------------------------------------------------------------------
# FlowKit → FlowPy
# ---------------------------------------------------------------------------

class _Importer:
    """Turns FlowKit gates (one sample's or a template's) into FlowPy gates."""

    def __init__(self, sample, transforms: dict, report: Report,
                 who: str):
        self.s, self.transforms, self.report, self.who = \
            sample, transforms, report, who
        self.by_pnn = ({v: k for k, v in sample.pnn.items()}
                       if sample is not None else {})
        self.approx = 0

    # --- channels and scales ---------------------------------------------

    def channel(self, dim_id: str) -> str:
        # FlowJo names compensated parameters "Comp-<channel>"
        if dim_id not in self.by_pnn and dim_id.startswith("Comp-"):
            dim_id = dim_id[len("Comp-"):]
        return self.by_pnn.get(dim_id, dim_id)

    def scale_of(self, tref):
        """(FlowPy transform name, exact?) for a FlowKit transform id."""
        if tref is None:
            return "Linear", True
        t = self.transforms.get(tref)
        name = type(t).__name__
        if name == "LogicleTransform" and np.allclose(
                [t.param_t, t.param_w, t.param_m, t.param_a],
                list(LOGICLE.values())):
            return "Logicle", True
        # "LogTransform" is Gating-ML's; "WSPLogTransform" is the log scale
        # FlowJo 10 itself writes. Both are plain log10 of the channel value
        # once their normalisation is undone, which is exactly FlowPy's
        # Log10 — see to_raw() for the inverse.
        if name in ("LogTransform", "WSPLogTransform"):
            return "Log10", True
        if name == "LinearTransform":
            return "Linear", False
        return "Logicle", False

    def to_raw(self, values, tref) -> np.ndarray:
        v = np.asarray(values, dtype=float)
        if tref is None:
            return v
        t = self.transforms[tref]
        # FlowKit's WSPLogTransform has no inverse() — it only goes forwards,
        # as x -> (log10(x) - log10(offset)) / decades. Without this, every
        # gate FlowJo drew on a log axis was dropped on import ("object has
        # no attribute 'inverse'") and nothing appeared in FlowPy.
        if type(t).__name__ == "WSPLogTransform":
            return float(t.offset) * 10.0 ** (v * float(t.decades))
        return np.asarray(t.inverse(v.reshape(-1, 1)), dtype=float).ravel()

    def to_display(self, values, tref, tr: str) -> np.ndarray:
        """FlowKit transformed coords → FlowPy display coords of *tr*."""
        name, exact = self.scale_of(tref)
        v = np.asarray(values, dtype=float)
        if exact and name == "Logicle":
            return v * LOGICLE_M
        if exact and name == "Log10" and \
                type(self.transforms.get(tref)).__name__ == "LogTransform":
            t = self.transforms[tref]
            return np.log10(t.param_t) + (v - 1.0) * t.param_m
        return _display(self.to_raw(v, tref), tr)

    # --- gates -------------------------------------------------------------

    def convert(self, g, name: str) -> list[Gate]:
        kind = type(g).__name__
        dims = getattr(g, "dimensions", None) or []
        if kind == "PolygonGate":
            return [self._polygon(name, dims, np.asarray(g.vertices, float))]
        if kind == "EllipsoidGate":
            return [self._ellipse(name, dims, g)]
        if kind == "RectangleGate":
            return [self._rectangle(name, dims)]
        if kind == "QuadrantGate":
            return self._quadrants(g)
        raise ValueError(f"{kind} is not supported")

    def _scales(self, dims):
        (tx, ex), (ty, ey) = (self.scale_of(d.transformation_ref)
                              for d in dims[:2])
        if not (ex and ey):
            self.approx += 1
        return tx, ty, ex and ey

    def _polygon(self, name, dims, verts) -> PolygonGate:
        dx, dy = dims[:2]
        tx, ty, exact = self._scales(dims)
        if not exact:
            # follow FlowJo's straight edges, which are curved on our scale
            closed = np.vstack([verts, verts[:1]])
            t = np.linspace(0, 1, 24, endpoint=False)[:, None]
            verts = np.vstack([a + (b - a) * t
                               for a, b in zip(closed[:-1], closed[1:])])
        xs = self.to_display(verts[:, 0], dx.transformation_ref, tx)
        ys = self.to_display(verts[:, 1], dy.transformation_ref, ty)
        g = PolygonGate(name, self.channel(dx.id), self.channel(dy.id),
                        [(float(a), float(b)) for a, b in
                         zip(apply_inverse(xs, tx), apply_inverse(ys, ty))])
        g.x_transform, g.y_transform = tx, ty
        return g

    def _ellipse(self, name, dims, g) -> Gate:
        dx, dy = dims[:2]
        tx, ty, exact = self._scales(dims)
        c = np.asarray(g.coordinates, float)
        cov = np.asarray(g.covariance_matrix, float) * float(g.distance_square)
        if not exact:                      # trace its outline as a polygon
            vals, vecs = np.linalg.eigh(cov)
            t = np.linspace(0, 2 * np.pi, 120, endpoint=False)
            pts = c + (vecs @ (np.sqrt(np.maximum(vals, 0))[:, None]
                               * np.vstack([np.cos(t), np.sin(t)]))).T
            self.approx -= 1               # counted again in _polygon
            return self._polygon(name, dims, pts)
        sx = LOGICLE_M if tx == "Logicle" else 1.0
        sy = LOGICLE_M if ty == "Logicle" else 1.0
        if tx == "Log10" or ty == "Log10":
            ms = [self.transforms[d.transformation_ref].param_m
                  if self.scale_of(d.transformation_ref)[0] == "Log10" else 1.0
                  for d in (dx, dy)]
            sx, sy = (ms[0] if tx == "Log10" else sx,
                      ms[1] if ty == "Log10" else sy)
        cx = self.to_display([c[0]], dx.transformation_ref, tx)[0]
        cy = self.to_display([c[1]], dy.transformation_ref, ty)[0]
        S = np.diag([sx, sy])
        vals, vecs = np.linalg.eigh(S @ cov @ S)
        major = vecs[:, 1]
        e = EllipseGate(name, self.channel(dx.id), self.channel(dy.id),
                        float(cx), float(cy), float(np.sqrt(vals[1])),
                        float(np.sqrt(max(vals[0], 1e-12))),
                        float(np.degrees(np.arctan2(major[1], major[0]))))
        e.x_transform, e.y_transform = tx, ty
        return e

    def _bound(self, value, tref, default):
        if value is None:
            return default
        return float(self.to_raw([value], tref)[0])

    def _rectangle(self, name, dims) -> Gate:
        lims = [(self._bound(d.min, d.transformation_ref, -_BIG),
                 self._bound(d.max, d.transformation_ref, _BIG))
                for d in dims]
        if len(dims) == 1:
            g = RangeGate(name, self.channel(dims[0].id), *lims[0])
        else:
            g = RectangleGate(name, self.channel(dims[0].id),
                              self.channel(dims[1].id), *lims[0], *lims[1])
        for d, attr in zip(dims, ("x_transform", "y_transform")):
            setattr(g, attr, self.scale_of(d.transformation_ref)[0])
        return g

    def _quadrants(self, g) -> list[Gate]:
        divs = list(g.dimensions)
        if len(divs) != 2 or any(len(d.values) != 1 for d in divs):
            raise ValueError("only 2-D quadrant gates with one split per "
                             "axis are supported")
        dx, dy = divs
        cx = float(self.to_raw([dx.values[0]], dx.transformation_ref)[0])
        cy = float(self.to_raw([dy.values[0]], dy.transformation_ref)[0])
        out = []
        for qid, q in g.quadrants.items():
            side = {}
            for ref, rng in zip(q.divider_refs, q.divider_ranges):
                lo, _hi = rng
                side[ref] = "high" if lo is not None else "low"
            h = "right" if side.get(dx.id) == "high" else "left"
            v = "top" if side.get(dy.id) == "high" else "bottom"
            quad = next(k for k, val in QuadrantGate.QUADRANTS.items()
                        if val == (h, v))
            qg = QuadrantGate(qid, self.channel(dx.dimension_ref),
                              self.channel(dy.dimension_ref), cx, cy, quad)
            qg.x_transform = self.scale_of(dx.transformation_ref)[0]
            qg.y_transform = self.scale_of(dy.transformation_ref)[0]
            out.append(qg)
        return out


def _set_coefficients(matrix_el, sample):
    """Write *sample*'s own spillover values into a copied FlowJo
    spilloverMatrix element (rows = spilling channel, as in FlowJo)."""
    cm = sample.comp_matrix
    idx = {sample.pnn.get(c, c): i for i, c in enumerate(cm.channels)}
    for sp in matrix_el:
        if not sp.tag.endswith("spillover"):
            continue
        row = idx.get(next(v for k, v in sp.attrib.items()
                           if k.endswith("parameter")))
        for coef in sp:
            col = idx.get(next(v for k, v in coef.attrib.items()
                               if k.endswith("parameter")))
            if row is not None and col is not None:
                key = next(k for k in coef.attrib if k.endswith("value"))
                coef.set(key, repr(float(cm.matrix[row, col])))


def apply_inverse(display, tr):
    from .transforms import inverse_transform
    return inverse_transform(np.asarray(display, float), tr)


def _build_tree(gate_ids, get_gate, importer: _Importer,
                report: Report) -> list[Gate]:
    """FlowKit (name, path) gate list → FlowPy gate forest."""
    from .gates import next_gate_color
    made: dict[tuple, Gate] = {}           # (name, path) → FlowPy gate
    roots: list[Gate] = []
    pending_bool = []
    for name, path in sorted(gate_ids, key=lambda gp: len(gp[1])):
        parent = made.get((path[-1], path[:-1])) if len(path) > 1 else None
        if len(path) > 1 and parent is None:
            continue                       # parent skipped
        fkg = get_gate(name, path)
        try:
            if type(fkg).__name__ == "BooleanGate":
                b = BooleanGate(name, "AND", ["?"])
                pending_bool.append((b, fkg))
                new = [b]
            else:
                new = importer.convert(fkg, name)
        except Exception as exc:
            report.notes.append(f"{importer.who}: '{name}' skipped ({exc})")
            continue
        color = next_gate_color()
        for g in new:
            g._color = color
            if parent is not None:
                parent.add_child(g)
            else:
                roots.append(g)
        # a FlowKit quadrant gate's quadrants are addressed as its children
        if len(new) == 1:
            made[(name, path)] = new[0]
        else:
            for g in new:
                made[(g.name, path + (name,))] = g
                made[(g.name, path)] = g
    for b, fkg in pending_bool:
        try:
            refs = fkg.gate_refs
            ids = [made[(r["ref"], tuple(r["path"]))].gate_id for r in refs]
            comps = [bool(r.get("complement")) for r in refs]
            if fkg.type == "not" or (fkg.type == "and" and all(comps)):
                b.operator = "NOT"
            elif not any(comps):
                b.operator = fkg.type.upper()
            else:
                raise ValueError("mixed NOT inside AND/OR")
            b.operand_ids = ids
        except Exception as exc:
            report.notes.append(f"{importer.who}: boolean '{b.name}' "
                                f"skipped ({exc})")
            _detach(b, roots)
    return roots


def _detach(g: Gate, roots: list[Gate]):
    if g.parent is not None:
        g.parent.remove_child(g)
    elif g in roots:
        roots.remove(g)


def wsp_sample_files(path: str | Path) -> dict[str, list[str]]:
    """{sample name in the workspace: the files it was recorded for, in the
    order FlowJo wrote them}.

    FlowJo tells its samples apart by an internal sampleID, so one workspace
    may hold two tubes called "124 - LB" from two experiments. FlowKit keys
    its samples by name, so of each such pair only the last one survives —
    silently. We read the paths ourselves to see which file a gate set
    really belongs to."""
    import xml.etree.ElementTree as ET
    from urllib.parse import unquote, urlparse

    def _path(uri: str) -> str:
        p = unquote(urlparse(uri).path if "//" in uri
                    else uri.split(":", 1)[-1])
        return str((Path(path).parent / p).resolve() if not p.startswith("/")
                   else Path(p).resolve())

    out: dict[str, list[str]] = {}
    try:
        root = ET.parse(str(path)).getroot()
    except ET.ParseError:
        return out
    for sm in root.iter():
        if sm.tag.split("}")[-1] != "Sample":
            continue
        uri = next((c.get("uri") for c in sm
                    if c.tag.split("}")[-1] == "DataSet" and c.get("uri")),
                   None)
        name = next((c.get("name") for c in sm
                     if c.tag.split("}")[-1] == "SampleNode"), None)
        if uri and name:
            out.setdefault(name, []).append(_path(uri))
    return out


def import_wsp(path: str | Path, samples) -> tuple[dict, Report]:
    """Gates (and compensation) for each loaded sample found in a FlowJo
    workspace. Returns ({sample: roots}, report); samples get compensated
    with the workspace's matrix."""
    from .compensation import CompensationMatrix
    fk = _fk()
    report = Report()
    # FlowKit opens every file it is handed, all at once. A sample whose FCS
    # file is no longer where it was opened from — a network share that has
    # since unmounted, data copied to another machine — therefore aborted
    # the whole import with a bare "No such file or directory". Everywhere
    # else in FlowPy an unreadable sample is skipped with a note, so do the
    # same here: import gates for the samples that can be read and say which
    # ones were left out.
    usable = []
    for s in samples:
        if Path(s.filepath).exists():
            usable.append(s)
        else:
            report.notes.append(f"{s.name}: skipped, its file is no longer "
                                f"where it was opened from ({s.filepath}) — "
                                "reconnect the drive, or open the file from "
                                "its new location and import again")
    if not usable:
        folders = sorted({str(Path(s.filepath).parent) for s in samples})
        raise FileNotFoundError(
            "None of the open samples can be read any more, so their gates "
            "cannot be checked against the workspace. The files are no "
            "longer at:\n\n" + "\n".join(folders[:5])
            + ("\n…" if len(folders) > 5 else "")
            + "\n\nIf they are on a network drive, reconnect it; otherwise "
              "open the files from where they are now and import again.")
    # Pair each sample with a gate set *before* building the Workspace.
    # FlowKit keys both the workspace's gate sets and the files it loads by
    # FlowJo's sample name, so two tubes sharing a name silently replace one
    # another: the gates of one experiment then land on the other's file and
    # the count comparison reads a third. We decide the pairing from the
    # paths the workspace records and then hand FlowKit one file per name.
    recorded = wsp_sample_files(path)
    survivor = {name: files[-1] for name, files in recorded.items() if files}
    pairs = []            # (sample, workspace sample name, or None)
    if not survivor:
        # The paths could not be read out of the workspace (an unusual or
        # damaged file): fall back to matching on the name alone, below.
        pairs = [(s, None) for s in usable]
    for s in usable if survivor else []:
        here = str(Path(s.filepath).resolve())
        sid = next((i for i in survivor if survivor[i] == here), None)
        if sid is None:
            sid = s.name if s.name in survivor else next(
                (i for i in survivor if Path(i).stem == Path(s.name).stem),
                None)
        if sid is None:
            # the workspace may still hold this file under a name it shares
            # with another tube, which the branch below explains
            sid = next((i for i, fs in recorded.items() if here in fs), None)
        if sid is None:
            report.notes.append(f"{s.name}: not in the workspace")
            continue
        kept = survivor.get(sid)
        if kept is not None and kept != here:
            shared = len(recorded.get(sid, []))
            mine = here in recorded.get(sid, [])
            report.notes.append(
                f"{s.name}: skipped — the workspace calls this sample "
                f"'{sid}', and {shared} tubes share that one name. The "
                f"gates stored under it belong to {kept}."
                + (" This file's own gates are in the workspace too, but "
                   "cannot be told apart while the name is shared — rename "
                   "one of the two tubes in FlowJo and save again."
                   if mine else ""))
            continue
        pairs.append((s, sid))
    if not pairs:
        report.notes.append("No sample in the workspace matched an open "
                            "file, so no gates were imported.")
        return {}, report
    ws = fk.Workspace(str(path), fcs_samples=[s.filepath for s, _ in pairs])
    wsp_ids = set(ws.get_sample_ids())
    result = {}
    for s, sid in pairs:
        if sid is None:
            sid = s.name if s.name in wsp_ids else next(
                (i for i in wsp_ids if Path(i).stem == Path(s.name).stem),
                None)
            if sid is None:
                report.notes.append(f"{s.name}: not in the workspace")
                continue
        if sid not in wsp_ids:
            report.notes.append(f"{s.name}: the workspace lists it as "
                                f"'{sid}', but FlowKit could not load it")
            continue
        try:
            m = ws.get_comp_matrix(sid)
        except Exception:
            m = None
        if m is not None:
            by_pnn = {v: k for k, v in s.pnn.items()}
            chans = [by_pnn.get(d, d) for d in m.detectors]
            if all(c in s.channels for c in chans):
                s.apply_compensation(CompensationMatrix(np.asarray(m.matrix,
                                                                   float),
                                                        chans))
        imp = _Importer(s, ws.get_transforms(sid), report, s.name)
        roots = _build_tree(ws.get_gate_ids(sid),
                            lambda n, p: ws.get_gate(sid, n, gate_path=p),
                            imp, report)
        if imp.approx:
            report.notes.append(
                f"{s.name}: {imp.approx} gate(s) were drawn on a FlowJo "
                "scale FlowPy doesn't have; their outlines were followed "
                "closely — see the count comparison")
        result[s] = roots
        # compare with FlowJo's own gating, through FlowKit
        try:
            ws.analyze_samples(sample_id=sid, use_mp=False)
            res = ws.get_gating_results(sid)
            masks = compute_masks(s, roots)
            for g in _bfs(roots):
                fk_path = tuple(["root"] + [a.name for a in g.ancestors()])
                try:
                    n_fk = int(res.get_gate_count(g.name, gate_path=fk_path))
                except Exception:
                    continue
                report.counts.append((s.name, " / ".join(
                    fk_path[1:] + (g.name,)), int(masks[g.gate_id].sum()),
                    n_fk))
        except Exception as exc:
            report.notes.append(f"{s.name}: could not recount with FlowKit "
                                f"({exc})")
    return result, report


def import_gatingml(path: str | Path, reference_sample=None):
    """Gating-ML 2.0 file → FlowPy gate forest (a protocol)."""
    fk = _fk()
    report = Report()
    gs = fk.parse_gating_xml(str(path))
    transforms = {tid: gs.get_transform(tid)
                  for tid in getattr(gs, "transformations", {})}
    imp = _Importer(reference_sample, transforms, report, Path(path).name)
    roots = _build_tree(gs.get_gate_ids(),
                        lambda n, p: gs.get_gate(n, gate_path=p), imp, report)
    if imp.approx:
        report.notes.append(f"{imp.approx} gate(s) use a scale FlowPy "
                            "doesn't have; their outlines were followed "
                            "closely")
    return roots, report
