"""Gating protocols — a gate tree saved on its own so it can be re-applied to
any number of samples (FlowJo calls this a gating template).

A protocol is plain JSON (``.fpg``)::

    {"format": "flowpy-protocol", "version": 1, "name": "...",
     "gates": [gate dicts, parents before children],
     "gate_parent_map": {child_id: parent_id}}

Gate ids are kept when a protocol is applied, so the same population has the
same id in every sample — layouts and batch statistics rely on that.
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

from .gates import Gate, BooleanGate, gate_from_dict
from .statistics import stat_label, sample_stats, stat_1d

PROTOCOL_FORMAT = "flowpy-protocol"


def _bfs(roots: list[Gate]) -> list[Gate]:
    result, queue = [], list(roots)
    while queue:
        g = queue.pop(0)
        result.append(g)
        queue.extend(g.children)
    return result


# ---------------------------------------------------------------------------
# (De)serialisation
# ---------------------------------------------------------------------------

def tree_to_dict(roots: list[Gate]) -> dict:
    gates = _bfs(roots)
    return {
        "gates": [g.to_dict() for g in gates],
        "gate_parent_map": {g.gate_id: g.parent.gate_id
                            for g in gates if g.parent is not None},
    }


def tree_from_dict(d: dict) -> list[Gate]:
    """Rebuild a gate forest; returns the root gates."""
    parent_map = d.get("gate_parent_map", {})
    by_id: dict[str, Gate] = {}
    for gd in d.get("gates", []):
        g = gate_from_dict(gd)
        by_id[g.gate_id] = g
    roots = []
    for gid, gate in by_id.items():
        par = parent_map.get(gid)
        if par and par in by_id:
            by_id[par].add_child(gate)
        else:
            roots.append(gate)
    return roots


def clone_tree(roots: list[Gate]) -> list[Gate]:
    """Independent copy of a gate forest (same ids, names and colours)."""
    return tree_from_dict(tree_to_dict(roots))


def save_protocol(path: str | Path, roots: list[Gate], name: str = "",
                  formulas: dict | None = None):
    """*formulas*: derived parameters the gates may use
    ({name: {"expr", "fluorescence"}}), recreated when applied."""
    d = {"format": PROTOCOL_FORMAT, "version": 1,
         "name": name or Path(path).stem, **tree_to_dict(roots)}
    if formulas:
        d["formulas"] = formulas
    with open(path, "w") as f:
        json.dump(d, f, indent=2)


def load_protocol(path: str | Path) -> list[Gate]:
    with open(path) as f:
        d = json.load(f)
    if d.get("format") != PROTOCOL_FORMAT and "gates" not in d:
        raise ValueError("Not a FlowPy gating protocol")
    # A workspace file also works: take its current sample's gates
    if "gates" not in d and d.get("samples"):
        d = d["samples"][0]
    return tree_from_dict(d)


def load_protocol_formulas(path: str | Path) -> dict:
    """Derived-parameter formulas stored in a protocol or workspace."""
    with open(path) as f:
        d = json.load(f)
    if "formulas" in d:
        return d["formulas"]
    derived = (d.get("samples") or [{}])[0].get("derived", {})
    return {n: {"expr": e["formula"], "fluorescence": e.get("fluorescence",
                                                            False)}
            for n, e in derived.items() if "formula" in e}


def add_formulas(sample, formulas: dict | None) -> list[str]:
    """Create the derived parameters *sample* doesn't have yet; returns
    the names that could not be created (e.g. missing channels)."""
    failed = []
    for name, spec in (formulas or {}).items():
        if name in sample.channels:
            continue
        try:
            sample.add_formula(name, spec["expr"],
                               spec.get("fluorescence", False))
        except ValueError:
            failed.append(name)
    return failed


# ---------------------------------------------------------------------------
# Applying to samples
# ---------------------------------------------------------------------------

def required_channels(roots: list[Gate]) -> set[str]:
    chans = set()
    for g in _bfs(roots):
        if isinstance(g, BooleanGate):
            continue
        chans.add(g.x_channel)
        if g.y_channel:
            chans.add(g.y_channel)
    return chans


def missing_channels(sample, roots: list[Gate]) -> list[str]:
    have = set(sample.channels)
    return sorted(c for c in required_channels(roots) if c not in have)


def gate_mask(sample, gate: Gate, masks: dict[str, np.ndarray]) -> np.ndarray:
    """Mask of one gate, given the masks already computed for its parent
    (and, for a boolean gate, its operands)."""
    par = masks.get(gate.parent.gate_id) if gate.parent else None
    if isinstance(gate, BooleanGate):
        m = gate.combine([masks[i] for i in gate.operand_ids])
        return m & par if par is not None else m
    return sample.events_for_gate(gate, par)


def compute_masks(sample, roots: list[Gate]) -> dict[str, np.ndarray]:
    """Evaluate every gate of the tree on *sample*. Parents come first, and
    boolean gates wait for their operands wherever those are in the tree.
    A boolean gate whose operands are gone gets an empty mask."""
    masks: dict[str, np.ndarray] = {}
    pending = _bfs(roots)
    while pending:
        waiting = []
        for g in pending:
            needs = [g.parent.gate_id] if g.parent else []
            if isinstance(g, BooleanGate):
                needs += g.operand_ids
            if all(n in masks for n in needs):
                masks[g.gate_id] = gate_mask(sample, g, masks)
            else:
                waiting.append(g)
        if len(waiting) == len(pending):          # unresolvable
            for g in waiting:
                masks[g.gate_id] = np.zeros(sample.n_events, dtype=bool)
            break
        pending = waiting
    return masks


def apply_protocol(sample, roots: list[Gate], formulas: dict | None = None):
    """Give *sample* its own copy of the gate tree and compute its masks,
    first creating any derived parameters in *formulas* it lacks.
    Raises ValueError if the sample lacks a channel the gates use."""
    add_formulas(sample, formulas)
    missing = missing_channels(sample, roots)
    if missing:
        raise ValueError(f"missing channel(s): {', '.join(missing)}")
    sample.gates = clone_tree(roots)
    sample.gate_masks = compute_masks(sample, sample.gates)



# ---------------------------------------------------------------------------
# Copy / paste of gates (through the system clipboard, so it works between
# workspaces and between two FlowPy windows)
# ---------------------------------------------------------------------------

CLIP_FORMAT = "flowpy-gates"
CLIP_MIME = "application/x-flowpy-gates"
PASTE_REPLACE, PASTE_ADD, PASTE_INSIDE = "replace", "add", "inside"


def gates_to_clip(roots: list[Gate], formulas: dict | None = None,
                  source: str = "", parent: str = "All Events") -> str:
    """Clipboard text for *roots* (and their children). *parent*: the
    population they were drawn in, so paste can put them back there."""
    d = {"format": CLIP_FORMAT, "version": 1, "source": source,
         "parent": parent, **tree_to_dict(roots)}
    if formulas:
        d["formulas"] = formulas
    return json.dumps(d)


def gates_from_clip(text: str) -> dict:
    """{roots, formulas, source, parent, n} from clipboard text (gates
    copied in FlowPy, or a protocol / workspace file's text). Raises
    ValueError if there are no gates in it."""
    try:
        d = json.loads(text or "")
    except (ValueError, TypeError):
        d = None
    if isinstance(d, dict) and "gates" not in d and d.get("samples"):
        d = d["samples"][0]                          # a workspace
    if not isinstance(d, dict) or not d.get("gates"):
        raise ValueError("The clipboard has no FlowPy gates. Copy them first "
                         "(select a gate, then Edit → Copy Gates).")
    roots = tree_from_dict(d)
    return {"roots": roots, "formulas": d.get("formulas") or {},
            "source": d.get("source", ""),
            "parent": d.get("parent", "All Events"),
            "n": len(_bfs(roots))}


def _with_new_ids(roots: list[Gate]) -> list[Gate]:
    """Copy of *roots* where every gate has a new id (AND/OR/NOT gates
    follow their operands)."""
    import uuid
    d = tree_to_dict(roots)
    new = {g["gate_id"]: str(uuid.uuid4())[:8] for g in d["gates"]}
    for g in d["gates"]:
        g["gate_id"] = new[g["gate_id"]]
        if "operand_ids" in g:
            g["operand_ids"] = [new.get(i, i) for i in g["operand_ids"]]
    d["gate_parent_map"] = {new.get(c, c): new.get(p, p)
                            for c, p in d["gate_parent_map"].items()}
    return tree_from_dict(d)


def paste_gates(sample, roots: list[Gate], mode: str = PASTE_ADD,
                inside: str | None = None, formulas: dict | None = None):
    """Paste a copy of *roots* into *sample*: replacing its gates
    (PASTE_REPLACE), next to them (PASTE_ADD) or inside its population
    *inside* ("Lymphocytes / Live", PASTE_INSIDE). Raises ValueError when
    the sample lacks a channel or that population."""
    add_formulas(sample, formulas)
    missing = missing_channels(sample, roots)
    if missing:
        raise ValueError(f"missing channel(s): {', '.join(missing)}")
    if mode == PASTE_REPLACE:
        sample.gates = clone_tree(roots)
    else:
        parent = None
        if mode == PASTE_INSIDE and inside and inside != "All Events":
            parent = next((g for g in _bfs(sample.gates)
                           if population_path(g) == inside), None)
            if parent is None:
                raise ValueError(f"no population '{inside}'")
        have = {g.gate_id for g in _bfs(sample.gates)}
        copies = (clone_tree(roots)
                  if not have & {g.gate_id for g in _bfs(roots)}
                  else _with_new_ids(roots))
        for g in copies:
            if parent is not None:
                parent.add_child(g)
            else:
                sample.gates.append(g)
    sample.gate_masks = compute_masks(sample, sample.gates)

# ---------------------------------------------------------------------------
# Batch statistics
# ---------------------------------------------------------------------------

def population_path(gate: Gate) -> str:
    return " / ".join(g.name for g in gate.ancestors() + [gate])


def batch_statistics(samples, stat: str = "Median",
                     pct: float = 50.0) -> pd.DataFrame:
    """One row per sample × population: count, % parent, % total and the
    *stat* (any of statistics.STAT_NAMES) of every fluorescence channel."""
    rows = []
    label = stat_label(stat, pct)
    for s in samples:
        n_total = s.n_events
        pops = [(None, "All Events", None)] + [
            (g, population_path(g), s.gate_masks.get(g.gate_id))
            for g in _bfs(s.gates)]
        for g, path, mask in pops:
            if g is not None and mask is None:
                continue
            par_mask = (s.gate_masks.get(g.parent.gate_id)
                        if g is not None and g.parent else None)
            n = int(mask.sum()) if mask is not None else n_total
            n_par = int(par_mask.sum()) if par_mask is not None else n_total
            row = {"Sample": s.name, "Population": path, "Count": n,
                   "% Parent": round(100.0 * n / n_par, 2) if n_par else 0.0,
                   "% Total": round(100.0 * n / n_total, 2) if n_total else 0.0}
            if s.fluorescence:
                vals = sample_stats(s, s.fluorescence, stat, pct, mask)
                for ch, v in zip(s.fluorescence, vals):
                    row[f"{ch} {label}"] = round(float(v), 2)
            rows.append(row)
    return pd.DataFrame(rows)


# Population statistics that don't need a channel
COUNT_STATS = ["Count", "% Parent", "% Total"]


def find_population(sample, path: str):
    """(gate, mask) of the population called *path* ("A / B") in *sample*;
    (None, None) for "All Events"; raises KeyError if it isn't there."""
    if path == "All Events":
        return None, None
    for g in _bfs(sample.gates):
        if population_path(g) == path:
            mask = sample.gate_masks.get(g.gate_id)
            if mask is None:
                break
            return g, mask
    raise KeyError(path)


def export_table(samples, populations: list[str], channels: list[str],
                 stats: list[str], pct: float = 50.0,
                 wide: bool = True,
                 meta_columns: list[str] | None = None) -> pd.DataFrame:
    """Chosen statistics of chosen channels in chosen populations.

    Populations are matched by path name, so samples gated separately
    still line up. Channel statistics use every event of the population.
    *wide*: one row per sample, one column per population × channel ×
    statistic. Otherwise one row per sample × population.
    *meta_columns*: sample metadata (group, condition…) added after Sample.
    Missing populations/channels give empty (NaN) cells."""
    count_stats = [s for s in stats if s in COUNT_STATS]
    chan_stats = [s for s in stats if s not in COUNT_STATS]
    rows = []
    for s in samples:
        n_total = s.n_events
        for path in populations:
            try:
                gate, mask = find_population(s, path)
            except KeyError:
                gate, mask = None, False
            vals: dict[str, float] = {}
            if mask is not False:
                n = int(mask.sum()) if mask is not None else n_total
                par = (s.gate_masks.get(gate.parent.gate_id)
                       if gate is not None and gate.parent else None)
                n_par = int(par.sum()) if par is not None else n_total
                counts = {"Count": n,
                          "% Parent": 100.0 * n / n_par if n_par else 0.0,
                          "% Total": 100.0 * n / n_total if n_total else 0.0}
                for st in count_stats:
                    vals[st] = counts[st]
                for ch in channels:
                    if ch not in s.channels:
                        continue
                    col = s.column(ch)
                    col = col[mask] if mask is not None else col
                    for st in chan_stats:
                        vals[f"{ch} {stat_label(st, pct)}"] = stat_1d(
                            col, st, pct)
            keys = count_stats + [f"{ch} {stat_label(st, pct)}"
                                  for ch in channels for st in chan_stats]
            meta = {c: getattr(s, "annotations", {}).get(c, "")
                    for c in (meta_columns or [])}
            rows.append({"Sample": s.name, **meta, "Population": path,
                         **{k: vals.get(k, np.nan) for k in keys}})
    df = pd.DataFrame(rows)
    if "Count" in df.columns:   # whole numbers, even with empty cells
        df["Count"] = df["Count"].astype("Int64")
    if not wide or df.empty:
        return df
    lead = ["Sample"] + list(meta_columns or [])
    out = df[lead].drop_duplicates("Sample").reset_index(drop=True)
    for path in populations:
        part = df[df["Population"] == path].drop(
            columns=["Population"] + list(meta_columns or []))
        prefix = f"{path} | " if len(populations) > 1 else ""
        part = part.rename(columns={c: prefix + c for c in part.columns
                                    if c != "Sample"})
        out = out.merge(part, on="Sample", how="left", sort=False)
    if len(populations) == 1:
        out.insert(len(lead), "Population", populations[0])
    return out
