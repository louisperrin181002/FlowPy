"""Sample — represents one FCS file loaded into memory."""

from __future__ import annotations
import base64
import threading
from pathlib import Path
import numpy as np
import pandas as pd

from .fcs_reader import read_fcs, read_fcs_header, extract_spillover
from .compensation import CompensationMatrix, guess_fluorescence_channels
from .gates import Gate
from . import formulas as fx


class Sample:
    """Holds all data and metadata for a single FCS file.

    Attributes
    ----------
    name         : display name (filename without path)
    filepath     : absolute path
    metadata     : raw FCS keyword/value dict
    raw_data     : original uncompensated events (DataFrame)
    data         : working dataset (compensated if comp matrix applied)
    channels     : list of all channel names in data
    scatter      : channels identified as scatter (FSC/SSC)
    fluorescence : channels identified as fluorescence
    comp_matrix  : CompensationMatrix or None
    gates        : top-level gates applied to this sample
    """

    def __init__(self, filepath: str | Path, lazy: bool = False):
        """*lazy*: read only the file's header now (channels, event count,
        keywords — a few KB, instant even over a network) and the events
        when first needed (see load())."""
        self.filepath = str(Path(filepath).resolve())
        self.name = Path(filepath).name
        self._lock = threading.RLock()
        self._raw_data: pd.DataFrame | None = None
        self._data: pd.DataFrame | None = None
        self._gate_masks: dict[str, np.ndarray] = {}
        self._ready = not lazy
        # computed channels from a workspace, added once events are read
        self._pending_derived: dict = {}
        self.load_error: str | None = None

        if lazy:
            self.metadata, columns, self._n = read_fcs_header(self.filepath)
        else:
            self.metadata, raw = read_fcs(self.filepath)
            # Uncompensated, the working data *is* the raw data (no copy:
            # a 10-million-event file stays one block of memory)
            self._raw_data = self._data = raw
            columns, self._n = list(raw.columns), len(raw)

        # Channel lists
        self.channels: list[str] = list(columns)
        # FCS channel names ($PnN) of the measured channels, for exchanging
        # gates with FlowJo / Gating-ML (which use $PnN)
        self.pnn: dict[str, str] = dict(zip(
            self.channels, self.metadata.get("_pnn_", self.channels)))
        self.fluorescence: list[str] = guess_fluorescence_channels(self.channels)
        self.scatter: list[str] = [c for c in self.channels
                                   if c not in self.fluorescence]

        # Per-channel numpy arrays of self.data, built lazily
        self._col_cache: dict[str, np.ndarray] = {}

        # Channels computed by FlowPy (UMAP, clusters, unmixed…), by name
        self.derived: dict[str, np.ndarray] = {}
        # Derived parameters defined by a formula (name -> expression); they
        # are recomputed whenever compensation changes
        self.formulas: dict[str, str] = {}

        # User metadata: {column: value}, e.g. {"Group": "LB", "Strain": "124"}
        self.annotations: dict[str, str] = {}

        # Compensation: the acquisition spillover matrix is applied on load
        # (as FlowJo does); Edit → Remove Compensation undoes it.
        self.comp_matrix: CompensationMatrix | None = None
        matrix, spill_channels = extract_spillover(self.metadata)
        if matrix is not None:
            # The matrix names channels by $PnN; columns may use markers
            by_pnn = {v: k for k, v in self.pnn.items()}
            spill_channels = [by_pnn.get(c, c) for c in spill_channels]
            mapped = _map_spill_channels(spill_channels, self.channels)
            if mapped:
                self.apply_compensation(CompensationMatrix(matrix, mapped))

        # This sample's own gate tree (root gates); the boolean event mask
        # of every gate, keyed by gate_id, is .gate_masks
        self.gates: list[Gate] = []

    # ------------------------------------------------------------------
    # Events (read on first use for a lazy sample)
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        """Events read, and everything set before applied to them."""
        return self._ready

    @property
    def _has_events(self) -> bool:
        return self._raw_data is not None

    def load(self):
        """Read the events if not done yet (thread-safe; other threads
        wait for the one reading). Then applies what was set before:
        compensation, computed channels, and the masks of its gates.
        Raises IOError if the file can't be read."""
        if self._ready:
            return
        with self._lock:
            # (also when called again from inside, while finishing)
            if self._ready or self._raw_data is not None:
                return
            if self.load_error:
                raise IOError(self.load_error)
            try:
                meta, raw = read_fcs(self.filepath)
                if list(raw.columns) != self.channels[:len(raw.columns)]:
                    raise IOError("the file changed since it was opened")
            except Exception as exc:
                self.load_error = str(exc)
                raise IOError(self.load_error) from exc
            self.metadata = meta
            self._n = len(raw)
            self._raw_data = self._data = raw
            try:
                if self.comp_matrix is not None:
                    self._data = self.comp_matrix.apply(raw)
                pending, self._pending_derived = self._pending_derived, {}
                if pending:
                    self.derived_from_dict(pending)
                if self.gates and not self._gate_masks:
                    from .protocol import compute_masks
                    try:
                        self._gate_masks = compute_masks(self, self.gates)
                    except Exception:
                        self._gate_masks = {}
            finally:
                self._ready = True

    def retry(self):
        """Forget a failed read, so load() tries the file again."""
        self.load_error = None

    @property
    def raw_data(self) -> pd.DataFrame:
        self.load()
        return self._raw_data

    @raw_data.setter
    def raw_data(self, df: pd.DataFrame):
        self._raw_data = df

    @property
    def data(self) -> pd.DataFrame:
        self.load()
        return self._data

    @data.setter
    def data(self, df: pd.DataFrame):
        self._data = df

    @property
    def gate_masks(self) -> dict[str, np.ndarray]:
        if not self._ready and self.gates:
            self.load()
        return self._gate_masks

    @gate_masks.setter
    def gate_masks(self, masks: dict[str, np.ndarray]):
        self._gate_masks = masks

    # ------------------------------------------------------------------
    # Compensation
    # ------------------------------------------------------------------

    def apply_compensation(self, matrix: CompensationMatrix | None = None):
        """Apply compensation matrix to raw data and update self.data."""
        m = matrix or self.comp_matrix
        if not self._has_events:
            self.comp_matrix = m             # applied when events are read
            return
        if m is not None:
            self.comp_matrix = m
            self.data = m.apply(self.raw_data)
        else:
            self.data = self.raw_data
        self._col_cache.clear()
        self._refresh_formulas()

    def remove_compensation(self):
        self.comp_matrix = None
        if not self._has_events:
            return
        self.data = self.raw_data
        self._col_cache.clear()
        self._refresh_formulas()

    def comp_state(self) -> dict | None:
        """Compensation as saved in a workspace (None = uncompensated)."""
        return self.comp_matrix.to_dict() if self.comp_matrix else None

    def restore_comp_state(self, state: dict | None):
        if state is None:
            self.remove_compensation()
        else:
            self.apply_compensation(CompensationMatrix.from_dict(state))

    # ------------------------------------------------------------------
    # Derived channels
    # ------------------------------------------------------------------

    def add_derived(self, name: str, values: np.ndarray,
                    fluorescence: bool = False):
        """Add (or replace) a computed channel. NaN marks events that were
        not part of the computation; they never fall inside a gate.
        *fluorescence* channels (e.g. unmixed) get statistics and the
        Logicle scale; others (UMAP, clusters) are treated like scatter."""
        values = np.asarray(values, dtype=np.float32)
        if len(values) != self.n_events:
            raise ValueError(f"{name}: expected {self.n_events} values, "
                             f"got {len(values)}")
        self.derived[name] = values
        self.raw_data[name] = values
        if self.data is not self.raw_data:
            self.data[name] = values
        if name not in self.channels:
            self.channels.append(name)
            (self.fluorescence if fluorescence else self.scatter).append(name)
        self._col_cache.pop(name, None)
        self._col_cache.pop("__lim__" + name, None)

    def add_formula(self, name: str, expr: str, fluorescence: bool = False):
        """Add (or redefine) a derived parameter computed from *expr*
        (see core.formulas). Raises FormulaError if it is invalid."""
        name = name.strip()
        if not name:
            raise fx.FormulaError("Give the parameter a name.")
        if name in self.channels and name not in self.formulas:
            raise fx.FormulaError(f"'{name}' is already a channel.")
        used = fx.validate(expr, [c for c in self.channels if c != name])
        if self._depends_on(used, name):
            raise fx.FormulaError("A formula can't use itself.")
        if not self._has_events:
            self._defer_derived({name: {"formula": expr,
                                        "fluorescence": fluorescence}})
            return
        self.add_derived(name, fx.evaluate(expr, self.column), fluorescence)
        self.formulas[name] = expr
        self._refresh_formulas(after=name)

    def _depends_on(self, used: list[str], name: str) -> bool:
        todo, seen = list(used), set()
        while todo:
            c = todo.pop()
            if c == name:
                return True
            if c in self.formulas and c not in seen:
                seen.add(c)
                todo.extend(fx.channels_in(self.formulas[c]))
        return False

    def formula_specs(self) -> dict:
        """{name: {"expr", "fluorescence"}} in definition order."""
        return {n: {"expr": e, "fluorescence": n in self.fluorescence}
                for n, e in self.formulas.items()}

    def remove_derived(self, name: str):
        """Delete a computed channel / derived parameter."""
        self.derived.pop(name, None)
        self.formulas.pop(name, None)
        self._pending_derived.pop(name, None)
        if self._has_events:
            for df in {id(self.raw_data): self.raw_data,
                       id(self.data): self.data}.values():
                if name in df.columns:
                    del df[name]
        for lst in (self.channels, self.fluorescence, self.scatter):
            if name in lst:
                lst.remove(name)
        self._col_cache.pop(name, None)
        self._col_cache.pop("__lim__" + name, None)

    def _refresh_formulas(self, after: str | None = None):
        """Recompute formula channels (all, or those defined after
        *after*, which may depend on it), in definition order."""
        names = list(self.formulas)
        if after is not None:
            names = names[names.index(after) + 1:]
        for n in names:
            try:
                vals = fx.evaluate(self.formulas[n], self.column)
            except Exception:
                vals = np.full(self.n_events, np.nan)
            self.add_derived(n, vals, n in self.fluorescence)

    def derived_to_dict(self) -> dict:
        """Computed channels for a workspace file (float32, base64);
        formula channels are stored as their formula."""
        if not self._has_events:
            return dict(self._pending_derived)
        out = {}
        for name, v in self.derived.items():
            entry = {"fluorescence": name in self.fluorescence}
            if name in self.formulas:
                entry["formula"] = self.formulas[name]
            else:
                entry["data"] = base64.b64encode(np.ascontiguousarray(
                    v, dtype="<f4").tobytes()).decode()
            out[name] = entry
        return out

    def _defer_derived(self, d: dict):
        """Computed channels for a sample whose events aren't read yet:
        added by load(). Formula channels are listed straight away."""
        self._pending_derived.update(d)
        for name, e in d.items():
            if "formula" in e:
                self.formulas[name] = e["formula"]
                if name not in self.channels:
                    self.channels.append(name)
                    (self.fluorescence if e.get("fluorescence")
                     else self.scatter).append(name)

    def derived_from_dict(self, d: dict):
        if not self._has_events:
            self._defer_derived(d)
            return
        for name, entry in d.items():
            if "formula" in entry:
                try:
                    self.add_formula(name, entry["formula"],
                                     entry.get("fluorescence", False))
                except fx.FormulaError:
                    pass
                continue
            vals = np.frombuffer(base64.b64decode(entry["data"]), dtype="<f4")
            if len(vals) == self.n_events:
                self.add_derived(name, vals.astype(np.float64),
                                 entry.get("fluorescence", False))

    # ------------------------------------------------------------------
    # Fast column access
    # ------------------------------------------------------------------

    def column(self, channel: str) -> np.ndarray:
        """*channel* of the working data as a numpy array (float32, a view
        of the data — treat it as read-only)."""
        arr = self._col_cache.get(channel)
        if arr is None:
            arr = self.data[channel].to_numpy()
            self._col_cache[channel] = arr
        return arr

    def channel_limits(self, channel: str) -> tuple[float, float]:
        """(min, max) of the raw channel values, ignoring non-finite."""
        key = "__lim__" + channel
        lim = self._col_cache.get(key)
        if lim is None:
            col = self.column(channel)
            col = col[np.isfinite(col)]
            lim = (np.array([col.min(), col.max()]) if len(col)
                   else np.array([0.0, 1.0]))
            self._col_cache[key] = lim
        return float(lim[0]), float(lim[1])

    # ------------------------------------------------------------------
    # Gating helpers
    # ------------------------------------------------------------------

    def events_for_gate(self, gate: Gate,
                        parent_mask: np.ndarray | None = None) -> np.ndarray:
        """Return boolean mask of events inside *gate*.

        If parent_mask is given, restricts evaluation to those events
        (events outside the parent gate are always False in the result).
        """
        x = self.column(gate.x_channel)
        y = self.column(gate.y_channel) if gate.y_channel else None

        if parent_mask is not None:
            # Only test events that passed the parent gate
            mask = np.zeros(len(x), dtype=bool)
            sub_x = x[parent_mask]
            sub_y = y[parent_mask] if y is not None else None
            sub_mask = gate.apply(sub_x, sub_y)
            mask[parent_mask] = sub_mask
            return mask
        else:
            return gate.apply(x, y)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_events(self) -> int:
        return self._n

    @property
    def is_compensated(self) -> bool:
        return self.comp_matrix is not None

    # ------------------------------------------------------------------
    # Repr / serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "filepath": self.filepath,
            "comp_matrix": self.comp_matrix.to_dict() if self.comp_matrix else None,
        }

    def __repr__(self):
        return (f"<Sample '{self.name}' "
                f"n={self.n_events} channels={len(self.channels)}>")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _map_spill_channels(spill_channels: list[str],
                        data_columns: list[str]) -> list[str] | None:
    """Try to match spillover channel names to actual FCS column names.

    Returns the matched names (may contain duplicates if ambiguous),
    or None if matching fails badly.
    """
    mapped = []
    cols_upper = {c.upper(): c for c in data_columns}
    for sc in spill_channels:
        exact = cols_upper.get(sc.upper())
        if exact:
            mapped.append(exact)
        else:
            # Try partial match on the short name (strip trailing -H / -A / -W)
            base = sc.split("-")[0].upper()
            candidates = [c for cu, c in cols_upper.items()
                          if cu.startswith(base)]
            mapped.append(candidates[0] if candidates else sc)
    return mapped if len(mapped) == len(spill_channels) else None
