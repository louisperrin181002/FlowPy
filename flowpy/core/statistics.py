"""Statistics for gated populations.

All functions work column-wise on an (events × channels) array, ignore
non-finite values, and are vectorised so whole populations are cheap.
"""

from __future__ import annotations
import numpy as np

# Statistic names as shown in the UI. "Percentile" takes a parameter.
STAT_NAMES = ["Mean", "Median", "Geometric Mean", "SD", "Robust SD", "CV",
              "Robust CV", "Min", "Max", "Percentile"]

# Robust SD (FlowJo definition): half the 15.87–84.13 percentile spread,
# which equals the SD for normally distributed data.
# Those two numbers are mu +/- 1 sigma expressed as percentiles, which
# is why half their spread equals the SD on normal data — and why it
# does NOT blow up on the outliers and debris a real cytometry
# population contains, unlike the ordinary SD.
_RSD_LO, _RSD_HI = 15.865, 84.135


def gates_in_order(root_gates: list) -> list:
    """Return all gates in BFS order (parents before children)."""
    # Breadth-first because a child's percentage is "of parent", so the
    # parent's count has to exist before the child is reached.
    # list() copies: queue is consumed below, and the caller's list of
    # root gates must not be emptied.
    result, queue = [], list(root_gates)
    while queue:
        g = queue.pop(0)
        result.append(g)
        queue.extend(g.children)
    return result


def stat_label(stat: str, pct: float = 50.0) -> str:
    return f"P{pct:g}" if stat == "Percentile" else stat


def _percentiles(x: np.ndarray, qs: list[float]) -> list[float]:
    """Percentiles of a NaN-free 1-D array, numpy's default ('linear')
    definition, from one partial sort; computed in float64."""
    # Each percentile falls at a fractional position in the sorted
    # array; the value is interpolated between the two neighbours.
    n = len(x)
    pos = [q / 100.0 * (n - 1) for q in qs]
    # partition() is O(n) where a full sort is O(n log n), and it only
    # needs to know which positions matter. Both neighbours of every
    # requested position are collected, deduplicated and passed in one
    # call, so three percentiles of ten million events cost one pass
    # rather than three sorts.
    kth = sorted({int(np.floor(p)) for p in pos} | {int(np.ceil(p))
                                                   for p in pos})
    part = np.partition(x, kth)
    out = []
    for p in pos:
        lo, hi = int(np.floor(p)), int(np.ceil(p))
        a, b = float(part[lo]), float(part[hi])
        out.append(a + (b - a) * (p - lo))
    return out


def stat_1d(col: np.ndarray, stat: str, pct: float = 50.0,
            mask: np.ndarray | None = None) -> float:
    """*stat* of one channel (optionally only the events in *mask*),
    ignoring non-finite values. Accumulates in float64 whatever the
    storage type."""
    x = col[mask] if mask is not None else col
    # FCS files often store events as 16- or 32-bit integers. Summing a
    # million int32 values overflows silently, so an integer column is
    # widened before anything is accumulated.
    if x.dtype.kind != "f":
        x = x.astype(np.float64)
    # Tested before being applied: the all() check is one cheap pass,
    # while the boolean index would copy the whole column — and on a
    # clean file there is nothing to remove.
    finite = np.isfinite(x)
    if not finite.all():
        x = x[finite]
    n = len(x)
    if n == 0:
        return np.nan
    # Warnings suppressed rather than printed: an empty or degenerate
    # population legitimately yields NaN here, and that is handled by
    # the return below. The UI shows a dash; a stream of numpy warnings
    # on the console would say nothing more.
    with np.errstate(all="ignore"):
        if stat == "Mean":
            v = np.mean(x, dtype=np.float64)
        elif stat == "Median":
            v = _percentiles(x, [50.0])[0]
        elif stat == "Percentile":
            v = _percentiles(x, [pct])[0]
        elif stat == "Geometric Mean":
            # Only positive values: log(0) is -inf and log of a negative
            # is NaN, and compensated fluorescence routinely goes
            # negative. Computed as exp(mean(log)) rather than as a
            # product, which would overflow long before a million
            # events.
            pos = x[x > 0]
            v = (np.exp(np.mean(np.log(pos.astype(np.float64))))
                 if len(pos) else np.nan)
        elif stat == "SD":
            # ddof=1 is the sample SD, the convention for measured data;
            # numpy's default ddof=0 would be the population SD. One
            # event has no spread, and ddof=1 would divide by zero.
            v = np.std(x, dtype=np.float64, ddof=1) if n > 1 else np.nan
        elif stat == "CV":
            # abs() on the mean: a compensated channel can have a
            # negative mean, and a negative CV would be reported as
            # though the spread itself were negative.
            v = (100.0 * np.std(x, dtype=np.float64, ddof=1)
                 / abs(np.mean(x, dtype=np.float64))) if n > 1 else np.nan
        elif stat in ("Robust SD", "Robust CV"):
            # All three percentiles in one call, so the robust CV costs
            # the same single partition as the robust SD.
            lo, med, hi = _percentiles(x, [_RSD_LO, 50.0, _RSD_HI])
            v = (hi - lo) / 2.0
            if stat == "Robust CV":
                # A median of exactly zero would divide by zero; it
                # happens on an integer channel where most events sit
                # at the baseline.
                v = 100.0 * v / abs(med) if med else np.nan
        elif stat == "Min":
            v = np.min(x)
        elif stat == "Max":
            v = np.max(x)
        else:
            raise ValueError(f"Unknown statistic: {stat}")
    # One exit point, and one kind of "no value": every degenerate
    # case above becomes NaN here, so callers test np.isfinite once
    # instead of checking for inf, -inf and NaN separately.
    v = float(v)
    return v if np.isfinite(v) else np.nan


# Above this many values per job, channels are computed in parallel
# (numpy's sorting and reductions release the GIL)
_PARALLEL_MIN = 2_000_000
# Built on first use and then kept: creating a thread pool costs more
# than the work it would do on a small population, and a module-level
# pool would start threads merely because this module was imported.
_POOL = None


def _pool():
    """The shared thread pool, created on first use."""
    global _POOL
    if _POOL is None:
        import os
        from concurrent.futures import ThreadPoolExecutor
        # Threads, not processes: numpy releases the GIL inside its
        # reductions and sorts, so these do run in parallel — and
        # without the cost of pickling a million-event column to a
        # subprocess, which would dwarf the computation.
        # Capped at 8; `or 2` covers cpu_count() returning None.
        _POOL = ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 2),
                                   thread_name_prefix="flowpy-stats")
    return _POOL


def channel_stats(columns: list[np.ndarray], stat: str, pct: float = 50.0,
                  mask: np.ndarray | None = None) -> np.ndarray:
    """*stat* of each 1-D array in *columns* (restricted to *mask*)."""
    n = int(mask.sum()) if mask is not None else (
        len(columns[0]) if columns else 0)
    # Total values, not event count: twenty channels of 100 000 events
    # is the same amount of work as one channel of two million, and
    # only the product says whether threads will pay for themselves.
    if len(columns) > 1 and n * len(columns) >= _PARALLEL_MIN:
        # The mask is converted to indices ONCE here. Passing the
        # boolean mask into each worker would make every thread rescan
        # it, which is the same cost again per channel.
        idx = np.flatnonzero(mask) if mask is not None else None
        vals = list(_pool().map(
            lambda c: stat_1d(c if idx is None else c[idx], stat, pct),
            columns))
    else:
        vals = [stat_1d(c, stat, pct, mask) for c in columns]
    return np.asarray(vals, dtype=np.float64)


def sample_stats(sample, channels: list[str], stat: str, pct: float = 50.0,
                 mask: np.ndarray | None = None) -> np.ndarray:
    """*stat* of *channels* of *sample* (in the population *mask*)."""
    return channel_stats([sample.column(c) for c in channels], stat, pct,
                         mask)


def column_stat(arr: np.ndarray, stat: str, pct: float = 50.0) -> np.ndarray:
    """*stat* of every column of *arr* (events × channels).

    Geometric mean uses positive values only (log is undefined otherwise).
    CV and robust CV are in percent.
    """
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.dtype.kind not in "fiu":
        arr = arr.astype(np.float64)
    if stat not in STAT_NAMES:
        raise ValueError(f"Unknown statistic: {stat}")
    return channel_stats([np.ascontiguousarray(arr[:, j])
                          for j in range(arr.shape[1])], stat, pct)
