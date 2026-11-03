"""Channel display transforms.

Linear, Log10, and three members of the biexponential family — Logicle,
FlowJo's Biex, and Arcsinh. All three compress a linear region around zero
into a log-like scale further out, so events at or below background stay
visible instead of piling against the axis; they differ in how that
compression is parameterised, and therefore in exactly where a population
sits on screen.

Which to use is mostly a matter of matching whatever the data was gated on
before: a gate drawn on one scale does not land in the same place on
another, which is why FlowPy records the scale with every gate.
"""

from __future__ import annotations
from functools import lru_cache

import numpy as np

TRANSFORM_NAMES = ["Linear", "Log10", "Logicle", "Biex (FlowJo)", "Arcsinh"]

#: Divisor for Arcsinh. 150 is the usual choice for fluorescence on a
#: 262,144-range instrument; 5 is common for mass cytometry. Larger values
#: widen the linear region around zero.
ARCSINH_COFACTOR = 150.0

#: FlowJo's own biex defaults, as written into its workspaces.
BIEX_DEFAULTS = dict(negative=0.0, width=-10.0, positive=4.418540,
                     max_value=262144.000029)

#: The biex parameters actually in use. Changed through set_biex_params(),
#: which the interface calls; kept module-level so a plot, a gate's
#: coordinates and an export all read the same scale rather than each
#: building its own.
_biex_params = dict(BIEX_DEFAULTS)


def apply_transform(data: np.ndarray, transform: str,
                    t: float = 262144.0, m: float = 4.5,
                    w: float = 0.5, a: float = 0.0) -> np.ndarray:
    """Apply the named transform to a 1-D array of raw channel values.

    Parameters
    ----------
    data      : raw values
    transform : one of "Linear", "Log10", "Logicle"
    t, m, w, a: logicle parameters (top-of-scale, decades, linear width, neg. decades)
    """
    if transform == "Linear":
        return data.copy()
    elif transform == "Log10":
        return _log10(data)
    elif transform == "Logicle":
        return _logicle(data, t=t, m=m, w=w, a=a)
    elif transform == "Biex (FlowJo)":
        return _biex(data)
    elif transform == "Arcsinh":
        return _arcsinh(data)
    return data.copy()


def inverse_transform(data: np.ndarray, transform: str,
                      t: float = 262144.0, m: float = 4.5,
                      w: float = 0.5, a: float = 0.0) -> np.ndarray:
    """Convert display-space values back to raw channel values."""
    if transform == "Linear":
        return data.copy()
    elif transform == "Log10":
        return _inv_log10(data)
    elif transform == "Logicle":
        return _inv_logicle(data, t=t, m=m, w=w, a=a)
    elif transform == "Biex (FlowJo)":
        return _inv_biex(data)
    elif transform == "Arcsinh":
        return _inv_arcsinh(data)
    return data.copy()


# ---------------------------------------------------------------------------
# Log10
# ---------------------------------------------------------------------------

def _log10(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x, dtype=np.float64)
    pos = x > 0
    out[pos] = np.log10(x[pos])
    out[~pos] = 0.0
    return out


def _inv_log10(y: np.ndarray) -> np.ndarray:
    return np.power(10.0, y)


# ---------------------------------------------------------------------------
# Logicle (Parks et al. 2006; Moore & Parks 2012)
# ---------------------------------------------------------------------------
# Computed by FlowUtils, the reference C implementation also used by FlowKit
# (validated against the Gating-ML 2.0 test suite). FlowUtils returns the
# display fraction [0, 1]; FlowPy shows it on [0, m] (raw 0 → w, raw t → m).

def _logicle(x: np.ndarray, t=262144.0, m=4.5, w=0.5, a=0.0) -> np.ndarray:
    """Forward logicle transform → display scale [0, m]."""
    from flowutils.transforms import logicle
    x = np.asarray(x, dtype=np.float64)
    out = logicle(x.reshape(-1, 1).copy(), [0], t=t, m=m, w=w, a=a)
    return np.asarray(out, dtype=np.float64).reshape(x.shape) * m


def _inv_logicle(y: np.ndarray, t=262144.0, m=4.5, w=0.5, a=0.0) -> np.ndarray:
    """Inverse logicle transform."""
    from flowutils.transforms import logicle_inverse
    y = np.asarray(y, dtype=np.float64)
    out = logicle_inverse((y / m).reshape(-1, 1).copy(), [0], t=t, m=m,
                          w=w, a=a)
    return np.asarray(out, dtype=np.float64).reshape(y.shape)


# ---------------------------------------------------------------------------
# Biex — FlowJo's biexponential
# ---------------------------------------------------------------------------
# FlowKit's WSPBiexTransform, which is the same lookup table FlowJo builds
# (ported from TreeStar's Java by way of cytolib). Used rather than
# reimplemented so that a plot here and a plot in FlowJo put a population
# in the same place — which is the only reason to offer this scale when
# Logicle is already available.
#
# Output is FlowJo channel space, 0 to 4096, not the 0-to-m of Logicle.
# Each transform carries its own display range and its own ticks.

BIEX_TOP = 4096.0


@lru_cache(maxsize=8)
def _biex_for(negative: float, width: float, positive: float,
              max_value: float):
    """The transform object for one set of parameters, built once.

    Constructing it generates a 4,097-point lookup table, which takes long
    enough to be worth never doing twice — on a plot that redraws as the
    mouse moves it would otherwise be rebuilt on every frame. Cached by
    parameters rather than held in one slot so that flipping between two
    settings stays instant.
    """
    import flowkit as fk
    return fk.transforms.WSPBiexTransform(
        negative=negative, width=width, positive=positive,
        max_value=max_value)


def _biex_transform():
    return _biex_for(**_biex_params)


def get_biex_params() -> dict:
    """The biex parameters in use."""
    return dict(_biex_params)


def set_biex_params(**kw) -> None:
    """Change the biex scale. Unknown keys are ignored.

    Note for callers: gates record *which* scale they were drawn on, not
    the parameters it had at the time, so changing these moves every gate
    already drawn on Biex. The interface says so before applying.
    """
    for k in ("negative", "width", "positive", "max_value"):
        if k in kw and kw[k] is not None:
            _biex_params[k] = float(kw[k])
    # Width is FlowJo's "width basis" and is negative by convention; a
    # positive one silently produces a scale with no linear region.
    if _biex_params["width"] > 0:
        _biex_params["width"] = -_biex_params["width"]


def fit_biex_range(low: float, high: float, tries: int = 4) -> dict:
    """Set the Biex scale to run from *low* to *high* in raw units.

    The parameters FlowJo exposes — width basis, decades, maximum — are
    what the lookup table is built from, not the ends it comes out with,
    so asking for "10^-3 to 10^9" cannot be answered by setting a field.
    It can be answered by searching: each end moves monotonically with one
    parameter, so a bisection on each converges in a few passes.

    The two interact slightly (width shifts the top by a fraction of a
    percent), hence the repeats. Returns the parameters chosen; the caller
    reads back biex_range() to see what was actually achieved, which is
    not always exactly what was asked for — a scale cannot put its floor
    above zero, and the top lands a little under the maximum by
    construction.
    """
    high = float(max(high, 10.0))
    # Aim a little past the requested floor. Landing exactly on it means
    # rounding can put it a hair above, and then the decade the operator
    # asked for is the one tick that gets dropped as unrepresentable —
    # ask for -10^2 and the axis comes back with no negative label at all.
    if low < 0:
        low = low * 1.08
    # Decades above the linear region. Setting this from the requested top
    # is what stops the linear region swallowing the axis: left at the
    # default 4.4 with a maximum of 10^9, the floor drops to -500,000.
    positive = float(max(1.0, np.log10(high)))
    set_biex_params(positive=positive, negative=0.0,
                    max_value=high, width=_biex_params["width"])

    def _bisect(key, lo_guess, hi_guess, target, read, rounds=40):
        lo, hi = lo_guess, hi_guess
        for _ in range(rounds):
            mid = (lo + hi) / 2.0
            set_biex_params(**{key: mid})
            if read() < target:
                lo = mid
            else:
                hi = mid
        set_biex_params(**{key: (lo + hi) / 2.0})

    for _ in range(tries):
        # Top rises with max_value.
        _bisect("max_value", high * 0.5, high * 3.0, high,
                lambda: biex_range()[1])
        if low < 0:
            # Floor falls as width basis goes more negative, so the search
            # runs backwards: bisect on -width.
            lo_w, hi_w = 0.1, 20000.0
            for _ in range(40):
                mid = (lo_w + hi_w) / 2.0
                set_biex_params(width=-mid)
                if biex_range()[0] > low:
                    lo_w = mid
                else:
                    hi_w = mid
            set_biex_params(width=-(lo_w + hi_w) / 2.0)
    return get_biex_params()


def biex_range() -> tuple:
    """The lowest and highest raw values this biex scale can show.

    What a person means by the minimum of a biex axis: below this the
    lookup table has no entries and everything lands on the same pixel.
    Found by inverting the ends of the output range rather than derived
    from the parameters, so it stays right if the parameters change
    meaning in a later FlowKit.
    """
    ends = np.array([0.0, BIEX_TOP])
    lo, hi = _inv_biex(ends)
    return float(lo), float(hi)


def _biex(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = _biex_transform().apply(x.reshape(-1, 1))
    return np.asarray(out, dtype=np.float64).reshape(x.shape)


def _inv_biex(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    out = _biex_transform().inverse(y.reshape(-1, 1))
    return np.asarray(out, dtype=np.float64).reshape(y.shape)


# ---------------------------------------------------------------------------
# Arcsinh
# ---------------------------------------------------------------------------
# The simplest of the biexponential family and the only one with a closed
# form: linear within about one cofactor of zero, logarithmic beyond it.
# Exact in both directions, with no lookup table and nothing to clip, so
# negative values survive a round trip that Biex would flatten.

def _arcsinh(x: np.ndarray, cofactor: float = ARCSINH_COFACTOR) -> np.ndarray:
    return np.arcsinh(np.asarray(x, dtype=np.float64) / cofactor)


def _inv_arcsinh(y: np.ndarray,
                 cofactor: float = ARCSINH_COFACTOR) -> np.ndarray:
    return np.sinh(np.asarray(y, dtype=np.float64)) * cofactor


def axis_ticks_labels(transform: str, vmin: float, vmax: float,
                      n_ticks: int = 6) -> tuple[list[float], list[str]]:
    """Return (tick_positions_in_display_space, tick_labels) for axis."""
    if transform == "Linear":
        ticks = np.linspace(vmin, vmax, n_ticks)
        labels = [f"{v:.0f}" for v in ticks]
        return list(ticks), labels

    elif transform == "Log10":
        # Decade ticks
        low = max(0, int(np.floor(vmin)))
        high = int(np.ceil(vmax)) + 1
        decades = list(range(low, high))
        ticks = [float(d) for d in decades if vmin <= d <= vmax]
        labels = [f"10{_superscript(d)}" for d in decades if vmin <= d <= vmax]
        return ticks, labels

    elif transform in ("Logicle", "Biex (FlowJo)", "Arcsinh"):
        return _decade_ticks(transform, vmin, vmax)

    return [], []


def _decade_ticks(transform: str, vmin: float,
                  vmax: float) -> tuple[list[float], list[str]]:
    """Decade ticks for a biexponential scale, over the range it covers.

    The decades are worked out from the axis itself rather than taken from
    a fixed list. A fixed list is fine until someone widens the scale: with
    Biex set to reach 10^9 the data was placed correctly and the axis still
    stopped labelling at 10^5, because 10^5 was the largest entry anyone
    had written down.

    Each decade is positioned by running its raw value back through the
    transform, so the three scales put 10^2 wherever each of them actually
    puts it rather than where the others do.
    """
    # What raw values the two ends of the axis correspond to. Inverting the
    # axis is what ties the ticks to the scale as configured, including any
    # parameters the operator has changed.
    try:
        raw_lo, raw_hi = inverse_transform(np.array([vmin, vmax]), transform)
    except Exception:                                   # pragma: no cover
        raw_lo, raw_hi = -1e2, 1e5
    raw_hi = float(max(raw_hi, 10.0))
    top = int(np.floor(np.log10(raw_hi)))
    bottom = (int(np.floor(np.log10(abs(float(raw_lo)))))
              if raw_lo < -1 else 1)

    # Below 10^2 the decades crowd into the linear region and the labels
    # collide, so that is where both wings stop — the same convention the
    # fixed list used.
    raw_vals = ([-(10.0 ** e) for e in range(bottom, 1, -1)] + [0.0] +
                [10.0 ** e for e in range(2, top + 1)])
    if len(raw_vals) == 1:                   # nothing but zero in range
        raw_vals = [0.0, 10.0 ** top]

    display = apply_transform(np.array(raw_vals, dtype=float), transform)
    # Biex is a lookup table and clips below its lowest entry, so every
    # negative decade past that point maps to the same value. Labelling the
    # clip floor "-10^4" would put a tick where no such value can be shown,
    # so saturated negatives are dropped rather than deduplicated into
    # whichever one came first.
    floor = float(apply_transform(np.array([-1e12]), transform)[0])
    ticks, labels, seen = [], [], set()
    for d, r in zip(display, raw_vals):
        if not (vmin <= d <= vmax):
            continue
        if r < 0 and d <= floor + 1e-9:
            continue
        key = round(float(d), 6)
        if key in seen:
            continue
        seen.add(key)
        ticks.append(float(d))
        if r == 0:
            labels.append("0")
        else:
            e = int(round(np.log10(abs(r))))
            labels.append(("-" if r < 0 else "") + f"10{_superscript(e)}")
    return ticks, labels


def _superscript(n: int) -> str:
    sups = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
    return str(n).translate(sups)
