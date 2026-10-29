"""Channel display transforms: Linear, Log10, Logicle (biexponential)."""

from __future__ import annotations
import numpy as np

TRANSFORM_NAMES = ["Linear", "Log10", "Logicle"]


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

    elif transform == "Logicle":
        # Decade ticks: -10⁴ … -10², 0, 10² … 10⁵
        raw_vals = ([-(10.0 ** e) for e in (4, 3, 2)] + [0.0] +
                    [10.0 ** e for e in (2, 3, 4, 5)])
        display = apply_transform(np.array(raw_vals), "Logicle")
        ticks, labels = [], []
        for d, r in zip(display, raw_vals):
            if vmin <= d <= vmax:
                ticks.append(float(d))
                if r == 0:
                    labels.append("0")
                else:
                    e = int(round(np.log10(abs(r))))
                    labels.append(("-" if r < 0 else "") + f"10{_superscript(e)}")
        return ticks, labels

    return [], []


def _superscript(n: int) -> str:
    sups = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
    return str(n).translate(sups)
