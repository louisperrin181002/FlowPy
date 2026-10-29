"""Computational analysis: dimensionality reduction (UMAP / t-SNE),
clustering (FlowSOM, k-means, Gaussian mixture), automatic gating with
Gaussian mixtures, and spectral unmixing from single-stain controls.

Built on established libraries — umap-learn, scikit-learn, the official
Python FlowSOM, FlowUtils' OLS unmixing — and free of Qt, so it can run in a
worker thread.
"""

from __future__ import annotations
import functools
import warnings
import numpy as np

from .transforms import apply_transform

SEED = 42


def _quiet(func):
    """numpy on macOS (Accelerate) emits spurious matmul RuntimeWarnings
    inside scikit-learn; results are unaffected."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore", RuntimeWarning)
            warnings.simplefilter("ignore", FutureWarning)
            warnings.simplefilter("ignore", UserWarning)
            return func(*args, **kwargs)
    return wrapper


def _is_scatter(ch: str) -> bool:
    return any(p in ch.upper() for p in ("FSC", "SSC", "TIME"))


# ---------------------------------------------------------------------------
# Input preparation
# ---------------------------------------------------------------------------

def feature_matrix(sample, idx: np.ndarray, channels: list[str]) -> np.ndarray:
    """Events × channels matrix ready for ML: fluorescence on the Logicle
    scale, scatter linear, then every column standardised."""
    cols = []
    for ch in channels:
        v = sample.column(ch)[idx]
        if not _is_scatter(ch):
            v = apply_transform(v, "Logicle")
        cols.append(v)
    x = np.column_stack(cols).astype(np.float64)
    x = np.where(np.isfinite(x), x, np.nan)
    med = np.nanmedian(x, axis=0)
    x = np.where(np.isnan(x), med, x)
    sd = x.std(axis=0)
    return (x - x.mean(axis=0)) / np.where(sd > 0, sd, 1.0)


def subsample(idx: np.ndarray, n: int, seed: int = SEED) -> np.ndarray:
    if len(idx) <= n:
        return idx
    return np.sort(np.random.default_rng(seed).choice(idx, n, replace=False))


# ---------------------------------------------------------------------------
# Dimensionality reduction
# ---------------------------------------------------------------------------

@_quiet
def run_umap(x: np.ndarray, n_neighbors: int = 15,
             min_dist: float = 0.1) -> np.ndarray:
    try:
        import umap
    except ImportError as exc:
        raise RuntimeError("UMAP needs the 'umap-learn' package "
                           "(pip install umap-learn)") from exc
    return umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                     random_state=SEED).fit_transform(x)


@_quiet
def run_tsne(x: np.ndarray, perplexity: float = 30.0) -> np.ndarray:
    from sklearn.manifold import TSNE
    return TSNE(n_components=2, perplexity=min(perplexity, len(x) / 4),
                init="pca", random_state=SEED).fit_transform(x)


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

CLUSTER_METHODS = ["FlowSOM", "K-means", "Gaussian mixture"]


@_quiet
def cluster(x: np.ndarray, k: int, method: str = "FlowSOM",
            train_max: int = 100_000) -> np.ndarray:
    """Labels 0…k-1 for every row of *x*.

    FlowSOM uses the official Python implementation (saeyslab/FlowSOM_Python):
    a 10×10 self-organising map, consensus-clustered into *k* metaclusters.
    Models are trained on at most *train_max* events, then every event is
    assigned to its nearest node / cluster.
    """
    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture

    rng = np.random.default_rng(SEED)
    train = x if len(x) <= train_max else x[rng.choice(len(x), train_max,
                                                       replace=False)]
    if method in ("FlowSOM", "FlowSOM-style"):
        from flowsom.models import FlowSOMEstimator
        try:
            from loguru import logger
            logger.disable("flowsom")
        except ImportError:
            pass
        som = FlowSOMEstimator(n_clusters=k, xdim=10, ydim=10, seed=SEED)
        som.fit(train)
        labels = np.asarray(som.predict(x), dtype=int)
    elif method == "K-means":
        labels = KMeans(n_clusters=k, random_state=SEED, n_init=4) \
            .fit(train).predict(x)
    elif method == "Gaussian mixture":
        labels = GaussianMixture(n_components=k, random_state=SEED) \
            .fit(train).predict(x)
    else:
        raise ValueError(f"Unknown clustering method: {method}")
    return _relabel_by_size(labels, k)


def _relabel_by_size(labels: np.ndarray, k: int) -> np.ndarray:
    """Cluster 0 = biggest, so numbering is stable and meaningful."""
    sizes = np.bincount(labels, minlength=k)
    order = np.argsort(-sizes)
    remap = np.empty(k, dtype=int)
    remap[order] = np.arange(k)
    return remap[labels]


def cluster_profiles(sample, idx: np.ndarray, labels: np.ndarray,
                     channels: list[str], k: int) -> np.ndarray:
    """k × channels median expression (Logicle scale for fluorescence)."""
    prof = np.full((k, len(channels)), np.nan)
    for j, ch in enumerate(channels):
        v = sample.column(ch)[idx]
        if not _is_scatter(ch):
            v = apply_transform(v, "Logicle")
        for c in range(k):
            sel = v[labels == c]
            sel = sel[np.isfinite(sel)]
            if len(sel):
                prof[c, j] = np.median(sel)
    return prof


# ---------------------------------------------------------------------------
# Automatic gating (Gaussian mixture → rotated ellipse gates)
# ---------------------------------------------------------------------------

# Mahalanobis radius that contains this fraction of a 2-D Gaussian
COVERAGE = {"68%": 1.5096, "90%": 2.1460, "95%": 2.4477, "99%": 3.0349}


@_quiet
def auto_gate(x: np.ndarray, y: np.ndarray, n_components: int | None = None,
              coverage: str = "95%", min_fraction: float = 0.02,
              max_components: int = 6) -> list[dict]:
    """Find populations on a 2-D plot (display coordinates).

    n_components=None picks the number (1…max_components) by BIC.
    Returns ellipses as dicts (cx, cy, semi_x, semi_y, angle_deg, fraction),
    largest population first.
    """
    from sklearn.mixture import GaussianMixture

    pts = np.column_stack([x, y])
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if len(pts) < 50:
        raise ValueError("Too few events to find populations")
    pts = subsample_rows(pts, 30_000)

    if n_components is None:
        best, best_bic = None, np.inf
        for n in range(1, max_components + 1):
            gm = GaussianMixture(n, covariance_type="full",
                                 random_state=SEED).fit(pts)
            bic = gm.bic(pts)
            if bic < best_bic:
                best, best_bic = gm, bic
        gm = best
    else:
        gm = GaussianMixture(n_components, covariance_type="full",
                             random_state=SEED).fit(pts)

    r = COVERAGE[coverage]
    out = []
    for w, mu, cov in zip(gm.weights_, gm.means_, gm.covariances_):
        if w < min_fraction:
            continue
        vals, vecs = np.linalg.eigh(cov)          # ascending eigenvalues
        major = vecs[:, 1]
        out.append(dict(cx=float(mu[0]), cy=float(mu[1]),
                        semi_x=float(r * np.sqrt(vals[1])),
                        semi_y=float(r * np.sqrt(vals[0])),
                        angle_deg=float(np.degrees(np.arctan2(major[1],
                                                              major[0]))),
                        fraction=float(w)))
    return sorted(out, key=lambda e: -e["fraction"])


def subsample_rows(a: np.ndarray, n: int) -> np.ndarray:
    if len(a) <= n:
        return a
    return a[np.random.default_rng(SEED).choice(len(a), n, replace=False)]


# ---------------------------------------------------------------------------
# Spectral unmixing (ordinary least squares)
# ---------------------------------------------------------------------------

def control_signature(control, detectors: list[str],
                      background: np.ndarray | None,
                      top_fraction: float = 0.05) -> np.ndarray:
    """Normalised emission signature of a single-stain control: median
    spectrum of its brightest events, minus the unstained background,
    scaled so the peak detector is 1."""
    y = np.column_stack([control.column(d) for d in detectors])
    y = y[np.all(np.isfinite(y), axis=1)]
    base = background if background is not None else np.zeros(len(detectors))
    total = (y - base).sum(axis=1)
    bright = y[total >= np.quantile(total, 1.0 - top_fraction)]
    sig = np.median(bright, axis=0) - base
    peak = np.max(np.abs(sig))
    if peak <= 0:
        raise ValueError(f"{control.name}: no signal above background")
    return sig / peak


def background_spectrum(unstained, detectors: list[str]) -> np.ndarray:
    y = np.column_stack([unstained.column(d) for d in detectors])
    return np.nanmedian(np.where(np.isfinite(y), y, np.nan), axis=0)


@_quiet
def unmix(sample, detectors: list[str], signatures: np.ndarray,
          background: np.ndarray | None) -> np.ndarray:
    """Per-event abundance of each fluorochrome (events × fluors), by
    ordinary least squares on (Y − background), using FlowUtils' OLS."""
    from flowutils.compensate import compensate_spectral_ols
    y = np.column_stack([sample.column(d) for d in detectors])
    if background is not None:
        y = y - background
    out = compensate_spectral_ols(np.ascontiguousarray(y), signatures,
                                  list(range(len(detectors))))
    return np.asarray(out, dtype=np.float64)[:, :len(signatures)]
