"""Sample metadata (group, condition, strain, timepoint…) and statistics
compared between groups.

Each sample carries ``sample.annotations`` ({column: value}); the workspace
keeps the ordered list of metadata columns.
"""

from __future__ import annotations
import csv
import re
from pathlib import Path
import numpy as np
import pandas as pd

from .protocol import find_population, COUNT_STATS
from .statistics import stat_1d, stat_label

DEFAULT_COLUMNS = ["Group"]
NO_GROUP = "(none)"


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def stem(name: str) -> str:
    """File name without .fcs."""
    return re.sub(r"\.fcs$", "", name, flags=re.IGNORECASE)


def split_name(name: str, sep: str, part: int) -> str:
    """Part *part* (1-based) of the file name split on *sep*; negative
    counts from the end. Empty when the name has fewer parts."""
    parts = [p.strip() for p in stem(name).split(sep)] if sep else [stem(name)]
    k = part - 1 if part > 0 else len(parts) + part
    return parts[k] if 0 <= k < len(parts) else ""


def parse_table_text(text: str) -> list[list[str]]:
    """Cells pasted from Excel / Numbers / Prism (tab-separated rows)."""
    rows = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    if rows and rows[-1] == "":
        rows = rows[:-1]
    return [r.split("\t") for r in rows]


def read_metadata_csv(path: str | Path, sample_names: list[str]
                      ) -> tuple[list[str], dict[str, dict[str, str]], list]:
    """Read a CSV (or tab-separated) table whose first column is the sample
    (file) name. Returns (columns, {sample name: {column: value}},
    unmatched rows). Names match with or without .fcs, ignoring case."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        text = f.read()
    dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t") \
        if text.strip() else csv.excel
    rows = [r for r in csv.reader(text.splitlines(), dialect) if any(r)]
    if not rows:
        return [], {}, []
    header = [h.strip() for h in rows[0]]
    columns = [h for h in header[1:] if h]
    lookup = {stem(n).lower(): n for n in sample_names}
    out, unmatched = {}, []
    for r in rows[1:]:
        key = stem(r[0].strip()).lower() if r else ""
        name = lookup.get(key)
        if name is None:
            unmatched.append(r[0] if r else "")
            continue
        out[name] = {h: (r[i].strip() if i < len(r) else "")
                     for i, h in enumerate(header) if i > 0 and h}
    return columns, out, unmatched


def write_metadata_csv(path: str | Path, samples, columns: list[str]):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["Sample"] + columns)
        for s in samples:
            w.writerow([s.name] + [s.annotations.get(c, "") for c in columns])


def _number(v: str) -> float | None:
    try:
        return float(str(v).strip().replace(",", "."))
    except ValueError:
        return None


def is_numeric(values) -> bool:
    vals = [v for v in values if v not in ("", NO_GROUP)]
    return bool(vals) and all(_number(v) is not None for v in vals)


def group_order(values: list[str]) -> list[str]:
    """Distinct values: numeric order when they are all numbers, otherwise
    order of first appearance; the no-group bucket goes last."""
    seen = list(dict.fromkeys(values))
    rest = [v for v in seen if v != NO_GROUP]
    if is_numeric(rest):
        rest.sort(key=_number)
    return rest + ([NO_GROUP] if NO_GROUP in seen else [])


# ---------------------------------------------------------------------------
# Values per sample
# ---------------------------------------------------------------------------

def sample_value(sample, population: str, channel: str | None, stat: str,
                 pct: float = 50.0) -> float:
    """One number for *sample*: a count statistic of *population*, or
    *stat* of *channel* in it. NaN if the population/channel is missing."""
    try:
        gate, mask = find_population(sample, population)
    except KeyError:
        return np.nan
    n_total = sample.n_events
    n = int(mask.sum()) if mask is not None else n_total
    if stat == "Count":
        return float(n)
    if stat == "% Total":
        return 100.0 * n / n_total if n_total else np.nan
    if stat == "% Parent":
        par = (sample.gate_masks.get(gate.parent.gate_id)
               if gate is not None and gate.parent else None)
        n_par = int(par.sum()) if par is not None else n_total
        return 100.0 * n / n_par if n_par else np.nan
    if not channel or channel not in sample.channels:
        return np.nan
    return stat_1d(sample.column(channel), stat, pct, mask)


def value_label(population: str, channel: str | None, stat: str,
                pct: float = 50.0) -> str:
    pop = population.split(" / ")[-1]
    if stat in COUNT_STATS:
        return f"{stat} of {pop}" if stat != "Count" else f"{pop} count"
    return f"{stat_label(stat, pct)} {channel} ({pop})"


def group_table(samples, group_col: str, hue_col: str | None,
                population: str, channel: str | None, stat: str,
                pct: float = 50.0, include_ungrouped: bool = False
                ) -> pd.DataFrame:
    """One row per sample: Sample, Group, [Hue], Value (+ every metadata
    value, handy in the exported CSV)."""
    rows = []
    for s in samples:
        g = s.annotations.get(group_col, "").strip() if group_col != \
            "Sample" else s.name
        if not g:
            if not include_ungrouped:
                continue
            g = NO_GROUP
        row = {"Sample": s.name, "Group": g}
        if hue_col:
            row["Hue"] = s.annotations.get(hue_col, "").strip() or NO_GROUP
        row["Value"] = sample_value(s, population, channel, stat, pct)
        rows.append(row)
    return pd.DataFrame(rows, columns=["Sample", "Group"] +
                        (["Hue"] if hue_col else []) + ["Value"])


# ---------------------------------------------------------------------------
# Summary and tests
# ---------------------------------------------------------------------------

ERRORS = ["SD", "SEM", "95% CI", "None"]
TESTS = ["None", "Parametric (t-test / ANOVA)",
         "Non-parametric (Mann-Whitney / Kruskal-Wallis)"]


def spread(values: np.ndarray, kind: str) -> float:
    v = values[np.isfinite(values)]
    n = len(v)
    if n < 2 or kind == "None":
        return 0.0 if kind == "None" else np.nan
    sd = float(np.std(v, ddof=1))
    if kind == "SD":
        return sd
    sem = sd / np.sqrt(n)
    if kind == "SEM":
        return sem
    from scipy.stats import t
    return float(t.ppf(0.975, n - 1) * sem)


def summarize(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """n, mean, SD, SEM, median, min, max of Value per group."""
    rows = []
    for keys, part in df.groupby(by, sort=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        v = part["Value"].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        n = len(v)
        rows.append({**dict(zip(by, keys)), "n": n,
                     "Mean": float(np.mean(v)) if n else np.nan,
                     "SD": float(np.std(v, ddof=1)) if n > 1 else np.nan,
                     "SEM": float(np.std(v, ddof=1) / np.sqrt(n))
                     if n > 1 else np.nan,
                     "Median": float(np.median(v)) if n else np.nan,
                     "Min": float(np.min(v)) if n else np.nan,
                     "Max": float(np.max(v)) if n else np.nan})
    return pd.DataFrame(rows)


def compare(groups: dict[str, np.ndarray], kind: str) -> tuple[str, float]:
    """(test name, p-value) across *groups*; groups with fewer than two
    values are left out. ('', nan) when no test applies."""
    from scipy import stats
    data = {k: v[np.isfinite(v)] for k, v in groups.items()}
    data = {k: v for k, v in data.items() if len(v) >= 2}
    if kind == TESTS[0] or len(data) < 2:
        return "", np.nan
    vals = list(data.values())
    parametric = kind == TESTS[1]
    with np.errstate(all="ignore"):
        if len(vals) == 2:
            if parametric:
                r = stats.ttest_ind(*vals, equal_var=False)
                return "Welch's t-test", float(r.pvalue)
            r = stats.mannwhitneyu(*vals, alternative="two-sided")
            return "Mann-Whitney U", float(r.pvalue)
        if parametric:
            return "One-way ANOVA", float(stats.f_oneway(*vals).pvalue)
        return "Kruskal-Wallis", float(stats.kruskal(*vals).pvalue)


def format_p(p: float) -> str:
    if not np.isfinite(p):
        return "n/a"
    if p < 1e-4:
        return "p < 0.0001"
    return f"p = {p:.4f}" if p < 0.01 else f"p = {p:.3f}"
