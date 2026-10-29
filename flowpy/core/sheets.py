"""Import sample metadata from Excel (.xlsx) or CSV — either a table (one row
per sample) or a plate map (the A–H × 1–12 grid of a 96-well plate, or
A–P × 1–24 of a 384-well plate).

Table: the first column is the sample — its file name (with or without
.fcs) or its well (B3 / B03) — and every other column becomes a metadata
column. A column called "Name" can rename the samples.

Plate map: every block that has 1, 2, 3… across and A, B, C… down is one
metadata column, named by the text in its top-left corner or just above
it (or by the sheet name). Samples are matched by well, read from the FCS
file ($WELLID / WELL ID keyword) or from the file name ("…_B03.fcs").
"""

from __future__ import annotations
import csv
import re
from pathlib import Path

from .groups import stem

_ROWS = "ABCDEFGHIJKLMNOP"
_WELL_IN_NAME = re.compile(
    r"(?<![A-Za-z0-9])([A-Pa-p])[ _-]?0?(2[0-4]|1[0-9]|[1-9])(?![0-9])")

# Plate formats: name → (rows, columns)
PLATE_FORMATS = {"6-well (2 × 3)": (2, 3), "12-well (3 × 4)": (3, 4),
                 "24-well (4 × 6)": (4, 6), "48-well (6 × 8)": (6, 8),
                 "96-well (8 × 12)": (8, 12), "384-well (16 × 24)": (16, 24)}
AUTO_FORMAT = "Auto-detect"
# Where a sample's well comes from
WELL_SOURCES = ["Sample name — first well-like part (e.g. 'A03 xxx')",
                "Sample name — last well-like part (e.g. '…_B03.fcs')",
                "FCS file (well keyword)",
                "Well column only (typed in Sample Metadata)"]
WELL_COLUMN = "Well"
NAME_COLUMNS = {"name", "sample name", "sample", "display name", "label"}


# ---------------------------------------------------------------------------
# Wells
# ---------------------------------------------------------------------------

def norm_well(text) -> str | None:
    """'b03', 'B3', 'B 3' → 'B3'; anything else → None."""
    m = re.fullmatch(r"\s*([A-Pa-p])\s*[-_ ]?0*(\d{1,2})\s*", str(text or ""))
    if not m or not 1 <= int(m.group(2)) <= 24:
        return None
    return f"{m.group(1).upper()}{int(m.group(2))}"


def well_in_text(text: str, which: str = "first") -> str | None:
    """A well written in *text*: 'A03 GFP' → A3, 'Specimen_001_B03' → B3."""
    hits = _WELL_IN_NAME.findall(stem(text or ""))
    if not hits:
        return None
    h = hits[0] if which == "first" else hits[-1]
    return f"{h[0].upper()}{int(h[1])}"


def keyword_well(sample) -> str | None:
    meta = getattr(sample, "metadata", {}) or {}
    for key in ("wellid", "well id", "well_id", "well", "$wellid",
                "plate well", "well name"):
        w = norm_well(meta.get(key))
        if w:
            return w
    return None


def sample_well(sample, source: str | None = None) -> str | None:
    """Well of *sample*. A Well metadata column always wins; otherwise
    *source* (one of WELL_SOURCES; default: FCS keyword, then the name)."""
    w = norm_well(getattr(sample, "annotations", {}).get(WELL_COLUMN))
    if w:
        return w
    names = [Path(getattr(sample, "filepath", "") or sample.name).name,
             sample.name]
    if source == WELL_SOURCES[0]:
        return next((x for x in (well_in_text(n, "first") for n in names)
                     if x), None)
    if source == WELL_SOURCES[1]:
        return next((x for x in (well_in_text(n, "last") for n in names)
                     if x), None)
    if source == WELL_SOURCES[2]:
        return keyword_well(sample)
    if source == WELL_SOURCES[3]:
        return None
    return keyword_well(sample) or next(
        (x for x in (well_in_text(n, "last") for n in names) if x), None)


def detect_format(wells) -> str:
    """96-well, or 384-well if a well lies beyond H12. (A few wells used
    don't make a small plate — pick 6- to 48-well by hand.)"""
    wells = [w for w in wells if w]
    if any(_ROWS.index(w[0]) >= 8 or int(w[1:]) > 12 for w in wells):
        return "384-well (16 × 24)"
    return "96-well (8 × 12)"


def fits(well: str, fmt: str) -> bool:
    r, c = PLATE_FORMATS[fmt]
    return _ROWS.index(well[0]) < r and int(well[1:]) <= c


# ---------------------------------------------------------------------------
# Reading files into grids of text
# ---------------------------------------------------------------------------

def _cell(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v).strip()


def read_grids(path: str | Path) -> list[tuple[str, list[list[str]]]]:
    """[(sheet name, rows of cell text)] for .xlsx/.xlsm or CSV/TSV."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext in (".xlsx", ".xlsm"):
        import openpyxl
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        out = []
        for ws in wb.worksheets:
            rows = [[_cell(v) for v in r] for r in ws.iter_rows(values_only=True)]
            out.append((ws.title, rows))
        wb.close()
        return out
    if ext == ".xls":
        raise ValueError("Old .xls files aren't supported — in Excel, use "
                         "File → Save As → Excel Workbook (.xlsx).")
    with open(path, newline="", encoding="utf-8-sig") as f:
        text = f.read()
    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel
    return [(path.stem, [[c.strip() for c in r]
                         for r in csv.reader(text.splitlines(), dialect)])]


# ---------------------------------------------------------------------------
# Plate maps
# ---------------------------------------------------------------------------

def _plate_blocks(rows: list[list[str]], sheet: str
                  ) -> list[tuple[str, dict[str, str]]]:
    """Every plate-map block in a sheet: [(column name, {well: value})]."""
    blocks = []
    for r, row in enumerate(rows):
        for c in range(len(row)):
            # header: 1, 2, 3… running to the right of column c
            n = 0
            while c + 1 + n < len(row) and row[c + 1 + n] == str(n + 1):
                n += 1
            if n < 4:
                continue
            values = {}
            rr = r + 1
            while rr < len(rows) and len(rows[rr]) > c and \
                    rows[rr][c].strip().upper() == _ROWS[rr - r - 1:rr - r]:
                letter = rows[rr][c].strip().upper()
                for k in range(n):
                    cc = c + 1 + k
                    v = rows[rr][cc] if cc < len(rows[rr]) else ""
                    if v:
                        values[f"{letter}{k + 1}"] = v
                rr += 1
            if rr == r + 1:
                continue                    # no A, B, C… below
            label = rows[r][c].strip()
            if not label and r > 0:
                above = [x for x in rows[r - 1][c:c + 3] if x.strip()] \
                    if len(rows[r - 1]) > c else []
                label = above[0].strip() if above else ""
            blocks.append((label or sheet, values))
            break                           # one block per header row
    # unique names
    seen: dict[str, int] = {}
    out = []
    for label, vals in blocks:
        seen[label] = seen.get(label, 0) + 1
        out.append((label if seen[label] == 1 else
                    f"{label} {seen[label]}", vals))
    return out


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------

class ImportResult:
    def __init__(self):
        self.columns: list[str] = []
        self.values: dict[int, dict[str, str]] = {}   # sample index → row
        self.names: dict[int, str] = {}               # from a Name column
        self.unmatched: list[str] = []                # rows / wells unused
        self.kind = ""                                 # "table" / "plate map"
        self.matched_by = ""                           # "file name" / "well"
        self.unmatched_note = ""

    @property
    def n_matched(self) -> int:
        return len(set(self.values) | set(self.names))


def import_metadata(path: str | Path, samples,
                    well_source: str | None = None) -> ImportResult:
    """Read *path* and match its rows / wells to *samples* (wells found
    as *well_source* says, see WELL_SOURCES)."""
    res = ImportResult()
    grids = read_grids(path)
    plates = [b for sheet, rows in grids for b in _plate_blocks(rows, sheet)]
    wells = {i: sample_well(s, well_source) for i, s in enumerate(samples)}
    if plates:
        res.kind, res.matched_by = "plate map", "well"
        used = set()
        for label, vals in plates:
            res.columns.append(label)
            for i, w in wells.items():
                if w and w in vals:
                    res.values.setdefault(i, {})[label] = vals[w]
                    used.add(w)
        all_wells = {w for _, vals in plates for w in vals}
        res.unmatched = sorted(all_wells - used,
                               key=lambda w: (w[0], int(w[1:])))
        res.unmatched_note = (f"{len(res.unmatched)} filled well(s) in the "
                              "map have no loaded sample")
        _split_names(res)
        return res

    # table: first non-empty sheet, first non-empty row = header
    rows = next((r for _, r in grids if any(any(c for c in x) for x in r)),
                [])
    rows = [r for r in rows if any(c for c in r)]
    if len(rows) < 2:
        raise ValueError("The sheet has no rows of samples. Put the sample "
                         "file names (or wells) in the first column and a "
                         "header row on top.")
    header = [h.strip() for h in rows[0]]
    cols = [(k, h) for k, h in enumerate(header) if k > 0 and h]
    res.kind = "table"
    by_name: dict[str, list[int]] = {}
    for i, s in enumerate(samples):
        # the file name, or the name shown if the sample was renamed
        for n in {stem(Path(s.filepath).name), stem(s.name)}:
            by_name.setdefault(n.lower(), []).append(i)
    by_well: dict[str, list[int]] = {}
    for i, w in wells.items():
        if w:
            by_well.setdefault(w, []).append(i)
    name_hits = well_hits = 0
    rows_matched = []
    for r in rows[1:]:
        key = r[0].strip() if r else ""
        idx = by_name.get(stem(key).lower())
        if idx:
            name_hits += 1
        else:
            w = norm_well(key)
            idx = by_well.get(w) if w else None
            if idx:
                well_hits += 1
        if not idx:
            if key:
                res.unmatched.append(key)
            continue
        rows_matched.append((idx, r))
    res.matched_by = ("well" if well_hits > name_hits else "file name")
    res.columns = [h for _, h in cols]
    for idx, r in rows_matched:
        row = {h: (r[k].strip() if k < len(r) else "") for k, h in cols}
        for i in idx:
            res.values.setdefault(i, {}).update(row)
    _split_names(res)
    return res


def _split_names(res: ImportResult):
    """A 'Name' column renames the samples instead of being metadata."""
    name_col = next((c for c in res.columns
                     if c.strip().lower() in NAME_COLUMNS), None)
    if name_col is None:
        return
    for i, row in res.values.items():
        v = row.pop(name_col, "").strip()
        if v:
            res.names[i] = v
    res.columns.remove(name_col)
