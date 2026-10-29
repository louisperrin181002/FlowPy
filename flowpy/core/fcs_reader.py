"""FCS file reader — uses FlowIO (established, maintained FCS 2.0/3.0/3.1
reader), with a minimal built-in fallback parser."""

from __future__ import annotations
import struct
import numpy as np
import pandas as pd
from pathlib import Path

from . import event_cache

RETRIES = 3          # network drives sometimes time out for a moment


class _File:
    """An FCS file handed to FlowIO, whose close() ignores errors: some
    network shares report a timeout when *closing* a file that was read
    without any problem (seen on SMB for files locked by another
    computer), which would otherwise throw away a complete read."""

    def __init__(self, path: str):
        import os
        self._f = open(path, "rb")
        self.name = os.path.basename(path)

    def read(self, *args):
        return self._f.read(*args)

    def seek(self, *args):
        return self._f.seek(*args)

    def tell(self):
        return self._f.tell()

    def close(self):
        try:
            self._f.close()
        except OSError:
            pass


def _flowdata(filepath: str, only_text: bool = False):
    import flowio
    fh = _File(filepath)
    try:
        return flowio.FlowData(fh, ignore_offset_discrepancy=True,
                               only_text=only_text)
    finally:
        fh.close()


def _retrying(fn, *args):
    """fn(*args), tried again (after 1 s, then 3 s) when the operating
    system reports a passing read error — usual on network drives."""
    import time
    for attempt in range(RETRIES):
        try:
            return fn(*args)
        except (TimeoutError, ConnectionError, InterruptedError):
            if attempt == RETRIES - 1:
                raise
        except OSError as exc:
            if exc.errno not in (5, 35, 60, 64, 65) or attempt == RETRIES - 1:
                raise
        time.sleep(1 + 2 * attempt)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def read_fcs(filepath: str | Path) -> tuple[dict, pd.DataFrame]:
    """Return (metadata_dict, events_DataFrame) for an FCS file.

    Events are preprocessed by FlowIO ($PnE log-amplification decoding and
    $PnG gain), so values are on the linear scale. Columns are named by
    the marker ($PnS) where there is one, else the channel ($PnN); if those
    names are not unique, all channels use $PnN. Metadata keys are the TEXT
    keywords, lower-case without "$" (e.g. "tot", "spill").
    """
    filepath = str(filepath)
    # Large files are memory-mapped from a temporary file (see
    # event_cache); one already read this session is not read again
    hit = event_cache.load(filepath)
    if hit is not None:
        meta, names, columns = hit
        return dict(meta), pd.DataFrame(columns.T, copy=False, columns=names)
    try:
        import flowio
    except ImportError:
        return _minimal_parse(filepath)
    try:
        fd = _retrying(_flowdata, filepath)
        columns = _preprocessed_columns(fd)
    except Exception as exc:
        raise IOError(f"Cannot read FCS file '{filepath}': {exc}") from exc
    names = channel_names(fd.pnn_labels, fd.pns_labels)
    meta = dict(fd.text)
    meta["_pnn_"] = list(fd.pnn_labels)
    del fd
    if columns.nbytes >= event_cache.MIN_FILE_BYTES:
        cached = event_cache.store(filepath, meta, names, columns)
        if cached is not None:
            columns = cached                 # the RAM copy is released
    # (channels × events) C-order, so every channel is one contiguous
    # float32 block that pandas uses without copying
    data = pd.DataFrame(columns.T, copy=False, columns=names)
    return meta, data


def read_fcs_header(filepath: str | Path) -> tuple[dict, list[str], int]:
    """(metadata, column names, event count) from the file's HEADER and
    TEXT segments only — a few KB, so it is instant even over a network.
    Same keywords and names as read_fcs()."""
    filepath = str(filepath)
    hit = event_cache.load(filepath)
    if hit is not None:
        meta, names, columns = hit
        return dict(meta), list(names), int(columns.shape[1])
    try:
        import flowio
    except ImportError:
        meta, data = _minimal_parse(filepath)
        return meta, list(data.columns), len(data)
    try:
        fd = _retrying(_flowdata, filepath, True)
    except Exception as exc:
        raise IOError(f"Cannot read FCS file '{filepath}': {exc}") from exc
    meta = dict(fd.text)
    meta["_pnn_"] = list(fd.pnn_labels)
    return (meta, channel_names(fd.pnn_labels, fd.pns_labels),
            int(fd.event_count))


def _preprocessed_columns(fd) -> np.ndarray:
    """Events as a (channels × events) float32 array, preprocessed exactly
    like FlowIO's as_array(preprocess=True) (time step, $PnE log decoding,
    $PnG gain) but one channel at a time, so a 10-million-event file
    doesn't need a full float64 copy. float32 is what cytometers record;
    FlowJo works in float32 too."""
    p = fd.channel_count
    raw = np.frombuffer(fd.events, dtype=fd.events.typecode).reshape(-1, p)
    out = np.empty((p, raw.shape[0]), dtype=np.float32)
    time_step = 1.0
    if "timestep" in fd.text and fd.time_index is not None:
        try:
            time_step = float(fd.text["timestep"])
        except ValueError:
            if fd.text["timestep"].strip():
                raise
    for num, ch in fd.channels.items():
        j = num - 1
        col = raw[:, j]
        decades, log0 = ch["pne"]
        gain = ch["png"]
        scale_time = j == fd.time_index and time_step != 1.0
        if decades > 0 or (gain != 1.0 and gain != 0) or scale_time:
            col = col.astype(np.float64)
            if scale_time:
                col = col * time_step
            if decades > 0:
                col = (10 ** (decades * col / ch["pnr"])) * log0
            if gain != 1.0 and gain != 0:
                col = col / gain
        out[j] = col
    return out


def channel_names(pnn: list[str], pns: list[str]) -> list[str]:
    """Marker name ($PnS) where set, else channel name ($PnN); all $PnN
    when that would give duplicates (the naming FlowPy has always used)."""
    pns = list(pns) if pns else [""] * len(pnn)
    names = [str(s) if s else str(n) for n, s in zip(pnn, pns)]
    return names if len(set(names)) == len(names) else [str(n) for n in pnn]


def extract_spillover(meta: dict) -> np.ndarray | None:
    """Extract the spillover / compensation matrix from FCS metadata.

    Returns an (N, N) float64 array, or None if no spillover is found.
    The returned matrix rows correspond to *channels* in the order given
    by the SPILL keyword's channel list.
    Also returns the channel name list via the second element of the tuple.
    """
    for key in ("SPILL", "$SPILL", "SPILLOVER", "$SPILLOVER",
                "spill", "spillover", "comp"):
        val = meta.get(key)
        if val:
            return _parse_spillover_string(val)
    return None, []


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_spillover_string(s: str):
    """Parse the standard SPILL string: 'N,ch1,ch2,...,v11,v12,...'.

    Fields are comma-separated (channel names may contain spaces). Returns
    (None, []) if the keyword is malformed, so the file still opens."""
    parts = [p.strip() for p in s.strip().split(",")]
    try:
        n = int(parts[0])
        channels = parts[1: n + 1]
        values = [float(v) for v in parts[n + 1: n + 1 + n * n]]
    except (ValueError, IndexError):
        return None, []
    if len(channels) != n or len(values) != n * n:
        return None, []
    matrix = np.array(values, dtype=np.float64).reshape(n, n)
    return matrix, channels


def _minimal_parse(filepath: str) -> tuple[dict, pd.DataFrame]:
    """Minimal FCS 3.x parser — used only when fcsparser is unavailable."""
    with open(filepath, "rb") as f:
        raw = f.read()

    version = raw[:6].decode("ascii", errors="replace").strip()
    text_start = int(raw[10:18].strip())
    text_end = int(raw[18:26].strip())
    data_start = int(raw[26:34].strip())
    data_end = int(raw[34:42].strip())

    # TEXT segment
    text_raw = raw[text_start: text_end + 1].decode("latin-1", errors="replace")
    delim = text_raw[0]
    pairs = text_raw[1:].split(delim)
    meta: dict = {}
    it = iter(pairs)
    for k in it:
        try:
            v = next(it)
        except StopIteration:
            break
        if k:
            meta[k.strip()] = v.strip()

    n_params = int(meta.get("$PAR", 0))
    n_events = int(meta.get("$TOT", 0))
    data_type = meta.get("$DATATYPE", "F").upper()
    byte_order = meta.get("$BYTEORD", "4,3,2,1")
    endian = "<" if byte_order.startswith("1") else ">"

    fmt_map = {"F": "f", "D": "d", "I": "i"}
    fmt = endian + fmt_map.get(data_type, "f")
    item_size = struct.calcsize(fmt)

    if data_start == 0:
        data_start = text_end + 1

    raw_data = raw[data_start: data_end + 1]
    n_values = n_events * n_params
    values = struct.unpack(f"{endian}{n_values}{fmt_map.get(data_type,'f')}",
                           raw_data[: n_values * item_size])
    arr = np.array(values, dtype=np.float64).reshape(n_events, n_params)

    # Channel names
    channels = []
    for i in range(1, n_params + 1):
        name = meta.get(f"$P{i}S") or meta.get(f"$P{i}N") or f"Ch{i}"
        channels.append(name)

    data = pd.DataFrame(arr, columns=channels)
    return meta, data
