"""Compensation (spectral unmixing) utilities."""

from __future__ import annotations
import numpy as np
import pandas as pd


class CompensationMatrix:
    """Wraps an N×N spillover matrix and applies it to FCS data.

    The matrix follows the standard FCS convention:
      - rows = fluorochromes / detectors
      - M[i][j] = fraction of fluorochrome i signal detected in channel j
      - diagonal entries are 1.0
    To compensate: data_comp = data_raw @ inv(M)
    """

    def __init__(self, matrix: np.ndarray, channels: list[str]):
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Spillover matrix must be square (N×N).")
        if len(channels) != matrix.shape[0]:
            raise ValueError(
                f"Channel list length {len(channels)} ≠ matrix size {matrix.shape[0]}."
            )
        self.channels: list[str] = list(channels)
        self.matrix: np.ndarray = matrix.astype(np.float64)

    # ------------------------------------------------------------------
    # Computation
    # ------------------------------------------------------------------

    @property
    def inv_matrix(self) -> np.ndarray:
        """Cached inverse of the spillover matrix."""
        try:
            return np.linalg.inv(self.matrix)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "Spillover matrix is singular; cannot compute compensation."
            ) from exc

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a new DataFrame with compensated fluorescence columns.

        Scatter channels (FSC, SSC, Time) are left unchanged.
        """
        # shallow: only the compensated columns get new memory (the
        # others stay shared with the raw data, e.g. a memory-mapped file)
        df = df.copy(deep=False)
        # Only compensate columns whose names appear in self.channels
        present = [c for c in self.channels if c in df.columns]
        if not present:
            return df

        idx = [self.channels.index(c) for c in present]
        sub_matrix = self.matrix[np.ix_(idx, idx)]
        try:
            inv = np.linalg.inv(sub_matrix)
        except np.linalg.LinAlgError:
            return df  # silently skip on singular sub-matrix

        # FlowUtils' compensation (the same routine FlowKit uses)
        from flowutils.compensate import compensate
        # In chunks of events, so huge files never need a full float64
        # copy; results are stored as float32 like the raw data
        cols = [df[c].to_numpy() for c in present]
        n = len(df)
        out = np.empty((len(present), n), dtype=np.float32)
        idx_all = list(range(len(present)))
        step = max(1, 4_000_000 // max(len(present), 1))
        for a in range(0, n, step):
            b = min(n, a + step)
            chunk = np.column_stack([c[a:b] for c in cols]).astype(np.float64)
            out[:, a:b] = compensate(chunk, sub_matrix, idx_all).T
        for k, c in enumerate(present):
            df[c] = out[k]
        return df

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "channels": self.channels,
            "matrix": self.matrix.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CompensationMatrix":
        return cls(np.array(d["matrix"], dtype=np.float64), d["channels"])

    @classmethod
    def identity(cls, channels: list[str]) -> "CompensationMatrix":
        return cls(np.eye(len(channels)), channels)

    # ------------------------------------------------------------------
    # Helpers for UI editing
    # ------------------------------------------------------------------

    def set_value(self, row: int, col: int, value: float):
        self.matrix[row, col] = value

    def get_value(self, row: int, col: int) -> float:
        return float(self.matrix[row, col])

    def __repr__(self):
        return f"<CompensationMatrix channels={self.channels}>"


def guess_fluorescence_channels(columns: list[str]) -> list[str]:
    """Return column names that are likely fluorescence channels
    (exclude FSC, SSC, Time and derived area/height/width scatter params)."""
    exclude_prefixes = ("FSC", "SSC", "Time", "time")
    return [c for c in columns
            if not any(c.upper().startswith(p.upper()) for p in exclude_prefixes)]
