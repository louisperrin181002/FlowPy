"""Compensation matrix editor dialog."""

from __future__ import annotations
import numpy as np

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QPushButton, QLabel, QHeaderView, QMessageBox, QFileDialog,
    QDoubleSpinBox, QAbstractItemView, QDialogButtonBox,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QBrush

from ..core.compensation import CompensationMatrix, guess_fluorescence_channels


class CompensationDialog(QDialog):
    """Modal dialog for viewing and editing the spillover/compensation matrix.

    Shows the N×N matrix with editable cells (the diagonal is locked at 1.0).
    Clicking OK returns the edited matrix.
    """

    def __init__(self, matrix: CompensationMatrix | None,
                 all_channels: list[str],
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle("Compensation Matrix Editor")
        self.setMinimumSize(600, 450)
        self.resize(750, 500)

        # Determine fluorescence channels
        fl_channels = guess_fluorescence_channels(all_channels)

        if matrix is not None:
            self._channels = matrix.channels
            self._matrix = matrix.matrix.copy()
        else:
            self._channels = fl_channels
            self._matrix = np.eye(len(fl_channels))

        self._build_ui()
        self._populate_table()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # Info label
        info = QLabel(
            "Edit the spillover matrix. Diagonal entries are fixed at 1.0.\n"
            "Each row = fluorochrome detected in its own channel; "
            "off-diagonals = spillover into other detectors."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        # Channel count label
        self._n_label = QLabel(f"Channels: {len(self._channels)}")
        layout.addWidget(self._n_label)

        # Table
        n = len(self._channels)
        self._table = QTableWidget(n, n)
        self._table.setHorizontalHeaderLabels(self._channels)
        self._table.setVerticalHeaderLabels(self._channels)
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents)
        self._table.verticalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents)
        self._table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection)
        layout.addWidget(self._table)

        # Button row
        btn_row = QHBoxLayout()

        self._reset_btn = QPushButton("Reset to Identity")
        self._reset_btn.clicked.connect(self._reset_identity)
        btn_row.addWidget(self._reset_btn)

        self._import_btn = QPushButton("Import CSV…")
        self._import_btn.clicked.connect(self._import_csv)
        btn_row.addWidget(self._import_btn)

        self._export_btn = QPushButton("Export CSV…")
        self._export_btn.clicked.connect(self._export_csv)
        btn_row.addWidget(self._export_btn)

        btn_row.addStretch()
        layout.addLayout(btn_row)

        # OK / Cancel
        bbox = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel
        )
        bbox.accepted.connect(self._on_accept)
        bbox.rejected.connect(self.reject)
        layout.addWidget(bbox)

    # ------------------------------------------------------------------
    # Table population
    # ------------------------------------------------------------------

    def _populate_table(self):
        n = len(self._channels)
        self._table.blockSignals(True)
        for r in range(n):
            for c in range(n):
                val = self._matrix[r, c]
                item = QTableWidgetItem(f"{val:.4f}")
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if r == c:
                    # Diagonal locked
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    item.setBackground(QBrush(QColor("#2a3a2a")))
                else:
                    item.setBackground(QBrush(QColor("#1e2e3e")))
                self._table.setItem(r, c, item)
        self._table.blockSignals(False)

    def _read_table(self) -> np.ndarray:
        n = len(self._channels)
        mat = np.eye(n)
        for r in range(n):
            for c in range(n):
                if r == c:
                    continue
                item = self._table.item(r, c)
                try:
                    mat[r, c] = float(item.text()) if item else 0.0
                except ValueError:
                    mat[r, c] = 0.0
        return mat

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _reset_identity(self):
        self._matrix = np.eye(len(self._channels))
        self._populate_table()

    def _import_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Compensation Matrix", "", "CSV files (*.csv)")
        if not path:
            return
        try:
            import pandas as pd
            df = pd.read_csv(path, index_col=0)
            if df.shape[0] != df.shape[1]:
                raise ValueError("Matrix must be square.")
            n = df.shape[0]
            self._channels = list(df.columns)
            self._matrix = df.to_numpy(dtype=np.float64)
            self._table.setRowCount(n)
            self._table.setColumnCount(n)
            self._table.setHorizontalHeaderLabels(self._channels)
            self._table.setVerticalHeaderLabels(self._channels)
            self._n_label.setText(f"Channels: {n}")
            self._populate_table()
        except Exception as exc:
            QMessageBox.critical(self, "Import Error", str(exc))

    def _export_csv(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Compensation Matrix", "compensation.csv",
            "CSV files (*.csv)")
        if not path:
            return
        try:
            import pandas as pd
            mat = self._read_table()
            df = pd.DataFrame(mat, index=self._channels, columns=self._channels)
            df.to_csv(path)
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))

    def _on_accept(self):
        try:
            mat = self._read_table()
            # Verify invertibility
            np.linalg.inv(mat)
            self._matrix = mat
            self.accept()
        except np.linalg.LinAlgError:
            QMessageBox.warning(
                self, "Invalid Matrix",
                "The compensation matrix is singular (non-invertible). "
                "Please correct the values."
            )

    # ------------------------------------------------------------------
    # Result accessor
    # ------------------------------------------------------------------

    def get_matrix(self) -> CompensationMatrix:
        """Return the edited CompensationMatrix. Call after exec() == Accepted."""
        return CompensationMatrix(self._matrix, self._channels)
