#!/usr/bin/env python3
"""FlowPy — Flow Cytometry Analysis Software
Entry point. Run with: python run_flowpy.py
"""

import sys
import os
import multiprocessing

# In the packaged app, worker processes (started by scikit-learn / joblib,
# numba…) re-run this executable. freeze_support() lets them do their job
# and exit; without it each one started another FlowPy and hung the app.
# It must run before the heavy imports below.
if __name__ == "__main__":
    multiprocessing.freeze_support()

# Ensure the package is importable from this directory
sys.path.insert(0, os.path.dirname(__file__))

# Persistent caches, set before matplotlib / numba are imported. In the
# packaged app PyInstaller points matplotlib at a fresh temporary folder on
# every launch, so its font cache was rebuilt each time (30–60 s startup).
_CACHE = os.path.join(os.path.expanduser("~"), "Library", "Caches", "FlowPy") \
    if sys.platform == "darwin" else \
    os.path.join(os.path.expanduser("~"), ".cache", "flowpy")
if getattr(sys, "frozen", False):
    os.environ["MPLCONFIGDIR"] = os.path.join(_CACHE, "matplotlib")
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
# UMAP compiles with numba, whose cache must be writable (the app bundle
# isn't)
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(_CACHE, "numba"))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon
from flowpy.ui.main_window import MainWindow

# Bundled files live in sys._MEIPASS when running as a packaged app
_BASE = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
ICON_PATH = os.path.join(_BASE, "assets", "flowpy.png")



def selftest() -> int:
    """`FlowPy --selftest`: check the scientific libraries work in this
    build (FCS I/O, Logicle, compensation, FlowSOM, UMAP, auto gating)."""
    import tempfile
    import numpy as np
    import flowio
    from flowpy.core.sample import Sample
    from flowpy.core.compensation import CompensationMatrix
    from flowpy.core.transforms import apply_transform, inverse_transform
    from flowpy.core import analysis

    rng = np.random.default_rng(0)
    x = np.vstack([rng.normal(c, 1, (500, 4)) for c in (0, 6, 12)])
    checks = {}
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "selftest.fcs")
        with open(path, "wb") as fh:
            flowio.create_fcs(fh, (x * 1000).astype(np.float32).ravel(),
                              ["FSC-A", "SSC-A", "FL1-A", "FL2-A"])
        s = Sample(path)
        checks["fcs"] = s.n_events == 1500 and np.allclose(
            s.column("FL1-A"), x[:, 2] * 1000, rtol=1e-5)
    v = np.array([-100.0, 0.0, 1e3, 1e5])
    checks["logicle"] = np.allclose(inverse_transform(
        apply_transform(v, "Logicle"), "Logicle"), v)
    comp = CompensationMatrix(np.array([[1, .1], [.05, 1]]), ["FL1-A", "FL2-A"])
    checks["compensation"] = np.allclose(
        comp.apply(s.data)[["FL1-A", "FL2-A"]].to_numpy(),
        s.data[["FL1-A", "FL2-A"]].to_numpy() @ np.linalg.inv(comp.matrix))
    checks["flowsom"] = len(set(analysis.cluster(x, 3, "FlowSOM"))) == 3
    checks["umap"] = analysis.run_umap(x, 15, 0.1).shape == (1500, 2)
    checks["autogate"] = len(analysis.auto_gate(x[:, 0], x[:, 1])) == 3
    # FlowJo / Gating-ML exchange: FlowKit recounts a gate, and a Gating-ML
    # file survives a round trip (needs FlowKit's bundled schemas)
    from flowpy.core.gates import RectangleGate
    from flowpy.core import flowjo_io, protocol
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "selftest.fcs")
        with open(path, "wb") as fh:
            flowio.create_fcs(fh, (x * 1000).astype(np.float32).ravel(),
                              ["FSC-A", "SSC-A", "FL1-A", "FL2-A"])
        s = Sample(path)
        s.gates = [RectangleGate("Pop", "FSC-A", "SSC-A", 3000, 9000,
                                 3000, 9000)]
        s.gate_masks = protocol.compute_masks(s, s.gates)
        report = flowjo_io.verify([s])
        xml = os.path.join(tmp, "selftest.xml")
        flowjo_io.export_gatingml(xml, s)
        roots, _ = flowjo_io.import_gatingml(xml, s)
        checks["flowkit"] = (len(report.counts) == 1
                             and not report.mismatches and len(roots) == 1)
    # group comparison tests (scipy.stats) and spectral detector detection
    from flowpy.core import groups, spectral
    _name, p = groups.compare({"a": x[:500, 2], "b": x[500:1000, 2]},
                              groups.TESTS[1])
    checks["groups"] = 0 <= p < 0.05 and len(spectral.spectral_detectors(
        [f"V{i}-A" for i in range(1, 17)])) == 16
    # Excel sample sheets / plate maps (openpyxl)
    from flowpy.core import sheets
    with tempfile.TemporaryDirectory() as tmp:
        import openpyxl
        wb = openpyxl.Workbook()
        wb.active.append(["File", "Group"])
        wb.active.append(["selftest.fcs", "A"])
        xl = os.path.join(tmp, "sheet.xlsx")
        wb.save(xl)
        res = sheets.import_metadata(xl, [s])
        checks["excel"] = res.values.get(0, {}).get("Group") == "A" and \
            sheets.well_in_text("A03 GFP") == "A3"
    # Figure export. Matplotlib loads the writer for a format on the first
    # save in that format, so a packaged build looks fine until someone
    # asks for SVG. Exercising every offered format here catches a missing
    # backend at build time instead of in front of a colleague.
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    fig = Figure(figsize=(2, 2))
    fig.add_subplot(111).plot([0, 1], [0, 1])
    with tempfile.TemporaryDirectory() as tmp:
        for fmt in ("png", "pdf", "svg", "tiff"):
            fig.savefig(os.path.join(tmp, "f." + fmt), format=fmt)
        checks["export"] = all(
            os.path.getsize(os.path.join(tmp, "f." + f)) > 200
            for f in ("png", "pdf", "svg", "tiff"))

    ok = all(checks.values())
    print("selftest", "OK" if ok else "FAILED", checks)
    return 0 if ok else 1


def main():
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    # Enable high-DPI scaling
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv)
    app.setApplicationName("FlowPy")
    app.setOrganizationName("FlowPy")
    app.setApplicationVersion("1.1.0")
    if os.path.exists(ICON_PATH):
        app.setWindowIcon(QIcon(ICON_PATH))

    window = MainWindow()
    window.show()

    # Open a workspace or FCS files passed as command-line arguments
    for arg in sys.argv[1:]:
        if arg.lower().endswith(".fpw"):
            window.load_workspace_file(arg)
    window.load_fcs_files([a for a in sys.argv[1:]
                           if a.lower().endswith(".fcs")])

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
