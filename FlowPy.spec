# -*- mode: python ; coding: utf-8 -*-
# PyInstaller recipe for FlowPy (same approach as BioStatExplorer).
#   macOS:   ./packaging/build_mac.sh        → dist/FlowPy.app
#   Windows: GitHub Actions (build-windows)  → dist/FlowPy.exe
import sys

ICON = "assets/flowpy.icns" if sys.platform == "darwin" else "assets/flowpy.ico"

from PyInstaller.utils.hooks import (
    collect_data_files, collect_submodules, copy_metadata,
)

a = Analysis(
    ["run_flowpy.py"],
    pathex=[],
    binaries=[],
    # FlowKit needs its Gating-ML schemas; FlowSOM/anndata their data files
    datas=[("assets/flowpy.png", "assets")]
          + collect_data_files("flowkit")
          + collect_data_files("flowsom")
          + collect_data_files("anndata")
          # scanpy (used by FlowSOM) reads installed-package metadata
          + copy_metadata("flowsom", recursive=True)
          + copy_metadata("flowkit", recursive=True)
          + copy_metadata("umap-learn", recursive=True),
    # Matplotlib chooses its output backend from the file extension at save
    # time, so nothing imports these statically and PyInstaller cannot see
    # them. Without them, exporting a figure as SVG fails with
    # "No module named 'matplotlib.backends.backend_svg'".
    hiddenimports=["matplotlib.backends.backend_svg",
                   "matplotlib.backends.backend_pdf",
                   "matplotlib.backends.backend_ps",
                   "matplotlib.backends.backend_agg",
                   # loaded lazily by UMAP / scikit-learn / FlowSOM / FlowKit
                   "umap", "pynndescent", "sklearn.manifold",
                   "sklearn.mixture", "sklearn.cluster",
                   "sklearn.utils._typedefs",
                   "sklearn.neighbors._partition_nodes"]
                  + collect_submodules("flowsom")
                  + collect_submodules("flowkit"),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # The build runs in a clean venv (packaging/build_mac.sh), so only a
    # few unneeded GUI / notebook packages are left to exclude
    excludes=["PyQt5", "PySide2", "PySide6", "tkinter", "IPython",
              "jupyter", "notebook", "pytest"],
    noarchive=False,
    optimize=0,
    # scanpy (used by FlowSOM) inspects its own source code on import
    module_collection_mode={"scanpy": "pyz+py", "flowsom": "pyz+py"},
)
pyz = PYZ(a.pure)

if sys.platform == "darwin":
    exe = EXE(
        pyz, a.scripts, [],
        exclude_binaries=True,
        name="FlowPy",
        debug=False,
        strip=False,
        upx=False,
        console=False,
        argv_emulation=False,
        icon=ICON,
    )
    coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False,
                   name="FlowPy")
    app = BUNDLE(
        coll,
        name="FlowPy.app",
        icon=ICON,
        bundle_identifier="uk.ac.earlham.flowpy",
        info_plist={
            "CFBundleShortVersionString": "1.1.0",
            "CFBundleDisplayName": "FlowPy",
            "NSHighResolutionCapable": True,
            "NSRequiresAquaSystemAppearance": True,   # light theme
        },
    )
else:
    # Single-file Windows executable
    exe = EXE(
        pyz, a.scripts, a.binaries, a.datas, [],
        name="FlowPy",
        debug=False,
        strip=False,
        upx=True,
        console=False,
        icon=ICON,
    )
