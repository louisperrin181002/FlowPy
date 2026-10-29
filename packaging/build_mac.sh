#!/bin/bash
# Build dist/FlowPy.app. Run from anywhere:  ./packaging/build_mac.sh
# Then drag dist/FlowPy.app into /Applications.
#
# Builds inside a private virtual environment (.venv-build) containing only
# FlowPy's dependencies. Building straight from Anaconda mixes in Anaconda's
# own Qt 5 plugins, and the resulting app fails to start.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PYTHON:-python3}
if [ ! -x .venv-build/bin/python ]; then
    "$PY" -m venv .venv-build
fi
VPY=.venv-build/bin/python
# Keep the shell's Anaconda Qt settings out of the build
unset QT_PLUGIN_PATH QT_QPA_PLATFORM_PLUGIN_PATH PYTHONPATH

"$VPY" -m pip install --quiet --upgrade pip
"$VPY" -m pip install --quiet -r requirements.txt pyinstaller pillow
[ -f assets/flowpy.icns ] || "$VPY" packaging/make_icon.py
"$VPY" -m PyInstaller --noconfirm --clean FlowPy.spec

echo
echo "Built: $(pwd)/dist/FlowPy.app  ($(du -sh dist/FlowPy.app | cut -f1))"
