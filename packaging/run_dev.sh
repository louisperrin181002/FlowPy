#!/bin/bash
# Run FlowPy from source in its own environment (.venv-build), creating it
# on first use. Arguments are passed on (e.g. a .fcs or .fpw file).
set -euo pipefail
cd "$(dirname "$0")/.."
if [ ! -x .venv-build/bin/python ]; then
    "${PYTHON:-python3}" -m venv .venv-build
    .venv-build/bin/python -m pip install --quiet --upgrade pip
    .venv-build/bin/python -m pip install --quiet -r requirements.txt
fi
unset QT_PLUGIN_PATH QT_QPA_PLATFORM_PLUGIN_PATH PYTHONPATH
exec .venv-build/bin/python run_flowpy.py "$@"
