#!/bin/bash
# Build a standalone PyInstaller executable for my_optimizer.
#
# Usage: bash build_pyinstaller.sh
#
# The resulting executable is placed at dist/my_optimizer.
# PyInstaller must be installed: pip install pyinstaller

set -e

PYTHON="${PYTHON:-python3}"
echo "Using: $PYTHON ($($PYTHON --version 2>&1))"

echo "=== Compiling C extension ==="
"$PYTHON" setup.py build_ext --inplace

if ! command -v pyinstaller &>/dev/null; then
    echo "ERROR: pyinstaller not found. Install with: pip install pyinstaller"
    exit 1
fi

echo "=== Building PyInstaller executable ==="
pyinstaller --onedir \
    --name my_optimizer \
    --add-data "bstar_core*.so:." \
    --hidden-import torch \
    --hidden-import numpy \
    --hidden-import Cython \
    --exclude-module iccad2026_evaluate \
    --exclude-module litetestLoader \
    --exclude-module liteLoader \
    --exclude-module lite_dataset \
    --exclude-module cost \
    --exclude-module utils \
    my_optimizer.py

echo "=== Done ==="
echo "Executable: dist/my_optimizer/my_optimizer"
echo ""
echo "Test with:"
echo "  python3 iccad2026_evaluate.py --evaluate op_wrapper.py --test-id 31"
echo ""
echo "Override Python version:"
echo "  PYTHON=python3.10 bash build_pyinstaller.sh"
