#!/usr/bin/env python3
"""Build the bstar_core C extension.

Usage:
    python3.13 setup.py build_ext --inplace

If bstar_core.pyx is present and Cython is installed, the extension is
generated from .pyx.  Otherwise the pre-generated bstar_core.c is used.
"""
import os
from setuptools import setup, Extension

have_pyx = os.path.isfile("bstar_core.pyx")

if have_pyx:
    try:
        from Cython.Build import cythonize
    except ImportError:
        have_pyx = False

ext = Extension("bstar_core", sources=["bstar_core.pyx" if have_pyx else "bstar_core.c"])

kwargs = {}
if have_pyx:
    kwargs["ext_modules"] = cythonize(
        [ext],
        language_level=3,
        compiler_directives={"boundscheck": False, "wraparound": False, "cdivision": True},
    )
else:
    kwargs["ext_modules"] = [ext]

setup(name="bstar_core", **kwargs)
