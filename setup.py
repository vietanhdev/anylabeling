"""
Minimal setup.py shim.

All static metadata lives in pyproject.toml (PEP 621).
This file only exists to handle the one dynamic behaviour that cannot be
expressed in static TOML: switching the package name to 'anylabeling-gpu'
and replacing onnxruntime with onnxruntime-gpu when __preferred_device__
is set to 'GPU' in anylabeling/app_info.py.
"""
import re
import platform

from setuptools import find_packages, setup


def _read_app_info(key):
    with open("anylabeling/app_info.py", encoding="utf-8") as f:
        content = f.read()
    match = re.search(rf"""^{key} = ['"]([^'"]*)['"]""", content, re.M)
    if not match:
        raise RuntimeError(f"anylabeling/app_info.py doesn't contain {key}")
    return match.group(1)


preferred_device = _read_app_info("__preferred_device__")
is_gpu = preferred_device == "GPU" and platform.system() != "Darwin"

# Only override what pyproject.toml cannot express dynamically.
# setuptools merges these kwargs with pyproject.toml metadata.
if is_gpu:
    setup(
        name="anylabeling-gpu",
        install_requires=[
            # swap CPU onnxruntime for the GPU build
            "onnxruntime-gpu==1.18.1",
        ],
    )
    print("Building AnyLabeling with GPU support")
else:
    setup()  # everything comes from pyproject.toml
    print("Building AnyLabeling without GPU support (CPU)")
