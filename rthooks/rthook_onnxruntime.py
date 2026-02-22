"""
Runtime hook: preload onnxruntime native DLLs before any Python import can fail.

The problem:
  onnxruntime_pybind11_state.pyd imports onnxruntime.dll, which during its
  DllMain calls LoadLibraryW("onnxruntime_providers_shared.dll") using the
  standard Win32 search order (SetDllDirectory / PATH), NOT the
  AddDllDirectory search order.  In a one-file PyInstaller bundle the DLLs
  land in _MEIPASS/onnxruntime/capi/ — a subdirectory that is NOT covered by
  either SetDllDirectory(_MEIPASS) or PATH in a clean deployment.

The fix:
  Load both DLLs by absolute path with ctypes.WinDLL() here, before Python's
  importlib ever tries to load the .pyd.  Once a DLL is loaded into the
  process the Windows loader returns the cached handle on any subsequent
  LoadLibraryW call for the same module name, so ORT's internal lookup
  succeeds regardless of search-path issues.
"""

import ctypes
import os
import sys

if sys.platform == "win32" and hasattr(sys, "_MEIPASS"):
    _capi = os.path.join(sys._MEIPASS, "onnxruntime", "capi")
    # Load in dependency order: shared lib first, then main DLL
    for _dll in ("onnxruntime_providers_shared.dll", "onnxruntime.dll"):
        _path = os.path.join(_capi, _dll)
        if os.path.isfile(_path):
            ctypes.WinDLL(_path)
    # Also register the directory so Python's importlib finds the .pyd
    if os.path.isdir(_capi):
        os.add_dll_directory(_capi)
