"""Regression tests for data files included in release executables."""

import ctypes
import os
import runpy
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


class TestPyInstallerSpec(unittest.TestCase):
    def test_windows_runtime_hook_preloads_ort_dlls_from_capi(self):
        with tempfile.TemporaryDirectory() as tmp:
            capi = Path(tmp) / "onnxruntime" / "capi"
            capi.mkdir(parents=True)
            shared = capi / "onnxruntime_providers_shared.dll"
            runtime_dll = capi / "onnxruntime.dll"
            shared.touch()
            runtime_dll.touch()

            hook_path = Path(__file__).parents[1] / "rthooks" / "rthook_onnxruntime.py"
            with (
                mock.patch.object(sys, "platform", "win32"),
                mock.patch.object(sys, "_MEIPASS", tmp, create=True),
                mock.patch.object(ctypes, "WinDLL", create=True) as win_dll,
                mock.patch.object(
                    os, "add_dll_directory", create=True
                ) as add_directory,
            ):
                runpy.run_path(str(hook_path))

            self.assertEqual(
                win_dll.call_args_list,
                [mock.call(str(shared)), mock.call(str(runtime_dll))],
            )
            add_directory.assert_called_once_with(str(capi))

    def test_bundles_osam_tokenizer_data(self):
        collected_data = [
            (
                "/site-packages/osam/_models/yoloworld/clip/vocab.gz",
                "osam/_models/yoloworld/clip",
            )
        ]
        collect_data_files = mock.Mock(return_value=collected_data)
        ort_binaries = [
            (
                "/site-packages/onnxruntime/capi/libonnxruntime_providers_cuda.so",
                "onnxruntime/capi",
            )
        ]
        collect_dynamic_libs = mock.Mock(return_value=ort_binaries)

        hooks = types.ModuleType("PyInstaller.utils.hooks")
        hooks.collect_data_files = collect_data_files
        hooks.collect_dynamic_libs = collect_dynamic_libs
        utils = types.ModuleType("PyInstaller.utils")
        utils.hooks = hooks
        pyinstaller = types.ModuleType("PyInstaller")
        pyinstaller.utils = utils

        analysis_arguments = {}

        def analysis(*_args, **kwargs):
            analysis_arguments.update(kwargs)
            return SimpleNamespace(
                pure=[],
                zipped_data=[],
                scripts=[],
                binaries=kwargs["binaries"],
                zipfiles=[],
                datas=kwargs["datas"],
            )

        fake_globals = {
            "Analysis": analysis,
            "PYZ": mock.Mock(return_value=object()),
            "EXE": mock.Mock(return_value=object()),
            "BUNDLE": mock.Mock(return_value=object()),
        }
        fake_modules = {
            "PyInstaller": pyinstaller,
            "PyInstaller.utils": utils,
            "PyInstaller.utils.hooks": hooks,
        }

        with tempfile.TemporaryDirectory() as tmp:
            conda_expat = Path(tmp) / "lib" / "libexpat.so.1"
            conda_expat.parent.mkdir(parents=True)
            conda_expat.touch()
            nvidia_root = Path(tmp) / "nvidia"
            library_dir = nvidia_root / "cublas" / "lib"
            library_dir.mkdir(parents=True)
            nvidia_library = library_dir / "libcublas.so.12"
            nvidia_library.touch()
            optional_duplicate = library_dir / "libnvblas.so.12"
            optional_duplicate.touch()
            compile_tool = nvidia_root / "cuda_nvrtc" / "lib" / "libnvrtc.so.12"
            compile_tool.parent.mkdir(parents=True)
            compile_tool.touch()
            nvidia_spec = SimpleNamespace(submodule_search_locations=[str(nvidia_root)])

            spec_path = Path(__file__).parents[1] / "anylabeling.spec"
            with (
                mock.patch.dict(sys.modules, fake_modules),
                mock.patch.object(sys, "platform", "linux"),
                mock.patch.object(sys, "prefix", tmp),
                mock.patch("importlib.util.find_spec", return_value=nvidia_spec),
            ):
                runpy.run_path(str(spec_path), init_globals=fake_globals)

        collect_data_files.assert_called_once_with("osam")
        collect_dynamic_libs.assert_called_once_with("onnxruntime")
        self.assertTrue(set(collected_data).issubset(analysis_arguments["datas"]))
        self.assertTrue(set(ort_binaries).issubset(analysis_arguments["binaries"]))
        self.assertIn(
            (str(conda_expat), "."),
            analysis_arguments["binaries"],
        )
        self.assertIn(
            (str(nvidia_library), "nvidia/cublas/lib"),
            analysis_arguments["binaries"],
        )
        self.assertNotIn(
            (str(optional_duplicate), "nvidia/cublas/lib"),
            analysis_arguments["binaries"],
        )
        self.assertNotIn(
            (str(compile_tool), "nvidia/cuda_nvrtc/lib"),
            analysis_arguments["binaries"],
        )


if __name__ == "__main__":
    unittest.main()
