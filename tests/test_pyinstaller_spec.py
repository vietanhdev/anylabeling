"""Regression tests for data files included in release executables."""

import runpy
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


class TestPyInstallerSpec(unittest.TestCase):
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
        nvidia_binaries = [
            (
                "/site-packages/nvidia/cublas/lib/libcublas.so.12",
                "nvidia/cublas/lib",
            )
        ]
        collect_dynamic_libs = mock.Mock(side_effect=[ort_binaries, nvidia_binaries])

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

        spec_path = Path(__file__).parents[1] / "anylabeling.spec"
        with (
            mock.patch.dict(sys.modules, fake_modules),
            mock.patch("importlib.util.find_spec", return_value=object()),
        ):
            runpy.run_path(str(spec_path), init_globals=fake_globals)

        collect_data_files.assert_called_once_with("osam")
        self.assertEqual(
            collect_dynamic_libs.call_args_list,
            [mock.call("onnxruntime"), mock.call("nvidia")],
        )
        self.assertTrue(set(collected_data).issubset(analysis_arguments["datas"]))
        self.assertTrue(
            set(ort_binaries + nvidia_binaries).issubset(analysis_arguments["binaries"])
        )


if __name__ == "__main__":
    unittest.main()
