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

        hooks = types.ModuleType("PyInstaller.utils.hooks")
        hooks.collect_data_files = collect_data_files
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
        with mock.patch.dict(sys.modules, fake_modules):
            runpy.run_path(str(spec_path), init_globals=fake_globals)

        collect_data_files.assert_called_once_with("osam")
        self.assertTrue(set(collected_data).issubset(analysis_arguments["datas"]))


if __name__ == "__main__":
    unittest.main()
