"""Regression checks for CPU/GPU release dependency selection."""

import unittest
from pathlib import Path


class TestReleaseWorkflow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        workflow = Path(__file__).parents[1] / ".github/workflows/release.yml"
        cls.source = workflow.read_text(encoding="utf-8")

    def test_nvidia_build_replaces_cpu_onnx_runtime(self):
        self.assertIn("pip uninstall -y onnxruntime", self.source)
        self.assertIn(
            'pip install "onnxruntime-gpu[cuda,cudnn]>=1.20.0,<1.27"',
            self.source,
        )
        self.assertIn("CUDAExecutionProvider", self.source)
        self.assertIn("--selection-only", self.source)
        self.assertIn("libegl1", self.source)
        self.assertIn("limit = 2 * 1024**3", self.source)

    def test_macos_gpu_build_installs_and_checks_coreml(self):
        self.assertIn('pip install -e ".[macos]"', self.source)
        self.assertIn("CoreMLExecutionProvider", self.source)
        self.assertIn('get_onnx_providers("${{ matrix.device }}")', self.source)
        self.assertIn("_MLModelProxy", self.source)

    def test_macos_archive_preserves_symlinks_and_launches(self):
        self.assertIn("ditto -c -k --sequesterRsrc --keepParent", self.source)
        self.assertNotIn("zip -r AnyLabeling-macOS-", self.source)
        self.assertIn("macOS archive lost PyInstaller symlinks", self.source)
        self.assertIn("process.wait(timeout=15)", self.source)
        self.assertIn("Extracted macOS application exited early", self.source)


if __name__ == "__main__":
    unittest.main()
