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
        self.assertIn("matrix.device == 'GPU' && 'conv'", self.source)
        self.assertIn("limit = 2 * 1024**3", self.source)

    def test_macos_gpu_build_installs_and_checks_coreml(self):
        self.assertIn('pip install -e ".[macos]"', self.source)
        self.assertIn("CoreMLExecutionProvider", self.source)
        self.assertIn("_MLModelProxy", self.source)


if __name__ == "__main__":
    unittest.main()
