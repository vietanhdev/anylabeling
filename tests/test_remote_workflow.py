import unittest
from pathlib import Path


class TestRemoteRealModelWorkflow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = (
            Path(__file__).parents[1] / ".github/workflows/remote-real-model-e2e.yml"
        ).read_text(encoding="utf-8")

    def test_runs_real_authenticated_model_on_all_desktop_operating_systems(self):
        self.assertIn("os: [ubuntu-latest, windows-latest, macos-latest]", self.source)
        self.assertIn("scripts/validate_remote_inference.py", self.source)
        self.assertIn("if: always()", self.source)
        self.assertIn("retention-days: 30", self.source)

    def test_pins_server_contract_and_model_checksums(self):
        self.assertIn("ref: 7cabc1e8caaec070410fbc47e8ee250ae50454ce", self.source)
        self.assertIn(
            "c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063",
            self.source,
        )
        self.assertIn(
            "5a9522051c3cec2bbd2f6323fccba32e8fbf3ddcc2b3e2fd46b04c720bc6f866",
            self.source,
        )


if __name__ == "__main__":
    unittest.main()
