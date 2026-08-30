"""Regression checks for the cross-platform test workflow."""

import unittest
from pathlib import Path


class TestCIWorkflow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        workflow = Path(__file__).parents[1] / ".github/workflows/tests.yml"
        cls.source = workflow.read_text(encoding="utf-8")

    def test_develop_pushes_and_pull_requests_run_the_matrix(self):
        self.assertIn("push:\n    branches: [main, master, develop]", self.source)
        self.assertIn(
            "pull_request:\n    branches: [main, master, develop]",
            self.source,
        )


if __name__ == "__main__":
    unittest.main()
