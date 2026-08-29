"""Tests for loading user configuration overrides."""

import tempfile
import unittest
from pathlib import Path

from anylabeling.config import get_config


class TestGetConfig(unittest.TestCase):
    def test_empty_config_file_uses_defaults(self):
        with tempfile.TemporaryDirectory() as config_dir:
            config_path = Path(config_dir) / "empty.yaml"
            config_path.touch()
            config = get_config(str(config_path))

        self.assertIsInstance(config, dict)
        self.assertIn("shortcuts", config)

    def test_empty_inline_yaml_uses_defaults(self):
        config = get_config("")

        self.assertIsInstance(config, dict)
        self.assertIn("shortcuts", config)

    def test_inline_mapping_still_overrides_defaults(self):
        config = get_config("labels: [cat, dog]")

        self.assertEqual(config["labels"], ["cat", "dog"])


if __name__ == "__main__":
    unittest.main()
