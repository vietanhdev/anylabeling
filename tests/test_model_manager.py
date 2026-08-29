import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

# Add paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from anylabeling.services.auto_labeling.model_manager import ModelManager


class TestModelManager(unittest.TestCase):
    @patch(
        "anylabeling.services.auto_labeling.model_manager.ModelManager.load_model_configs"
    )
    def test_set_text_prompt_delegation(self, mock_load):
        # Prevent actual loading of configs which might be slow/blocking
        manager = ModelManager()
        mock_model = MagicMock()
        manager.loaded_model_config = {"model": mock_model, "type": "segment_anything"}

        manager.set_text_prompt("new prompt")
        mock_model.set_text_prompt.assert_called_once_with("new prompt")

    @patch(
        "anylabeling.services.auto_labeling.model_manager.ModelManager.load_model_configs"
    )
    def test_unload_model(self, mock_load):
        manager = ModelManager()
        mock_model = MagicMock()
        manager.loaded_model_config = {"model": mock_model}

        manager.unload_model()
        mock_model.unload.assert_called_once()
        self.assertIsNone(manager.loaded_model_config)

    @patch(
        "anylabeling.services.auto_labeling.model_manager.ModelManager.load_model_configs"
    )
    def test_invalid_custom_model_reports_completion(self, mock_load):
        manager = ModelManager()
        statuses = []
        completions = []
        manager.new_model_status.connect(statuses.append)
        manager.model_loaded.connect(completions.append)

        with tempfile.TemporaryDirectory() as temp_dir:
            invalid_configs = {
                "missing.yaml": None,
                "empty.yaml": "",
                "malformed.yaml": "type: [unterminated",
                "wrong-shape.yaml": "- not\n- a\n- mapping\n",
                "missing-fields.yaml": "type: yolov8\n",
            }
            for filename, content in invalid_configs.items():
                with self.subTest(filename=filename):
                    path = os.path.join(temp_dir, filename)
                    if content is not None:
                        with open(path, "w", encoding="utf-8") as config_file:
                            config_file.write(content)
                    statuses.clear()
                    completions.clear()

                    manager.load_custom_model(path)

                    self.assertEqual(len(statuses), 1)
                    self.assertTrue(
                        statuses[0].startswith("Error in loading custom model:")
                    )
                    self.assertEqual(completions, [{}])

    @patch(
        "anylabeling.services.auto_labeling.model_manager.ModelManager.load_model_configs"
    )
    def test_unknown_model_reports_completion(self, mock_load):
        manager = ModelManager()
        completions = []
        manager.model_loaded.connect(completions.append)

        manager.load_model("unknown-config.yaml")

        self.assertEqual(completions, [{}])

    @patch(
        "anylabeling.services.auto_labeling.model_manager.ModelManager.load_model_configs"
    )
    def test_download_error_is_reported_without_escaping_worker(self, mock_load):
        manager = ModelManager()
        manager.model_configs = [
            {
                "display_name": "Broken download",
                "has_downloaded": False,
                "type": "yolov8",
            }
        ]
        statuses = []
        completions = []
        manager.new_model_status.connect(statuses.append)
        manager.model_loaded.connect(completions.append)

        with (
            patch.object(
                manager,
                "_download_and_extract_model",
                side_effect=ValueError("invalid model archive"),
            ),
            patch("anylabeling.services.auto_labeling.model_manager.logging.exception"),
        ):
            result = manager._load_model(0)

        self.assertIsNone(result)
        self.assertEqual(statuses, ["Error in loading model: invalid model archive"])
        manager.on_model_download_finished()
        self.assertEqual(completions, [{}])


if __name__ == "__main__":
    unittest.main()
