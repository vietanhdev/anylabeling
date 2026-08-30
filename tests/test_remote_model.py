import os
import unittest
from unittest.mock import MagicMock, patch

from PyQt6.QtGui import QColor, QImage

from anylabeling.services.auto_labeling.remote_model import RemoteModel, _wire_prompts


class TestRemoteModel(unittest.TestCase):
    @patch("anylabeling.services.auto_labeling.remote_model.RemoteInferenceClient")
    def test_real_qimage_is_losslessly_encoded_and_shapes_are_editable(
        self, client_cls
    ):
        client = client_cls.return_value
        client.capabilities.promptable = False
        client.predict.return_value = {
            "shapes": [
                {
                    "type": "rectangle",
                    "points": [{"x": 1.5, "y": 2.5}, {"x": 8.5, "y": 7.5}],
                    "label": "cat",
                    "score": 0.75,
                    "group_id": 0,
                    "attributes": {"class_id": 15},
                }
            ]
        }
        config = {
            "type": "remote",
            "name": "shared-model",
            "display_name": "Shared model",
            "server_url": "https://inference.example.com",
            "model_id": "model-1",
            "password_env": "ANYLABELING_TEST_REMOTE_PASSWORD",
            "parameters": {"confidence": 0.5},
        }
        image = QImage(10, 10, QImage.Format.Format_RGB32)
        image.fill(QColor("red"))
        with patch.dict(
            os.environ,
            {"ANYLABELING_TEST_REMOTE_PASSWORD": "correct horse battery staple"},
        ):
            model = RemoteModel(config, MagicMock())
            result = model.predict_shapes(image)
        encoded, media_type = client.predict.call_args.args[:2]
        self.assertTrue(encoded.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertEqual(media_type, "image/png")
        self.assertEqual(
            client.predict.call_args.kwargs["parameters"], {"confidence": 0.5}
        )
        shape = result.shapes[0]
        self.assertEqual(
            (shape.shape_type, shape.label, shape.group_id), ("rectangle", "cat", 0)
        )
        self.assertEqual(
            shape.other_data, {"score": 0.75, "attributes": {"class_id": 15}}
        )

    def test_prompt_conversion_rejects_invalid_geometry(self):
        self.assertEqual(
            _wire_prompts(
                [
                    {"type": "point", "data": [1, 2], "label": 1},
                    {"type": "rectangle", "data": [3, 4, 8, 9]},
                ]
            ),
            [
                {"type": "point", "point": {"x": 1.0, "y": 2.0}, "foreground": True},
                {
                    "type": "box",
                    "top_left": {"x": 3.0, "y": 4.0},
                    "bottom_right": {"x": 8.0, "y": 9.0},
                },
            ],
        )
        with self.assertRaisesRegex(ValueError, "positive area"):
            _wire_prompts([{"type": "rectangle", "data": [3, 4, 3, 9]}])

    @patch("anylabeling.services.auto_labeling.remote_model.RemoteInferenceClient")
    def test_cancel_does_not_unload_client(self, client_cls):
        client = client_cls.return_value
        client.capabilities.promptable = False
        config = {
            "type": "remote",
            "name": "shared-model",
            "display_name": "Shared model",
            "server_url": "https://inference.example.com",
            "model_id": "model-1",
            "password_env": "ANYLABELING_TEST_REMOTE_PASSWORD",
        }
        with patch.dict(
            os.environ,
            {"ANYLABELING_TEST_REMOTE_PASSWORD": "correct horse battery staple"},
        ):
            model = RemoteModel(config, MagicMock())
        model.cancel_prediction()
        client.cancel.assert_called_once_with()
        self.assertIs(model.client, client)

    @patch("anylabeling.services.auto_labeling.remote_model.RemoteInferenceClient")
    def test_promptable_remote_model_defaults_to_polygon(self, client_cls):
        client_cls.return_value.capabilities.promptable = True
        config = {
            "type": "remote",
            "name": "shared-promptable",
            "display_name": "Shared promptable model",
            "server_url": "https://inference.example.com",
            "model_id": "model-1",
            "password_env": "ANYLABELING_TEST_REMOTE_PASSWORD",
        }
        with patch.dict(
            os.environ,
            {"ANYLABELING_TEST_REMOTE_PASSWORD": "correct horse battery staple"},
        ):
            model = RemoteModel(config, MagicMock())
        self.assertTrue(model.supports_interactive_prompts)
        self.assertEqual(model.output_mode, "polygon")
        self.assertIn("button_add_point", model.get_required_widgets())


if __name__ == "__main__":
    unittest.main()
