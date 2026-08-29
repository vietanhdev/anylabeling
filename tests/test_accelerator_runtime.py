"""Tests for accelerator selection and ONNX Runtime DNN execution."""

import os
import unittest
from unittest import mock

import numpy as np

from anylabeling.services.auto_labeling.runtime import (
    OnnxRuntimeModel,
    get_onnx_providers,
    select_onnx_providers,
)


class TestSelectOnnxProviders(unittest.TestCase):
    def test_cpu_build_does_not_enable_available_accelerators(self):
        providers = select_onnx_providers(
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
            "CPU",
        )

        self.assertEqual(providers, ["CPUExecutionProvider"])

    def test_gpu_prefers_cuda_over_optional_tensorrt(self):
        providers = select_onnx_providers(
            [
                "TensorrtExecutionProvider",
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
            "GPU",
        )

        self.assertEqual(
            providers,
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

    def test_gpu_uses_coreml_on_apple_silicon(self):
        providers = select_onnx_providers(
            ["CoreMLExecutionProvider", "CPUExecutionProvider"],
            "GPU",
        )

        self.assertEqual(
            providers,
            ["CoreMLExecutionProvider", "CPUExecutionProvider"],
        )

    def test_explicit_directml_alias_is_supported(self):
        providers = select_onnx_providers(
            ["DmlExecutionProvider", "CPUExecutionProvider"],
            "DIRECTML",
        )

        self.assertEqual(
            providers,
            ["DmlExecutionProvider", "CPUExecutionProvider"],
        )

    def test_explicit_tensorrt_keeps_cuda_fallback(self):
        providers = select_onnx_providers(
            [
                "TensorrtExecutionProvider",
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
            "TENSORRT",
        )

        self.assertEqual(
            providers,
            [
                "TensorrtExecutionProvider",
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
        )

    def test_unavailable_requested_accelerator_falls_back_to_cpu(self):
        providers = select_onnx_providers(
            ["CPUExecutionProvider"],
            "CUDA",
        )

        self.assertEqual(providers, ["CPUExecutionProvider"])

    @mock.patch(
        "anylabeling.services.auto_labeling.runtime.onnxruntime.get_available_providers"
    )
    def test_environment_overrides_build_preference(self, get_available_providers):
        get_available_providers.return_value = [
            "CoreMLExecutionProvider",
            "CPUExecutionProvider",
        ]

        with mock.patch.dict(os.environ, {"ANYLABELING_DEVICE": "coreml"}):
            providers = get_onnx_providers(preferred_device="CPU")

        self.assertEqual(
            providers,
            ["CoreMLExecutionProvider", "CPUExecutionProvider"],
        )


class TestOnnxRuntimeModel(unittest.TestCase):
    @mock.patch("anylabeling.services.auto_labeling.runtime.create_inference_session")
    def test_matches_opencv_single_output_interface(self, create_session):
        session = create_session.return_value
        session.get_inputs.return_value = [mock.Mock(name="images")]
        session.get_inputs.return_value[0].name = "images"
        session.get_outputs.return_value = [mock.Mock(name="detections")]
        session.get_outputs.return_value[0].name = "detections"
        expected = np.ones((1, 6, 10), dtype=np.float32)
        session.run.return_value = [expected]
        model = OnnxRuntimeModel("model.onnx")
        blob = np.zeros((1, 3, 640, 640), dtype=np.float32)

        model.setInput(blob)
        output = model.forward()

        session.run.assert_called_once_with(None, {"images": blob})
        np.testing.assert_array_equal(output, expected)

    @mock.patch("anylabeling.services.auto_labeling.runtime.create_inference_session")
    def test_returns_named_output_list_for_yolov5(self, create_session):
        session = create_session.return_value
        session.get_inputs.return_value = [mock.Mock(name="images")]
        session.get_inputs.return_value[0].name = "images"
        session.get_outputs.return_value = [mock.Mock(name="output")]
        session.get_outputs.return_value[0].name = "output"
        expected = [np.ones((1, 10, 85), dtype=np.float32)]
        session.run.return_value = expected
        model = OnnxRuntimeModel("model.onnx")
        blob = np.zeros((1, 3, 640, 640), dtype=np.float32)

        model.setInput(blob)
        output_names = model.getUnconnectedOutLayersNames()
        output = model.forward(output_names)

        self.assertEqual(output_names, ["output"])
        session.run.assert_called_once_with(output_names, {"images": blob})
        self.assertIs(output, expected)


if __name__ == "__main__":
    unittest.main()
