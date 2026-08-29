"""Runtime helpers shared by ONNX-backed auto-labeling models."""

import logging
import os

import onnxruntime

from anylabeling.app_info import __preferred_device__

CPU_PROVIDER = "CPUExecutionProvider"
TENSORRT_PROVIDER = "TensorrtExecutionProvider"
CUDA_PROVIDER = "CUDAExecutionProvider"

_PROVIDER_ALIASES = {
    "CUDA": CUDA_PROVIDER,
    "COREML": "CoreMLExecutionProvider",
    "DIRECTML": "DmlExecutionProvider",
    "DML": "DmlExecutionProvider",
    "ROCM": "ROCMExecutionProvider",
    "MIGRAPHX": "MIGraphXExecutionProvider",
    "OPENVINO": "OpenVINOExecutionProvider",
    "TENSORRT": TENSORRT_PROVIDER,
    "CANN": "CANNExecutionProvider",
    "QNN": "QNNExecutionProvider",
    "VITISAI": "VitisAIExecutionProvider",
    "WEBGPU": "WebGpuExecutionProvider",
}

# Prefer broadly available providers that do not need an additional runtime.
# TensorRT is deliberately after CUDA because the ONNX Runtime GPU wheel can
# advertise TensorRT even when the TensorRT shared libraries are not installed.
_ACCELERATOR_PRIORITY = [
    CUDA_PROVIDER,
    "CoreMLExecutionProvider",
    "DmlExecutionProvider",
    "ROCMExecutionProvider",
    "MIGraphXExecutionProvider",
    "OpenVINOExecutionProvider",
    "CANNExecutionProvider",
    "QNNExecutionProvider",
    "VitisAIExecutionProvider",
    "WebGpuExecutionProvider",
    TENSORRT_PROVIDER,
]


def _with_cpu_fallback(providers, available_providers):
    selected = list(providers)
    if CPU_PROVIDER in available_providers and CPU_PROVIDER not in selected:
        selected.append(CPU_PROVIDER)
    return selected


def select_onnx_providers(available_providers, preferred_device):
    """Select an accelerator deterministically, followed by CPU fallback."""
    available = list(available_providers)
    if not available:
        return []

    requested = (preferred_device or "CPU").strip().upper()
    if requested == "CPU":
        if CPU_PROVIDER in available:
            return [CPU_PROVIDER]
        return available

    if requested in {"GPU", "AUTO"}:
        for provider in _ACCELERATOR_PRIORITY:
            if provider in available:
                return _with_cpu_fallback([provider], available)
        logging.warning(
            "No supported accelerator provider is available; falling back to CPU"
        )
        return [CPU_PROVIDER] if CPU_PROVIDER in available else available

    provider = _PROVIDER_ALIASES.get(requested)
    if provider is None:
        provider = next(
            (candidate for candidate in available if candidate.upper() == requested),
            None,
        )

    if provider not in available:
        logging.warning(
            "Requested ONNX Runtime provider %s is unavailable; falling back to CPU",
            preferred_device,
        )
        return [CPU_PROVIDER] if CPU_PROVIDER in available else available

    selected = [provider]
    if provider == TENSORRT_PROVIDER and CUDA_PROVIDER in available:
        selected.append(CUDA_PROVIDER)
    return _with_cpu_fallback(selected, available)


def get_onnx_providers(preferred_device=None):
    """Return providers for the build preference or runtime environment override."""
    device = os.environ.get(
        "ANYLABELING_DEVICE",
        preferred_device or __preferred_device__,
    )
    return select_onnx_providers(
        onnxruntime.get_available_providers(),
        device,
    )


def create_inference_session(model_path, preferred_device=None, **kwargs):
    """Create an ONNX Runtime session with controlled accelerator fallback."""
    providers = get_onnx_providers(preferred_device)
    logging.info("Loading %s with ONNX providers %s", model_path, providers)
    return onnxruntime.InferenceSession(model_path, providers=providers, **kwargs)


class OnnxRuntimeModel:
    """Small OpenCV-DNN-compatible adapter backed by ONNX Runtime."""

    def __init__(self, model_path):
        self.session = create_inference_session(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        self.input_blob = None

    def setInput(self, blob):
        self.input_blob = blob

    def getUnconnectedOutLayersNames(self):
        return self.output_names

    def forward(self, output_names=None):
        if self.input_blob is None:
            raise RuntimeError("Model input has not been set")
        outputs = self.session.run(
            output_names,
            {self.input_name: self.input_blob},
        )
        if output_names is None and len(outputs) == 1:
            return outputs[0]
        return outputs
