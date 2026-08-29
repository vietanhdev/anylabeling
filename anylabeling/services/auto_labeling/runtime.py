"""Runtime helpers shared by ONNX-backed auto-labeling models."""

import logging
import os

import onnxruntime

from anylabeling.app_info import __preferred_device__

CPU_PROVIDER = "CPUExecutionProvider"
TENSORRT_PROVIDER = "TensorrtExecutionProvider"
CUDA_PROVIDER = "CUDAExecutionProvider"
OPENVINO_PROVIDER = "OpenVINOExecutionProvider"
QNN_PROVIDER = "QNNExecutionProvider"
VITISAI_PROVIDER = "VitisAIExecutionProvider"
CANN_PROVIDER = "CANNExecutionProvider"

_PROVIDER_ALIASES = {
    "CUDA": CUDA_PROVIDER,
    "COREML": "CoreMLExecutionProvider",
    "DIRECTML": "DmlExecutionProvider",
    "DML": "DmlExecutionProvider",
    "ROCM": "ROCMExecutionProvider",
    "MIGRAPHX": "MIGraphXExecutionProvider",
    "OPENVINO": OPENVINO_PROVIDER,
    "OPENVINO_NPU": OPENVINO_PROVIDER,
    "INTEL_NPU": OPENVINO_PROVIDER,
    "TENSORRT": TENSORRT_PROVIDER,
    "CANN": CANN_PROVIDER,
    "ASCEND_NPU": CANN_PROVIDER,
    "QNN": QNN_PROVIDER,
    "QUALCOMM_NPU": QNN_PROVIDER,
    "SNAPDRAGON_NPU": QNN_PROVIDER,
    "VITISAI": VITISAI_PROVIDER,
    "AMD_NPU": VITISAI_PROVIDER,
    "RYZENAI": VITISAI_PROVIDER,
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
    OPENVINO_PROVIDER,
    CANN_PROVIDER,
    QNN_PROVIDER,
    VITISAI_PROVIDER,
    "WebGpuExecutionProvider",
    TENSORRT_PROVIDER,
]

_NPU_PRIORITY = [
    QNN_PROVIDER,
    OPENVINO_PROVIDER,
    VITISAI_PROVIDER,
    CANN_PROVIDER,
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

    if requested in {"GPU", "AUTO", "NPU"}:
        priority = _NPU_PRIORITY if requested == "NPU" else _ACCELERATOR_PRIORITY
        for provider in priority:
            if provider in available:
                return _with_cpu_fallback([provider], available)
        logging.warning(
            "No supported %s provider is available; falling back to CPU",
            requested,
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


def _get_requested_device(preferred_device=None):
    return os.environ.get(
        "ANYLABELING_DEVICE",
        preferred_device or __preferred_device__,
    )


def get_onnx_providers(preferred_device=None):
    """Return providers for the build preference or runtime environment override."""
    return select_onnx_providers(
        onnxruntime.get_available_providers(),
        _get_requested_device(preferred_device),
    )


def get_onnx_provider_options(providers, preferred_device=None):
    """Return hardware-specific options aligned with the provider list."""
    requested = _get_requested_device(preferred_device).strip().upper()
    options = [{} for _provider in providers]

    if requested in {"NPU", "OPENVINO_NPU", "INTEL_NPU"}:
        if OPENVINO_PROVIDER in providers:
            options[providers.index(OPENVINO_PROVIDER)] = {"device_type": "NPU"}

    if requested in {"NPU", "QNN", "QUALCOMM_NPU", "SNAPDRAGON_NPU"}:
        if QNN_PROVIDER in providers:
            options[providers.index(QNN_PROVIDER)] = {"backend_type": "htp"}

    return options


def create_inference_session(model_path, preferred_device=None, **kwargs):
    """Create an ONNX Runtime session with controlled accelerator fallback."""
    providers = get_onnx_providers(preferred_device)
    provider_options = get_onnx_provider_options(providers, preferred_device)
    if any(provider_options) and "provider_options" not in kwargs:
        kwargs["provider_options"] = provider_options
    if CUDA_PROVIDER in providers:
        preload_dlls = getattr(onnxruntime, "preload_dlls", None)
        if preload_dlls is not None:
            try:
                # Empty string means NVIDIA's pip-installed runtime packages.
                preload_dlls(directory="")
            except Exception as error:
                logging.warning("Could not preload CUDA runtime libraries: %s", error)
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
