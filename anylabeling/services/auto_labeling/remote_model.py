"""AnyLabeling adapter for authenticated AnyLearning inference servers."""

from __future__ import annotations

import math
import os
import re
from typing import Any

from PyQt6 import QtCore
from PyQt6.QtCore import QBuffer, QCoreApplication, QIODevice

from anylabeling.views.labeling.shape import Shape

from .model import Model
from .registry import ModelRegistry
from .remote_client import RemoteInferenceClient, RemoteInferenceError
from .types import AutoLabelingResult

_ENVIRONMENT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")


@ModelRegistry.register("remote")
class RemoteModel(Model):
    """Run auto-labeling through a preconfigured remote ONNX model."""

    class Meta:
        required_config_names = [
            "type",
            "name",
            "display_name",
            "server_url",
            "model_id",
            "password_env",
        ]
        widgets = ["button_run"]
        output_modes = {
            "polygon": QCoreApplication.translate("Model", "Polygon"),
            "rectangle": QCoreApplication.translate("Model", "Rectangle"),
        }
        default_output_mode = "rectangle"

    _PROMPT_WIDGETS = [
        "output_label",
        "output_select_combobox",
        "button_run",
        "button_add_point",
        "button_remove_point",
        "button_add_rect",
        "button_clear",
        "button_finish_object",
    ]

    def __init__(self, model_config, on_message) -> None:
        super().__init__(model_config, on_message)
        environment_name = self.config["password_env"]
        if not isinstance(environment_name, str) or not _ENVIRONMENT_NAME.fullmatch(
            environment_name
        ):
            raise ValueError("Remote password_env is not a valid environment name")
        password = os.environ.get(environment_name)
        if password is None:
            raise ValueError(
                "Remote inference password environment variable is not set: "
                f"{environment_name}"
            )
        parameters = _wire_parameters(self.config.get("parameters", {}))
        self.client: RemoteInferenceClient | None = RemoteInferenceClient(
            self.config["server_url"],
            self.config["model_id"],
            password,
            prediction_timeout_seconds=self.config.get(
                "prediction_timeout_seconds", 120
            ),
            poll_interval_seconds=self.config.get("poll_interval_seconds", 0.1),
            max_image_bytes=self.config.get("max_image_bytes", 32 * 1024**2),
        )
        self.supports_interactive_prompts = self.client.capabilities.promptable
        self.marks: list[dict[str, Any]] = []
        self.parameters = parameters
        self.output_mode = (
            "polygon" if self.supports_interactive_prompts else "rectangle"
        )

    def get_required_widgets(self):
        return list(
            self._PROMPT_WIDGETS
            if self.supports_interactive_prompts
            else self.Meta.widgets
        )

    def set_auto_labeling_marks(self, marks):
        if not isinstance(marks, list) or len(marks) > 10_000:
            raise ValueError("Remote prompt list is invalid")
        self.marks = marks

    def predict_shapes(self, image, filename=None) -> AutoLabelingResult:
        del filename
        if image is None or image.isNull():
            return AutoLabelingResult([], replace=True)
        client = self.client
        if client is None:
            raise RemoteInferenceError("Remote inference model is unloaded")
        result = client.predict(
            _encode_png(image),
            "image/png",
            prompts=_wire_prompts(self.marks)
            if self.supports_interactive_prompts
            else [],
            output_shape=self.output_mode
            if self.supports_interactive_prompts
            else None,
            parameters=self.parameters,
        )
        return AutoLabelingResult(
            [_labeling_shape(value) for value in result["shapes"]], replace=True
        )

    def cancel_prediction(self) -> None:
        if self.client is not None:
            self.client.cancel()

    def unload(self) -> None:
        if self.client is not None:
            self.client.close()
            self.client = None


def _encode_png(image) -> bytes:
    buffer = QBuffer()
    if not buffer.open(QIODevice.OpenModeFlag.WriteOnly):
        raise RemoteInferenceError("Could not allocate the remote image buffer")
    try:
        if not image.save(buffer, "PNG"):
            raise RemoteInferenceError("Could not encode the image for inference")
        encoded = bytes(buffer.data())
    finally:
        buffer.close()
    if not encoded:
        raise RemoteInferenceError("Could not encode the image for inference")
    return encoded


def _wire_prompts(marks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prompts = []
    for index, mark in enumerate(marks):
        if not isinstance(mark, dict):
            raise ValueError(f"Remote prompt {index} must be an object")
        kind, data = mark.get("type"), mark.get("data")
        if kind == "point":
            if not _finite_coordinates(data, 2):
                raise ValueError(f"Remote point prompt {index} is invalid")
            label = mark.get("label")
            if label not in (0, 1):
                raise ValueError(f"Remote point prompt {index} label must be 0 or 1")
            prompts.append(
                {
                    "type": "point",
                    "point": {"x": float(data[0]), "y": float(data[1])},
                    "foreground": label == 1,
                }
            )
        elif kind == "rectangle":
            if not _finite_coordinates(data, 4):
                raise ValueError(f"Remote rectangle prompt {index} is invalid")
            x1, y1, x2, y2 = (float(item) for item in data)
            if x2 <= x1 or y2 <= y1:
                raise ValueError(
                    f"Remote rectangle prompt {index} must have positive area"
                )
            prompts.append(
                {
                    "type": "box",
                    "top_left": {"x": x1, "y": y1},
                    "bottom_right": {"x": x2, "y": y2},
                }
            )
        else:
            raise ValueError(f"Remote prompt {index} has an unsupported type")
    return prompts


def _finite_coordinates(value: Any, length: int) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == length
        and all(
            isinstance(item, (int, float))
            and not isinstance(item, bool)
            and math.isfinite(item)
            for item in value
        )
    )


def _wire_parameters(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or len(value) > 128:
        raise ValueError("Remote inference parameters must be a bounded mapping")
    parameters = {}
    for key, item in value.items():
        if not isinstance(key, str) or not 1 <= len(key) <= 128:
            raise ValueError("Remote inference parameter names are invalid")
        if item is None or isinstance(item, (str, bool)):
            if isinstance(item, str) and len(item) > 2_048:
                raise ValueError(f"Remote inference parameter {key!r} is too long")
            parameters[key] = item
        elif isinstance(item, int):
            if not -(2**63) <= item <= 2**63 - 1:
                raise ValueError(f"Remote inference parameter {key!r} is invalid")
            parameters[key] = item
        elif isinstance(item, float) and math.isfinite(item):
            parameters[key] = item
        elif (
            isinstance(item, (list, tuple))
            and len(item) <= 256
            and (
                all(isinstance(member, str) and len(member) <= 2_048 for member in item)
                or all(
                    isinstance(member, int)
                    and not isinstance(member, bool)
                    and -(2**63) <= member <= 2**63 - 1
                    for member in item
                )
            )
        ):
            parameters[key] = list(item)
        else:
            raise ValueError(f"Remote inference parameter {key!r} is invalid")
    return parameters


def _labeling_shape(value: dict[str, Any]) -> Shape:
    shape_type = "polygon" if value["type"] == "rotated_rectangle" else value["type"]
    shape = Shape(
        label=value.get("label") or "AUTOLABEL_OBJECT",
        shape_type=shape_type,
        flags={},
        group_id=value.get("group_id"),
    )
    for point in value["points"]:
        shape.add_point(QtCore.QPointF(point["x"], point["y"]))
    if shape_type == "polygon":
        shape.close()
    if value.get("score") is not None:
        shape.other_data["score"] = value["score"]
    if value.get("attributes"):
        shape.other_data["attributes"] = value["attributes"]
    return shape


__all__ = ["RemoteModel"]
