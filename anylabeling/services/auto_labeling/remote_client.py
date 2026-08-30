"""Bounded client for the public AnyLearning inference protocol."""

from __future__ import annotations

import base64
import hashlib
import ipaddress
import json
import math
import secrets
import ssl
import threading
import time
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit, urlunsplit
from urllib.request import (
    HTTPHandler,
    HTTPRedirectHandler,
    HTTPSHandler,
    ProxyHandler,
    Request,
    build_opener,
)

PROTOCOL_VERSION = "1.0"
_REQUEST_HEADER = "X-AnyLearning-Request"
_MAX_METADATA_BYTES = 8 * 1024
_MAX_JSON_RESPONSE_BYTES = 9 * 1024**2
_MAX_IMAGE_BYTES = 32 * 1024**2
_MAX_SHAPES = 10_000
_MAX_POINTS = 100_000
_STATES = frozenset(
    {"queued", "running", "succeeded", "failed", "cancelled", "timed_out"}
)


class RemoteInferenceError(RuntimeError):
    """A public, credential-free remote inference failure."""


class _NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, request, file_pointer, code, message, headers, url):
        del request, file_pointer, code, message, headers, url
        return None


@dataclass(frozen=True)
class RemoteModelCapabilities:
    model_id: str
    model_revision: str
    tasks: tuple[str, ...]
    metadata: dict[str, Any]

    @property
    def promptable(self) -> bool:
        return "promptable_segmentation" in self.tasks


class RemoteInferenceClient:
    """Authenticate, discover one model, and run token-owned prediction jobs."""

    def __init__(
        self,
        server_url: str,
        model_id: str,
        password: str,
        *,
        prediction_timeout_seconds: float = 120,
        poll_interval_seconds: float = 0.1,
        max_image_bytes: int = _MAX_IMAGE_BYTES,
        max_response_bytes: int = _MAX_JSON_RESPONSE_BYTES,
    ) -> None:
        self._server_url = _validate_server_url(server_url)
        if not isinstance(model_id, str) or not 1 <= len(model_id) <= 512:
            raise ValueError("remote model_id must contain 1 to 512 characters")
        if not isinstance(password, str) or not 12 <= len(password.encode()) <= 1_024:
            raise ValueError("remote password must contain 12 to 1024 UTF-8 bytes")
        if not 1 <= prediction_timeout_seconds <= 3_600:
            raise ValueError("prediction timeout must be between 1 and 3600 seconds")
        if not 0.02 <= poll_interval_seconds <= 2:
            raise ValueError("poll interval must be between 0.02 and 2 seconds")
        if not 1_024 <= max_image_bytes <= 512 * 1024**2:
            raise ValueError("remote image byte limit is invalid")
        if not 1_024 <= max_response_bytes <= 256 * 1024**2:
            raise ValueError("remote response byte limit is invalid")

        context = ssl.create_default_context()
        self._opener = build_opener(
            ProxyHandler({}),
            HTTPHandler(),
            HTTPSHandler(context=context),
            _NoRedirects(),
        )
        self._model_id = model_id
        self._password: str | None = password
        self._prediction_timeout = float(prediction_timeout_seconds)
        self._poll_interval = float(poll_interval_seconds)
        self._max_image_bytes = max_image_bytes
        self._max_response_bytes = max_response_bytes
        self._token: str | None = None
        self._token_expires_at = 0.0
        self._cancelled = threading.Event()
        self._prediction_lock = threading.Lock()
        self._authenticate()
        self.capabilities = self._discover_model()

    def close(self) -> None:
        self._cancelled.set()
        self._token = None
        self._password = None
        self._token_expires_at = 0.0

    def cancel(self) -> None:
        self._cancelled.set()

    def predict(
        self,
        encoded_image: bytes,
        media_type: str,
        *,
        prompts: list[dict[str, Any]] | None = None,
        output_shape: str | None = None,
        parameters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not isinstance(encoded_image, bytes) or not encoded_image:
            raise ValueError("remote image must be non-empty bytes")
        if len(encoded_image) > self._max_image_bytes:
            raise RemoteInferenceError("Encoded image exceeds the client limit")
        if media_type not in {"image/jpeg", "image/png", "image/webp"}:
            raise ValueError("remote image media type is unsupported")
        if not self._prediction_lock.acquire(blocking=False):
            raise RemoteInferenceError("A remote prediction is already running")

        job_id: str | None = None
        try:
            self._cancelled.clear()
            self._ensure_token_lifetime()
            request_id = secrets.token_hex(16)
            source_id = f"content-sha256:{hashlib.sha256(encoded_image).hexdigest()}"
            request_payload: dict[str, Any] = {
                "protocol_version": PROTOCOL_VERSION,
                "request_id": request_id,
                "source_id": source_id,
                "model_id": self.capabilities.model_id,
                "model_revision": self.capabilities.model_revision,
                "prompts": prompts or [],
                "parameters": parameters or {},
            }
            if output_shape is not None:
                request_payload["output_shape"] = output_shape
            metadata = json.dumps(
                request_payload,
                ensure_ascii=True,
                allow_nan=False,
                separators=(",", ":"),
            ).encode("ascii")
            if len(metadata) > _MAX_METADATA_BYTES:
                raise RemoteInferenceError("Inference request metadata is too large")
            encoded_metadata = base64.urlsafe_b64encode(metadata).rstrip(b"=")
            submitted = self._json_request(
                "POST",
                "/v1/predictions",
                body=encoded_image,
                headers={
                    "Authorization": f"Bearer {self._required_token()}",
                    "Content-Type": media_type,
                    _REQUEST_HEADER: encoded_metadata.decode("ascii"),
                },
                expected_status=202,
            )
            job_id = _bounded_text(submitted.get("job_id"), "job_id", 512)
            state = self._validate_job(submitted, job_id, request_id)
            if state == "succeeded":
                result = submitted.get("result")
            elif state in {"queued", "running"}:
                result = self._poll(job_id, request_id)
            else:
                raise RemoteInferenceError(
                    _bounded_text(submitted.get("error"), "prediction error", 2_048)
                    if isinstance(submitted.get("error"), str)
                    else "Remote prediction failed"
                )
            return _validate_result(
                result,
                request_id=request_id,
                source_id=source_id,
                capabilities=self.capabilities,
            )
        finally:
            if job_id is not None:
                self._delete_job(job_id)
            self._prediction_lock.release()

    def _poll(self, job_id: str, request_id: str) -> Any:
        deadline = time.monotonic() + self._prediction_timeout
        while True:
            if self._cancelled.is_set():
                raise RemoteInferenceError("Remote prediction was cancelled")
            if time.monotonic() >= deadline:
                raise RemoteInferenceError("Remote prediction exceeded its deadline")
            snapshot = self._json_request(
                "GET",
                f"/v1/predictions/{quote(job_id, safe='')}",
                headers={"Authorization": f"Bearer {self._required_token()}"},
                expected_status=200,
            )
            state = self._validate_job(snapshot, job_id, request_id)
            if state == "succeeded":
                return snapshot.get("result")
            if state not in {"queued", "running"}:
                raise RemoteInferenceError(
                    _bounded_text(snapshot.get("error"), "prediction error", 2_048)
                    if isinstance(snapshot.get("error"), str)
                    else "Remote prediction failed"
                )
            self._cancelled.wait(self._poll_interval)

    def _validate_job(self, value: Any, job_id: str, request_id: str) -> str:
        if not isinstance(value, dict):
            raise RemoteInferenceError("Server returned an invalid prediction job")
        if value.get("job_id") != job_id or value.get("request_id") != request_id:
            raise RemoteInferenceError("Server prediction identity did not match")
        state = value.get("state")
        if state not in _STATES:
            raise RemoteInferenceError("Server returned an invalid prediction state")
        return state

    def _authenticate(self) -> None:
        if self._password is None:
            raise RemoteInferenceError("Remote inference client is closed")
        payload = json.dumps(
            {"password": self._password}, ensure_ascii=True, separators=(",", ":")
        ).encode("utf-8")
        response = self._json_request(
            "POST",
            "/v1/auth/token",
            body=payload,
            headers={"Content-Type": "application/json"},
            expected_status=200,
            maximum=16 * 1024,
        )
        token = _bounded_text(response.get("access_token"), "access token", 4_096)
        if len(token.encode("ascii", errors="ignore")) != len(token) or not token:
            raise RemoteInferenceError("Server returned an invalid access token")
        if response.get("token_type") != "bearer":
            raise RemoteInferenceError("Server returned an invalid token type")
        expires_in = response.get("expires_in")
        if type(expires_in) is not int or not 30 <= expires_in <= 3_600:
            raise RemoteInferenceError("Server returned an invalid token lifetime")
        self._token = token
        self._token_expires_at = time.monotonic() + expires_in

    def _ensure_token_lifetime(self) -> None:
        required = self._prediction_timeout + 5
        if self._token is None or self._token_expires_at - time.monotonic() < required:
            self._authenticate()
        if self._token_expires_at - time.monotonic() < required:
            raise RemoteInferenceError(
                "Server token lifetime is shorter than the prediction timeout"
            )

    def _discover_model(self) -> RemoteModelCapabilities:
        response = self._json_request(
            "GET",
            f"/v1/models/{quote(self._model_id, safe='')}",
            headers={"Authorization": f"Bearer {self._required_token()}"},
            expected_status=200,
            maximum=256 * 1024,
        )
        model_id = _bounded_text(response.get("model_id"), "model_id", 512)
        revision = _bounded_text(response.get("model_revision"), "model_revision", 512)
        if (
            model_id != self._model_id
            or response.get("protocol_version") != PROTOCOL_VERSION
        ):
            raise RemoteInferenceError("Server model identity is incompatible")
        raw_tasks = response.get("tasks")
        if (
            not isinstance(raw_tasks, list)
            or not 1 <= len(raw_tasks) <= 16
            or any(
                not isinstance(item, str) or not 1 <= len(item) <= 128
                for item in raw_tasks
            )
            or len(raw_tasks) != len(set(raw_tasks))
        ):
            raise RemoteInferenceError("Server returned invalid model tasks")
        metadata = response.get("metadata", {})
        if not isinstance(metadata, dict) or len(metadata) > 128:
            raise RemoteInferenceError("Server returned invalid model metadata")
        _validate_metadata(metadata, "model metadata")
        return RemoteModelCapabilities(model_id, revision, tuple(raw_tasks), metadata)

    def _delete_job(self, job_id: str) -> None:
        if self._token is None:
            return
        try:
            self._raw_request(
                "DELETE",
                f"/v1/predictions/{quote(job_id, safe='')}",
                headers={"Authorization": f"Bearer {self._token}"},
                expected_status=204,
                maximum=1_024,
            )
        except RemoteInferenceError:
            pass

    def _required_token(self) -> str:
        if self._token is None:
            raise RemoteInferenceError("Remote inference client is not authenticated")
        return self._token

    def _json_request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        payload = self._raw_request(method, path, **kwargs)
        try:
            value = json.loads(
                payload,
                parse_constant=_reject_json_constant,
                object_pairs_hook=_unique_json_object,
            )
        except (UnicodeError, json.JSONDecodeError, ValueError) as error:
            raise RemoteInferenceError("Server returned invalid JSON") from error
        if not isinstance(value, dict):
            raise RemoteInferenceError("Server returned an invalid JSON object")
        return value

    def _raw_request(
        self,
        method: str,
        path: str,
        *,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
        expected_status: int,
        maximum: int | None = None,
    ) -> bytes:
        if not path.startswith("/") or "?" in path or "#" in path:
            raise ValueError("remote API path is invalid")
        request = Request(
            self._server_url + path,
            data=body,
            headers={"Accept": "application/json", **(headers or {})},
            method=method,
        )
        try:
            response = self._opener.open(
                request, timeout=min(30.0, max(1.0, self._prediction_timeout))
            )
        except HTTPError as error:
            status = error.code
            error.close()
            if status == 401:
                raise RemoteInferenceError("Remote authentication failed") from error
            if status == 404:
                raise RemoteInferenceError(
                    "Remote model or prediction was not found"
                ) from error
            if status == 429:
                raise RemoteInferenceError(
                    "Remote inference capacity was reached"
                ) from error
            raise RemoteInferenceError(
                f"Remote server rejected the request (HTTP {status})"
            ) from error
        except (OSError, TimeoutError, URLError) as error:
            raise RemoteInferenceError(
                "Could not reach the remote inference server"
            ) from error

        with response:
            if response.status != expected_status:
                raise RemoteInferenceError(
                    f"Remote server returned HTTP {response.status}"
                )
            limit = self._max_response_bytes if maximum is None else maximum
            lengths = response.headers.get_all("Content-Length", failobj=[])
            if len(lengths) > 1:
                raise RemoteInferenceError("Server returned invalid response framing")
            if lengths:
                declared = lengths[0]
                if not declared.isascii() or not declared.isdecimal():
                    raise RemoteInferenceError(
                        "Server returned invalid response framing"
                    )
                if int(declared) > limit:
                    raise RemoteInferenceError(
                        "Server response exceeds the client limit"
                    )
            payload = response.read(limit + 1)
            if len(payload) > limit:
                raise RemoteInferenceError("Server response exceeds the client limit")
            if (
                expected_status != 204
                and response.headers.get_content_type() != "application/json"
            ):
                raise RemoteInferenceError("Server response is not JSON")
            return payload


def _validate_server_url(value: str) -> str:
    if not isinstance(value, str) or not 1 <= len(value) <= 2_048:
        raise ValueError("remote server URL is invalid")
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("remote server URL must be an exact HTTP(S) origin")
    try:
        port = parsed.port
    except ValueError as error:
        raise ValueError("remote server URL port is invalid") from error
    if parsed.scheme == "http" and not _is_loopback(parsed.hostname):
        raise ValueError("non-loopback remote inference requires HTTPS")
    host = f"[{parsed.hostname}]" if ":" in parsed.hostname else parsed.hostname
    authority = f"{host}:{port}" if port is not None else host
    return urlunsplit((parsed.scheme, authority, "", "", ""))


def _is_loopback(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _bounded_text(value: Any, name: str, maximum: int) -> str:
    if not isinstance(value, str) or not 1 <= len(value) <= maximum:
        raise RemoteInferenceError(f"Server returned an invalid {name}")
    return value


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant is not accepted: {value}")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Duplicate JSON object keys are not accepted")
        value[key] = item
    return value


def _validate_result(
    value: Any,
    *,
    request_id: str,
    source_id: str,
    capabilities: RemoteModelCapabilities,
) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("protocol_version") != PROTOCOL_VERSION:
        raise RemoteInferenceError("Server returned an invalid inference result")
    expected = {
        "request_id": request_id,
        "source_id": source_id,
        "model_id": capabilities.model_id,
        "model_revision": capabilities.model_revision,
    }
    if any(
        value.get(field) != expected_value for field, expected_value in expected.items()
    ):
        raise RemoteInferenceError("Server inference result identity did not match")
    shapes = value.get("shapes")
    if not isinstance(shapes, list) or len(shapes) > _MAX_SHAPES:
        raise RemoteInferenceError("Server returned an invalid shape list")
    for shape in shapes:
        _validate_shape(shape)
    warnings = value.get("warnings", [])
    if (
        not isinstance(warnings, list)
        or len(warnings) > 128
        or any(not isinstance(item, str) or len(item) > 2_048 for item in warnings)
    ):
        raise RemoteInferenceError("Server returned invalid inference warnings")
    timings = value.get("timings_ms", {})
    if not isinstance(timings, dict) or len(timings) > 64:
        raise RemoteInferenceError("Server returned invalid inference timings")
    for key, timing in timings.items():
        if (
            not isinstance(key, str)
            or not 1 <= len(key) <= 128
            or not isinstance(timing, (int, float))
            or isinstance(timing, bool)
            or not math.isfinite(timing)
            or timing < 0
        ):
            raise RemoteInferenceError("Server returned invalid inference timings")
    return value


def _validate_shape(value: Any) -> None:
    if not isinstance(value, dict):
        raise RemoteInferenceError("Server returned an invalid shape")
    if not {"type", "points"} <= set(value) or not set(value) <= {
        "type",
        "points",
        "label",
        "score",
        "group_id",
        "attributes",
    }:
        raise RemoteInferenceError("Server returned an invalid shape")
    expected_points = {
        "point": (1, 1),
        "rectangle": (2, 2),
        "polygon": (3, _MAX_POINTS),
        "rotated_rectangle": (4, 4),
    }.get(value.get("type"))
    points = value.get("points")
    if expected_points is None or not isinstance(points, list):
        raise RemoteInferenceError("Server returned an unsupported shape")
    if not expected_points[0] <= len(points) <= expected_points[1]:
        raise RemoteInferenceError("Server returned invalid shape points")
    for point in points:
        if not isinstance(point, dict) or set(point) != {"x", "y"}:
            raise RemoteInferenceError("Server returned invalid point coordinates")
        if any(
            not isinstance(point[axis], (int, float))
            or isinstance(point[axis], bool)
            or not math.isfinite(point[axis])
            or abs(point[axis]) > 1_000_000_000
            for axis in ("x", "y")
        ):
            raise RemoteInferenceError("Server returned invalid point coordinates")
    label = value.get("label")
    if label is not None and (not isinstance(label, str) or len(label) > 1_024):
        raise RemoteInferenceError("Server returned an invalid shape label")
    score = value.get("score")
    if score is not None and (
        not isinstance(score, (int, float))
        or isinstance(score, bool)
        or not math.isfinite(score)
        or not 0 <= score <= 1
    ):
        raise RemoteInferenceError("Server returned an invalid shape score")
    group_id = value.get("group_id")
    if group_id is not None and not (
        (isinstance(group_id, str) and len(group_id) <= 2_048)
        or (
            isinstance(group_id, int)
            and not isinstance(group_id, bool)
            and -(2**63) <= group_id <= 2**63 - 1
        )
    ):
        raise RemoteInferenceError("Server returned an invalid shape group_id")
    attributes = value.get("attributes", {})
    if not isinstance(attributes, dict) or len(attributes) > 128:
        raise RemoteInferenceError("Server returned invalid shape attributes")
    _validate_metadata(attributes, "shape attributes")


def _validate_metadata(value: dict[Any, Any], name: str) -> None:
    for key, item in value.items():
        if not isinstance(key, str) or not 1 <= len(key) <= 128:
            raise RemoteInferenceError(f"Server returned invalid {name}")
        if item is None or isinstance(item, (str, bool)):
            if isinstance(item, str) and len(item) > 2_048:
                raise RemoteInferenceError(f"Server returned invalid {name}")
            continue
        if isinstance(item, int):
            if not -(2**63) <= item <= 2**63 - 1:
                raise RemoteInferenceError(f"Server returned invalid {name}")
            continue
        if isinstance(item, float) and math.isfinite(item):
            continue
        raise RemoteInferenceError(f"Server returned invalid {name}")
