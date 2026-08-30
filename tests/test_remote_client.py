import base64
import hashlib
import json
import threading
import unittest
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from anylabeling.services.auto_labeling.remote_client import (
    RemoteInferenceClient,
    RemoteInferenceError,
)


class _State:
    def __init__(
        self, *, redirect_auth=False, invalid_attributes=False, hold_running=False
    ):
        self.redirect_auth = redirect_auth
        self.invalid_attributes = invalid_attributes
        self.hold_running = hold_running
        self.deleted = threading.Event()
        self.polled = threading.Event()
        self.request = None
        self.image = None
        self.authorization = []


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, _format, *args):
        del args

    @property
    def state(self):
        return self.server.protocol_state

    def _json(self, status, value):
        body = json.dumps(value, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self):
        value = self.headers.get("Authorization")
        self.state.authorization.append(value)
        return value == "Bearer test-token"

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        if self.path == "/v1/auth/token":
            if self.state.redirect_auth:
                self.send_response(307)
                self.send_header("Location", "/redirected-auth")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            if json.loads(body) != {"password": "correct horse battery staple"}:
                self._json(401, {"detail": "denied"})
                return
            self._json(
                200,
                {
                    "access_token": "test-token",
                    "token_type": "bearer",
                    "expires_in": 300,
                },
            )
            return
        if self.path == "/v1/predictions" and self._authorized():
            encoded = self.headers["X-AnyLearning-Request"]
            encoded += "=" * (-len(encoded) % 4)
            self.state.request = json.loads(base64.urlsafe_b64decode(encoded))
            self.state.image = body
            self._json(
                202,
                {
                    "job_id": "job-1",
                    "request_id": self.state.request["request_id"],
                    "state": "queued",
                },
            )
            return
        self._json(404, {"detail": "not found"})

    def do_GET(self):
        if self.path == "/v1/models/model-1" and self._authorized():
            self._json(
                200,
                {
                    "protocol_version": "1.0",
                    "model_id": "model-1",
                    "model_revision": "sha256:revision",
                    "tasks": ["detection"],
                    "supports_batch": False,
                    "supports_cancellation": True,
                    "max_batch_size": 1,
                    "metadata": {"backend": "yolo_onnx"},
                },
            )
            return
        if self.path == "/v1/predictions/job-1" and self._authorized():
            request = self.state.request
            if self.state.hold_running:
                self.state.polled.set()
                self._json(
                    200,
                    {
                        "job_id": "job-1",
                        "request_id": request["request_id"],
                        "state": "running",
                    },
                )
                return
            attributes = (
                {"nested": ["rejected"]}
                if self.state.invalid_attributes
                else {"class_id": 16}
            )
            self._json(
                200,
                {
                    "job_id": "job-1",
                    "request_id": request["request_id"],
                    "state": "succeeded",
                    "result": {
                        "protocol_version": "1.0",
                        "request_id": request["request_id"],
                        "source_id": request["source_id"],
                        "model_id": "model-1",
                        "model_revision": "sha256:revision",
                        "shapes": [
                            {
                                "type": "rectangle",
                                "points": [
                                    {"x": 10.5, "y": 20.25},
                                    {"x": 30.75, "y": 40.5},
                                ],
                                "label": "dog",
                                "score": 0.9,
                                "group_id": 0,
                                "attributes": attributes,
                            }
                        ],
                        "warnings": [],
                        "timings_ms": {"inference": 1.25},
                    },
                },
            )
            return
        self._json(404, {"detail": "not found"})

    def do_DELETE(self):
        if self.path == "/v1/predictions/job-1" and self._authorized():
            self.state.deleted.set()
            self.send_response(204)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self._json(404, {"detail": "not found"})


@contextmanager
def _server(**options):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.protocol_state = _State(**options)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, server.protocol_state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


class TestRemoteInferenceClient(unittest.TestCase):
    def test_authenticated_round_trip_validates_identity_and_deletes_job(self):
        image = b"\x89PNG\r\n\x1a\nreal-image-payload"
        with _server() as (server, state):
            client = RemoteInferenceClient(
                f"http://127.0.0.1:{server.server_port}",
                "model-1",
                "correct horse battery staple",
                prediction_timeout_seconds=2,
                poll_interval_seconds=0.02,
            )
            result = client.predict(image, "image/png")
        self.assertEqual(result["shapes"][0]["label"], "dog")
        self.assertEqual(state.image, image)
        self.assertEqual(
            state.request["source_id"],
            f"content-sha256:{hashlib.sha256(image).hexdigest()}",
        )
        self.assertTrue(state.deleted.is_set())
        self.assertTrue(
            all(value == "Bearer test-token" for value in state.authorization)
        )

    def test_non_loopback_plain_http_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "requires HTTPS"):
            RemoteInferenceClient(
                "http://example.com", "model-1", "correct horse battery staple"
            )

    def test_redirected_authentication_is_not_followed(self):
        with _server(redirect_auth=True) as (server, _state):
            with self.assertRaisesRegex(
                RemoteInferenceError, r"rejected the request \(HTTP 307\)"
            ):
                RemoteInferenceClient(
                    f"http://127.0.0.1:{server.server_port}",
                    "model-1",
                    "correct horse battery staple",
                )

    def test_nested_shape_attributes_are_rejected_and_job_is_deleted(self):
        with _server(invalid_attributes=True) as (server, state):
            client = RemoteInferenceClient(
                f"http://127.0.0.1:{server.server_port}",
                "model-1",
                "correct horse battery staple",
                prediction_timeout_seconds=2,
                poll_interval_seconds=0.02,
            )
            with self.assertRaisesRegex(RemoteInferenceError, "shape attributes"):
                client.predict(b"valid bytes", "image/png")
        self.assertTrue(state.deleted.is_set())

    def test_cancellation_interrupts_polling_and_deletes_job(self):
        with _server(hold_running=True) as (server, state):
            client = RemoteInferenceClient(
                f"http://127.0.0.1:{server.server_port}",
                "model-1",
                "correct horse battery staple",
                prediction_timeout_seconds=5,
                poll_interval_seconds=0.02,
            )
            captured = []

            def run_prediction():
                try:
                    client.predict(b"valid bytes", "image/png")
                except Exception as error:  # noqa: BLE001
                    captured.append(error)

            thread = threading.Thread(target=run_prediction)
            thread.start()
            self.assertTrue(state.polled.wait(timeout=2))
            client.cancel()
            thread.join(timeout=2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(captured), 1)
        self.assertIsInstance(captured[0], RemoteInferenceError)
        self.assertIn("cancelled", str(captured[0]))
        self.assertTrue(state.deleted.is_set())


if __name__ == "__main__":
    unittest.main()
