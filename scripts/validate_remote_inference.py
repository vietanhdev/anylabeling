#!/usr/bin/env python3
"""Validate AnyLabeling against an authenticated AnyLearning TCP server."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import secrets
import socket
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import psutil
import uvicorn
from PyQt6.QtGui import QImage

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from anylabeling.services.auto_labeling.remote_model import RemoteModel

_PASSWORD = "real-remote-validation-password"
_PASSWORD_ENV = "ANYLABELING_REAL_REMOTE_PASSWORD"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _start_server(app: Any) -> tuple[uvicorn.Server, threading.Thread, int]:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(128)
    port = listener.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning", access_log=False))
    thread = threading.Thread(
        target=server.run,
        kwargs={"sockets": [listener]},
        name="anylearning-real-remote-server",
        daemon=True,
    )
    thread.start()
    deadline = time.monotonic() + 30
    while not server.started:
        if not thread.is_alive():
            raise RuntimeError("AnyLearning validation server stopped during startup")
        if time.monotonic() >= deadline:
            server.should_exit = True
            thread.join(timeout=10)
            raise TimeoutError("AnyLearning validation server did not start in time")
        time.sleep(0.02)
    return server, thread, port


def _stop_server(server: uvicorn.Server, thread: threading.Thread) -> None:
    server.should_exit = True
    thread.join(timeout=20)
    if thread.is_alive():
        server.force_exit = True
        thread.join(timeout=10)
    if thread.is_alive():
        raise RuntimeError("AnyLearning validation server did not stop cleanly")


def _adapt_result(model, result, *, request_id: str):
    from anylearning.inference import InferenceResult, InferenceShape, Point

    capabilities = model.client.capabilities
    return InferenceResult(
        request_id=request_id,
        source_id="anylabeling-qimage-png",
        model_id=capabilities.model_id,
        model_revision=capabilities.model_revision,
        shapes=tuple(
            InferenceShape(
                type=shape.shape_type,
                points=tuple(Point(x=point.x(), y=point.y()) for point in shape.points),
                label=shape.label,
                score=shape.other_data.get("score"),
                group_id=shape.group_id,
                attributes=shape.other_data.get("attributes", {}),
            )
            for shape in result.shapes
        ),
    )


def run_validation(
    anylearning_root: Path,
    manifest_path: Path,
    model_path: Path,
    image_path: Path,
    output_root: Path,
) -> Path:
    sys.path.insert(0, str(anylearning_root.resolve(strict=True)))
    from anylearning.inference.validation import (
        _annotate,
        _check_expectations,
        _load_rgb,
        _result_digest,
        load_validation_manifest,
    )
    from anylearning.server import (
        ServerModelDefinition,
        ServerSettings,
        create_server_app,
        hash_password,
    )

    manifest_path = manifest_path.resolve(strict=True)
    model_path = model_path.resolve(strict=True)
    image_path = image_path.resolve(strict=True)
    manifest = load_validation_manifest(manifest_path)
    if manifest.backend != "yolo_onnx" or len(manifest.images) != 1:
        raise ValueError("remote validation requires one YOLO ONNX image case")
    config = dict(manifest.config)
    config.update(config_file=str(manifest_path), model_path=str(model_path))
    definition = ServerModelDefinition(backend=manifest.backend, config=config)
    settings = ServerSettings(
        password_hash=hash_password(_PASSWORD),
        token_secret=secrets.token_bytes(32),
        token_ttl_seconds=300,
        prediction_timeout_seconds=120,
        prediction_result_ttl_seconds=300,
    )
    server, thread, port = _start_server(
        create_server_app(settings, model_definitions=(definition,))
    )
    remote_model = None
    previous_password = os.environ.get(_PASSWORD_ENV)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = (
        output_root.resolve() / f"{stamp}-anylabeling-remote-{secrets.token_hex(4)}"
    )
    output_dir.mkdir(parents=True, mode=0o700)
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    started = time.perf_counter()
    try:
        os.environ[_PASSWORD_ENV] = _PASSWORD
        image_case = manifest.images[0]
        remote_model = RemoteModel(
            {
                "type": "remote",
                "name": "real-remote-yolox",
                "display_name": "Real remote YOLOX-S",
                "server_url": f"http://127.0.0.1:{port}",
                "model_id": config["name"],
                "password_env": _PASSWORD_ENV,
                "prediction_timeout_seconds": 120,
                "poll_interval_seconds": 0.02,
                "parameters": dict(image_case.request_parameters),
            },
            lambda _message: None,
        )
        image = QImage(str(image_path))
        if image.isNull():
            raise ValueError("validation image could not be decoded by Qt")
        results, round_trip_ms = [], []
        for run in range(manifest.runs):
            run_started = time.perf_counter()
            labeling_result = remote_model.predict_shapes(image, str(image_path))
            round_trip_ms.append((time.perf_counter() - run_started) * 1000)
            results.append(
                _adapt_result(remote_model, labeling_result, request_id=f"run-{run}")
            )
            peak_rss = max(peak_rss, process.memory_info().rss)
        digests = [
            _result_digest(result.model_copy(update={"request_id": "canonical"}))
            for result in results
        ]
        failures = _check_expectations(results[0], image_case.expected)
        if len(set(digests)) != 1:
            failures.append("AnyLabeling remote results changed across identical runs")
        annotated_name = "000-dog-anylabeling-remote.png"
        if not cv2.imwrite(
            str(output_dir / annotated_name),
            _annotate(_load_rgb(image_path), results[0]),
        ):
            raise OSError("could not write annotated remote validation image")
        summary = {
            "schema_version": 1,
            "passed": not failures,
            "created_at": datetime.now(UTC).isoformat(),
            "transport": "authenticated TCP HTTP loopback",
            "client": "AnyLabeling RemoteModel",
            "server": "AnyLearning inference server",
            "manifest": manifest_path.name,
            "model_sha256": _sha256(model_path),
            "image_sha256": _sha256(image_path),
            "provenance": manifest.provenance.model_dump(mode="json"),
            "runs": manifest.runs,
            "shape_count": len(results[0].shapes),
            "labels": [shape.label for shape in results[0].shapes],
            "consistent_runs": len(set(digests)) == 1,
            "consistency_digest": digests[0],
            "round_trip_ms": round_trip_ms,
            "total_elapsed_ms": (time.perf_counter() - started) * 1000,
            "peak_observed_rss_bytes": peak_rss,
            "annotated_image": annotated_name,
            "failures": failures,
        }
        _write_json(output_dir / "summary.json", summary)
        _write_json(
            output_dir / "results.json",
            [result.model_dump(mode="json") for result in results],
        )
        (output_dir / "index.html").write_text(
            "<!doctype html><meta charset=utf-8><title>AnyLabeling remote validation</title>"
            f"<h1>{'PASS' if not failures else 'FAIL'}: authenticated remote YOLOX-S</h1>"
            f"<p>{'<br>'.join(html.escape(item) for item in failures)}</p>"
            f'<img src="{html.escape(annotated_name)}" style="max-width:100%;height:auto">',
            encoding="utf-8",
        )
        if failures:
            raise AssertionError("; ".join(failures))
        return output_dir
    except Exception as error:
        _write_json(
            output_dir / "failure.json",
            {
                "schema_version": 1,
                "passed": False,
                "created_at": datetime.now(UTC).isoformat(),
                "error_type": type(error).__name__,
                "error": str(error)[:2_048],
            },
        )
        raise
    finally:
        if remote_model is not None:
            remote_model.unload()
        if previous_password is None:
            os.environ.pop(_PASSWORD_ENV, None)
        else:
            os.environ[_PASSWORD_ENV] = previous_password
        _stop_server(server, thread)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anylearning-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("validation-results"))
    args = parser.parse_args()
    print(
        run_validation(
            args.anylearning_root,
            args.manifest,
            args.model,
            args.image,
            args.output_root,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
