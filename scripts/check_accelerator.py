"""Run a small real ONNX graph and verify its execution provider."""

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

from anylabeling.services.auto_labeling.runtime import (
    create_inference_session,
    get_onnx_providers,
)


def _make_model(path: Path, operation: str) -> None:
    if operation == "conv":
        weights = numpy_helper.from_array(
            np.ones((2, 3, 3, 3), dtype=np.float32), "weights"
        )
        nodes = [
            helper.make_node(
                "Conv",
                ["input", "weights"],
                ["output"],
                pads=[1, 1, 1, 1],
            )
        ]
        inputs = [
            helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 16, 16])
        ]
        outputs = [
            helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2, 16, 16])
        ]
    else:
        weights = numpy_helper.from_array(np.eye(4, dtype=np.float32), "weights")
        nodes = [helper.make_node("MatMul", ["input", "weights"], ["output"])]
        inputs = [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])]
        outputs = [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])]

    graph = helper.make_graph(
        nodes,
        "accelerator-smoke-test",
        inputs,
        outputs,
        [weights],
    )
    model = helper.make_model(
        graph,
        ir_version=8,
        opset_imports=[helper.make_opsetid("", 13)],
    )
    onnx.save(model, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--expect", default=None)
    parser.add_argument("--operation", choices=("matmul", "conv"), default="matmul")
    args = parser.parse_args()

    selected = get_onnx_providers(args.device)
    if not selected:
        raise SystemExit("ONNX Runtime reports no execution providers")
    if args.expect and selected[0] != args.expect:
        raise SystemExit(
            f"Expected {args.expect}, selected {selected[0]} from "
            f"{ort.get_available_providers()}"
        )

    with tempfile.TemporaryDirectory(prefix="anylabeling-accelerator-") as tmp:
        model_path = Path(tmp) / f"{args.operation}.onnx"
        _make_model(model_path, args.operation)
        options = ort.SessionOptions()
        options.enable_profiling = True
        options.profile_file_prefix = str(Path(tmp) / "profile")
        session = create_inference_session(
            str(model_path),
            preferred_device=args.device,
            sess_options=options,
        )
        if args.operation == "conv":
            value = np.ones((1, 3, 16, 16), dtype=np.float32)
        else:
            value = np.arange(4, dtype=np.float32).reshape(1, 4)
        output = session.run(None, {"input": value})[0]
        if args.operation == "matmul":
            np.testing.assert_array_equal(output, value)
        elif output.shape != (1, 2, 16, 16):
            raise AssertionError(f"Unexpected convolution output shape: {output.shape}")
        profile_path = Path(session.end_profiling())
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        configured = session.get_providers()
        executed_by = {
            event.get("args", {}).get("provider")
            for event in profile
            if event.get("cat") == "Node" and event.get("args", {}).get("provider")
        }

    if args.expect and args.expect not in executed_by:
        raise SystemExit(
            f"Graph did not execute on {args.expect}; node providers: "
            f"{sorted(executed_by)}"
        )
    print("Available providers:", ort.get_available_providers())
    print("Selected providers:", configured)
    print("Node execution providers:", sorted(executed_by))
    print("Inference result: OK")


if __name__ == "__main__":
    main()
