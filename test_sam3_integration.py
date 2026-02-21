"""Integration test for SAM3 with the anylabeling ONNX wrapper.

Uses the exported ONNX models from
  samexporter/output_models/sam3/

Requires:
  - conda env ``anylabeling`` with onnxruntime, cv2, numpy, osam
  - ONNX models already present on disk

Run:
  conda run -n anylabeling python test_sam3_integration.py
"""

import sys
import os

import cv2
import numpy as np

# Make anylabeling importable without PyQt5 by monkey-patching it out.
# (PyQt5 is imported transitively via anylabeling.services.auto_labeling.model.)
from unittest.mock import MagicMock
_qt_mods = [
    "PyQt5", "PyQt5.QtCore", "PyQt5.QtGui", "PyQt5.QtWidgets",
    "qimage2ndarray",
    "anylabeling.views.labeling.shape",
    "anylabeling.views.labeling.label_file",
    "anylabeling.utils",
]
for _m in _qt_mods:
    sys.modules.setdefault(_m, MagicMock())

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(project_root, "anylabeling"))

from anylabeling.services.auto_labeling.sam3_onnx import SegmentAnything3ONNX


def test_sam3_anylabeling_integration():
    print("Testing SAM3 AnyLabeling integration...")

    encoder_path = os.path.join(
        project_root, "samexporter", "output_models", "sam3", "sam3_image_encoder.onnx"
    )
    decoder_path = os.path.join(
        project_root, "samexporter", "output_models", "sam3", "sam3_decoder.onnx"
    )
    language_path = os.path.join(
        project_root, "samexporter", "output_models", "sam3", "sam3_language_encoder.onnx"
    )

    for p in (encoder_path, decoder_path, language_path):
        if not os.path.exists(p):
            print(f"SKIP: model not found at {p}")
            return

    # ── Model loading ──────────────────────────────────────────────────────
    model = SegmentAnything3ONNX(encoder_path, decoder_path, language_path)
    print(f"Tokenizer: {model.language_encoder._tokenize}")

    # ── Load test image (BGR from cv2, convert to RGB for the model) ───────
    image_path = os.path.join(project_root, "samexporter", "images", "truck.jpg")
    bgr = cv2.imread(image_path)
    if bgr is None:
        print(f"SKIP: test image not found at {image_path}")
        return
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    H, W = rgb.shape[:2]
    print(f"Image: {image_path}  {W}×{H}")

    # ── Test 1: Encode with text prompt ───────────────────────────────────
    print("\nTest 1: Encoding with text prompt 'truck'...")
    embedding = model.encode(rgb, text_prompt="truck")
    assert embedding["language_mask"] is not None, "language_mask should not be None"
    assert embedding["language_features"] is not None, "language_features should not be None"
    assert embedding["language_mask"].shape == (1, 32), \
        f"Unexpected language_mask shape: {embedding['language_mask'].shape}"
    assert embedding["language_features"].shape == (32, 1, 256), \
        f"Unexpected language_features shape: {embedding['language_features'].shape}"
    print("  Encoding successful.")
    print(f"  language_mask shape:    {embedding['language_mask'].shape}")
    print(f"  language_features shape:{embedding['language_features'].shape}")

    # ── Test 2: Point prompt ───────────────────────────────────────────────
    print("\nTest 2: Point prompt (center-ish of image)...")
    prompt_pt = [{"type": "point", "data": [W // 2, H // 2], "label": 1}]
    masks_pt = model.predict_masks(embedding, prompt_pt, confidence_threshold=0.0)
    print(f"  Raw masks shape: {masks_pt.shape}  dtype: {masks_pt.dtype}")
    assert masks_pt.ndim == 4, f"Expected 4-D masks, got shape {masks_pt.shape}"
    assert masks_pt.dtype == np.bool_, f"Expected bool masks, got {masks_pt.dtype}"
    # With a real tokeniser the point prompt should find something.
    masks_pt_filtered = model.predict_masks(embedding, prompt_pt, confidence_threshold=0.5)
    print(f"  Filtered masks (thr=0.5): {masks_pt_filtered.shape[0]} masks")
    assert masks_pt_filtered.shape[0] > 0, (
        "Point prompt with text 'truck' returned 0 masks above threshold. "
        "Is osam installed? Is the tokeniser working?"
    )

    # ── Test 3: Rectangle prompt ───────────────────────────────────────────
    print("\nTest 3: Rectangle prompt...")
    # A box covering most of the truck in the 1800×1200 truck.jpg.
    prompt_rect = [{"type": "rectangle", "data": [100, 100, 900, 600]}]
    masks_rect = model.predict_masks(embedding, prompt_rect, confidence_threshold=0.5)
    print(f"  Filtered masks (thr=0.5): {masks_rect.shape[0]} masks")
    assert masks_rect.ndim == 4, f"Expected 4-D masks, got {masks_rect.shape}"
    assert masks_rect.shape[0] > 0, "Rectangle prompt returned 0 masks above threshold."
    assert masks_rect.shape[1] == 1, "Expected channel dim of 1"
    assert masks_rect.shape[2] == H and masks_rect.shape[3] == W, (
        f"Masks should match original image size ({H}×{W}), "
        f"got {masks_rect.shape[2]}×{masks_rect.shape[3]}"
    )
    print(f"  Mask shape: {masks_rect.shape}  (N=1 first result)")

    # ── Test 4: Bool mask conversion to uint8 (simulates post_process) ────
    print("\nTest 4: Bool → float32 → uint8 conversion (post_process sim)...")
    mask_2d = masks_rect[0, 0].astype(np.float32)
    mask_2d[mask_2d > 0.0] = 255
    mask_2d[mask_2d <= 0.0] = 0
    mask_uint8 = mask_2d.astype(np.uint8)
    n_positive = int(mask_uint8.max())
    print(f"  Mask max: {n_positive}  (should be 255)")
    assert n_positive == 255, "Bool→uint8 conversion did not produce 255 foreground"

    # ── Test 5: No prompt (text-only mode) ────────────────────────────────
    print("\nTest 5: Text-only mode (no geometric prompt)...")
    masks_text = model.predict_masks(embedding, [], confidence_threshold=0.5)
    print(f"  Text-only masks: {masks_text.shape[0]} masks")
    assert masks_text.ndim == 4, f"Expected 4-D masks, got {masks_text.shape}"
    # Text-only should still produce at least one detection with a good tokeniser.
    assert masks_text.shape[0] > 0, (
        "Text-only ('truck') returned 0 masks. Check osam tokeniser."
    )

    print("\nAll integration tests passed!")


if __name__ == "__main__":
    test_sam3_anylabeling_integration()
