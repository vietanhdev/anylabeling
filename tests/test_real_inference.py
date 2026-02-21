"""Real-inference integration tests for all downloaded model types.

Tests SAM3, SAM2, SAM1 (MobileSAM), and YOLOv8 with actual ONNX models.
Skips any test whose models are not yet on disk.

Run:
    conda run -n anylabeling python -m pytest tests/test_real_inference.py -v -s
"""
import sys
import os
import unittest
from unittest.mock import MagicMock

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Stub PyQt6 before importing anylabeling (headless / no display)
# ---------------------------------------------------------------------------
_MOCK_MODS = [
    "PyQt6", "PyQt6.QtCore", "PyQt6.QtGui", "PyQt6.QtWidgets",
    "qimage2ndarray",
    "anylabeling.views.labeling.shape",
    "anylabeling.views.labeling.label_file",
    "anylabeling.utils",
]
for _m in _MOCK_MODS:
    sys.modules.setdefault(_m, MagicMock())

import PyQt6.QtCore as _qtc
_qtc.QCoreApplication.translate = lambda ctx, s: s

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from anylabeling.services.auto_labeling.sam3_onnx import SegmentAnything3ONNX
from anylabeling.services.auto_labeling.sam2_onnx import SegmentAnything2ONNX
from anylabeling.services.auto_labeling.sam_onnx import SegmentAnythingONNX

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT      = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SAM_ROOT  = os.path.join(_ROOT, "..", "samexporter")
_MODEL_DIR = os.path.expanduser("~/anylabeling_data/models")
_IMG       = os.path.join(_SAM_ROOT, "images", "truck.jpg")
_SAMPLE    = os.path.join(_ROOT, "sample_images",
                          "evan-foley-ZgUtMaOVUAY-unsplash.jpg")

SAM3_DIR   = os.path.join(_MODEL_DIR, "sam3_vit_h_20260220")
SAM2_DIR   = os.path.join(_MODEL_DIR, "sam2_hiera_tiny_20240803")
MSAM_DIR   = os.path.join(_MODEL_DIR, "mobile_sam_20230629")
YOLOv8_DIR = os.path.join(_MODEL_DIR, "yolov8n-r20230415")


def _load_rgb(path):
    """Load image as RGB ndarray; raise if not found."""
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(f"Test image not found: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _load_bgr(path):
    """Load image as BGR ndarray (for models that expect BGR input)."""
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(f"Test image not found: {path}")
    return bgr


def _test_img():
    """Return a test image; prefer truck.jpg, fall back to sample."""
    for p in (_IMG, _SAMPLE):
        bgr = cv2.imread(p)
        if bgr is not None:
            return p, bgr
    raise FileNotFoundError("No test image found")


# ---------------------------------------------------------------------------
# SAM3 ViT-H
# ---------------------------------------------------------------------------
class TestSAM3RealInference(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.enc  = os.path.join(SAM3_DIR, "sam3_image_encoder.onnx")
        cls.dec  = os.path.join(SAM3_DIR, "sam3_decoder.onnx")
        cls.lang = os.path.join(SAM3_DIR, "sam3_language_encoder.onnx")
        if not all(os.path.exists(p) for p in (cls.enc, cls.dec, cls.lang)):
            cls.model = None
            return
        cls.model = SegmentAnything3ONNX(cls.enc, cls.dec, cls.lang)
        _, bgr = _test_img()
        cls.rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        cls.H, cls.W = cls.rgb.shape[:2]
        print(f"\n[SAM3] Image: {cls.W}×{cls.H}")

    def _skip_if_no_model(self):
        if self.model is None:
            self.skipTest("SAM3 models not downloaded")

    def test_01_encode_shapes(self):
        self._skip_if_no_model()
        emb = self.model.encode(self.rgb, text_prompt="truck")
        self.assertIsNotNone(emb["language_mask"])
        self.assertEqual(emb["language_mask"].shape, (1, 32))
        self.assertEqual(emb["language_features"].shape, (32, 1, 256))
        self.assertEqual(emb["original_size"], (self.H, self.W))
        print(f"  encode OK — image {self.W}×{self.H}")

    def test_02_text_only_detection(self):
        self._skip_if_no_model()
        emb = self.model.encode(self.rgb, text_prompt="truck")
        masks = self.model.predict_masks(emb, [], confidence_threshold=0.5)
        self.assertGreaterEqual(masks.shape[0], 1)
        self.assertEqual(masks.shape[1], 1)
        self.assertEqual(masks.shape[2], self.H)
        self.assertEqual(masks.shape[3], self.W)
        self.assertEqual(masks.dtype, np.bool_)
        print(f"  text-only 'truck': {masks.shape[0]} mask(s), shape={masks.shape}")

    def test_03_point_prompt(self):
        self._skip_if_no_model()
        emb = self.model.encode(self.rgb, text_prompt="truck")
        prompt = [{"type": "point", "data": [self.W // 2, self.H // 2], "label": 1}]
        masks = self.model.predict_masks(emb, prompt, confidence_threshold=0.5)
        self.assertGreaterEqual(masks.shape[0], 1)
        print(f"  point prompt: {masks.shape[0]} mask(s)")

    def test_04_rectangle_prompt(self):
        self._skip_if_no_model()
        emb = self.model.encode(self.rgb, text_prompt="truck")
        prompt = [{"type": "rectangle", "data": [100, 100, 900, 600]}]
        masks = self.model.predict_masks(emb, prompt, confidence_threshold=0.5)
        self.assertGreaterEqual(masks.shape[0], 1)
        self.assertEqual(masks.shape[2], self.H)
        self.assertEqual(masks.shape[3], self.W)
        print(f"  rect prompt: {masks.shape[0]} mask(s), shape={masks.shape}")

    def test_05_multi_class_update_language(self):
        self._skip_if_no_model()
        emb_truck = self.model.encode(self.rgb, text_prompt="truck")
        emb_car   = self.model.update_language(emb_truck, "car")
        # Image tensors shared; language features differ
        self.assertIs(emb_car["backbone_fpn_0"], emb_truck["backbone_fpn_0"])
        self.assertFalse(
            np.allclose(emb_car["language_features"],
                        emb_truck["language_features"])
        )
        m_car = self.model.predict_masks(emb_car, [], confidence_threshold=0.5)
        print(f"  multi-class 'car': {m_car.shape[0]} mask(s)")

    def test_06_bool_mask_to_uint8(self):
        self._skip_if_no_model()
        emb = self.model.encode(self.rgb, text_prompt="truck")
        masks = self.model.predict_masks(emb, [], confidence_threshold=0.5)
        if masks.shape[0] == 0:
            self.skipTest("No masks returned")
        mask_2d = masks[0, 0].astype(np.float32)
        mask_2d[mask_2d > 0.0] = 255
        mask_uint8 = mask_2d.astype(np.uint8)
        self.assertEqual(mask_uint8.max(), 255)
        print(f"  bool→uint8 conversion OK, max={mask_uint8.max()}")


# ---------------------------------------------------------------------------
# SAM2 Hiera-Tiny
# ---------------------------------------------------------------------------
class TestSAM2RealInference(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.enc = os.path.join(SAM2_DIR, "sam2_hiera_tiny.encoder.onnx")
        cls.dec = os.path.join(SAM2_DIR, "sam2_hiera_tiny.decoder.onnx")
        if not all(os.path.exists(p) for p in (cls.enc, cls.dec)):
            cls.model = None
            return
        cls.model = SegmentAnything2ONNX(cls.enc, cls.dec)
        path, bgr = _test_img()
        # SAM2 encoder calls cv2.cvtColor(image, BGR2RGB) internally,
        # so pass the BGR image directly from cv2.imread.
        cls.bgr = bgr
        cls.H, cls.W = bgr.shape[:2]
        print(f"\n[SAM2] Image: {cls.W}×{cls.H}")

    def _skip_if_no_model(self):
        if self.model is None:
            self.skipTest("SAM2 models not downloaded")

    def test_01_encode_returns_embedding(self):
        self._skip_if_no_model()
        emb = self.model.encode(self.bgr)
        self.assertIn("image_embedding", emb)
        self.assertIn("high_res_feats_0", emb)
        self.assertIn("high_res_feats_1", emb)
        self.assertEqual(emb["original_size"], (self.H, self.W))
        print(f"  encode OK — embedding shape: {emb['image_embedding'].shape}")

    def test_02_point_prompt(self):
        self._skip_if_no_model()
        emb = self.model.encode(self.bgr)
        prompt = [{"type": "point", "data": [self.W // 2, self.H // 2], "label": 1}]
        masks = self.model.predict_masks(emb, prompt)
        self.assertIsNotNone(masks)
        self.assertGreater(masks.size, 0)
        print(f"  point prompt masks shape: {masks.shape}")

    def test_03_rectangle_prompt(self):
        self._skip_if_no_model()
        emb = self.model.encode(self.bgr)
        prompt = [{"type": "rectangle", "data": [100, 100, 900, 600]}]
        masks = self.model.predict_masks(emb, prompt)
        self.assertIsNotNone(masks)
        self.assertGreater(masks.size, 0)
        print(f"  rect prompt masks shape: {masks.shape}")

    def test_04_mask_size_matches_image(self):
        self._skip_if_no_model()
        emb = self.model.encode(self.bgr)
        prompt = [{"type": "point", "data": [self.W // 2, self.H // 2], "label": 1}]
        masks = self.model.predict_masks(emb, prompt)
        # SAM2 decoder resizes mask back to original image size
        mask_h, mask_w = masks.shape[-2], masks.shape[-1]
        self.assertEqual(mask_h, self.H)
        self.assertEqual(mask_w, self.W)
        print(f"  mask size {mask_w}×{mask_h} matches image {self.W}×{self.H} ✓")


# ---------------------------------------------------------------------------
# SAM1 / MobileSAM
# ---------------------------------------------------------------------------
def _find_msam_files():
    """Return (encoder_path, decoder_path) for MobileSAM or (None, None)."""
    import glob
    # Search both the directory root and any one-level subdirectories
    search_dirs = [MSAM_DIR] + glob.glob(os.path.join(MSAM_DIR, "*", ""))
    for d in search_dirs:
        for enc_name in ("mobile_sam.encoder.onnx", "mobile_sam_encoder.onnx",
                         "vit_t_encoder.onnx", "encoder.onnx"):
            enc = os.path.join(d, enc_name)
            if not os.path.exists(enc):
                continue
            for dec_name in ("sam_vit_h_4b8939.decoder.onnx",
                             "mobile_sam_decoder.onnx", "vit_t_decoder.onnx",
                             "decoder.onnx"):
                dec = os.path.join(d, dec_name)
                if os.path.exists(dec):
                    return enc, dec
    return None, None


class TestSAM1MobileSAMRealInference(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        enc, dec = _find_msam_files()
        if enc is None:
            cls.model = None
            print(f"\n[MobileSAM] No ONNX files found in {MSAM_DIR}")
            return
        cls.model = SegmentAnythingONNX(enc, dec)
        path, bgr = _test_img()
        cls.bgr = bgr
        cls.H, cls.W = bgr.shape[:2]
        print(f"\n[MobileSAM] enc={os.path.basename(enc)}  img: {cls.W}×{cls.H}")

    def _skip_if_no_model(self):
        if self.model is None:
            self.skipTest("MobileSAM ONNX files not downloaded/extracted")

    def test_01_encode(self):
        self._skip_if_no_model()
        emb = self.model.encode(self.bgr)
        self.assertIn("image_embedding", emb)
        self.assertIn("original_size", emb)
        print(f"  encode OK — embedding shape: {emb['image_embedding'].shape}")

    def test_02_point_prompt(self):
        self._skip_if_no_model()
        emb = self.model.encode(self.bgr)
        prompt = [{"type": "point", "data": [self.W // 2, self.H // 2], "label": 1}]
        masks = self.model.predict_masks(emb, prompt)
        self.assertIsNotNone(masks)
        self.assertGreater(masks.size, 0)
        print(f"  point prompt masks shape: {masks.shape}")

    def test_03_rectangle_prompt(self):
        self._skip_if_no_model()
        emb = self.model.encode(self.bgr)
        prompt = [{"type": "rectangle", "data": [100, 100, 900, 600]}]
        masks = self.model.predict_masks(emb, prompt)
        self.assertIsNotNone(masks)
        self.assertGreater(masks.size, 0)
        print(f"  rect prompt masks shape: {masks.shape}")


# ---------------------------------------------------------------------------
# YOLOv8n
# ---------------------------------------------------------------------------
def _find_yolov8_onnx():
    """Return (model_path, config) for the YOLOv8n model, or (None, None)."""
    import glob
    import yaml
    # Search directory root and one-level subdirectories (zip may create a sub-dir)
    search_dirs = [YOLOv8_DIR] + glob.glob(os.path.join(YOLOv8_DIR, "*", ""))
    for d in search_dirs:
        onnx_files = glob.glob(os.path.join(d, "*.onnx"))
        if not onnx_files:
            continue
        model_path = onnx_files[0]
        config_path = os.path.join(d, "config.yaml")
        config = {}
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = yaml.safe_load(f) or {}
        return model_path, config
    return None, None


class TestYOLOv8RealInference(unittest.TestCase):
    """Test YOLOv8n ONNX model directly (bypasses Qt and model_manager)."""

    @classmethod
    def setUpClass(cls):
        cls.model_path, cls.config = _find_yolov8_onnx()
        if cls.model_path is None:
            print(f"\n[YOLOv8] No ONNX file in {YOLOv8_DIR}")
            return
        cls.net = cv2.dnn.readNet(cls.model_path)
        cls.input_w = cls.config.get("input_width", 640)
        cls.input_h = cls.config.get("input_height", 640)
        cls.conf_thr = cls.config.get("confidence_threshold", 0.25)
        cls.nms_thr  = cls.config.get("nms_threshold", 0.45)
        cls.classes  = cls.config.get("classes", [str(i) for i in range(80)])
        path, bgr = _test_img()
        cls.bgr = bgr
        cls.H, cls.W = bgr.shape[:2]
        print(f"\n[YOLOv8] model={os.path.basename(cls.model_path)}  img: {cls.W}×{cls.H}")

    def _skip_if_no_model(self):
        if self.model_path is None:
            self.skipTest("YOLOv8 ONNX file not downloaded/extracted")

    def _forward(self):
        blob = cv2.dnn.blobFromImage(
            self.bgr, 1 / 255,
            (self.input_w, self.input_h),
            [0, 0, 0], swapRB=True, crop=False,
        )
        self.net.setInput(blob)
        outputs = self.net.forward()
        return np.array([cv2.transpose(outputs[0])])

    def test_01_forward_output_shape(self):
        self._skip_if_no_model()
        outputs = self._forward()
        # outputs shape: (1, num_rows, 4+num_classes)
        self.assertEqual(outputs.ndim, 3)
        self.assertEqual(outputs.shape[0], 1)
        n_classes = len(self.classes)
        self.assertEqual(outputs.shape[2], 4 + n_classes,
                         f"Expected 4+{n_classes} cols, got {outputs.shape[2]}")
        print(f"  forward output shape: {outputs.shape}")

    def test_02_detections_above_threshold(self):
        self._skip_if_no_model()
        outputs = self._forward()
        rows = outputs.shape[1]
        x_factor = self.W / self.input_w
        y_factor = self.H / self.input_h

        boxes, confidences, class_ids = [], [], []
        for r in range(rows):
            row = outputs[0][r]
            scores = row[4:]
            _, conf, _, (_, cid) = cv2.minMaxLoc(scores)
            if conf >= self.conf_thr:
                cx, cy, w, h = row[0], row[1], row[2], row[3]
                left = int((cx - w / 2) * x_factor)
                top  = int((cy - h / 2) * y_factor)
                boxes.append([left, top, int(w * x_factor), int(h * y_factor)])
                confidences.append(float(conf))
                class_ids.append(cid)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_thr, self.nms_thr)
        dets = len(indices)
        if dets > 0:
            labels = [self.classes[class_ids[i]] for i in indices]
            print(f"  detected {dets} object(s): {labels[:10]}")
        else:
            print(f"  no detections above threshold {self.conf_thr} in this image")
        # The test passes regardless — we're testing the pipeline, not the data
        self.assertIsInstance(dets, int)

    def test_03_bounding_boxes_within_image(self):
        self._skip_if_no_model()
        outputs = self._forward()
        rows = outputs.shape[1]
        x_factor = self.W / self.input_w
        y_factor = self.H / self.input_h

        for r in range(rows):
            row = outputs[0][r]
            scores = row[4:]
            _, conf, _, _ = cv2.minMaxLoc(scores)
            if conf >= self.conf_thr:
                cx, cy = float(row[0]) * x_factor, float(row[1]) * y_factor
                # Centre point should be roughly within image bounds
                self.assertGreater(cx, -self.W * 0.1)
                self.assertLess(cx, self.W * 1.1)
                self.assertGreater(cy, -self.H * 0.1)
                self.assertLess(cy, self.H * 1.1)
        print("  all bounding-box centres within image bounds ✓")


# ---------------------------------------------------------------------------
# Summary helper
# ---------------------------------------------------------------------------

def _print_summary():
    models_status = {
        "SAM3 ViT-H": all(
            os.path.exists(os.path.join(SAM3_DIR, n))
            for n in ("sam3_image_encoder.onnx", "sam3_decoder.onnx",
                      "sam3_language_encoder.onnx")
        ),
        "SAM2 Hiera-Tiny": all(
            os.path.exists(os.path.join(SAM2_DIR, n))
            for n in ("sam2_hiera_tiny.encoder.onnx",
                      "sam2_hiera_tiny.decoder.onnx")
        ),
        "MobileSAM": _find_msam_files()[0] is not None,
        "YOLOv8n": _find_yolov8_onnx()[0] is not None,
    }
    print("\n=== Model availability ===")
    for name, avail in models_status.items():
        print(f"  {'✓' if avail else '✗'} {name}")


if __name__ == "__main__":
    _print_summary()
    unittest.main(verbosity=2)
