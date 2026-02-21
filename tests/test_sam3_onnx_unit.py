"""Unit tests for anylabeling SAM3 ONNX classes.

All ONNX model I/O is mocked so no model files are required.
"""
import sys
import os
import unittest
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from anylabeling.services.auto_labeling.sam3_onnx import (
    SAM3ImageEncoder,
    SAM3ImageDecoder,
    SAM3LanguageEncoder,
    SegmentAnything3ONNX,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_encoder_session(input_shape, input_type="tensor(uint8)"):
    """Create a mock onnxruntime session for the image encoder."""
    session = MagicMock()
    inp = MagicMock()
    inp.name = "image"
    inp.shape = input_shape
    inp.type = input_type
    session.get_inputs.return_value = [inp]
    # Return 6 dummy feature maps (matching SAM3 encoder output)
    session.run.return_value = [
        np.zeros((1, 64, 64, 256), dtype=np.float32),  # vision_pos_enc_0
        np.zeros((1, 32, 32, 256), dtype=np.float32),  # vision_pos_enc_1
        np.zeros((1, 16, 16, 256), dtype=np.float32),  # vision_pos_enc_2
        np.zeros((1, 256, 64, 64), dtype=np.float32),  # backbone_fpn_0
        np.zeros((1, 256, 32, 32), dtype=np.float32),  # backbone_fpn_1
        np.zeros((1, 256, 16, 16), dtype=np.float32),  # backbone_fpn_2
    ]
    return session


def _make_decoder_session(input_names, H=100, W=100):
    """Create a mock onnxruntime session for the decoder."""
    session = MagicMock()
    inputs = []
    for name in input_names:
        inp = MagicMock()
        inp.name = name
        inputs.append(inp)
    session.get_inputs.return_value = inputs
    # Return (boxes, scores, masks) – ONNX export order
    n = 2
    session.run.return_value = [
        np.zeros((n, 4), dtype=np.float32),             # boxes
        np.array([0.9, 0.3], dtype=np.float32),         # scores
        np.ones((n, 1, H, W), dtype=np.bool_),          # masks
    ]
    return session


def _make_language_session():
    """Create a mock onnxruntime session for the language encoder."""
    session = MagicMock()
    inp = MagicMock()
    inp.name = "tokens"
    session.get_inputs.return_value = [inp]
    session.run.return_value = [
        np.zeros((1, 32), dtype=np.bool_),       # text_attention_mask
        np.zeros((32, 1, 256), dtype=np.float32), # text_memory
        np.zeros((32, 1, 1024), dtype=np.float32),# text_embeds
    ]
    return session


# ---------------------------------------------------------------------------
# SAM3ImageEncoder.prepare_input
# ---------------------------------------------------------------------------

class TestSAM3ImageEncoderPrepareInput(unittest.TestCase):

    def _make_encoder(self, input_shape, input_type):
        with patch("onnxruntime.InferenceSession", return_value=_make_encoder_session(input_shape, input_type)):
            enc = SAM3ImageEncoder.__new__(SAM3ImageEncoder)
            enc.session = _make_encoder_session(input_shape, input_type)
            enc.input_name = "image"
            enc.input_shape = input_shape
            enc.input_type = input_type
            if len(input_shape) == 3:
                enc.input_height = int(input_shape[1]) or 1008
                enc.input_width = int(input_shape[2]) or 1008
            else:
                enc.input_height = int(input_shape[2]) or 1008
                enc.input_width = int(input_shape[3]) or 1008
        return enc

    def test_uint8_model_returns_uint8_tensor(self):
        enc = self._make_encoder([3, 1008, 1008], "tensor(uint8)")
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        out = enc.prepare_input(image)
        self.assertEqual(out.dtype, np.uint8)
        self.assertEqual(out.shape, (3, 1008, 1008))

    def test_float_model_returns_float32_in_range(self):
        enc = self._make_encoder([3, 1008, 1008], "tensor(float)")
        image = np.full((480, 640, 3), 128, dtype=np.uint8)
        out = enc.prepare_input(image)
        self.assertEqual(out.dtype, np.float32)
        self.assertEqual(out.shape, (3, 1008, 1008))
        # (128/255 - 0.5) / 0.5 ≈ 0.004
        self.assertTrue(np.all(out >= -1.0) and np.all(out <= 1.0))

    def test_float_model_zero_maps_to_minus_one(self):
        enc = self._make_encoder([3, 1008, 1008], "tensor(float)")
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        out = enc.prepare_input(image)
        # (0/255 - 0.5)/0.5 == -1
        np.testing.assert_allclose(out, -1.0, atol=1e-5)

    def test_float_model_255_maps_to_one(self):
        enc = self._make_encoder([3, 1008, 1008], "tensor(float)")
        image = np.full((480, 640, 3), 255, dtype=np.uint8)
        out = enc.prepare_input(image)
        np.testing.assert_allclose(out, 1.0, atol=1e-5)

    def test_3d_input_shape_sets_height_width(self):
        enc = self._make_encoder([3, 512, 512], "tensor(uint8)")
        self.assertEqual(enc.input_height, 512)
        self.assertEqual(enc.input_width, 512)

    def test_4d_input_shape_sets_height_width(self):
        enc = self._make_encoder([1, 3, 512, 512], "tensor(uint8)")
        self.assertEqual(enc.input_height, 512)
        self.assertEqual(enc.input_width, 512)

    def test_output_is_chw_not_hwc(self):
        enc = self._make_encoder([3, 1008, 1008], "tensor(uint8)")
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        out = enc.prepare_input(image)
        # First dimension should be channels (3)
        self.assertEqual(out.shape[0], 3)


# ---------------------------------------------------------------------------
# SAM3LanguageEncoder._fallback_tokenize
# ---------------------------------------------------------------------------

class TestSAM3LanguageEncoderFallback(unittest.TestCase):

    def _make_lang_encoder_no_osam(self):
        """Create encoder that always uses the fallback tokenizer."""
        with patch("onnxruntime.InferenceSession", return_value=_make_language_session()):
            enc = SAM3LanguageEncoder.__new__(SAM3LanguageEncoder)
            enc.session = _make_language_session()
            enc._tokenize = enc._fallback_tokenize.__get__(enc, SAM3LanguageEncoder)
        return enc

    def test_fallback_returns_zeros(self):
        enc = self._make_lang_encoder_no_osam()
        result = enc._fallback_tokenize(["cat"], context_length=32)
        self.assertEqual(result.shape, (1, 32))
        np.testing.assert_array_equal(result, 0)

    def test_fallback_multiple_texts(self):
        enc = self._make_lang_encoder_no_osam()
        result = enc._fallback_tokenize(["cat", "dog"], context_length=32)
        self.assertEqual(result.shape, (2, 32))

    def test_fallback_dtype_int64(self):
        enc = self._make_lang_encoder_no_osam()
        result = enc._fallback_tokenize(["cat"])
        self.assertEqual(result.dtype, np.int64)

    def test_fallback_custom_context_length(self):
        enc = self._make_lang_encoder_no_osam()
        result = enc._fallback_tokenize(["test"], context_length=16)
        self.assertEqual(result.shape, (1, 16))


# ---------------------------------------------------------------------------
# SAM3ImageDecoder dummy language inputs
# ---------------------------------------------------------------------------

FULL_INPUT_NAMES = [
    "original_height", "original_width",
    "vision_pos_enc_0", "vision_pos_enc_1", "vision_pos_enc_2",
    "backbone_fpn_0", "backbone_fpn_1", "backbone_fpn_2",
    "language_mask", "language_features", "language_embeds",
    "box_coords", "box_labels", "box_masks",
]

SIMPLIFIED_INPUT_NAMES = [
    "original_height", "original_width",
    "vision_pos_enc_2",
    "backbone_fpn_0", "backbone_fpn_1", "backbone_fpn_2",
    "language_mask", "language_features",
    "box_coords", "box_labels", "box_masks",
]


def _make_decoder(input_names, H=100, W=100):
    dec = SAM3ImageDecoder.__new__(SAM3ImageDecoder)
    dec.session = _make_decoder_session(input_names, H, W)
    dec.input_names = input_names
    return dec


def _dummy_embedding(H=100, W=100):
    return {
        "original_size": (H, W),
        "vision_pos_enc_0": np.zeros((1, 64, 64, 256), np.float32),
        "vision_pos_enc_1": np.zeros((1, 32, 32, 256), np.float32),
        "vision_pos_enc_2": np.zeros((1, 16, 16, 256), np.float32),
        "backbone_fpn_0": np.zeros((1, 256, 64, 64), np.float32),
        "backbone_fpn_1": np.zeros((1, 256, 32, 32), np.float32),
        "backbone_fpn_2": np.zeros((1, 256, 16, 16), np.float32),
        "language_mask": None,
        "language_features": None,
        "language_embeds": None,
    }


class TestSAM3ImageDecoderDummyInputs(unittest.TestCase):

    def test_dummy_language_mask_shape(self):
        dec = _make_decoder(FULL_INPUT_NAMES)
        emb = _dummy_embedding()
        dec(*([emb["original_size"]] +
              [emb[k] for k in ("vision_pos_enc_0", "vision_pos_enc_1", "vision_pos_enc_2",
                                "backbone_fpn_0", "backbone_fpn_1", "backbone_fpn_2")] +
              [emb["language_mask"], emb["language_features"], emb["language_embeds"]] +
              [np.zeros((1, 1, 4), np.float32),
               np.zeros((1, 1), np.int64),
               np.ones((1, 1), np.bool_)]))
        # Check that the session was called with the correct dummy language_mask shape
        call_kwargs = dec.session.run.call_args[0][1]
        self.assertIn("language_mask", call_kwargs)
        self.assertEqual(call_kwargs["language_mask"].shape, (1, 32))
        self.assertIn("language_features", call_kwargs)
        self.assertEqual(call_kwargs["language_features"].shape, (32, 1, 256))
        self.assertIn("language_embeds", call_kwargs)
        self.assertEqual(call_kwargs["language_embeds"].shape, (32, 1, 1024))

    def test_real_language_inputs_not_replaced(self):
        dec = _make_decoder(FULL_INPUT_NAMES)
        real_mask = np.ones((1, 32), dtype=np.bool_)
        real_features = np.ones((32, 1, 256), dtype=np.float32)
        real_embeds = np.ones((32, 1, 1024), dtype=np.float32)
        emb = _dummy_embedding()
        emb["language_mask"] = real_mask
        emb["language_features"] = real_features
        emb["language_embeds"] = real_embeds

        dec(*([emb["original_size"]] +
              [emb[k] for k in ("vision_pos_enc_0", "vision_pos_enc_1", "vision_pos_enc_2",
                                "backbone_fpn_0", "backbone_fpn_1", "backbone_fpn_2")] +
              [real_mask, real_features, real_embeds] +
              [np.zeros((1, 1, 4), np.float32),
               np.zeros((1, 1), np.int64),
               np.ones((1, 1), np.bool_)]))
        call_kwargs = dec.session.run.call_args[0][1]
        np.testing.assert_array_equal(call_kwargs["language_mask"], real_mask)

    def test_simplified_model_skips_missing_inputs(self):
        """onnxsim-simplified decoder may not have vision_pos_enc_0/1."""
        dec = _make_decoder(SIMPLIFIED_INPUT_NAMES)
        emb = _dummy_embedding()
        dec(*([emb["original_size"]] +
              [emb[k] for k in ("vision_pos_enc_0", "vision_pos_enc_1", "vision_pos_enc_2",
                                "backbone_fpn_0", "backbone_fpn_1", "backbone_fpn_2")] +
              [None, None, None] +
              [np.zeros((1, 1, 4), np.float32),
               np.zeros((1, 1), np.int64),
               np.ones((1, 1), np.bool_)]))
        call_kwargs = dec.session.run.call_args[0][1]
        # These keys were removed by onnxsim; they must not appear in the call
        self.assertNotIn("vision_pos_enc_0", call_kwargs)
        self.assertNotIn("vision_pos_enc_1", call_kwargs)
        self.assertNotIn("language_embeds", call_kwargs)

    def test_returns_masks_scores_boxes_order(self):
        """__call__ should return (masks, scores, boxes)."""
        dec = _make_decoder(FULL_INPUT_NAMES)
        emb = _dummy_embedding()
        masks, scores, boxes = dec(
            *([emb["original_size"]] +
              [emb[k] for k in ("vision_pos_enc_0", "vision_pos_enc_1", "vision_pos_enc_2",
                                "backbone_fpn_0", "backbone_fpn_1", "backbone_fpn_2")] +
              [None, None, None] +
              [np.zeros((1, 1, 4), np.float32),
               np.zeros((1, 1), np.int64),
               np.ones((1, 1), np.bool_)]))
        # The session returns [boxes, scores, masks]; __call__ returns (masks, scores, boxes)
        self.assertEqual(masks.shape[-2:], (100, 100))  # (N, 1, H, W)
        self.assertEqual(scores.shape, (2,))
        self.assertEqual(boxes.shape, (2, 4))


# ---------------------------------------------------------------------------
# SegmentAnything3ONNX.predict_masks  (coordinate computation)
# ---------------------------------------------------------------------------

class TestSAM3PredictMasksCoords(unittest.TestCase):
    """Test that predict_masks converts prompts to correct normalized coords."""

    def _make_model(self, H=200, W=400):
        model = SegmentAnything3ONNX.__new__(SegmentAnything3ONNX)
        model.image_encoder = MagicMock()
        model.language_encoder = None
        dec = _make_decoder(FULL_INPUT_NAMES, H=H, W=W)
        model.decoder = dec
        return model, H, W

    def _embedding(self, H, W):
        emb = _dummy_embedding(H, W)
        emb["original_size"] = (H, W)
        return emb

    def test_rectangle_coords_normalized(self):
        model, H, W = self._make_model()
        emb = self._embedding(H, W)
        x1, y1, x2, y2 = 40, 20, 200, 100
        model.predict_masks(emb, [{"type": "rectangle", "data": [x1, y1, x2, y2]}],
                            confidence_threshold=0.0)
        call_kwargs = model.decoder.session.run.call_args[0][1]
        box_coords = call_kwargs["box_coords"]  # (1, 1, 4)
        cx, cy, w, h = box_coords[0, 0]
        self.assertAlmostEqual(cx, (x1 + x2) / 2.0 / W, places=5)
        self.assertAlmostEqual(cy, (y1 + y2) / 2.0 / H, places=5)
        self.assertAlmostEqual(w, (x2 - x1) / W, places=5)
        self.assertAlmostEqual(h, (y2 - y1) / H, places=5)

    def test_rectangle_box_mask_false(self):
        model, H, W = self._make_model()
        emb = self._embedding(H, W)
        model.predict_masks(emb, [{"type": "rectangle", "data": [10, 10, 100, 100]}],
                            confidence_threshold=0.0)
        call_kwargs = model.decoder.session.run.call_args[0][1]
        self.assertFalse(call_kwargs["box_masks"][0, 0])

    def test_point_coords_normalized(self):
        model, H, W = self._make_model(H=300, W=600)
        emb = self._embedding(300, 600)
        px, py = 120, 90
        model.predict_masks(emb, [{"type": "point", "data": [px, py], "label": 1}],
                            confidence_threshold=0.0)
        call_kwargs = model.decoder.session.run.call_args[0][1]
        box_coords = call_kwargs["box_coords"]  # (1, 1, 4)
        cx, cy, bw, bh = box_coords[0, 0]
        self.assertAlmostEqual(cx, px / 600, places=5)
        self.assertAlmostEqual(cy, py / 300, places=5)
        # Point is represented as a 1%-size box
        self.assertAlmostEqual(bw, 0.01, places=5)
        self.assertAlmostEqual(bh, 0.01, places=5)

    def test_point_box_mask_false(self):
        model, H, W = self._make_model()
        emb = self._embedding(H, W)
        model.predict_masks(emb, [{"type": "point", "data": [50, 50], "label": 1}],
                            confidence_threshold=0.0)
        call_kwargs = model.decoder.session.run.call_args[0][1]
        self.assertFalse(call_kwargs["box_masks"][0, 0])

    def test_empty_prompt_uses_dummy_box(self):
        """No geometric prompt → dummy box with box_masks=True (text-only mode)."""
        model, H, W = self._make_model()
        emb = self._embedding(H, W)
        model.predict_masks(emb, [], confidence_threshold=0.0)
        call_kwargs = model.decoder.session.run.call_args[0][1]
        self.assertTrue(call_kwargs["box_masks"][0, 0])

    def test_confidence_threshold_filtering(self):
        """Masks with score ≤ threshold should be removed."""
        model, H, W = self._make_model()
        emb = self._embedding(H, W)
        # Decoder returns scores [0.9, 0.3]
        result = model.predict_masks(emb, [], confidence_threshold=0.5)
        # Only first mask (score=0.9) should pass
        self.assertEqual(result.shape[0], 1)

    def test_all_filtered_returns_empty(self):
        model, H, W = self._make_model()
        emb = self._embedding(H, W)
        result = model.predict_masks(emb, [], confidence_threshold=0.99)
        self.assertEqual(result.shape[0], 0)

    def test_no_threshold_returns_all(self):
        model, H, W = self._make_model()
        emb = self._embedding(H, W)
        result = model.predict_masks(emb, [], confidence_threshold=0.0)
        # Both masks (scores 0.9 and 0.3) pass threshold > 0.0
        self.assertEqual(result.shape[0], 2)


# ---------------------------------------------------------------------------
# SegmentAnything3ONNX.encode (no language encoder)
# ---------------------------------------------------------------------------

class TestSAM3EncodeNoLanguageEncoder(unittest.TestCase):

    def test_encode_sets_language_keys_to_none(self):
        model = SegmentAnything3ONNX.__new__(SegmentAnything3ONNX)
        enc_session = _make_encoder_session([3, 1008, 1008])
        with patch("onnxruntime.InferenceSession", return_value=enc_session):
            model.image_encoder = SAM3ImageEncoder.__new__(SAM3ImageEncoder)
            model.image_encoder.session = enc_session
            model.image_encoder.input_name = "image"
            model.image_encoder.input_shape = [3, 1008, 1008]
            model.image_encoder.input_type = "tensor(uint8)"
            model.image_encoder.input_height = 1008
            model.image_encoder.input_width = 1008
        model.language_encoder = None
        model.decoder = MagicMock()

        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        embedding = model.encode(image, text_prompt=None)

        self.assertIsNone(embedding["language_mask"])
        self.assertIsNone(embedding["language_features"])
        self.assertIsNone(embedding["language_embeds"])

    def test_encode_returns_all_required_keys(self):
        model = SegmentAnything3ONNX.__new__(SegmentAnything3ONNX)
        enc_session = _make_encoder_session([3, 1008, 1008])
        model.image_encoder = SAM3ImageEncoder.__new__(SAM3ImageEncoder)
        model.image_encoder.session = enc_session
        model.image_encoder.input_name = "image"
        model.image_encoder.input_shape = [3, 1008, 1008]
        model.image_encoder.input_type = "tensor(uint8)"
        model.image_encoder.input_height = 1008
        model.image_encoder.input_width = 1008
        model.language_encoder = None
        model.decoder = MagicMock()

        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        embedding = model.encode(image)

        required = ["vision_pos_enc_0", "vision_pos_enc_1", "vision_pos_enc_2",
                    "backbone_fpn_0", "backbone_fpn_1", "backbone_fpn_2",
                    "original_size", "language_mask", "language_features", "language_embeds"]
        for key in required:
            self.assertIn(key, embedding, f"Missing key: {key}")

    def test_encode_original_size_matches_image(self):
        model = SegmentAnything3ONNX.__new__(SegmentAnything3ONNX)
        enc_session = _make_encoder_session([3, 1008, 1008])
        model.image_encoder = SAM3ImageEncoder.__new__(SAM3ImageEncoder)
        model.image_encoder.session = enc_session
        model.image_encoder.input_name = "image"
        model.image_encoder.input_shape = [3, 1008, 1008]
        model.image_encoder.input_type = "tensor(uint8)"
        model.image_encoder.input_height = 1008
        model.image_encoder.input_width = 1008
        model.language_encoder = None
        model.decoder = MagicMock()

        H, W = 480, 640
        image = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
        embedding = model.encode(image)

        self.assertEqual(embedding["original_size"], (H, W))


if __name__ == "__main__":
    unittest.main()
