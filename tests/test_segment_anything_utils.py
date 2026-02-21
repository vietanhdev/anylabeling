"""Tests for SegmentAnything.detect_model_variant and post_process.

PyQt5 is mocked out so no display is required.
"""
import sys
import os
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

# ---------------------------------------------------------------------------
# Stub out PyQt5 and anylabeling UI modules before importing segment_anything
# ---------------------------------------------------------------------------
_MOCK_MODS = [
    "PyQt5", "PyQt5.QtCore", "PyQt5.QtGui", "PyQt5.QtWidgets",
    "anylabeling.utils",
    "anylabeling.views.labeling.shape",
    "anylabeling.views.labeling.utils",
    "anylabeling.views.labeling.utils.opencv",
    "qimage2ndarray",
]
for _mod in _MOCK_MODS:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

# QCoreApplication.translate must return the first arg (msgid) so strings work
import PyQt5.QtCore as _qt_core
_qt_core.QCoreApplication.translate = lambda ctx, s: s

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import onnx  # real onnx
from anylabeling.services.auto_labeling.segment_anything import SegmentAnything


# ---------------------------------------------------------------------------
# detect_model_variant
# ---------------------------------------------------------------------------

def _mock_onnx_model(input_names):
    """Return a mock onnx.ModelProto with the given decoder input names."""
    # Do NOT use spec= here — onnx.ModelProto is a protobuf class and its
    # attributes are not enumerable, so MagicMock(spec=...) blocks them.
    model = MagicMock()
    inputs = []
    for name in input_names:
        inp = MagicMock()
        inp.name = name
        inputs.append(inp)
    model.graph.input = inputs
    return model


class TestDetectModelVariant(unittest.TestCase):

    def _instance(self):
        """Get a bare SegmentAnything instance without __init__ side effects."""
        return SegmentAnything.__new__(SegmentAnything)

    @patch("anylabeling.services.auto_labeling.segment_anything.onnx.load")
    def test_sam3_detected_by_backbone_fpn(self, mock_load):
        mock_load.return_value = _mock_onnx_model(
            ["backbone_fpn_0", "backbone_fpn_1", "backbone_fpn_2",
             "vision_pos_enc_2", "box_coords"]
        )
        result = self._instance().detect_model_variant("any_path.onnx")
        self.assertEqual(result, "sam3")

    @patch("anylabeling.services.auto_labeling.segment_anything.onnx.load")
    def test_sam3_detected_by_language_mask(self, mock_load):
        mock_load.return_value = _mock_onnx_model(
            ["language_mask", "vision_pos_enc_2", "box_coords"]
        )
        result = self._instance().detect_model_variant("any_path.onnx")
        self.assertEqual(result, "sam3")

    @patch("anylabeling.services.auto_labeling.segment_anything.onnx.load")
    def test_sam2_detected_by_high_res_feats(self, mock_load):
        mock_load.return_value = _mock_onnx_model(
            ["high_res_feats_0", "high_res_feats_1", "image_embed"]
        )
        result = self._instance().detect_model_variant("any_path.onnx")
        self.assertEqual(result, "sam2")

    @patch("anylabeling.services.auto_labeling.segment_anything.onnx.load")
    def test_sam1_detected_as_fallback(self, mock_load):
        mock_load.return_value = _mock_onnx_model(
            ["image_embeddings", "point_coords", "point_labels"]
        )
        result = self._instance().detect_model_variant("any_path.onnx")
        self.assertEqual(result, "sam")

    @patch("anylabeling.services.auto_labeling.segment_anything.onnx.load")
    def test_priority_sam3_over_sam2(self, mock_load):
        """If both SAM3 and SAM2 inputs are present, SAM3 wins (checked first)."""
        mock_load.return_value = _mock_onnx_model(
            ["backbone_fpn_0", "high_res_feats_0"]
        )
        result = self._instance().detect_model_variant("any_path.onnx")
        self.assertEqual(result, "sam3")


# ---------------------------------------------------------------------------
# post_process – mask conversion and shape creation
# ---------------------------------------------------------------------------

class _FakeShape:
    """Minimal stand-in for anylabeling Shape for post_process tests."""
    def __init__(self, **kwargs):
        self.points = []
        self.shape_type = None
        self.closed = False
        self.fill_color = None
        self.line_color = None
        self.line_width = None
        self.label = None
        self.selected = None
        self.flags = kwargs.get("flags", {})

    def add_point(self, pt):
        self.points.append(pt)


class _FakeQPointF:
    def __init__(self, x, y):
        self.x_ = x
        self.y_ = y


def _sa_with_mocks(output_mode="polygon"):
    """Create a minimal SegmentAnything instance suitable for post_process."""
    sa = SegmentAnything.__new__(SegmentAnything)
    sa.output_mode = output_mode

    # Patch the Shape class and QPointF inside the module
    import anylabeling.services.auto_labeling.segment_anything as _sa_mod
    _sa_mod.Shape = _FakeShape
    import PyQt5.QtCore as _qtc
    _qtc.QPointF = _FakeQPointF

    return sa


class TestPostProcess(unittest.TestCase):

    def _simple_mask(self, H=100, W=100, fill_value=1):
        """Create a solid white rectangle mask centred in a black image."""
        mask = np.zeros((H, W), dtype=np.float32)
        mask[20:80, 20:80] = fill_value
        return mask

    def test_bool_mask_produces_shapes(self):
        sa = _sa_with_mocks("polygon")
        mask = self._simple_mask().astype(np.bool_)
        shapes = sa.post_process(mask)
        self.assertGreater(len(shapes), 0)

    def test_float_mask_produces_shapes(self):
        sa = _sa_with_mocks("polygon")
        mask = self._simple_mask()  # float32, values 0 or 1
        shapes = sa.post_process(mask)
        self.assertGreater(len(shapes), 0)

    def test_uint8_mask_produces_shapes(self):
        sa = _sa_with_mocks("polygon")
        mask = (self._simple_mask() * 255).astype(np.uint8)
        shapes = sa.post_process(mask)
        self.assertGreater(len(shapes), 0)

    def test_blank_mask_returns_no_shapes(self):
        sa = _sa_with_mocks("polygon")
        mask = np.zeros((100, 100), dtype=np.float32)
        shapes = sa.post_process(mask)
        self.assertEqual(len(shapes), 0)

    def test_polygon_shape_type(self):
        sa = _sa_with_mocks("polygon")
        mask = self._simple_mask()
        shapes = sa.post_process(mask)
        for s in shapes:
            self.assertEqual(s.shape_type, "polygon")

    def test_rectangle_shape_type(self):
        sa = _sa_with_mocks("rectangle")
        mask = self._simple_mask()
        shapes = sa.post_process(mask)
        self.assertGreater(len(shapes), 0)
        self.assertEqual(shapes[0].shape_type, "rectangle")

    def test_polygon_has_at_least_3_points(self):
        sa = _sa_with_mocks("polygon")
        mask = self._simple_mask()
        shapes = sa.post_process(mask)
        for s in shapes:
            self.assertGreaterEqual(len(s.points), 3)

    def test_label_propagated(self):
        sa = _sa_with_mocks("polygon")
        mask = self._simple_mask()
        shapes = sa.post_process(mask, label="my_object")
        for s in shapes:
            self.assertEqual(s.label, "my_object")

    def test_default_label(self):
        sa = _sa_with_mocks("polygon")
        mask = self._simple_mask()
        shapes = sa.post_process(mask)
        for s in shapes:
            self.assertEqual(s.label, "AUTOLABEL_OBJECT")

    def test_mask_values_255_also_detected(self):
        sa = _sa_with_mocks("polygon")
        mask = (self._simple_mask() * 255).astype(np.float32)
        shapes = sa.post_process(mask)
        self.assertGreater(len(shapes), 0)

    def test_rectangle_mode_returns_one_shape(self):
        """Rectangle mode merges all contours into one bounding box."""
        sa = _sa_with_mocks("rectangle")
        mask = self._simple_mask()
        shapes = sa.post_process(mask)
        self.assertEqual(len(shapes), 1)

    def test_rectangle_shape_has_two_points(self):
        sa = _sa_with_mocks("rectangle")
        mask = self._simple_mask()
        shapes = sa.post_process(mask)
        self.assertEqual(len(shapes[0].points), 2)


if __name__ == "__main__":
    unittest.main()
