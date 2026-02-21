"""Tests for AutoLabelingResult and AutoLabelingMode."""
import sys
import os
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from anylabeling.services.auto_labeling.types import AutoLabelingResult, AutoLabelingMode


class TestAutoLabelingResult(unittest.TestCase):

    def test_default_replace_true(self):
        result = AutoLabelingResult(shapes=[])
        self.assertTrue(result.replace)

    def test_replace_false(self):
        result = AutoLabelingResult(shapes=[], replace=False)
        self.assertFalse(result.replace)

    def test_shapes_stored(self):
        shapes = ["a", "b", "c"]
        result = AutoLabelingResult(shapes=shapes)
        self.assertIs(result.shapes, shapes)

    def test_empty_shapes(self):
        result = AutoLabelingResult(shapes=[])
        self.assertEqual(result.shapes, [])


class TestAutoLabelingMode(unittest.TestCase):

    def test_class_constants(self):
        self.assertEqual(AutoLabelingMode.OBJECT, "AUTOLABEL_OBJECT")
        self.assertEqual(AutoLabelingMode.ADD, "AUTOLABEL_ADD")
        self.assertEqual(AutoLabelingMode.REMOVE, "AUTOLABEL_REMOVE")
        self.assertEqual(AutoLabelingMode.POINT, "point")
        self.assertEqual(AutoLabelingMode.RECTANGLE, "rectangle")

    def test_equality_same(self):
        a = AutoLabelingMode(AutoLabelingMode.ADD, AutoLabelingMode.POINT)
        b = AutoLabelingMode(AutoLabelingMode.ADD, AutoLabelingMode.POINT)
        self.assertEqual(a, b)

    def test_equality_different_mode(self):
        a = AutoLabelingMode(AutoLabelingMode.ADD, AutoLabelingMode.POINT)
        b = AutoLabelingMode(AutoLabelingMode.REMOVE, AutoLabelingMode.POINT)
        self.assertNotEqual(a, b)

    def test_equality_different_shape(self):
        a = AutoLabelingMode(AutoLabelingMode.ADD, AutoLabelingMode.POINT)
        b = AutoLabelingMode(AutoLabelingMode.ADD, AutoLabelingMode.RECTANGLE)
        self.assertNotEqual(a, b)

    def test_equality_non_instance(self):
        a = AutoLabelingMode(AutoLabelingMode.ADD, AutoLabelingMode.POINT)
        self.assertNotEqual(a, "not a mode")

    def test_get_default_mode(self):
        mode = AutoLabelingMode.get_default_mode()
        self.assertIsInstance(mode, AutoLabelingMode)
        self.assertEqual(mode.edit_mode, AutoLabelingMode.ADD)
        self.assertEqual(mode.shape_type, AutoLabelingMode.POINT)

    def test_none_singleton(self):
        none_mode = AutoLabelingMode.NONE
        self.assertIsInstance(none_mode, AutoLabelingMode)
        self.assertIsNone(none_mode.edit_mode)
        self.assertIsNone(none_mode.shape_type)

    def test_none_not_equal_to_default(self):
        self.assertNotEqual(AutoLabelingMode.NONE, AutoLabelingMode.get_default_mode())


if __name__ == "__main__":
    unittest.main()
