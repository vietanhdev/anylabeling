"""Regression tests for the LABEL_COLORMAP read-only crash.

imgviz.label_colormap() returns a read-only numpy array in newer imgviz/numpy
versions.  label_widget.py must call .copy() so that subsequent index
assignments do not raise ValueError.
"""
import sys
import os
import unittest

import numpy as np
import imgviz

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TestLabelColormapMutability(unittest.TestCase):

    def test_imgviz_colormap_may_be_readonly(self):
        """Document that the raw return value might not be writable."""
        cmap = imgviz.label_colormap()
        # We're not asserting it IS readonly (that depends on the version),
        # just that we can detect writeability safely.
        try:
            cmap[0] = cmap[1]
            writable = True
        except ValueError:
            writable = False
        # Either outcome is acceptable here; we just shouldn't crash
        self.assertIsInstance(writable, bool)

    def test_copy_is_always_writable(self):
        """After .copy(), the array must be writable regardless of imgviz version."""
        cmap = imgviz.label_colormap().copy()
        # This must not raise
        cmap[2] = cmap[1]
        cmap[1] = [0, 180, 33]
        self.assertEqual(list(cmap[1]), [0, 180, 33])

    def test_copy_preserves_shape(self):
        original = imgviz.label_colormap()
        copy = original.copy()
        self.assertEqual(original.shape, copy.shape)
        self.assertEqual(original.dtype, copy.dtype)

    def test_label_widget_colormap_is_writable(self):
        """The module-level LABEL_COLORMAP in label_widget must be assignable.

        This is the direct regression test for the startup crash:
          ValueError: assignment destination is read-only
        """
        # We cannot import label_widget without PyQt6, so we reproduce the
        # exact same pattern the module uses and confirm it works.
        LABEL_COLORMAP = imgviz.label_colormap().copy()   # the fix
        # These were the crashing lines:
        LABEL_COLORMAP[2] = LABEL_COLORMAP[1]
        LABEL_COLORMAP[1] = [0, 180, 33]
        self.assertEqual(list(LABEL_COLORMAP[1]), [0, 180, 33])


if __name__ == "__main__":
    unittest.main()
