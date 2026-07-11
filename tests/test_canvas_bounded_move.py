"""Regression tests for Canvas.bounded_move_shapes edge-clamping.

Covers two bugs in the same lines, found while fixing #237/#238/#240/#241:
  1. QPoint/QPointF TypeError when a selected shape is dragged past the
     pixmap border (fixed in #241).
  2. int() truncation silently dropping sub-pixel corrections, and a
     related off-by-one against out_off_pixmap's (w-1, h-1) bound, both
     of which could leave a shape slightly outside the pixmap after the
     "correction" ran.

Run:
    QT_QPA_PLATFORM=offscreen python -m unittest tests.test_canvas_bounded_move -v
"""
import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6 import QtCore, QtGui
from PyQt6.QtWidgets import QApplication

_APP = QApplication.instance() or QApplication([])

from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.widgets.canvas import Canvas


class TestBoundedMoveShapes(unittest.TestCase):
    def _make_canvas(self, w=100, h=100):
        canvas = Canvas(parent=None)
        canvas.pixmap = QtGui.QPixmap(w, h)
        return canvas

    def test_does_not_raise_typeerror_at_left_edge(self):
        canvas = self._make_canvas()
        shape = Shape(shape_type="polygon")
        shape.add_point(QtCore.QPointF(5, 5))
        shape.add_point(QtCore.QPointF(15, 5))
        shape.add_point(QtCore.QPointF(5, 15))
        canvas.offsets = (QtCore.QPointF(-5, -5), QtCore.QPointF(5, 5))
        canvas.prev_point = QtCore.QPointF(3, 5)
        canvas.bounded_move_shapes([shape], QtCore.QPointF(3, 5))  # must not raise

    def test_sub_pixel_overflow_at_left_top_is_corrected(self):
        canvas = self._make_canvas()
        shape = Shape(shape_type="polygon")
        shape.add_point(QtCore.QPointF(5.3, 5.3))
        canvas.offsets = (QtCore.QPointF(-5.6, -5.6), QtCore.QPointF(5.6, 5.6))
        canvas.prev_point = QtCore.QPointF(5.3, 5.3)
        pos = QtCore.QPointF(5.3, 5.3)
        self.assertTrue(canvas.out_off_pixmap(pos + canvas.offsets[0]))

        canvas.bounded_move_shapes([shape], pos)

        o1_after = canvas.prev_point + canvas.offsets[0]
        self.assertFalse(
            canvas.out_off_pixmap(o1_after),
            "sub-pixel overflow at the left/top edge must be corrected",
        )

    def test_sub_pixel_overflow_at_right_bottom_is_corrected(self):
        canvas = self._make_canvas()
        shape = Shape(shape_type="polygon")
        shape.add_point(QtCore.QPointF(94.0, 94.0))
        canvas.offsets = (QtCore.QPointF(-2.0, -2.0), QtCore.QPointF(5.3, 5.3))
        canvas.prev_point = QtCore.QPointF(94.0, 94.0)
        pos = QtCore.QPointF(94.0, 94.0)
        self.assertTrue(canvas.out_off_pixmap(pos + canvas.offsets[1]))

        canvas.bounded_move_shapes([shape], pos)

        o2_after = canvas.prev_point + canvas.offsets[1]
        self.assertFalse(
            canvas.out_off_pixmap(o2_after),
            "sub-pixel overflow at the right/bottom edge must be corrected",
        )
        # Corrected position should land exactly on the boundary out_off_pixmap
        # itself uses (w - 1), not one past it.
        self.assertAlmostEqual(o2_after.x(), canvas.pixmap.width() - 1)


if __name__ == "__main__":
    unittest.main()
