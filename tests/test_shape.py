"""Regression tests for Shape's QPainterPath caching (paint()/contains_point()).

Run:
    QT_QPA_PLATFORM=offscreen python -m unittest tests.test_shape -v
"""
import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6 import QtCore, QtGui
from PyQt6.QtWidgets import QApplication

_APP = QApplication.instance() or QApplication([])

from anylabeling.views.labeling.shape import Shape


def _make_triangle(shape_type="polygon"):
    s = Shape(label="x", shape_type=shape_type)
    s.add_point(QtCore.QPointF(0, 0))
    s.add_point(QtCore.QPointF(10, 0))
    s.add_point(QtCore.QPointF(10, 10))
    s.close()
    return s


class TestShapePaintCache(unittest.TestCase):
    def _paint(self, shape):
        img = QtGui.QImage(100, 100, QtGui.QImage.Format.Format_ARGB32)
        painter = QtGui.QPainter(img)
        shape.paint(painter)
        painter.end()

    def test_vertex_handles_appear_after_selecting_an_already_painted_shape(self):
        """Canvas repaints every shape on every frame, so in real usage a
        shape is almost always painted once unselected before the user
        selects it. Vertex handles must still show up once selected."""
        control = _make_triangle()
        control.selected = True
        self._paint(control)  # selected from the very first paint

        shape = _make_triangle()
        shape.selected = False
        self._paint(shape)  # first paint, unselected (the normal case)
        shape.selected = True  # plain attribute set, as canvas.py does
        self._paint(shape)  # second paint, now selected

        self.assertEqual(
            shape._vrtx_path.elementCount(),
            control._vrtx_path.elementCount(),
            "vertex handles must render the same whether or not the shape "
            "was painted unselected before being selected",
        )

    def test_vertex_handles_disappear_after_deselecting(self):
        shape = _make_triangle()
        shape.selected = True
        self._paint(shape)
        selected_count = shape._vrtx_path.elementCount()

        shape.selected = False
        self._paint(shape)

        self.assertLess(shape._vrtx_path.elementCount(), selected_count)

    def test_contains_point_before_paint_does_not_corrupt_later_paint(self):
        """contains_point() must not cache incompatible geometry into the
        shared _path used by paint() — regression test for a shape_type
        ("point") where make_path() and paint()'s own path diverge."""
        shape = Shape(label="pt", shape_type="point")
        shape.add_point(QtCore.QPointF(5, 5))

        shape.contains_point(QtCore.QPointF(5, 5))  # called before any paint()
        self._paint(shape)  # must not raise, and must draw the vertex

        self.assertGreater(shape._vrtx_path.elementCount(), 0)

    def test_contains_point_matches_paint_geometry_for_closed_polygon(self):
        shape = _make_triangle()
        # contains_point() before paint() ...
        inside_before = shape.contains_point(QtCore.QPointF(7, 3))
        self._paint(shape)
        # ... and after paint() must agree.
        inside_after = shape.contains_point(QtCore.QPointF(7, 3))
        self.assertEqual(inside_before, inside_after)


if __name__ == "__main__":
    unittest.main()
