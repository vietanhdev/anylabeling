"""Regression tests for painting transient Canvas drawing state."""

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6 import QtCore, QtGui, QtWidgets

from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.widgets.canvas import Canvas

_APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class TestCanvasPaint(unittest.TestCase):
    def tearDown(self):
        for widget in QtWidgets.QApplication.topLevelWidgets():
            widget.close()
        _APP.processEvents()

    def test_mode_change_does_not_paint_rectangle_as_polygon_preview(self):
        canvas = Canvas(parent=None)
        canvas.resize(100, 100)
        canvas.load_pixmap(QtGui.QPixmap(100, 100))
        canvas.set_fill_drawing(True)

        rectangle = Shape(shape_type="rectangle")
        rectangle.add_point(QtCore.QPointF(10, 10))
        rectangle.add_point(QtCore.QPointF(40, 40))
        canvas.current = rectangle
        canvas.line.points = [
            QtCore.QPointF(40, 40),
            QtCore.QPointF(50, 50),
        ]

        # A mode switch can occur before the transient current shape is reset.
        canvas.create_mode = "polygon"
        canvas.show()
        _APP.processEvents()

        rendered = canvas.grab()
        self.assertFalse(rendered.isNull())


if __name__ == "__main__":
    unittest.main()
