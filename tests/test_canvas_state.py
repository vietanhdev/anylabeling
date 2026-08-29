"""Regression tests for transient Canvas state across image changes."""

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6 import QtCore, QtGui
from PyQt6.QtWidgets import QApplication

_APP = QApplication.instance() or QApplication([])

from anylabeling.views.labeling.shape import Shape  # noqa: E402
from anylabeling.views.labeling.widgets.canvas import Canvas  # noqa: E402


class TestCanvasState(unittest.TestCase):
    def test_reset_state_clears_selection_and_keyboard_move(self):
        canvas = Canvas(parent=None)
        shape = Shape(shape_type="rectangle")
        shape.add_point(QtCore.QPointF(1, 1))
        shape.add_point(QtCore.QPointF(10, 10))
        shape.selected = True

        canvas.shapes = [shape]
        canvas.selected_shapes = [shape]
        canvas.selected_shapes_copy = [shape.copy()]
        canvas.moving_shape = True

        canvas.reset_state()
        canvas.load_pixmap(QtGui.QPixmap(20, 20))

        self.assertEqual(canvas.selected_shapes, [])
        self.assertEqual(canvas.selected_shapes_copy, [])
        self.assertFalse(canvas.moving_shape)

        event = QtGui.QKeyEvent(
            QtCore.QEvent.Type.KeyRelease,
            QtCore.Qt.Key.Key_Down,
            QtCore.Qt.KeyboardModifier.NoModifier,
        )
        canvas.keyReleaseEvent(event)  # must not look up the old shape


if __name__ == "__main__":
    unittest.main()
