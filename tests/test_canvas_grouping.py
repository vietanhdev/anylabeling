"""Regression tests for Canvas grouping and persisted group IDs."""

import io
import os
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image
from PyQt6 import QtCore, QtWidgets

from anylabeling.views.labeling.label_file import LabelFile
from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.widgets.canvas import Canvas

_APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def make_shape(offset, group_id=None):
    shape = Shape(label="part", shape_type="rectangle", group_id=group_id)
    shape.add_point(QtCore.QPointF(offset, offset))
    shape.add_point(QtCore.QPointF(offset + 5, offset + 5))
    shape.close()
    return shape


class TestShapeGrouping(unittest.TestCase):
    def test_grouping_marks_canvas_changed_and_is_undoable(self):
        canvas = Canvas(parent=None)
        shapes = [make_shape(1), make_shape(10)]
        canvas.load_shapes(shapes)
        canvas.selected_shapes = shapes
        changes = []
        canvas.shape_moved.connect(lambda: changes.append(True))

        canvas.group_selected_shapes()

        self.assertEqual([shape.group_id for shape in canvas.shapes], [1, 1])
        self.assertEqual(len(changes), 1)
        self.assertTrue(canvas.is_shape_restorable)

        canvas.restore_shape()
        self.assertEqual([shape.group_id for shape in canvas.shapes], [None, None])

    def test_ungrouping_updates_every_shape_in_the_selected_group(self):
        canvas = Canvas(parent=None)
        shapes = [make_shape(1, group_id=7), make_shape(10, group_id=7)]
        canvas.load_shapes(shapes)
        canvas.selected_shapes = [shapes[0]]
        changes = []
        canvas.shape_moved.connect(lambda: changes.append(True))

        canvas.ungroup_selected_shapes()

        self.assertEqual([shape.group_id for shape in canvas.shapes], [None, None])
        self.assertEqual(len(changes), 1)
        self.assertTrue(canvas.is_shape_restorable)

    def test_group_id_survives_label_file_round_trip(self):
        shapes = [
            {
                "label": "part",
                "text": "",
                "points": [[1, 1], [6, 6]],
                "group_id": 3,
                "shape_type": "rectangle",
                "flags": {},
            },
            {
                "label": "part",
                "text": "",
                "points": [[10, 10], [15, 15]],
                "group_id": 3,
                "shape_type": "rectangle",
                "flags": {},
            },
        ]
        image_buffer = io.BytesIO()
        Image.new("RGB", (20, 20)).save(image_buffer, format="PNG")

        with tempfile.TemporaryDirectory() as directory:
            filename = os.path.join(directory, "labels.json")
            LabelFile().save(
                filename=filename,
                shapes=shapes,
                image_path="image.png",
                image_height=20,
                image_width=20,
                image_data=image_buffer.getvalue(),
            )

            loaded = LabelFile(filename)

        self.assertEqual([shape["group_id"] for shape in loaded.shapes], [3, 3])


if __name__ == "__main__":
    unittest.main()
