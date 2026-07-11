"""Tests for YOLO label I/O with polygon (segmentation) support."""

import os
import tempfile
import unittest

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from anylabeling.views.labeling.yolo_io import (
    read_yolo_label,
    write_yolo_label,
)


def _temp_path(suffix=".txt"):
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    return path


class TestReadYoloLabel(unittest.TestCase):
    """Tests for read_yolo_label() — detection and segmentation formats."""

    def setUp(self):
        self.img_w = 640
        self.img_h = 480
        self.id_to_label = {0: "person", 1: "car"}
        self.tmp_path = _temp_path()

    def tearDown(self):
        os.unlink(self.tmp_path)

    def _write(self, lines):
        with open(self.tmp_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def test_read_detection_format(self):
        """Detection format (5 values) → rectangle shape."""
        self._write(["0 0.5 0.5 0.4 0.3"])
        shapes = read_yolo_label(
            self.tmp_path, self.img_w, self.img_h, self.id_to_label
        )
        self.assertEqual(len(shapes), 1)
        self.assertEqual(shapes[0]["shape_type"], "rectangle")
        self.assertEqual(shapes[0]["label"], "person")
        self.assertEqual(len(shapes[0]["points"]), 2)

    def test_read_segmentation_format_three_points(self):
        """Segmentation format (7 values = 3 points) → polygon shape."""
        self._write(["0 0.1 0.1 0.5 0.1 0.3 0.4"])
        shapes = read_yolo_label(
            self.tmp_path, self.img_w, self.img_h, self.id_to_label
        )
        self.assertEqual(len(shapes), 1)
        self.assertEqual(shapes[0]["shape_type"], "polygon")
        self.assertEqual(shapes[0]["label"], "person")
        self.assertEqual(len(shapes[0]["points"]), 3)

    def test_read_segmentation_format_five_points(self):
        """Segmentation format (11 values = 5 points) → polygon shape."""
        self._write(["1 0.1 0.1 0.5 0.1 0.5 0.4 0.3 0.5 0.1 0.4"])
        shapes = read_yolo_label(
            self.tmp_path, self.img_w, self.img_h, self.id_to_label
        )
        self.assertEqual(len(shapes), 1)
        self.assertEqual(shapes[0]["shape_type"], "polygon")
        self.assertEqual(shapes[0]["label"], "car")
        self.assertEqual(len(shapes[0]["points"]), 5)

    def test_read_mixed_formats(self):
        """A single file with both detection and segmentation lines."""
        self._write([
            "0 0.5 0.5 0.4 0.3",
            "1 0.1 0.1 0.5 0.1 0.3 0.4",
        ])
        shapes = read_yolo_label(
            self.tmp_path, self.img_w, self.img_h, self.id_to_label
        )
        self.assertEqual(len(shapes), 2)
        self.assertEqual(shapes[0]["shape_type"], "rectangle")
        self.assertEqual(shapes[1]["shape_type"], "polygon")

    def test_read_normalised_coordinates(self):
        """Check coordinate conversion from normalised → absolute pixels."""
        self._write(["0 0.5 0.5 0.4 0.3"])
        shapes = read_yolo_label(
            self.tmp_path, self.img_w, self.img_h, self.id_to_label
        )
        # x_center=0.5*640=320, width=0.4*640=256 → x1=320-128=192, x2=320+128=448
        # y_center=0.5*480=240, height=0.3*480=144 → y1=240-72=168, y2=240+72=312
        self.assertAlmostEqual(shapes[0]["points"][0][0], 192.0)
        self.assertAlmostEqual(shapes[0]["points"][0][1], 168.0)
        self.assertAlmostEqual(shapes[0]["points"][1][0], 448.0)
        self.assertAlmostEqual(shapes[0]["points"][1][1], 312.0)

    def test_read_normalised_coordinates_polygon(self):
        """Check polygon coordinate conversion."""
        self._write(["0 0.1 0.2 0.5 0.2 0.3 0.6"])
        shapes = read_yolo_label(
            self.tmp_path, self.img_w, self.img_h, self.id_to_label
        )
        # x: 0.1*640=64, 0.5*640=320, 0.3*640=192
        # y: 0.2*480=96, 0.2*480=96, 0.6*480=288
        expected = [[64.0, 96.0], [320.0, 96.0], [192.0, 288.0]]
        for i, (x, y) in enumerate(expected):
            with self.subTest(point=i):
                self.assertAlmostEqual(shapes[0]["points"][i][0], x)
                self.assertAlmostEqual(shapes[0]["points"][i][1], y)

    def test_read_populates_label_to_id(self):
        """label_to_id is populated for new labels."""
        self._write(["3 0.5 0.5 0.4 0.3"])
        label_to_id = {}
        shapes = read_yolo_label(
            self.tmp_path, self.img_w, self.img_h,
            self.id_to_label, label_to_id=label_to_id,
        )
        self.assertEqual(len(shapes), 1)
        self.assertEqual(shapes[0]["label"], "class_3")
        self.assertIn("class_3", label_to_id)
        self.assertEqual(label_to_id["class_3"], 3)

    def test_read_unknown_class_id(self):
        """An unseen class_id gets a fallback label."""
        self._write(["99 0.5 0.5 0.4 0.3"])
        shapes = read_yolo_label(
            self.tmp_path, self.img_w, self.img_h, self.id_to_label
        )
        self.assertEqual(shapes[0]["label"], "class_99")

    def test_read_skips_short_lines(self):
        """Lines with < 5 values are skipped."""
        self._write([
            "0 0.5 0.5 0.4 0.3",
            "1 0.1 0.2",
            "2 0.5 0.5 0.4",
        ])
        shapes = read_yolo_label(
            self.tmp_path, self.img_w, self.img_h, self.id_to_label
        )
        self.assertEqual(len(shapes), 1)

    def test_read_skips_odd_coord_count(self):
        """Segmentation-like line with odd coord count is skipped."""
        self._write(["0 0.1 0.2 0.3 0.4 0.5"])
        shapes = read_yolo_label(
            self.tmp_path, self.img_w, self.img_h, self.id_to_label
        )
        self.assertEqual(len(shapes), 0)

    def test_read_skips_comments_and_blanks(self):
        """Comments and blank lines are ignored."""
        self._write([
            "# this is a comment",
            "",
            "0 0.5 0.5 0.4 0.3",
            "  ",
            "# another comment",
        ])
        shapes = read_yolo_label(
            self.tmp_path, self.img_w, self.img_h, self.id_to_label
        )
        self.assertEqual(len(shapes), 1)

    def test_read_nonexistent_file(self):
        """Missing file returns empty list."""
        shapes = read_yolo_label(
            "/nonexistent/path.txt", self.img_w, self.img_h, self.id_to_label
        )
        self.assertEqual(shapes, [])


class TestWriteYoloLabel(unittest.TestCase):
    """Tests for write_yolo_label() — mixed-mode output."""

    def setUp(self):
        self.img_w = 640
        self.img_h = 480
        self.tmp_path = _temp_path()

    def tearDown(self):
        os.unlink(self.tmp_path)

    def _read_lines(self):
        with open(self.tmp_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    def test_write_rectangle(self):
        """Rectangle shapes → detection format (5 values)."""
        shapes = [{
            "label": "person", "shape_type": "rectangle",
            "points": [[100, 80], [500, 320]],
            "text": "", "group_id": None, "flags": {},
        }]
        label_to_id = {"person": 0}
        write_yolo_label(self.tmp_path, shapes, self.img_w, self.img_h, label_to_id)
        lines = self._read_lines()
        self.assertEqual(len(lines), 1)
        parts = lines[0].split()
        self.assertEqual(len(parts), 5)
        self.assertEqual(parts[0], "0")

    def test_write_polygon(self):
        """Polygon shapes → segmentation format (variable values)."""
        shapes = [{
            "label": "person", "shape_type": "polygon",
            "points": [[100, 80], [500, 80], [300, 320]],
            "text": "", "group_id": None, "flags": {},
        }]
        label_to_id = {"person": 0}
        write_yolo_label(self.tmp_path, shapes, self.img_w, self.img_h, label_to_id)
        lines = self._read_lines()
        self.assertEqual(len(lines), 1)
        parts = lines[0].split()
        self.assertEqual(len(parts), 7)  # class_id + 6 coords = 7
        self.assertEqual(parts[0], "0")

    def test_write_mixed_shapes(self):
        """Mixed rectangle + polygon → mixed output."""
        shapes = [
            {
                "label": "person", "shape_type": "rectangle",
                "points": [[100, 80], [500, 320]],
                "text": "", "group_id": None, "flags": {},
            },
            {
                "label": "car", "shape_type": "polygon",
                "points": [[50, 60], [200, 60], [200, 180], [50, 180]],
                "text": "", "group_id": None, "flags": {},
            },
        ]
        label_to_id = {"person": 0, "car": 1}
        write_yolo_label(self.tmp_path, shapes, self.img_w, self.img_h, label_to_id)
        lines = self._read_lines()
        self.assertEqual(len(lines), 2)
        parts0 = lines[0].split()
        parts1 = lines[1].split()
        self.assertEqual(len(parts0), 5)
        self.assertEqual(len(parts1), 9)  # class_id + 8 coords = 9
        self.assertEqual(parts0[0], "0")
        self.assertEqual(parts1[0], "1")

    def test_write_auto_assigns_new_ids(self):
        """Unseen labels get the next unused class ID."""
        shapes = [{
            "label": "dog", "shape_type": "rectangle",
            "points": [[100, 80], [500, 320]],
            "text": "", "group_id": None, "flags": {},
        }]
        label_to_id = {"person": 0, "car": 1}
        write_yolo_label(self.tmp_path, shapes, self.img_w, self.img_h, label_to_id)
        self.assertEqual(label_to_id["dog"], 2)

    def test_write_skips_circle_shape(self):
        """Non-rectangle/non-polygon shapes are skipped."""
        shapes = [
            {
                "label": "person", "shape_type": "rectangle",
                "points": [[100, 80], [500, 320]],
                "text": "", "group_id": None, "flags": {},
            },
            {
                "label": "dot", "shape_type": "point",
                "points": [[200, 150]],
                "text": "", "group_id": None, "flags": {},
            },
        ]
        label_to_id = {"person": 0}
        write_yolo_label(self.tmp_path, shapes, self.img_w, self.img_h, label_to_id)
        lines = self._read_lines()
        self.assertEqual(len(lines), 1)
        self.assertNotIn("dot", " ".join(lines))

    def test_write_empty_shapes(self):
        """Empty shapes list produces empty (newline-free) file."""
        write_yolo_label(self.tmp_path, [], self.img_w, self.img_h, {})
        with open(self.tmp_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertEqual(content, "")


class TestYoloRoundTrip(unittest.TestCase):
    """End-to-end round-trip: read → write → read yields same data."""

    def setUp(self):
        self.img_w = 640
        self.img_h = 480
        self.id_to_label = {0: "person", 1: "car", 2: "dog"}
        self.tmp_path = _temp_path()

    def tearDown(self):
        os.unlink(self.tmp_path)

    def test_roundtrip_detection_only(self):
        """Detection-format file round-trips with rectangle shapes."""
        original_lines = [
            "0 0.5 0.5 0.4 0.3",
            "1 0.2 0.3 0.1 0.2",
        ]
        with open(self.tmp_path, "w", encoding="utf-8") as f:
            f.write("\n".join(original_lines) + "\n")

        shapes = read_yolo_label(
            self.tmp_path, self.img_w, self.img_h, self.id_to_label
        )
        label_to_id = {"person": 0, "car": 1}
        write_yolo_label(
            self.tmp_path, shapes, self.img_w, self.img_h, label_to_id
        )

        shapes2 = read_yolo_label(
            self.tmp_path, self.img_w, self.img_h, self.id_to_label
        )
        self.assertEqual(len(shapes), len(shapes2))
        for s1, s2 in zip(shapes, shapes2):
            self.assertEqual(s1["shape_type"], s2["shape_type"])
            self.assertEqual(s1["label"], s2["label"])
            for p1, p2 in zip(s1["points"], s2["points"]):
                self.assertAlmostEqual(p1[0], p2[0], places=5)
                self.assertAlmostEqual(p1[1], p2[1], places=5)

    def test_roundtrip_segmentation_only(self):
        """Segmentation-format file round-trips with polygon shapes."""
        original_lines = [
            "0 0.1 0.1 0.5 0.1 0.3 0.4",
            "2 0.2 0.2 0.6 0.2 0.6 0.6 0.2 0.6",
        ]
        with open(self.tmp_path, "w", encoding="utf-8") as f:
            f.write("\n".join(original_lines) + "\n")

        shapes = read_yolo_label(
            self.tmp_path, self.img_w, self.img_h, self.id_to_label
        )
        self.assertEqual(len(shapes), 2)
        label_to_id = {"person": 0, "dog": 2}
        write_yolo_label(
            self.tmp_path, shapes, self.img_w, self.img_h, label_to_id
        )

        shapes2 = read_yolo_label(
            self.tmp_path, self.img_w, self.img_h, self.id_to_label
        )
        self.assertEqual(len(shapes), len(shapes2))
        for i, (s1, s2) in enumerate(zip(shapes, shapes2)):
            with self.subTest(shape=i):
                self.assertEqual(s1["shape_type"], s2["shape_type"])
                self.assertEqual(s1["label"], s2["label"])
                self.assertEqual(len(s1["points"]), len(s2["points"]))
                for j, (p1, p2) in enumerate(zip(s1["points"], s2["points"])):
                    with self.subTest(point=j):
                        self.assertAlmostEqual(p1[0], p2[0], places=5)
                        self.assertAlmostEqual(p1[1], p2[1], places=5)

    def test_roundtrip_mixed(self):
        """Mixed detection+segmentation file round-trips correctly."""
        original_lines = [
            "0 0.5 0.5 0.4 0.3",
            "1 0.1 0.1 0.5 0.1 0.3 0.4",
        ]
        with open(self.tmp_path, "w", encoding="utf-8") as f:
            f.write("\n".join(original_lines) + "\n")

        shapes = read_yolo_label(
            self.tmp_path, self.img_w, self.img_h, self.id_to_label
        )
        self.assertEqual(len(shapes), 2)
        self.assertEqual(shapes[0]["shape_type"], "rectangle")
        self.assertEqual(shapes[1]["shape_type"], "polygon")

        label_to_id = {"person": 0, "car": 1}
        write_yolo_label(
            self.tmp_path, shapes, self.img_w, self.img_h, label_to_id
        )

        shapes2 = read_yolo_label(
            self.tmp_path, self.img_w, self.img_h, self.id_to_label
        )
        self.assertEqual(len(shapes), len(shapes2))
        for s1, s2 in zip(shapes, shapes2):
            self.assertEqual(s1["shape_type"], s2["shape_type"])
            self.assertEqual(len(s1["points"]), len(s2["points"]))


if __name__ == "__main__":
    unittest.main()
