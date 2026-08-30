"""Tests for reliable label-file reads and overwrites."""

import io
import os
import tempfile
import unittest

from PIL import Image

from anylabeling.views.labeling.label_file import LabelFile, io_open


class TestLabelFileIO(unittest.TestCase):
    def test_io_open_closes_file_after_context(self):
        with tempfile.TemporaryDirectory() as directory:
            filename = os.path.join(directory, "labels.json")

            with io_open(filename, "w") as file:
                file.write("{}")
                opened_file = file

            self.assertTrue(opened_file.closed)

    def test_existing_label_file_can_be_overwritten_and_removed(self):
        image_buffer = io.BytesIO()
        Image.new("RGB", (2, 2)).save(image_buffer, format="PNG")

        with tempfile.TemporaryDirectory() as directory:
            filename = os.path.join(directory, "labels.json")
            label_file = LabelFile()
            for revision in (1, 2):
                label_file.save(
                    filename=filename,
                    shapes=[],
                    image_path="image.png",
                    image_height=2,
                    image_width=2,
                    image_data=image_buffer.getvalue(),
                    other_data={"revision": revision},
                )

            loaded = LabelFile(filename)
            self.assertEqual(loaded.other_data["revision"], 2)

            os.remove(filename)
            self.assertFalse(os.path.exists(filename))


if __name__ == "__main__":
    unittest.main()
