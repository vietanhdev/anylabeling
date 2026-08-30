import tempfile
import unittest
from pathlib import Path

from PyQt6.QtGui import QImage

from anylabeling.services.auto_labeling.model import Model


class TestModelImageLoading(unittest.TestCase):
    def test_corrupt_image_returns_none(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "corrupt.png"
            path.write_bytes(b"not an image")

            self.assertIsNone(Model.load_image_from_filename(str(path)))

    def test_valid_image_returns_non_null_qimage(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "valid.png"
            image = QImage(2, 2, QImage.Format.Format_RGB32)
            self.assertTrue(image.save(str(path)))

            loaded = Model.load_image_from_filename(str(path))

            self.assertIsInstance(loaded, QImage)
            self.assertFalse(loaded.isNull())


if __name__ == "__main__":
    unittest.main()
