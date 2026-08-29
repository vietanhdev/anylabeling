"""Regression tests for brightness/contrast image modes."""

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import PIL.Image
import PIL.ImageEnhance
from PyQt6.QtWidgets import QApplication

_APP = QApplication.instance() or QApplication([])

from anylabeling.views.labeling.widgets.brightness_contrast_dialog import (  # noqa: E402
    BrightnessContrastDialog,
    enhance_image,
)


class TestBrightnessContrastDialog(unittest.TestCase):
    def test_adjusts_16_bit_grayscale_image_without_crashing(self):
        image = PIL.Image.fromarray(
            np.array([[0, 16384], [32768, 65535]], dtype=np.uint16)
        )
        rendered_images = []
        dialog = BrightnessContrastDialog(image, rendered_images.append)

        dialog.on_new_value(None)

        self.assertEqual(len(rendered_images), 1)
        self.assertFalse(rendered_images[0].isNull())
        self.assertEqual(rendered_images[0].width(), 2)
        self.assertEqual(rendered_images[0].height(), 2)

    def test_16_bit_identity_preserves_pixel_values(self):
        for dtype in ("<u2", ">u2"):
            with self.subTest(dtype=dtype):
                values = np.array([[0, 16384], [32768, 65535]], dtype=dtype)
                image = PIL.Image.fromarray(values)

                adjusted = enhance_image(image, brightness=1.0, contrast=1.0)

                np.testing.assert_array_equal(np.asarray(adjusted), values)

    def test_16_bit_adjustment_clips_to_unsigned_range(self):
        values = np.array([[0, 32768, 65535]], dtype=np.uint16)
        image = PIL.Image.fromarray(values)

        adjusted = enhance_image(image, brightness=2.0, contrast=1.0)

        np.testing.assert_array_equal(
            np.asarray(adjusted),
            np.array([[0, 65535, 65535]], dtype=np.uint16),
        )

    def test_rgb_adjustment_matches_existing_pillow_behavior(self):
        image = PIL.Image.fromarray(
            np.array([[[20, 40, 60], [100, 120, 140]]], dtype=np.uint8)
        )
        expected = PIL.ImageEnhance.Brightness(image).enhance(1.4)
        expected = PIL.ImageEnhance.Contrast(expected).enhance(0.6)

        adjusted = enhance_image(image, brightness=1.4, contrast=0.6)

        np.testing.assert_array_equal(np.asarray(adjusted), np.asarray(expected))


if __name__ == "__main__":
    unittest.main()
