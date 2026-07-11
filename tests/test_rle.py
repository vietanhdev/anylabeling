"""Tests for anylabeling.views.labeling.utils.encode_rle / decode_rle."""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from anylabeling.views.labeling.utils import decode_rle, encode_rle


class TestRLE(unittest.TestCase):
    def test_round_trip_simple(self):
        data = [0, 0, 0, 1, 1, 2, 2, 2, 2]
        self.assertEqual(decode_rle(encode_rle(data)), data)

    def test_round_trip_no_repeats(self):
        data = [0, 1, 2, 3, 4]
        self.assertEqual(decode_rle(encode_rle(data)), data)

    def test_empty(self):
        self.assertEqual(encode_rle([]), [])
        self.assertEqual(decode_rle([]), [])

    def test_single_run(self):
        self.assertEqual(encode_rle([5, 5, 5]), [5, 3])
        self.assertEqual(decode_rle([5, 3]), [5, 5, 5])

    def test_decode_rejects_odd_length(self):
        with self.assertRaises(ValueError):
            decode_rle([1, 2, 3])

    def test_decode_rejects_negative_count(self):
        with self.assertRaises(ValueError):
            decode_rle([1, -2])

    def test_decode_rejects_non_int_count(self):
        with self.assertRaises(ValueError):
            decode_rle([1, 2.5])


if __name__ == "__main__":
    unittest.main()
