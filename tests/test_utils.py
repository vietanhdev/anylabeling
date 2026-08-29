"""Tests for background worker cleanup."""

import unittest
from unittest import mock

from anylabeling.utils import GenericWorker


class TestGenericWorker(unittest.TestCase):
    def test_finished_is_emitted_when_task_raises(self):
        finished = []

        def fail():
            raise RuntimeError("background failure")

        worker = GenericWorker(fail)
        worker.finished.connect(lambda: finished.append(True))

        with mock.patch("anylabeling.utils.logging.exception") as log_exception:
            worker.run()

        self.assertEqual(finished, [True])
        log_exception.assert_called_once_with("Unhandled error in background task")


if __name__ == "__main__":
    unittest.main()
