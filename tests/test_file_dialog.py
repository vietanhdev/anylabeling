"""Regression tests for the Ctrl+O file dialog."""

import os
import unittest
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6 import QtCore, QtWidgets

from anylabeling.views.labeling.label_widget import LabelingWidget


class TestOpenFileDialog(unittest.TestCase):
    @mock.patch("anylabeling.views.labeling.label_widget.FileDialogPreview")
    def test_uses_pyqt6_scoped_enums(self, dialog_class):
        dialog = dialog_class.return_value
        dialog.exec.return_value = False
        widget = SimpleNamespace(
            filename=None,
            may_continue=lambda: True,
            tr=lambda text: text,
        )

        LabelingWidget.open_file(widget)

        dialog.setFileMode.assert_called_once_with(
            QtWidgets.QFileDialog.FileMode.ExistingFile
        )
        dialog.setViewMode.assert_called_once_with(
            QtWidgets.QFileDialog.ViewMode.Detail
        )

    @mock.patch("anylabeling.views.labeling.label_widget.FileDialogPreview")
    @mock.patch(
        "anylabeling.views.labeling.label_widget."
        "QtGui.QImageReader.supportedImageFormats",
        return_value=[QtCore.QByteArray(b"png"), QtCore.QByteArray(b"svg")],
    )
    def test_does_not_advertise_unsupported_svg(self, _formats, dialog_class):
        dialog = dialog_class.return_value
        dialog.exec.return_value = False
        widget = SimpleNamespace(
            filename=None,
            may_continue=lambda: True,
            tr=lambda text: text,
        )

        LabelingWidget.open_file(widget)

        name_filter = dialog.setNameFilter.call_args.args[0]
        self.assertIn("*.png", name_filter)
        self.assertNotIn("*.svg", name_filter)


if __name__ == "__main__":
    unittest.main()
