"""Regression tests for Escape handling in the labeling workspace."""

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6 import QtCore, QtTest, QtWidgets

from anylabeling.views.labeling.label_widget import LabelingWidget
from anylabeling.views.labeling.widgets.label_dialog import LabelDialog

_APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class MinimalLabelingWidget(LabelingWidget):
    """Exercise LabelingWidget's dialog behavior without loading the full UI."""

    def __init__(self):
        QtWidgets.QDialog.__init__(self)
        layout = QtWidgets.QVBoxLayout(self)
        self.model_selector = QtWidgets.QComboBox(self)
        self.model_selector.addItem("SAM2 Hiera-Tiny")
        layout.addWidget(self.model_selector)

    def resizeEvent(self, event):
        event.accept()

    def closeEvent(self, event):
        event.accept()


class TestLabelingWidgetEscape(unittest.TestCase):
    def tearDown(self):
        for widget in QtWidgets.QApplication.topLevelWidgets():
            widget.close()
        _APP.processEvents()

    def test_escape_from_child_does_not_hide_labeling_workspace(self):
        widget = MinimalLabelingWidget()
        widget.show()
        widget.model_selector.setFocus()
        _APP.processEvents()

        QtTest.QTest.keyClick(widget.model_selector, QtCore.Qt.Key.Key_Escape)
        _APP.processEvents()

        self.assertTrue(widget.isVisible())

    def test_escape_still_closes_label_dialog(self):
        dialog = LabelDialog()
        dialog.show()
        dialog.edit.setFocus()
        _APP.processEvents()

        QtTest.QTest.keyClick(dialog.edit, QtCore.Qt.Key.Key_Escape)
        _APP.processEvents()

        self.assertFalse(dialog.isVisible())


if __name__ == "__main__":
    unittest.main()
