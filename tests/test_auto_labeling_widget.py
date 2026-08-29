"""Regression tests for leaving the Auto Labeling panel."""

import os
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6 import QtWidgets

from anylabeling.views.labeling.widgets.auto_labeling.auto_labeling import (
    AutoLabelingWidget,
)

_APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class MinimalAutoLabelingWidget(AutoLabelingWidget):
    """Exercise panel closing without loading the model-management UI."""

    def __init__(self, parent):
        QtWidgets.QWidget.__init__(self)
        self.parent = parent
        self.model_select_combobox = QtWidgets.QComboBox(self)
        self.model_select_combobox.addItems(["No Model", "SAM"])
        self.model_select_combobox.setCurrentIndex(1)
        self.model_select_combobox.currentIndexChanged.connect(lambda: None)


class TestAutoLabelingWidget(unittest.TestCase):
    def tearDown(self):
        if hasattr(self, "widget"):
            self.widget.hide()
            self.widget.deleteLater()
            _APP.processEvents()

    def test_close_returns_parent_to_edit_mode(self):
        parent = mock.Mock()
        self.widget = MinimalAutoLabelingWidget(parent)
        self.widget.show()
        _APP.processEvents()

        self.widget.unload_and_hide()
        _APP.processEvents()

        parent.set_edit_mode.assert_called_once_with()
        self.assertEqual(self.widget.model_select_combobox.currentIndex(), 0)
        self.assertFalse(self.widget.isVisible())

    def test_failed_model_load_reenables_model_picker(self):
        self.widget = MinimalAutoLabelingWidget(mock.Mock())
        self.widget.model_select_combobox.setEnabled(False)

        self.widget.on_new_model_loaded({})

        self.assertTrue(self.widget.model_select_combobox.isEnabled())
        self.assertEqual(self.widget.model_select_combobox.currentIndex(), 0)


if __name__ == "__main__":
    unittest.main()
