import os

from PyQt5 import uic
from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import pyqtSignal

from anylabeling.services.auto_labeling.model_manager import ModelManager


class AutoLabelingWidget(QWidget):
    new_model_selected = pyqtSignal(str)

    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        current_dir = os.path.dirname(__file__)
        uic.loadUi(os.path.join(current_dir, "auto_labeling.ui"), self)

        self.model_manager = ModelManager()
        self.model_manager.new_model_status.connect(self.on_new_model_status)
        self.new_model_selected.connect(self.model_manager.load_model)
        self.model_manager.model_loaded.connect(self.update_visible_buttons)

        # Add models to combobox
        self.model_select_combobox.clear()
        self.model_select_combobox.addItem("No Model", userData=None)
        for model_info in self.model_manager.get_model_infos().values():
            self.model_select_combobox.addItem(
                model_info["display_name"], userData=model_info["name"]
            )

        # Hide labeling buttons by default
        self.hide_labeling_buttons()

        # Handle close button
        self.button_close.clicked.connect(self.hide)

        # Handle model select combobox
        self.model_select_combobox.currentIndexChanged.connect(
            self.on_model_select_combobox_changed
        )

    def on_new_model_status(self, status):
        self.model_status_label.setText(status)

    def on_model_select_combobox_changed(self, index):
        model_name = self.model_select_combobox.itemData(index)
        if model_name is None:
            self.hide_labeling_buttons()
            self.new_model_selected.emit(None)
        else:
            self.hide_labeling_buttons()
            self.new_model_selected.emit(model_name)

    def update_visible_buttons(self, buttons):
        """Update button status"""
        for button in buttons:
            getattr(self, button).show()

    def hide_labeling_buttons(self):
        """Hide labeling buttons by default"""
        buttons = [
            "button_run",
            "button_add_point",
            "button_remove_point",
            "button_add_rect",
            "button_undo",
            "button_clear",
            "button_finish_object",
        ]
        for button in buttons:
            getattr(self, button).hide()

    def on_open(self):
        pass

    def on_close(self):
        return True
