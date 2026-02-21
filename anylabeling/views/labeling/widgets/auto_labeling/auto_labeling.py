import os

from PyQt6 import uic
from PyQt6.QtCore import pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import QWidget, QFileDialog

from anylabeling.services.auto_labeling.model_manager import ModelManager
from anylabeling.services.auto_labeling.types import AutoLabelingMode
from anylabeling.styles.theme import AppTheme


class AutoLabelingWidget(QWidget):
    new_model_selected = pyqtSignal(str)
    new_custom_model_selected = pyqtSignal(str)
    auto_segmentation_requested = pyqtSignal()
    auto_segmentation_disabled = pyqtSignal()
    auto_labeling_mode_changed = pyqtSignal(AutoLabelingMode)
    clear_auto_labeling_action_requested = pyqtSignal()
    finish_auto_labeling_object_action_requested = pyqtSignal()

    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        current_dir = os.path.dirname(__file__)
        uic.loadUi(os.path.join(current_dir, "auto_labeling.ui"), self)

        self.model_manager = ModelManager()
        self.model_manager.model_configs_changed.connect(
            lambda model_list: self.update_model_configs(model_list)
        )
        self.model_manager.new_model_status.connect(self.on_new_model_status)
        self.new_model_selected.connect(self.model_manager.load_model)
        self.new_custom_model_selected.connect(self.model_manager.load_custom_model)
        self.model_manager.model_loaded.connect(self.update_visible_widgets)
        self.model_manager.model_loaded.connect(self.on_new_model_loaded)
        self.model_manager.new_auto_labeling_result.connect(
            lambda auto_labeling_result: self.parent.new_shapes_from_auto_labeling(
                auto_labeling_result
            )
        )
        self.model_manager.auto_segmentation_model_selected.connect(
            self.auto_segmentation_requested
        )
        self.model_manager.auto_segmentation_model_unselected.connect(
            self.auto_segmentation_disabled
        )
        self.model_manager.output_modes_changed.connect(self.on_output_modes_changed)
        self.output_select_combobox.currentIndexChanged.connect(
            lambda: self.model_manager.set_output_mode(
                self.output_select_combobox.currentData()
            )
        )

        self.update_model_configs(self.model_manager.get_model_configs())

        # Disable tools when inference is running
        def set_enable_tools(enable):
            self.model_select_combobox.setEnabled(enable)
            self.output_select_combobox.setEnabled(enable)
            self.edit_prompt.setEnabled(enable)
            self.button_add_point.setEnabled(enable)
            self.button_remove_point.setEnabled(enable)
            self.button_add_rect.setEnabled(enable)
            self.button_clear.setEnabled(enable)
            self.button_finish_object.setEnabled(enable)

        self.model_manager.prediction_started.connect(lambda: set_enable_tools(False))
        self.model_manager.prediction_finished.connect(lambda: set_enable_tools(True))

        # Prompt input
        self.edit_prompt.textChanged.connect(self.on_prompt_changed)
        self.edit_prompt.returnPressed.connect(self.run_prediction)

        # Prompt mode
        self.combobox_prompt_mode.currentIndexChanged.connect(self.on_prompt_mode_changed)

        # Confidence
        self.double_spin_box_confidence.valueChanged.connect(self.on_confidence_changed)

        # Auto labeling buttons
        self.button_run.setShortcut("I")
        self.button_run.clicked.connect(self.run_prediction)
        self.button_add_point.clicked.connect(
            lambda: self.set_auto_labeling_mode(
                AutoLabelingMode.ADD, AutoLabelingMode.POINT
            )
        )
        self.button_add_point.setShortcut("Q")
        self.button_remove_point.clicked.connect(
            lambda: self.set_auto_labeling_mode(
                AutoLabelingMode.REMOVE, AutoLabelingMode.POINT
            )
        )
        self.button_remove_point.setShortcut("E")
        self.button_add_rect.clicked.connect(
            lambda: self.set_auto_labeling_mode(
                AutoLabelingMode.ADD, AutoLabelingMode.RECTANGLE
            )
        )
        self.button_clear.clicked.connect(self.clear_auto_labeling_action_requested)
        self.button_finish_object.clicked.connect(
            self.finish_auto_labeling_object_action_requested
        )
        self.button_finish_object.setShortcut("F")

        # Hide labeling widgets by default
        self.hide_labeling_widgets()

        # Handle close button
        self.button_close.clicked.connect(self.unload_and_hide)

        # Handle model select combobox
        self.model_select_combobox.currentIndexChanged.connect(
            self.on_model_select_combobox_changed
        )

        self.auto_labeling_mode_changed.connect(self.update_button_colors)
        self.auto_labeling_mode = AutoLabelingMode.NONE
        self.auto_labeling_mode_changed.emit(self.auto_labeling_mode)

    def update_model_configs(self, model_list):
        """Update model list"""
        # Add models to combobox
        self.model_select_combobox.clear()
        self.model_select_combobox.addItem(self.tr("No Model"), userData=None)
        self.model_select_combobox.addItem(
            self.tr("...Load Custom Model"), userData="load_custom_model"
        )
        for model_config in model_list:
            self.model_select_combobox.addItem(
                (
                    self.tr("(User) ")
                    if model_config.get("is_custom_model", False)
                    else ""
                )
                + model_config["display_name"],
                userData=model_config["config_file"],
            )

    @pyqtSlot()
    def update_button_colors(self):
        """Update button colors based on current theme and mode"""
        style_sheet = """
            text-align: center;
            margin-right: 3px;
            border-radius: 5px;
            padding: 4px 8px;
            border: 1px solid {border_color};
        """

        border_color = AppTheme.get_color("border")
        normal_bg_color = AppTheme.get_color("button")
        normal_text_color = AppTheme.get_color("button_text")
        active_bg_color = AppTheme.get_color("success")
        remove_bg_color = AppTheme.get_color("error")
        highlighted_text_color = AppTheme.get_color("highlighted_text")

        normal_style = (
            style_sheet.format(border_color=border_color)
            + f"background-color: {normal_bg_color}; color: {normal_text_color};"
        )

        for button in [
            self.button_add_point,
            self.button_remove_point,
            self.button_add_rect,
            self.button_clear,
            self.button_finish_object,
        ]:
            button.setStyleSheet(normal_style)

        if self.auto_labeling_mode == AutoLabelingMode.NONE:
            return

        if self.auto_labeling_mode.edit_mode == AutoLabelingMode.ADD:
            if self.auto_labeling_mode.shape_type == AutoLabelingMode.POINT:
                self.button_add_point.setStyleSheet(
                    style_sheet.format(border_color=border_color)
                    + f"background-color: {active_bg_color}; color: {highlighted_text_color};"
                )
            elif self.auto_labeling_mode.shape_type == AutoLabelingMode.RECTANGLE:
                self.button_add_rect.setStyleSheet(
                    style_sheet.format(border_color=border_color)
                    + f"background-color: {active_bg_color}; color: {highlighted_text_color};"
                )
        elif self.auto_labeling_mode.edit_mode == AutoLabelingMode.REMOVE:
            if self.auto_labeling_mode.shape_type == AutoLabelingMode.POINT:
                self.button_remove_point.setStyleSheet(
                    style_sheet.format(border_color=border_color)
                    + f"background-color: {remove_bg_color}; color: {highlighted_text_color};"
                )

    def set_auto_labeling_mode(self, edit_mode, shape_type=None):
        """Set auto labeling mode"""
        if edit_mode is None:
            self.auto_labeling_mode = AutoLabelingMode.NONE
        else:
            self.auto_labeling_mode = AutoLabelingMode(edit_mode, shape_type)
        self.auto_labeling_mode_changed.emit(self.auto_labeling_mode)

    def run_prediction(self):
        """Run prediction"""
        if self.parent.filename is not None:
            self.model_manager.predict_shapes_threading(
                self.parent.image, self.parent.filename
            )

    def unload_and_hide(self):
        """Unload model and hide widget"""
        self.model_select_combobox.setCurrentIndex(0)
        self.hide()

    def on_new_model_status(self, status):
        self.model_status_label.setText(status)

    def on_new_model_loaded(self, model_config):
        """Enable model select combobox"""
        self.model_select_combobox.currentIndexChanged.disconnect()
        if "config_file" not in model_config:
            self.model_select_combobox.setCurrentIndex(0)
        else:
            config_file = model_config["config_file"]
            self.model_select_combobox.setCurrentIndex(
                self.model_select_combobox.findData(config_file)
            )
        self.model_select_combobox.currentIndexChanged.connect(
            self.on_model_select_combobox_changed
        )
        self.model_select_combobox.setEnabled(True)

    def on_output_modes_changed(self, output_modes, default_output_mode):
        """Handle output modes changed"""
        # Disconnect onIndexChanged signal to prevent triggering
        # on model select combobox change
        self.output_select_combobox.currentIndexChanged.disconnect()

        self.output_select_combobox.clear()
        for output_mode, display_name in output_modes.items():
            self.output_select_combobox.addItem(display_name, userData=output_mode)
        self.output_select_combobox.setCurrentIndex(
            self.output_select_combobox.findData(default_output_mode)
        )

        # Reconnect onIndexChanged signal
        self.output_select_combobox.currentIndexChanged.connect(
            lambda: self.model_manager.set_output_mode(
                self.output_select_combobox.currentData()
            )
        )

    def on_model_select_combobox_changed(self, index):
        """Handle model select combobox change"""
        self.clear_auto_labeling_action_requested.emit()
        config_path = self.model_select_combobox.itemData(index)

        # Load custom model?
        if config_path == "load_custom_model":
            # Unload current model
            self.model_manager.unload_model()
            # Open file dialog to select "config.yaml" file for model
            file_dialog = QFileDialog(self)
            file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
            file_dialog.setNameFilter("Config file (*.yaml)")
            if file_dialog.exec():
                config_file = file_dialog.selectedFiles()[0]
                # Disable combobox while loading model
                if config_path:
                    self.model_select_combobox.setEnabled(False)
                self.hide_labeling_widgets()
                self.model_manager.load_custom_model(config_file)
            else:
                self.model_select_combobox.setCurrentIndex(0)
            return

        # Disable combobox while loading model
        if config_path:
            self.model_select_combobox.setEnabled(False)
        self.hide_labeling_widgets()
        self.new_model_selected.emit(config_path)

    def update_visible_widgets(self, model_config):
        """Update widget status"""
        if not model_config or "model" not in model_config:
            return
        model = model_config["model"]
        widgets = model.get_required_widgets()

        is_sam3 = getattr(model, "_is_sam3", False)

        # Always check if prompt mode selection should be shown
        if "label_prompt" in widgets or "edit_prompt" in widgets:
            self.label_prompt_mode.show()
            self.combobox_prompt_mode.show()
            self.label_confidence.show()
            self.double_spin_box_confidence.show()

            if not is_sam3:
                # Force Visual mode for non-SAM3 models
                visual_index = self.combobox_prompt_mode.findText(
                    self.tr("Visual")
                )
                if visual_index >= 0:
                    self.combobox_prompt_mode.setCurrentIndex(visual_index)
                self.combobox_prompt_mode.setEnabled(False)
            else:
                self.combobox_prompt_mode.setEnabled(True)

        prompt_mode = self.combobox_prompt_mode.currentText().lower()

        for widget in widgets:
            widget_obj = getattr(self, widget)

            # Filter based on prompt mode
            if prompt_mode == "visual":
                if widget in ["label_prompt", "edit_prompt"]:
                    widget_obj.hide()
                    continue
            elif prompt_mode == "text":
                # In text mode hide the geometric-prompt buttons.
                # Inference is triggered by pressing Enter, changing
                # the prompt text, or clicking the Run button.
                if widget in [
                    "button_add_point", "button_remove_point",
                    "button_add_rect", "button_clear", "button_finish_object",
                ]:
                    widget_obj.hide()
                    continue

            widget_obj.show()

        # Set initial values for widgets
        if hasattr(model_config["model"], "text_prompt"):
            self.edit_prompt.setText(model_config["model"].text_prompt)
        if hasattr(model_config["model"], "confidence_threshold"):
            self.double_spin_box_confidence.setValue(model_config["model"].confidence_threshold)

    def hide_labeling_widgets(self):
        """Hide labeling widgets by default"""
        widgets = [
            "output_label",
            "output_select_combobox",
            "label_prompt_mode",
            "combobox_prompt_mode",
            "label_confidence",
            "double_spin_box_confidence",
            "label_prompt",
            "edit_prompt",
            "button_run",
            "button_add_point",
            "button_remove_point",
            "button_add_rect",
            "button_clear",
            "button_finish_object",
        ]
        for widget in widgets:
            getattr(self, widget).hide()

    def on_prompt_changed(self, text):
        """Handle prompt changed"""
        self.model_manager.set_text_prompt(text)

    def on_confidence_changed(self, value):
        """Handle confidence changed"""
        self.model_manager.set_confidence_threshold(value)

    def on_prompt_mode_changed(self, index):
        """Handle prompt mode changed"""
        mode = self.combobox_prompt_mode.currentText().lower()
        self.model_manager.set_prompt_mode(mode)

        if mode == "visual":
            # Clear and reset the text prompt when switching to visual mode so
            # old text does not linger in the model's language encoder.
            self.edit_prompt.blockSignals(True)
            self.edit_prompt.clear()
            self.edit_prompt.blockSignals(False)
            self.model_manager.set_text_prompt("")

        # Refresh widget visibility
        if self.model_manager.loaded_model_config:
            self.update_visible_widgets(self.model_manager.loaded_model_config)

    def on_new_marks(self, marks):
        """Handle new marks"""
        self.model_manager.set_auto_labeling_marks(marks)
        self.run_prediction()

    def on_open(self):
        pass

    def on_close(self):
        return True
