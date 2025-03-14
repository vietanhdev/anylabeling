"""Export dialog widget for exporting annotations to different formats."""

import os
from PyQt5.QtCore import Qt, QThreadPool
from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QRadioButton,
    QButtonGroup,
    QPushButton,
    QFileDialog,
    QCheckBox,
    QGroupBox,
    QProgressBar,
    QSpinBox,
    QMessageBox,
    QSizePolicy,
    QLineEdit,
)

from ..utils.export_worker import ExportWorker


class ExportDialog(QDialog):
    """Dialog for exporting annotations to different formats."""

    def __init__(self, parent=None, current_folder=None):
        super().__init__(parent)
        self.current_folder = current_folder
        self.source_folder = current_folder
        self.output_folder = None
        self.thread_pool = QThreadPool()
        self.export_worker = None
        self.setMinimumWidth(600)
        self.setMinimumHeight(400)
        self.setWindowTitle("Export Annotations")
        self.setup_ui()

    def setup_ui(self):
        """Set up the UI elements."""
        layout = QVBoxLayout()
        self.setLayout(layout)

        # -- Format selection --
        format_group = QGroupBox("Format")
        format_layout = QVBoxLayout()
        format_group.setLayout(format_layout)

        self.format_combo = QComboBox()
        self.format_combo.addItem("YOLO (.txt)", "yolo")
        self.format_combo.addItem("COCO (.json)", "coco")
        self.format_combo.addItem("Pascal VOC (.xml)", "pascal_voc")
        self.format_combo.addItem("CreateML", "createml")
        format_layout.addWidget(self.format_combo)

        layout.addWidget(format_group)

        # -- Source selection --
        source_group = QGroupBox("Source")
        source_layout = QVBoxLayout()
        source_group.setLayout(source_layout)

        self.current_folder_radio = QRadioButton("Current Folder")
        self.current_folder_radio.setChecked(True)
        self.select_folder_radio = QRadioButton("Select Folder")
        self.source_button_group = QButtonGroup()
        self.source_button_group.addButton(self.current_folder_radio)
        self.source_button_group.addButton(self.select_folder_radio)

        source_layout.addWidget(self.current_folder_radio)

        select_folder_layout = QHBoxLayout()
        select_folder_layout.addWidget(self.select_folder_radio)
        self.source_folder_edit = QLineEdit()
        self.source_folder_edit.setReadOnly(True)
        self.source_folder_edit.setEnabled(False)
        select_folder_layout.addWidget(self.source_folder_edit)
        self.source_browse_button = QPushButton("Browse...")
        self.source_browse_button.setEnabled(False)
        select_folder_layout.addWidget(self.source_browse_button)

        source_layout.addLayout(select_folder_layout)

        # Add recursive search option
        self.recursive_check = QCheckBox("Search recursively in subfolders")
        self.recursive_check.setChecked(True)
        source_layout.addWidget(self.recursive_check)

        layout.addWidget(source_group)

        # -- Output folder selection --
        output_group = QGroupBox("Output Folder")
        output_layout = QHBoxLayout()
        output_group.setLayout(output_layout)

        self.output_folder_edit = QLineEdit()
        self.output_folder_edit.setReadOnly(True)
        output_layout.addWidget(self.output_folder_edit)
        self.output_browse_button = QPushButton("Browse...")
        output_layout.addWidget(self.output_browse_button)

        layout.addWidget(output_group)

        # -- Data split options --
        split_group = QGroupBox("Data Split")
        split_layout = QVBoxLayout()
        split_group.setLayout(split_layout)

        self.split_check = QCheckBox("Split data into train/val/test sets")
        split_layout.addWidget(self.split_check)

        # Split ratio layout
        ratio_layout = QHBoxLayout()

        train_layout = QHBoxLayout()
        train_layout.addWidget(QLabel("Train:"))
        self.train_spin = QSpinBox()
        self.train_spin.setRange(1, 99)
        self.train_spin.setValue(70)
        self.train_spin.setSuffix("%")
        self.train_spin.setEnabled(False)
        train_layout.addWidget(self.train_spin)
        ratio_layout.addLayout(train_layout)

        val_layout = QHBoxLayout()
        val_layout.addWidget(QLabel("Val:"))
        self.val_spin = QSpinBox()
        self.val_spin.setRange(1, 99)
        self.val_spin.setValue(20)
        self.val_spin.setSuffix("%")
        self.val_spin.setEnabled(False)
        val_layout.addWidget(self.val_spin)
        ratio_layout.addLayout(val_layout)

        test_layout = QHBoxLayout()
        test_layout.addWidget(QLabel("Test:"))
        self.test_spin = QSpinBox()
        self.test_spin.setRange(0, 98)
        self.test_spin.setValue(10)
        self.test_spin.setSuffix("%")
        self.test_spin.setEnabled(False)
        test_layout.addWidget(self.test_spin)
        ratio_layout.addLayout(test_layout)

        split_layout.addLayout(ratio_layout)

        layout.addWidget(split_group)

        # -- Progress bar --
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout()
        progress_group.setLayout(progress_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)

        self.progress_label = QLabel("Ready")
        progress_layout.addWidget(self.progress_label)

        layout.addWidget(progress_group)

        # -- Dialog buttons --
        button_layout = QHBoxLayout()

        self.export_button = QPushButton("Export")
        self.export_button.setDefault(True)
        self.cancel_button = QPushButton("Cancel")

        button_layout.addStretch()
        button_layout.addWidget(self.export_button)
        button_layout.addWidget(self.cancel_button)

        layout.addLayout(button_layout)

        # -- Connect signals --
        self.connect_signals()

    def connect_signals(self):
        """Connect signals to slots."""
        # Source folder selection
        self.select_folder_radio.toggled.connect(self.on_source_radio_toggled)
        self.source_browse_button.clicked.connect(
            self.on_source_browse_clicked
        )

        # Output folder selection
        self.output_browse_button.clicked.connect(
            self.on_output_browse_clicked
        )

        # Data split options
        self.split_check.toggled.connect(self.on_split_check_toggled)
        self.train_spin.valueChanged.connect(self.on_ratio_changed)
        self.val_spin.valueChanged.connect(self.on_ratio_changed)
        self.test_spin.valueChanged.connect(self.on_ratio_changed)

        # Dialog buttons
        self.export_button.clicked.connect(self.on_export_clicked)
        self.cancel_button.clicked.connect(self.on_cancel_clicked)

    def on_source_radio_toggled(self, checked):
        """Handle toggling of source radio buttons."""
        self.source_folder_edit.setEnabled(checked)
        self.source_browse_button.setEnabled(checked)
        if checked and not self.source_folder_edit.text():
            self.source_folder = None

    def on_source_browse_clicked(self):
        """Handle clicking of source browse button."""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Source Folder",
            self.current_folder or os.path.expanduser("~"),
        )
        if folder:
            self.source_folder = folder
            self.source_folder_edit.setText(folder)

    def on_output_browse_clicked(self):
        """Handle clicking of output browse button."""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Output Folder",
            self.current_folder or os.path.expanduser("~"),
        )
        if folder:
            self.output_folder = folder
            self.output_folder_edit.setText(folder)

    def on_split_check_toggled(self, checked):
        """Handle toggling of split checkbox."""
        self.train_spin.setEnabled(checked)
        self.val_spin.setEnabled(checked)
        self.test_spin.setEnabled(checked)

    def on_ratio_changed(self, value):
        """Handle changes to split ratios."""
        if self.sender() == self.train_spin:
            # Adjust val and test to maintain 100%
            total = (
                self.train_spin.value()
                + self.val_spin.value()
                + self.test_spin.value()
            )
            if total != 100:
                # Keep val and test proportional to each other
                val_test_total = self.val_spin.value() + self.test_spin.value()
                if val_test_total > 0:
                    val_ratio = self.val_spin.value() / val_test_total
                    remaining = 100 - self.train_spin.value()
                    self.val_spin.blockSignals(True)
                    self.test_spin.blockSignals(True)
                    self.val_spin.setValue(int(remaining * val_ratio))
                    self.test_spin.setValue(remaining - self.val_spin.value())
                    self.val_spin.blockSignals(False)
                    self.test_spin.blockSignals(False)
        elif self.sender() == self.val_spin:
            # Adjust train and test to maintain 100%
            total = (
                self.train_spin.value()
                + self.val_spin.value()
                + self.test_spin.value()
            )
            if total != 100:
                # Prioritize reducing tes
                self.test_spin.blockSignals(True)
                test_value = (
                    100 - self.train_spin.value() - self.val_spin.value()
                )
                if test_value < 0:
                    # If test would go negative, reduce train instead
                    self.train_spin.blockSignals(True)
                    self.train_spin.setValue(100 - self.val_spin.value())
                    self.test_spin.setValue(0)
                    self.train_spin.blockSignals(False)
                else:
                    self.test_spin.setValue(test_value)
                self.test_spin.blockSignals(False)
        elif self.sender() == self.test_spin:
            # Adjust train and val to maintain 100%
            total = (
                self.train_spin.value()
                + self.val_spin.value()
                + self.test_spin.value()
            )
            if total != 100:
                # Prioritize reducing val
                self.val_spin.blockSignals(True)
                val_value = (
                    100 - self.train_spin.value() - self.test_spin.value()
                )
                if val_value < 0:
                    # If val would go negative, reduce train instead
                    self.train_spin.blockSignals(True)
                    self.train_spin.setValue(100 - self.test_spin.value())
                    self.val_spin.setValue(0)
                    self.train_spin.blockSignals(False)
                else:
                    self.val_spin.setValue(val_value)
                self.val_spin.blockSignals(False)

    def validate_inputs(self):
        """Validate user inputs before starting export."""
        # Check if source folder is selected
        if self.select_folder_radio.isChecked() and not self.source_folder:
            QMessageBox.warning(
                self, "Missing Source", "Please select a source folder."
            )
            return False

        # Check if output folder is selected
        if not self.output_folder:
            QMessageBox.warning(
                self, "Missing Output", "Please select an output folder."
            )
            return False

        # Check if source and output folders are different
        source = (
            self.current_folder
            if self.current_folder_radio.isChecked()
            else self.source_folder
        )
        if source == self.output_folder:
            result = QMessageBox.question(
                self,
                "Same Folder",
                "Source and output folders are the same. This may overwrite files. Continue?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if result == QMessageBox.No:
                return False

        # Check split ratios
        if self.split_check.isChecked():
            total = (
                self.train_spin.value()
                + self.val_spin.value()
                + self.test_spin.value()
            )
            if total != 100:
                QMessageBox.warning(
                    self,
                    "Invalid Split Ratio",
                    "Split ratios must add up to 100%.",
                )
                return False

        return True

    def on_export_clicked(self):
        """Handle clicking of export button."""
        if not self.validate_inputs():
            return

        # Disable controls during export
        self.set_controls_enabled(False)

        # Get export parameters
        export_format = self.format_combo.currentData()
        input_dir = (
            self.current_folder
            if self.current_folder_radio.isChecked()
            else self.source_folder
        )
        output_dir = self.output_folder
        split_data = self.split_check.isChecked()
        train_ratio = self.train_spin.value() / 100.0
        val_ratio = self.val_spin.value() / 100.0
        test_ratio = self.test_spin.value() / 100.0
        recursive = self.recursive_check.isChecked()

        # Create and start export worker
        self.export_worker = ExportWorker(
            export_format,
            input_dir,
            output_dir,
            split_data,
            train_ratio,
            val_ratio,
            test_ratio,
            recursive,
        )

        # Connect worker signals
        self.export_worker.signals.started.connect(self.on_export_started)
        self.export_worker.signals.finished.connect(self.on_export_finished)
        self.export_worker.signals.progress.connect(self.on_export_progress)
        self.export_worker.signals.error.connect(self.on_export_error)

        # Start worker
        self.thread_pool.start(self.export_worker)

    def on_cancel_clicked(self):
        """Handle clicking of cancel button."""
        if self.export_worker and self.export_worker.running:
            result = QMessageBox.question(
                self,
                "Cancel Export",
                "Export is in progress. Are you sure you want to cancel?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if result == QMessageBox.Yes:
                self.export_worker.stop()
                self.progress_label.setText("Cancelling...")
        else:
            self.reject()

    def on_export_started(self):
        """Handle export started signal."""
        self.progress_label.setText("Starting export...")
        self.progress_bar.setValue(0)

    def on_export_finished(self):
        """Handle export finished signal."""
        self.set_controls_enabled(True)
        self.progress_label.setText("Export completed!")
        self.progress_bar.setValue(100)

        QMessageBox.information(
            self,
            "Export Completed",
            f"Annotations have been exported to {self.output_folder}",
        )

    def on_export_progress(self, progress, message):
        """Handle export progress signal."""
        self.progress_bar.setValue(progress)
        self.progress_label.setText(message)

    def on_export_error(self, error_message):
        """Handle export error signal."""
        self.set_controls_enabled(True)
        self.progress_label.setText(f"Error: {error_message}")

        QMessageBox.critical(
            self,
            "Export Error",
            f"An error occurred during export:\n{error_message}",
        )

    def set_controls_enabled(self, enabled):
        """Enable or disable controls during export."""
        # Format selection
        self.format_combo.setEnabled(enabled)

        # Source selection
        self.current_folder_radio.setEnabled(enabled)
        self.select_folder_radio.setEnabled(enabled)
        self.source_folder_edit.setEnabled(
            enabled and self.select_folder_radio.isChecked()
        )
        self.source_browse_button.setEnabled(
            enabled and self.select_folder_radio.isChecked()
        )
        self.recursive_check.setEnabled(enabled)

        # Output folder selection
        self.output_folder_edit.setEnabled(enabled)
        self.output_browse_button.setEnabled(enabled)

        # Data split options
        self.split_check.setEnabled(enabled)
        split_enabled = enabled and self.split_check.isChecked()
        self.train_spin.setEnabled(split_enabled)
        self.val_spin.setEnabled(split_enabled)
        self.test_spin.setEnabled(split_enabled)

        # Dialog buttons
        self.export_button.setEnabled(enabled)
        self.cancel_button.setText("Close" if enabled else "Cancel")
