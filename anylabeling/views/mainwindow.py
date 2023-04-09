"""This module defines the main application window"""

from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from .labeling.label_wrapper import LabelingWrapper


class MainWindow(QMainWindow):
    """Main application window"""

    def __init__(self, app):
        super().__init__()
        self.app = app

        self.setContentsMargins(0, 0, 0, 0)
        self.setWindowTitle("AnyLabeling")

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        self.labeling_widget = LabelingWrapper(self)
        main_layout.addWidget(self.labeling_widget)
        widget = QWidget()
        widget.setLayout(main_layout)
        self.setCentralWidget(widget)

        status_bar = QStatusBar()
        status_bar.showMessage(
            "AnyLabeling - Effortless data labeling with AI support"
        )
        self.setStatusBar(status_bar)
