"""This module defines labelme wrapper and related functions"""

from PyQt5.QtWidgets import QVBoxLayout, QWidget

from .label_widget import LabelmeWidget


class LabelingWrapper(QWidget):
    """Wrapper widget for labelme module"""

    def __init__(self, parent):
        super().__init__()
        self.parent = parent

        # Create a labelme widget
        view = LabelmeWidget(self)

        # Create the main layout and put labelme into
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(view)
        self.setLayout(main_layout)
