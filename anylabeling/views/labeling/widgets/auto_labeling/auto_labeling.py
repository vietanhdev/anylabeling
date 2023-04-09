import os

from PyQt5 import uic
from PyQt5.QtCore import QThread
from PyQt5.QtWidgets import QWidget

class AutoLabelingWidget(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        current_dir = os.path.dirname(__file__)
        uic.loadUi(os.path.join(current_dir, "auto_labeling.ui"), self)

    def on_open(self):
        pass

    def on_close(self):
        return True