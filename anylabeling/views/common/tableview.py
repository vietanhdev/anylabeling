"""Defines table view widget as a common component"""

import os

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QTableView


class TableViewWidget(QTableView):
    """Table view widget with drag/drop support"""

    new_data_items = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.parent = parent

    def dragEnterEvent(self, event):
        """Filter only drag enter events with urls"""
        if event.mimeData().hasUrls:
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        """Filter only drag move events with urls"""
        if event.mimeData().hasUrls():
            event.setDropAction(Qt.CopyAction)
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        """Filter only drop events with urls"""
        if event.mimeData().hasUrls():
            event.setDropAction(Qt.CopyAction)
            event.accept()

            links = []
            for url in event.mimeData().urls():
                link = str(url.toLocalFile())
                link = os.path.normpath(link)
                links.append(link)

            self.new_data_items.emit(links)
        else:
            event.ignore()
