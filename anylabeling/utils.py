import os

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

MESH_EXTENSIONS = [".obj", ".stl", ".ply"]


def is_mesh_file(filename):
    """Check if the filename is a mesh file"""
    if not filename:
        return False
    return os.path.splitext(filename)[1].lower() in MESH_EXTENSIONS


class GenericWorker(QObject):
    finished = pyqtSignal()

    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs

    @pyqtSlot()
    def run(self):
        self.func(*self.args, **self.kwargs)
        self.finished.emit()
