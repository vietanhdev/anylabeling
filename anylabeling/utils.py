import os

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

# Common polygonal-mesh formats pyvista.read() supports. Deliberately a
# curated subset, not pyvista's full reader list — that list also includes
# non-mesh formats (images, volumes, CFD data) that must not be routed
# through the 3D mesh-loading path.
MESH_EXTENSIONS = [".obj", ".stl", ".ply", ".vtk", ".vtp", ".vtu", ".glb", ".gltf"]


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
