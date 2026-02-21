from .model import Model
from .registry import ModelRegistry

# Import models to ensure they register themselves
from . import yolov5
from . import yolov8
from . import segment_anything
