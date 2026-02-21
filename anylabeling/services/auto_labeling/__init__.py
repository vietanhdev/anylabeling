# Import models to ensure they register themselves via @ModelRegistry.register
from . import segment_anything as segment_anything  # noqa: F401
from . import yolov5 as yolov5  # noqa: F401
from . import yolov8 as yolov8  # noqa: F401
from .model import Model as Model
from .registry import ModelRegistry as ModelRegistry
