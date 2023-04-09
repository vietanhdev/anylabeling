import copy
import yaml
from threading import Lock
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtCore import QObject, QThread, pyqtSignal, pyqtSlot
from anylabeling.utils import GenericWorker
import importlib.resources as pkg_resources

from anylabeling import configs as anylabeling_configs


class ModelManager(QObject):
    """Model manager"""

    new_model_status = pyqtSignal(str)
    model_loaded = pyqtSignal(list)
    new_prediction_shapes = pyqtSignal(list)

    model_configs = {
        "segment_anything_vit_b": "autolabel_segment_anything.yaml",
        "yolov5s": "autolabel_yolov5s.yaml",
        "yolov5m": "autolabel_yolov5m.yaml",
    }

    def __init__(self):
        super().__init__()
        self.model_infos = {}
        for model_name, config_name in ModelManager.model_configs.items():
            self.model_infos[model_name] = {}
            with pkg_resources.open_text(
                anylabeling_configs, config_name
            ) as f:
                config = yaml.safe_load(f)
            self.model_infos[model_name] = config

        self.loaded_model_info = None
        self.loaded_model_info_lock = Lock()

        self.model_download_thread = None

    def get_model_infos(self):
        """Return model infos"""
        return self.model_infos

    def get_model_names(self):
        """Return model names"""
        return list(self.model_infos.keys())

    @pyqtSlot()
    def on_model_download_finished(self):
        """Handle model download thread finished"""
        self.new_model_status.emit("Model loaded. Ready for labeling.")
        if self.loaded_model_info and self.loaded_model_info["model"]:
            self.model_loaded.emit(
                self.loaded_model_info["model"].get_required_buttons()
            )

    def load_model(self, model_name):
        """Run model loading in a thread"""
        if (
            self.model_download_thread is not None
            and self.model_download_thread.isRunning()
        ):
            print(
                "Another model is being loaded. Please wait for it to finish."
            )
            return
        if not model_name:
            if self.model_download_worker is not None:
                self.model_download_worker.finished.disconnect(
                    self.on_model_download_finished
                )
            self.unload_model()
            self.new_model_status.emit("No model selected.")
            return
        self.model_download_thread = QThread()
        self.new_model_status.emit(
            f"Loading model: {self.model_infos[model_name]['display_name']}."
            " Please wait..."
        )
        self.model_download_worker = GenericWorker(
            self._load_model, model_name
        )
        self.model_download_worker.finished.connect(
            self.on_model_download_finished
        )
        self.model_download_worker.finished.connect(
            self.model_download_thread.quit
        )
        self.model_download_worker.moveToThread(self.model_download_thread)
        self.model_download_thread.started.connect(
            self.model_download_worker.run
        )
        self.model_download_thread.start()

    def _load_model(self, model_name):
        """Load and return model info"""
        if self.loaded_model_info is not None:
            self.loaded_model_info["model"].unload()
            self.loaded_model_info = None

        model_info = copy.deepcopy(self.model_infos[model_name])

        if model_info["type"] == "yolov5":
            from .yolov5 import YOLOv5

            model_info["model"] = YOLOv5(model_info)
        elif model_info["type"] == "segment_anything":
            from .segment_anything import SegmentAnything

            model_info["model"] = SegmentAnything(model_info)
        else:
            raise Exception(f"Unknown model type: {model_info['type']}")

        self.loaded_model_info = model_info
        return self.loaded_model_info

    def unload_model(self):
        """Unload model"""
        if self.loaded_model_info is not None:
            self.loaded_model_info["model"].unload()
            self.loaded_model_info = None

    @pyqtSlot()
    def predict_shapes(self, image):
        """Predict shapes"""
        if self.loaded_model_info is None:
            raise Exception("Model is not loaded")
        shapes = self.loaded_model_info["model"].predict_shapes(image)
        self.new_prediction_shapes.emit(shapes)
