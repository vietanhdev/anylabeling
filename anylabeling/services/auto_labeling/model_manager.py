import copy
import importlib.resources as pkg_resources
from threading import Lock

import yaml
from PyQt5.QtCore import QObject, QThread, pyqtSignal, pyqtSlot

from anylabeling.configs import auto_labeling as auto_labeling_configs
from anylabeling.services.auto_labeling.types import AutoLabelingResult
from anylabeling.utils import GenericWorker


class ModelManager(QObject):
    """Model manager"""

    new_model_status = pyqtSignal(str)
    model_loaded = pyqtSignal(list)
    new_auto_labeling_result = pyqtSignal(AutoLabelingResult)
    auto_segmentation_model_selected = pyqtSignal()
    auto_segmentation_model_unselected = pyqtSignal()
    prediction_started = pyqtSignal()
    prediction_finished = pyqtSignal()
    request_next_files_requested = pyqtSignal()
    output_modes_changed = pyqtSignal(dict, str)

    model_configs = {}

    def __init__(self):
        super().__init__()
        self.model_infos = {}
        self.model_list_config = {}

        # Load list of models
        with pkg_resources.open_text(
            auto_labeling_configs, "models.yaml"
        ) as f:
            self.model_list_config = yaml.safe_load(f)
        for model_config in self.model_list_config:
            self.model_configs[model_config["model_name"]] = model_config[
                "config_file"
            ]

        # Load model configs
        for model_name, config_name in ModelManager.model_configs.items():
            self.model_infos[model_name] = {}
            with pkg_resources.open_text(
                auto_labeling_configs, config_name
            ) as f:
                config = yaml.safe_load(f)
            self.model_infos[model_name] = config

        self.loaded_model_info = None
        self.loaded_model_info_lock = Lock()

        self.model_download_thread = None
        self.model_execution_thread = None
        self.model_execution_thread_lock = Lock()

    def get_model_infos(self):
        """Return model infos"""
        return self.model_infos

    def get_model_names(self):
        """Return model names"""
        return list(self.model_infos.keys())

    def set_output_mode(self, mode):
        """Set output mode"""
        if self.loaded_model_info and self.loaded_model_info["model"]:
            self.loaded_model_info["model"].set_output_mode(mode)

    @pyqtSlot()
    def on_model_download_finished(self):
        """Handle model download thread finished"""
        self.new_model_status.emit(
            self.tr("Model loaded. Ready for labeling.")
        )
        if self.loaded_model_info and self.loaded_model_info["model"]:
            self.model_loaded.emit(
                self.loaded_model_info["model"].get_required_widgets()
            )
            self.output_modes_changed.emit(
                self.loaded_model_info["model"].Meta.output_modes,
                self.loaded_model_info["model"].Meta.default_output_mode,
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
            self.new_model_status.emit(self.tr("No model selected."))
            return
        self.model_download_thread = QThread()
        self.new_model_status.emit(
            self.tr("Loading model: {model_name}. Please wait...").format(
                model_name=self.model_infos[model_name]["display_name"]
            )
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
            self.auto_segmentation_model_unselected.emit()

        model_info = copy.deepcopy(self.model_infos[model_name])
        if model_info["type"] == "yolov5":
            from .yolov5 import YOLOv5

            model_info["model"] = YOLOv5(
                model_info, on_message=self.new_model_status.emit
            )
            self.auto_segmentation_model_unselected.emit()
        elif model_info["type"] == "yolov8":
            from .yolov8 import YOLOv8

            model_info["model"] = YOLOv8(
                model_info, on_message=self.new_model_status.emit
            )
            self.auto_segmentation_model_unselected.emit()
        elif model_info["type"] == "segment_anything":
            from .segment_anything import SegmentAnything

            model_info["model"] = SegmentAnything(
                model_info, on_message=self.new_model_status.emit
            )
            self.auto_segmentation_model_selected.emit()

            # Request next files for prediction
            self.request_next_files_requested.emit()
        else:
            raise Exception(f"Unknown model type: {model_info['type']}")

        self.loaded_model_info = model_info
        return self.loaded_model_info

    def set_auto_labeling_marks(self, marks):
        """Set auto labeling marks
        (For example, for segment_anything model, it is the marks for)
        """
        if (
            self.loaded_model_info is None
            or self.loaded_model_info["type"] != "segment_anything"
        ):
            return
        self.loaded_model_info["model"].set_auto_labeling_marks(marks)

    def unload_model(self):
        """Unload model"""
        if self.loaded_model_info is not None:
            self.loaded_model_info["model"].unload()
            self.loaded_model_info = None

    def predict_shapes(self, image, filename=None):
        """Predict shapes.
        NOTE: This function is blocking. The model can take a long time to
        predict. So it is recommended to use predict_shapes_threading instead.
        """
        if self.loaded_model_info is None:
            self.new_model_status.emit(
                self.tr("Model is not loaded. Choose a mode to continue.")
            )
            self.prediction_finished.emit()
            return
        try:
            auto_labeling_result = self.loaded_model_info[
                "model"
            ].predict_shapes(image, filename)
            self.new_auto_labeling_result.emit(auto_labeling_result)
        except Exception as e:  # noqa
            print(f"Error in predict_shapes: {e}")
            self.new_model_status.emit(
                self.tr("Error in model prediction. Please check the model.")
            )
        self.new_model_status.emit(
            self.tr("Finished inferencing AI model. Check the result.")
        )
        self.prediction_finished.emit()

    @pyqtSlot()
    def predict_shapes_threading(self, image, filename=None):
        """Predict shapes.
        This function starts a thread to run the prediction.
        """
        if self.loaded_model_info is None:
            self.new_model_status.emit(
                self.tr("Model is not loaded. Choose a mode to continue.")
            )
            return
        self.new_model_status.emit(
            self.tr("Inferencing AI model. Please wait...")
        )
        self.prediction_started.emit()

        with self.model_execution_thread_lock:
            if (
                self.model_execution_thread is not None
                and self.model_execution_thread.isRunning()
            ):
                self.new_model_status.emit(
                    self.tr(
                        "Another model is being executed."
                        " Please wait for it to finish."
                    )
                )
                self.prediction_finished.emit()
                return

            self.model_execution_thread = QThread()
            self.model_execution_worker = GenericWorker(
                self.predict_shapes, image, filename
            )
            self.model_execution_worker.finished.connect(
                self.model_execution_thread.quit
            )
            self.model_execution_worker.moveToThread(
                self.model_execution_thread
            )
            self.model_execution_thread.started.connect(
                self.model_execution_worker.run
            )
            self.model_execution_thread.start()

    def on_next_files_changed(self, next_files):
        """Run prediction on next files in advance to save inference time later"""
        if self.loaded_model_info is None:
            return

        # Currently only segment_anything model supports this feature
        if self.loaded_model_info["type"] != "segment_anything":
            return

        self.loaded_model_info["model"].on_next_files_changed(next_files)
