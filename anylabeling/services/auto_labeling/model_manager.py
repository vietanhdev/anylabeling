import os
import copy
import time
import shutil
import pathlib
import logging
import tempfile
import zipfile
import importlib.resources as pkg_resources
from threading import Lock
import urllib.request

import yaml
from PyQt5.QtCore import QObject, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtCore import QCoreApplication

from anylabeling.configs import auto_labeling as auto_labeling_configs
from anylabeling.services.auto_labeling.types import AutoLabelingResult
from anylabeling.utils import GenericWorker

from anylabeling.config import get_config, save_config

import ssl

ssl._create_default_https_context = (
    ssl._create_unverified_context
)  # Prevent issue when downloading models behind a proxy


class ModelManager(QObject):
    """Model manager"""

    MAX_NUM_CUSTOM_MODELS = 5

    model_configs_changed = pyqtSignal(list)
    new_model_status = pyqtSignal(str)
    model_loaded = pyqtSignal(dict)
    new_auto_labeling_result = pyqtSignal(AutoLabelingResult)
    auto_segmentation_model_selected = pyqtSignal()
    auto_segmentation_model_unselected = pyqtSignal()
    prediction_started = pyqtSignal()
    prediction_finished = pyqtSignal()
    request_next_files_requested = pyqtSignal()
    output_modes_changed = pyqtSignal(dict, str)

    def __init__(self):
        super().__init__()
        self.model_configs = []

        self.loaded_model_config = None
        self.loaded_model_config_lock = Lock()

        self.model_download_worker = None
        self.model_download_thread = None
        self.model_execution_thread = None
        self.model_execution_worker = None
        self.model_execution_thread_lock = Lock()

        self.load_model_configs()

    def load_model_configs(self):
        """Load model configs"""
        # Load list of default models
        with pkg_resources.open_text(
            auto_labeling_configs, "models.yaml"
        ) as f:
            model_list = yaml.safe_load(f)
            for model in model_list:
                model["is_custom_model"] = False

            # Check downloaded
            for model in model_list:
                home_dir = os.path.expanduser("~")
                model_download_path = os.path.join(
                    home_dir, "anylabeling_data", "models", model["name"]
                )
                pathlib.Path(model_download_path).mkdir(
                    parents=True, exist_ok=True
                )
                config_file = os.path.join(model_download_path, "config.yaml")
                model["config_file"] = config_file

                # Initialize model config if needed
                if not os.path.isfile(config_file):
                    model["has_downloaded"] = False
                    with open(config_file, "w") as f:
                        yaml.dump(model, f)

        # Load list of custom models
        custom_models = get_config().get("custom_models", [])
        for custom_model in custom_models:
            custom_model["is_custom_model"] = True
            custom_model["has_downloaded"] = True

        # Remove invalid/not found custom models
        custom_models = [
            custom_model
            for custom_model in custom_models
            if os.path.isfile(custom_model.get("config_file", ""))
        ]
        config = get_config()
        config["custom_models"] = custom_models
        save_config(config)

        model_list += custom_models

        # Load model configs
        model_configs = []
        for model in model_list:
            model_config = copy.deepcopy(model)
            config_file = model.get("config_file", None)
            if config_file:
                with open(config_file, "r") as f:
                    model_config = yaml.safe_load(f)
                    model_config["config_file"] = os.path.normpath(
                        os.path.abspath(config_file)
                    )
                    model_config["is_custom_model"] = model.get(
                        "is_custom_model", False
                    )
            model_configs.append(model_config)

        # Sort by last used
        for i, model_config in enumerate(model_configs):
            # Keep order for integrated models
            if not model_config.get("is_custom_model", False):
                model_config["last_used"] = -i
            else:
                model_config["last_used"] = model_config.get(
                    "last_used", time.time()
                )
        model_configs.sort(key=lambda x: x.get("last_used", 0), reverse=True)

        self.model_configs = model_configs
        self.model_configs_changed.emit(model_configs)

    def get_model_configs(self):
        """Return model infos"""
        return self.model_configs

    def set_output_mode(self, mode):
        """Set output mode"""
        if self.loaded_model_config and self.loaded_model_config["model"]:
            self.loaded_model_config["model"].set_output_mode(mode)

    @pyqtSlot()
    def on_model_download_finished(self):
        """Handle model download thread finished"""
        if self.loaded_model_config and self.loaded_model_config["model"]:
            self.new_model_status.emit(
                self.tr("Model loaded. Ready for labeling.")
            )
            self.model_loaded.emit(self.loaded_model_config)
            self.output_modes_changed.emit(
                self.loaded_model_config["model"].Meta.output_modes,
                self.loaded_model_config["model"].Meta.default_output_mode,
            )
        else:
            self.model_loaded.emit({})

    def load_custom_model(self, config_file):
        """Run custom model loading in a thread"""
        config_file = os.path.normpath(os.path.abspath(config_file))
        if (
            self.model_download_thread is not None
            and self.model_download_thread.isRunning()
        ):
            print(
                "Another model is being loaded. Please wait for it to finish."
            )
            return

        # Check config file path
        if not config_file or not os.path.isfile(config_file):
            self.new_model_status.emit(
                self.tr("Error in loading custom model: Invalid path.")
            )
            return

        # Check config file content
        model_config = {}
        with open(config_file, "r") as f:
            model_config = yaml.safe_load(f)
            model_config["config_file"] = os.path.abspath(config_file)
        if not model_config:
            self.new_model_status.emit(
                self.tr("Error in loading custom model: Invalid config file.")
            )
            return
        if (
            "type" not in model_config
            or "display_name" not in model_config
            or "name" not in model_config
            or model_config["type"]
            not in ["segment_anything", "yolov5", "yolov8"]
        ):
            self.new_model_status.emit(
                self.tr(
                    "Error in loading custom model: Invalid config file format."
                )
            )
            return

        # Add or replace custom model
        custom_models = get_config().get("custom_models", [])
        matched_index = None
        for i, model in enumerate(custom_models):
            if os.path.normpath(model["config_file"]) == os.path.normpath(
                config_file
            ):
                matched_index = i
                break
        if matched_index is not None:
            model_config["last_used"] = time.time()
            custom_models[matched_index] = model_config
        else:
            if len(custom_models) >= self.MAX_NUM_CUSTOM_MODELS:
                custom_models.sort(
                    key=lambda x: x.get("last_used", 0), reverse=True
                )
                removed_model = custom_models.pop()
                # Remove old model folder
                config_file = removed_model["config_file"]
                if os.path.exists(config_file):
                    try:
                        pathlib.Path(config_file).parent.rmdir()
                    except OSError:
                        pass
            custom_models = [model_config] + custom_models

        # Save config
        config = get_config()
        config["custom_models"] = custom_models
        save_config(config)

        # Reload model configs
        self.load_model_configs()

        # Load model
        self.load_model(model_config["config_file"])

    def load_model(self, config_file):
        """Run model loading in a thread"""
        if (
            self.model_download_thread is not None
            and self.model_download_thread.isRunning()
        ):
            print(
                "Another model is being loaded. Please wait for it to finish."
            )
            return
        if not config_file:
            if self.model_download_worker is not None:
                try:
                    self.model_download_worker.finished.disconnect(
                        self.on_model_download_finished
                    )
                except TypeError:
                    pass
            self.unload_model()
            self.new_model_status.emit(self.tr("No model selected."))
            return

        # Check and get model id
        model_id = None
        for i, model_config in enumerate(self.model_configs):
            if model_config["config_file"] == config_file:
                model_id = i
                break
        if model_id is None:
            self.new_model_status.emit(
                self.tr("Error in loading model: Invalid model name.")
            )
            return

        self.model_download_thread = QThread()
        self.new_model_status.emit(
            self.tr("Loading model: {model_name}. Please wait...").format(
                model_name=self.model_configs[model_id]["display_name"]
            )
        )
        self.model_download_worker = GenericWorker(self._load_model, model_id)
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

    def _download_and_extract_model(self, model_config):
        """Download and extract a model from model config"""
        config_file = model_config["config_file"]
        # Check if model is already downloaded
        if not os.path.exists(config_file):
            raise ValueError(self.tr("Error in loading config file."))
        with open(config_file, "r") as f:
            model_config = yaml.safe_load(f)
        if model_config.get("has_downloaded", False):
            return

        # Download model
        download_url = model_config.get("download_url", None)
        if not download_url:
            raise ValueError(self.tr("Missing download_url in config file."))
        tmp_dir = tempfile.mkdtemp()
        zip_model_path = os.path.join(tmp_dir, "model.zip")

        # Download url
        ellipsis_download_url = download_url
        if len(download_url) > 40:
            ellipsis_download_url = (
                download_url[:20] + "..." + download_url[-20:]
            )
        logging.info(
            "Downloading %s to %s", ellipsis_download_url, zip_model_path
        )
        try:
            # Download and show progress
            def _progress(count, block_size, total_size):
                percent = int(count * block_size * 100 / total_size)
                self.new_model_status.emit(
                    QCoreApplication.translate(
                        "Model", "Downloading {download_url}: {percent}%"
                    ).format(
                        download_url=ellipsis_download_url, percent=percent
                    )
                )

            urllib.request.urlretrieve(
                download_url, zip_model_path, reporthook=_progress
            )
        except Exception as e:  # noqa
            print(f"Could not download {download_url}: {e}")
            self.new_model_status.emit(f"Could not download {download_url}")
            return None

        # Extract model
        tmp_extract_dir = os.path.join(tmp_dir, "extract")
        extract_dir = os.path.dirname(config_file)
        with zipfile.ZipFile(zip_model_path, "r") as zip_ref:
            zip_ref.extractall(tmp_extract_dir)

        # Find model folder (containing config.yaml)
        model_folder = None
        for root, _, files in os.walk(tmp_extract_dir):
            if "config.yaml" in files:
                model_folder = root
                break
        if model_folder is None:
            raise ValueError(
                self.tr("Could not find config.yaml in zip file.")
            )

        # Move model folder to correct location
        shutil.rmtree(extract_dir)
        shutil.move(model_folder, extract_dir)

        # Clean up
        shutil.rmtree(tmp_dir)

        # Update config file
        with open(config_file, "r") as f:
            model_config = yaml.safe_load(f)
        model_config["has_downloaded"] = True
        model_config["config_file"] = config_file
        with open(config_file, "w") as f:
            yaml.dump(model_config, f)

        return model_config

    def _load_model(self, model_id):
        """Load and return model info"""
        if self.loaded_model_config is not None:
            self.loaded_model_config["model"].unload()
            self.loaded_model_config = None
            self.auto_segmentation_model_unselected.emit()

        model_config = copy.deepcopy(self.model_configs[model_id])

        # Download and extract model
        if not model_config.get("has_downloaded", True):
            model_config = self._download_and_extract_model(model_config)
            if model_config is None:
                return

            self.model_configs[model_id].update(model_config)

        if model_config["type"] == "yolov5":
            from .yolov5 import YOLOv5

            try:
                model_config["model"] = YOLOv5(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_unselected.emit()
            except Exception as e:  # noqa
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                print(
                    "Error in loading model: {error_message}".format(
                        error_message=str(e)
                    )
                )
                return
        elif model_config["type"] == "yolov8":
            from .yolov8 import YOLOv8

            try:
                model_config["model"] = YOLOv8(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_unselected.emit()
            except Exception as e:  # noqa
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                print(
                    "Error in loading model: {error_message}".format(
                        error_message=str(e)
                    )
                )
                return
        elif model_config["type"] == "segment_anything":
            from .segment_anything import SegmentAnything

            try:
                model_config["model"] = SegmentAnything(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_selected.emit()
            except Exception as e:  # noqa
                print(
                    "Error in loading model: {error_message}".format(
                        error_message=str(e)
                    )
                )
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                return

            # Request next files for prediction
            self.request_next_files_requested.emit()
        else:
            raise Exception(f"Unknown model type: {model_config['type']}")

        self.loaded_model_config = model_config
        return self.loaded_model_config

    def set_auto_labeling_marks(self, marks):
        """Set auto labeling marks
        (For example, for segment_anything model, it is the marks for)
        """
        if (
            self.loaded_model_config is None
            or self.loaded_model_config["type"] != "segment_anything"
        ):
            return
        self.loaded_model_config["model"].set_auto_labeling_marks(marks)

    def unload_model(self):
        """Unload model"""
        if self.loaded_model_config is not None:
            self.loaded_model_config["model"].unload()
            self.loaded_model_config = None

    def predict_shapes(self, image, filename=None):
        """Predict shapes.
        NOTE: This function is blocking. The model can take a long time to
        predict. So it is recommended to use predict_shapes_threading instead.
        """
        if self.loaded_model_config is None:
            self.new_model_status.emit(
                self.tr("Model is not loaded. Choose a mode to continue.")
            )
            self.prediction_finished.emit()
            return
        try:
            auto_labeling_result = self.loaded_model_config[
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
        if self.loaded_model_config is None:
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
        if self.loaded_model_config is None:
            return

        # Currently only segment_anything model supports this feature
        if self.loaded_model_config["type"] != "segment_anything":
            return

        self.loaded_model_config["model"].on_next_files_changed(next_files)
