import logging
import os
import pathlib
import yaml
import onnx
import urllib.request

# Temporarily disable SSL verification
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

from abc import abstractmethod


from PyQt5.QtCore import QFile, QObject
from PyQt5.QtGui import QImage
from PyQt5.QtCore import QCoreApplication

from .types import AutoLabelingResult
from anylabeling.views.labeling.label_file import LabelFile, LabelFileError


class Model(QObject):
    BASE_DOWNLOAD_URL = (
        "https://github.com/vietanhdev/anylabeling-assets/raw/main/"
    )

    class Meta(QObject):
        required_config_names = []
        widgets = ["button_run"]
        output_modes = {
            "rectangle": QCoreApplication.translate("Model", "Rectangle"),
        }
        default_output_mode = "rectangle"

    def __init__(self, model_config, on_message) -> None:
        super().__init__()
        self.on_message = on_message
        # Load and check config
        if isinstance(model_config, str):
            if not os.path.isfile(model_config):
                raise Exception(f"Config file not found: {model_config}")
            with open(model_config, "r") as f:
                self.config = yaml.safe_load(f)
        elif isinstance(model_config, dict):
            self.config = model_config
        else:
            raise Exception(f"Unknown config type: {type(model_config)}")
        self.check_missing_config(
            config_names=self.Meta.required_config_names,
            config=self.config,
        )
        self.output_mode = self.Meta.default_output_mode

    def get_required_widgets(self):
        """
        Get required widgets for showing in UI
        """
        return self.Meta.widgets

    def get_model_abs_path(self, model_path, model_folder_name):
        """
        Get model absolute path from config path or download from url
        """
        # Try getting model path from config folder
        model_abs_path = os.path.abspath(model_path)
        if os.path.exists(model_abs_path):
            return model_abs_path

        self.on_message(
            QCoreApplication.translate(
                "Model", "Downloading model from registry..."
            )
        )

        # Build download url
        filename = os.path.basename(model_path)
        if model_path.startswith("anylabeling_assets/"):
            download_url = (
                self.BASE_DOWNLOAD_URL
                + model_path[len("anylabeling_assets/") :]
            )
        elif model_path.startswith(("http://", "https://")):
            download_url = model_path
        else:
            raise Exception(
                f"Unknown model path: {model_path}. "
                "Model path must start with anylabeling_assets/ or "
                "http:// or https://"
            )

        # Create model folder
        home_dir = os.path.expanduser("~")
        model_abs_path = os.path.abspath(
            os.path.join(
                home_dir,
                "anylabeling_data",
                "models",
                model_folder_name,
                filename,
            )
        )
        if os.path.exists(model_abs_path):
            if model_abs_path.lower().endswith(".onnx"):
                try:
                    onnx.checker.check_model(model_abs_path)
                except onnx.checker.ValidationError as e:
                    logging.warning("The model is invalid: %s", str(e))
                    logging.warning("Action: Delete and redownload...")
                    os.remove(model_abs_path)
                else:
                    return model_abs_path
            else:
                return model_abs_path
        pathlib.Path(model_abs_path).parent.mkdir(parents=True, exist_ok=True)

        # Download url
        ellipsis_download_url = download_url
        if len(download_url) > 40:
            ellipsis_download_url = (
                download_url[:20] + "..." + download_url[-20:]
            )
        logging.info(
            "Downloading %s to %s", ellipsis_download_url, model_abs_path
        )
        try:
            # Download and show progress
            def _progress(count, block_size, total_size):
                percent = int(count * block_size * 100 / total_size)
                self.on_message(
                    QCoreApplication.translate(
                        "Model", "Downloading {download_url}: {percent}%"
                    ).format(
                        download_url=ellipsis_download_url, percent=percent
                    )
                )

            urllib.request.urlretrieve(
                download_url, model_abs_path, reporthook=_progress
            )
        except Exception as e:  # noqa
            self.on_message(f"Could not download {download_url}")
            raise Exception(f"Could not download {download_url}: {e}") from e

        return model_abs_path

    def check_missing_config(self, config_names, config):
        """
        Check if config has all required config names
        """
        for name in config_names:
            if name not in config:
                raise Exception(f"Missing config: {name}")

    @abstractmethod
    def predict_shapes(self, image, filename=None) -> AutoLabelingResult:
        """
        Predict image and return AnyLabeling shapes
        """
        raise NotImplementedError

    @abstractmethod
    def unload(self):
        """
        Unload memory
        """
        raise NotImplementedError

    @staticmethod
    def load_image_from_filename(filename):
        """Load image from labeling file and return image data and image path."""
        label_file = os.path.splitext(filename)[0] + ".json"
        if QFile.exists(label_file) and LabelFile.is_label_file(label_file):
            try:
                label_file = LabelFile(label_file)
            except LabelFileError as e:
                logging.error("Error reading {}: {}".format(label_file, e))
                return None, None
            image_data = label_file.image_data
        else:
            image_data = LabelFile.load_image_file(filename)
        image = QImage.fromData(image_data)
        if image.isNull():
            logging.error("Error reading {}".format(filename))
        return image

    def on_next_files_changed(self, next_files):
        """
        Handle next files changed. This function can preload next files
        and run inference to save time for user.
        """
        pass

    def set_output_mode(self, mode):
        """
        Set output mode
        """
        self.output_mode = mode
