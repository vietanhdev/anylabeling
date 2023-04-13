import logging
import os
import pathlib
import urllib.request
from abc import abstractmethod

import yaml

from .types import AutoLabelingResult


class Model:
    BASE_DOWNLOAD_URL = (
        "https://github.com/hdnh2006/anylabeling-assets/releases/download/v0.0.0/"
    )

    class Meta:
        required_config_names = []
        buttons = ["button_run"]

    def __init__(self, model_config, on_message) -> None:
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

    def get_required_buttons(self):
        """
        Get required buttons for showing in UI
        """
        return self.Meta.buttons

    def get_model_abs_path(self, model_path):
        """
        Get model absolute path from config path or download from url
        """
        # Try getting model path from config folder
        model_abs_path = os.path.abspath(model_path)
        if os.path.exists(model_abs_path):
            return model_abs_path

        # Try download model from url
        if model_path.startswith("anylabeling_assets/"):
            self.on_message(
                "Downloading model from model registry. This may take a"
                " while..."
            )
            relative_path = model_path.replace("anylabeling_assets/", "")
            download_url = self.BASE_DOWNLOAD_URL + relative_path.replace("models/yolov5/","").replace("models/segment_anything/","")
            home_dir = os.getcwd()
            model_abs_path = os.path.abspath(
                os.path.join(home_dir, "anylabeling_data", relative_path)
            )
            if os.path.exists(model_abs_path):
                return model_abs_path
            pathlib.Path(model_abs_path).parent.mkdir(
                parents=True, exist_ok=True
            )

            # Download model from url
            logging.info(
                f"Downloading model from {download_url} to {model_abs_path}"
            )

            try:
                data = urllib.request.urlopen(download_url).read()
                with open(model_abs_path, "wb") as f:
                    f.write(data)
            except Exception as e:  # noqa
                self.on_message(
                    f"Could not downloading model from {download_url}"
                )
                raise Exception(
                    f"Could not downloading model from {download_url}: {e}"
                ) from e

            return model_abs_path

        return None

    def check_missing_config(self, config_names, config):
        """
        Check if config has all required config names
        """
        for name in config_names:
            if name not in config:
                raise Exception(f"Missing config: {name}")

    @abstractmethod
    def predict_shapes(self, image) -> AutoLabelingResult:
        """
        Predict image and return AnyLabeling shapes
        """
        raise NotImplementedError

    @abstractmethod
    def unload(self):
        """
        Unload model from memory
        """
        raise NotImplementedError
