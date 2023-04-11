from abc import abstractmethod
import logging
import os
import pathlib
import urllib.request
import yaml

from .types import AutoLabelingResult


class Model:
    BASE_DOWNLOAD_URL = (
        "https://github.com/vietanhdev/anylabeling-assets/raw/main/"
    )

    class Meta:
        required_config_names = []
        buttons = ["button_run"]

    def __init__(self, model_config) -> None:
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
            relative_path = model_path.replace("anylabeling_assets/", "")
            download_url = self.BASE_DOWNLOAD_URL + relative_path
            model_abs_path = os.path.join(
                os.path.abspath("data"), relative_path
            )
            if os.path.exists(model_abs_path):
                return model_abs_path
            pathlib.Path(os.path.dirname(model_abs_path)).mkdir(
                parents=True, exist_ok=True
            )

            # Download model from url
            logging.info(
                f"Downloading model from {download_url} to {model_abs_path}"
            )
            data = urllib.request.urlopen(download_url).read()
            with open(model_abs_path, "wb") as f:
                f.write(data)

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
