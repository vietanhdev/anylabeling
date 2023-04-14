import logging
import os
import pathlib
import urllib.request
from abc import abstractmethod

import yaml

from .types import AutoLabelingResult


class Model:
    BASE_DOWNLOAD_URL = (
        "https://github.com/vietanhdev/anylabeling-assets/raw/main/"
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

    def get_model_abs_path(self, model_path, model_folder_name):
        """
        Get model absolute path from config path or download from url
        """
        # Try getting model path from config folder
        model_abs_path = os.path.abspath(model_path)
        if os.path.exists(model_abs_path):
            return model_abs_path

        self.on_message(
            "Downloading model from model registry. This may take a while..."
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
            return model_abs_path
        pathlib.Path(model_abs_path).parent.mkdir(parents=True, exist_ok=True)

        # Download model from url
        ellipsis_download_url = download_url
        if len(download_url) > 40:
            ellipsis_download_url = (
                download_url[:20] + "..." + download_url[-20:]
            )
        logging.info(
            f"Downloading model from {ellipsis_download_url} to {model_abs_path}"
        )
        try:
            # Download and show progress
            def _progress(count, block_size, total_size):
                percent = int(count * block_size * 100 / total_size)
                self.on_message(
                    f"Downloading model from {ellipsis_download_url}: {percent}%"
                )

            urllib.request.urlretrieve(
                download_url, model_abs_path, reporthook=_progress
            )
        except Exception as e:  # noqa
            self.on_message(f"Could not download model from {download_url}")
            raise Exception(
                f"Could not download model from {download_url}: {e}"
            ) from e

        return model_abs_path

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
