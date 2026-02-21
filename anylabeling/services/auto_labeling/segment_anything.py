import logging
import os
import traceback

import cv2
import numpy as np
import onnx
from PyQt6 import QtCore
from PyQt6.QtCore import QCoreApplication, QThread

from anylabeling.utils import GenericWorker
from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.utils.opencv import qt_img_to_rgb_cv_img

from .lru_cache import LRUCache
from .model import Model
from .registry import ModelRegistry
from .sam2_onnx import SegmentAnything2ONNX
from .sam3_onnx import SegmentAnything3ONNX
from .sam_onnx import SegmentAnythingONNX
from .types import AutoLabelingResult


@ModelRegistry.register("segment_anything")
class SegmentAnything(Model):
    """Segmentation model using SegmentAnything"""

    class Meta:
        required_config_names = [
            "type",
            "name",
            "display_name",
            "encoder_model_path",
            "decoder_model_path",
        ]
        widgets = [
            "output_label",
            "output_select_combobox",
            "label_prompt",
            "edit_prompt",
            "button_run",
            "button_add_point",
            "button_remove_point",
            "button_add_rect",
            "button_clear",
            "button_finish_object",
        ]
        output_modes = {
            "polygon": QCoreApplication.translate("Model", "Polygon"),
            "rectangle": QCoreApplication.translate("Model", "Rectangle"),
        }
        default_output_mode = "polygon"

    def __init__(self, config_path, on_message) -> None:
        # Run the parent class's init method
        super().__init__(config_path, on_message)
        self.input_size = self.config["input_size"]
        self.max_width = self.config["max_width"]
        self.max_height = self.config["max_height"]
        self.text_prompt = "visual"
        self.prompt_mode = "visual"
        self.confidence_threshold = self.config.get("confidence_threshold", 0.5)

        # Get encoder and decoder model paths
        encoder_model_abs_path = self.get_model_abs_path(
            self.config, "encoder_model_path"
        )
        if not encoder_model_abs_path or not (
            os.path.isfile(encoder_model_abs_path)
            or os.path.isdir(encoder_model_abs_path)
        ):
            raise FileNotFoundError(
                QCoreApplication.translate(
                    "Model",
                    "Could not download or initialize encoder of Segment Anything.",
                )
            )
        decoder_model_abs_path = self.get_model_abs_path(
            self.config, "decoder_model_path"
        )
        if not decoder_model_abs_path or not (
            os.path.isfile(decoder_model_abs_path)
            or os.path.isdir(decoder_model_abs_path)
        ):
            raise FileNotFoundError(
                QCoreApplication.translate(
                    "Model",
                    "Could not download or initialize decoder of Segment Anything.",
                )
            )

        # Detect the model variant once and cache it – avoids re-loading the
        # ONNX graph on every call to predict_shapes.
        self._model_variant: str = self.detect_model_variant(decoder_model_abs_path)
        self._is_sam3: bool = self._model_variant == "sam3"

        # Load models
        if "coreml" in decoder_model_abs_path:
            from .sam2_coreml import SegmentAnything2CoreML  # macOS-only

            config_folder = os.path.dirname(decoder_model_abs_path)
            self.model = SegmentAnything2CoreML(config_folder)
        elif "language_encoder_path" in self.config or self._is_sam3:
            language_encoder_abs_path = self.get_model_abs_path(
                self.config, "language_encoder_path"
            )
            self.model = SegmentAnything3ONNX(
                encoder_model_abs_path,
                decoder_model_abs_path,
                language_encoder_abs_path,
            )
            self._is_sam3 = True  # Ensure flag is set when config triggers SAM3
        elif self._model_variant == "sam2":
            self.model = SegmentAnything2ONNX(
                encoder_model_abs_path, decoder_model_abs_path
            )
        else:
            self.model = SegmentAnythingONNX(
                encoder_model_abs_path, decoder_model_abs_path
            )

        # Mark for auto labeling
        # points, rectangles
        self.marks = []

        # Cache for image embedding
        self.cache_size = 10
        self.preloaded_size = self.cache_size - 3
        self.image_embedding_cache = LRUCache(self.cache_size)

        # Pre-inference worker
        self.pre_inference_thread = None
        self.pre_inference_worker = None
        self.stop_inference = False

    def detect_model_variant(self, decoder_model_abs_path: str) -> str:
        """Detect SAM model variant from the decoder ONNX graph.

        Detection heuristics (based on unique input names):
          - SAM3  → has ``backbone_fpn_0`` *or* ``language_mask``
          - SAM2  → has ``high_res_feats_0``
          - SAM   → anything else

        Note: after onnxsim simplification the decoder may *not* contain
        ``vision_pos_enc_0``/``vision_pos_enc_1`` (they are optimised away),
        so those names are not used for detection.
        """
        model = onnx.load(decoder_model_abs_path)
        input_names = {inp.name for inp in model.graph.input}
        if "backbone_fpn_0" in input_names or "language_mask" in input_names:
            return "sam3"
        if "high_res_feats_0" in input_names:
            return "sam2"
        return "sam"

    def set_text_prompt(self, text_prompt):
        """Set text prompt"""
        if self.text_prompt != text_prompt:
            self.text_prompt = text_prompt
            # Clear cache when text prompt changed
            self.image_embedding_cache.clear()

    def set_prompt_mode(self, prompt_mode):
        """Set prompt mode"""
        self.prompt_mode = prompt_mode

    def set_confidence_threshold(self, threshold):
        """Set confidence threshold"""
        self.confidence_threshold = threshold

    def set_auto_labeling_marks(self, marks):
        """Set auto labeling marks"""
        self.marks = marks

    def post_process(self, masks, label="AUTOLABEL_OBJECT"):
        """Post-process a single 2-D mask into AnyLabeling Shape objects.

        Parameters
        ----------
        masks:
            2-D array of shape ``(H, W)``.  May be bool, float, or uint8.
        """
        # Ensure the mask is 2D
        while len(masks.shape) > 2:
            masks = masks[0]

        # Ensure the mask is a float/uint8 array so that assignment of the
        # value 255 works correctly (bool arrays raise an error in NumPy ≥ 2).
        masks = masks.astype(np.float32)
        masks[masks > 0.0] = 255
        masks[masks <= 0.0] = 0
        masks = masks.astype(np.uint8)
        contours, _ = cv2.findContours(masks, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Refine contours
        approx_contours = []
        for contour in contours:
            # Approximate contour
            epsilon = 0.001 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            approx_contours.append(approx)

        # Remove too big contours ( >90% of image size)
        if len(approx_contours) > 1:
            image_size = masks.shape[0] * masks.shape[1]
            areas = [cv2.contourArea(contour) for contour in approx_contours]
            approx_contours = [
                contour
                for contour, area in zip(approx_contours, areas)
                if area < image_size * 0.9
            ]

        # Remove small contours (area < 20% of average area)
        if len(approx_contours) > 1:
            areas = [cv2.contourArea(contour) for contour in approx_contours]
            avg_area = np.mean(areas)

            approx_contours = [
                contour
                for contour, area in zip(approx_contours, areas)
                if area > avg_area * 0.2
            ]

        # Contours to shapes
        shapes = []
        if self.output_mode == "polygon":
            for approx in approx_contours:
                # Scale points
                points = approx.reshape(-1, 2)
                points[:, 0] = points[:, 0]
                points[:, 1] = points[:, 1]
                points = points.tolist()
                if len(points) < 3:
                    continue
                points.append(points[0])

                # Create shape
                shape = Shape(flags={})
                for point in points:
                    point[0] = int(point[0])
                    point[1] = int(point[1])
                    shape.add_point(QtCore.QPointF(point[0], point[1]))
                shape.shape_type = "polygon"
                shape.closed = True
                shape.fill_color = "#000000"
                shape.line_color = "#000000"
                shape.line_width = 1
                shape.label = label
                shape.selected = False
                shapes.append(shape)
        elif self.output_mode == "rectangle":
            x_min = 100000000
            y_min = 100000000
            x_max = 0
            y_max = 0
            for approx in approx_contours:
                # Scale points
                points = approx.reshape(-1, 2)
                points[:, 0] = points[:, 0]
                points[:, 1] = points[:, 1]
                points = points.tolist()
                if len(points) < 3:
                    continue

                # Get min/max
                for point in points:
                    x_min = min(x_min, point[0])
                    y_min = min(y_min, point[1])
                    x_max = max(x_max, point[0])
                    y_max = max(y_max, point[1])

            # Create shape
            shape = Shape(flags={})
            shape.add_point(QtCore.QPointF(x_min, y_min))
            shape.add_point(QtCore.QPointF(x_max, y_max))
            shape.shape_type = "rectangle"
            shape.closed = True
            shape.fill_color = "#000000"
            shape.line_color = "#000000"
            shape.line_width = 1
            shape.label = label
            shape.selected = False
            shapes.append(shape)

        return shapes

    def predict_shapes(self, image, filename=None) -> AutoLabelingResult:
        """
        Predict shapes from image
        """
        self.stop_inference = False
        if image is None or (not self.marks and self.prompt_mode != "text"):
            return AutoLabelingResult([], replace=False)

        shapes = []
        try:
            # Use cached image embedding if possible
            cached_data = self.image_embedding_cache.get(filename)
            if cached_data is not None:
                image_embedding = cached_data
            else:
                cv_image = qt_img_to_rgb_cv_img(image, filename)
                if self.stop_inference:
                    return AutoLabelingResult([], replace=False)
                # For SAM3, pass the text prompt so the language encoder runs
                # during the same encode() call as the image encoder.
                if self._is_sam3:
                    image_embedding = self.model.encode(
                        cv_image, text_prompt=self.text_prompt
                    )
                else:
                    image_embedding = self.model.encode(cv_image)
                self.image_embedding_cache.put(filename, image_embedding)

            if self.stop_inference:
                return AutoLabelingResult([], replace=False)

            if self._is_sam3:
                # ── SAM3 path ─────────────────────────────────────────────
                # Text mode supports comma-separated multi-class prompts.
                # E.g. "egg, bottle" detects ALL eggs AND ALL bottles and
                # labels each shape with its own class name.
                # Visual mode (point/rect) uses text_prompt as a single cue.
                if self.prompt_mode == "text":
                    class_terms = [
                        t.strip() for t in self.text_prompt.split(",") if t.strip()
                    ]
                    if not class_terms:
                        class_terms = [self.text_prompt or "visual"]
                else:
                    class_terms = [self.text_prompt or "visual"]

                inference_marks = self.marks if self.prompt_mode != "text" else []

                for term in class_terms:
                    # Re-run language encoder for this specific class term.
                    # Image features are reused from the cached embedding (fast).
                    term_embedding = self.model.update_language(image_embedding, term)
                    term_masks = self.model.predict_masks(
                        term_embedding,
                        inference_marks,
                        confidence_threshold=self.confidence_threshold,
                    )
                    if term_masks is None or len(term_masks) == 0:
                        continue

                    label = term if self.prompt_mode == "text" else "AUTOLABEL_OBJECT"
                    # SAM3 returns one mask per detected object instance.
                    # Iterate ALL masks so every matching object gets a shape.
                    for i in range(len(term_masks)):
                        mask_2d = term_masks[i, 0]  # (N, 1, H, W) → (H, W)
                        shapes.extend(self.post_process(mask_2d, label=label))
            else:
                # ── SAM1 / SAM2 path ──────────────────────────────────────
                # The decoder returns 3 quality-level candidates for the same
                # prompt; use only the first (highest-quality) mask.
                inference_marks = self.marks if self.prompt_mode != "text" else []
                masks = self.model.predict_masks(image_embedding, inference_marks)

                if masks is None or len(masks) == 0:
                    return AutoLabelingResult([], replace=False)

                mask_2d = masks
                while len(mask_2d.shape) > 2:
                    mask_2d = mask_2d[0]
                shapes = self.post_process(mask_2d, label="AUTOLABEL_OBJECT")
        except Exception as e:  # noqa
            logging.warning("Could not inference model")
            logging.warning(e)
            traceback.print_exc()
            return AutoLabelingResult([], replace=False)

        result = AutoLabelingResult(shapes, replace=False)
        return result

    def unload(self):
        self.stop_inference = True
        if self.pre_inference_thread:
            try:
                self.pre_inference_thread.quit()
                # Wait for the thread to actually finish (increased to 3s)
                if not self.pre_inference_thread.wait(3000):
                    logging.warning("Pre-inference thread did not stop in time")
                else:
                    self.pre_inference_thread = None
            except RuntimeError:
                self.pre_inference_thread = None

    def preload_worker(self, files):
        """
        Preload next files, run inference and cache results
        """
        files = files[: self.preloaded_size]
        for filename in files:
            if self.image_embedding_cache.find(filename):
                continue
            image = self.load_image_from_filename(filename)
            if image is None:
                continue
            if self.stop_inference:
                return
            cv_image = qt_img_to_rgb_cv_img(image)
            image_embedding = self.model.encode(cv_image)
            self.image_embedding_cache.put(
                filename,
                image_embedding,
            )

    def on_next_files_changed(self, next_files):
        """
        Handle next files changed. This function can preload next files
        and run inference to save time for user.
        """
        if (
            self.pre_inference_thread is None
            or not self.pre_inference_thread.isRunning()
        ):
            self.pre_inference_thread = QThread()
            self.pre_inference_worker = GenericWorker(self.preload_worker, next_files)
            self.pre_inference_worker.finished.connect(self.pre_inference_thread.quit)
            self.pre_inference_worker.finished.connect(
                self.pre_inference_worker.deleteLater
            )
            # Reset reference when the thread actually finishes
            self.pre_inference_thread.finished.connect(
                lambda: setattr(self, "pre_inference_thread", None)
            )
            self.pre_inference_thread.finished.connect(
                self.pre_inference_thread.deleteLater
            )
            self.pre_inference_worker.moveToThread(self.pre_inference_thread)
            self.pre_inference_thread.started.connect(self.pre_inference_worker.run)
            self.pre_inference_thread.start()
