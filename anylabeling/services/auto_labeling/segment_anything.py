import logging
import os
from copy import deepcopy

import onnxruntime
import cv2
import numpy as np
from PyQt5 import QtCore

from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.utils.opencv import qt_img_to_cv_img
from .model import Model


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
        buttons = [
            "button_run",
            "button_add_point",
            "button_remove_point",
            "button_add_rect",
            "button_undo",
            "button_clear",
            "button_finish_object",
        ]

    def __init__(self, config_path) -> None:
        # Run the parent class's init method
        super().__init__(config_path)
        self.input_size = self.config["input_size"]

        # Get encoder and decoder model paths
        encoder_model_abs_path = self.get_model_abs_path(
            self.config["encoder_model_path"]
        )
        if not os.path.isfile(encoder_model_abs_path):
            raise Exception(f"Encoder not found: {encoder_model_abs_path}")
        decoder_model_abs_path = self.get_model_abs_path(
            self.config["decoder_model_path"]
        )
        if not os.path.isfile(decoder_model_abs_path):
            raise Exception(f"Decoder not found: {decoder_model_abs_path}")

        # Load models
        self.encoder_session = onnxruntime.InferenceSession(
            encoder_model_abs_path
        )
        self.decoder_session = onnxruntime.InferenceSession(
            decoder_model_abs_path
        )

    def pre_process(self, image):
        image_size = self.input_size

        # Resize longest side
        self.original_size = image.shape[:2]
        h, w = image.shape[:2]
        if h > w:
            new_h, new_w = image_size, int(w * image_size / h)
        else:
            new_h, new_w = int(h * image_size / w), image_size
        input_image = cv2.resize(image, (new_w, new_h))

        # Normalize
        pixel_mean = np.array([123.675, 116.28, 103.53]).reshape(1, 1, -1)
        pixel_std = np.array([58.395, 57.12, 57.375]).reshape(1, 1, -1)
        x = (input_image - pixel_mean) / pixel_std

        # Padding to square
        h, w = x.shape[:2]
        padh = image_size - h
        padw = image_size - w
        x = np.pad(x, ((0, padh), (0, padw), (0, 0)), mode="constant")
        x = x.astype(np.float32)

        # Transpose
        x = x.transpose(2, 0, 1)[None, :, :, :]

        encoder_inputs = {
            "x": x,
        }
        return encoder_inputs

    def run_encoder(self, encoder_inputs):
        output = self.encoder_session.run(None, encoder_inputs)
        image_embedding = output[0]
        return image_embedding

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int):
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)

    @staticmethod
    def apply_coords(
        coords: np.ndarray, original_size, target_length
    ) -> np.ndarray:
        """
        Expects a numpy array of length 2 in the final dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = SegmentAnything.get_preprocess_shape(
            original_size[0], original_size[1], target_length
        )
        coords = deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def run_decoder(self, image_embedding):
        input_point = np.array([[100, 100]])
        input_label = np.array([1])

        # Add a batch index, concatenate a padding point, and transform.
        onnx_coord = np.concatenate(
            [input_point, np.array([[0.0, 0.0]])], axis=0
        )[None, :, :]
        onnx_label = np.concatenate([input_label, np.array([-1])], axis=0)[
            None, :
        ].astype(np.float32)
        onnx_coord = self.apply_coords(
            onnx_coord, self.original_size, self.input_size
        ).astype(np.float32)

        # Create an empty mask input and an indicator for no mask.
        onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
        onnx_has_mask_input = np.zeros(1, dtype=np.float32)

        decoder_inputs = {
            "image_embeddings": image_embedding,
            "point_coords": onnx_coord,
            "point_labels": onnx_label,
            "mask_input": onnx_mask_input,
            "has_mask_input": onnx_has_mask_input,
            "orig_im_size": np.array(self.original_size, dtype=np.float32),
        }
        masks, _, low_res_logits = self.decoder_session.run(
            None, decoder_inputs
        )
        masks = masks > 0.0
        masks = masks.reshape(self.original_size)
        return masks

    def post_process(self, masks):
        """
        Post process masks
        """
        shapes = []
        # Find contours
        contours, _ = cv2.findContours(
            masks.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        for contour in contours:
            # Approximate contour
            epsilon = 0.001 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            points = approx.reshape(-1, 2).tolist()
            if len(points) < 3:
                continue
            points.append(points[0])

            # Create shape
            shape = Shape(flags={})
            for point in points:
                point[0] = int(point[0])
                point[1] = int(point[1])
                shape.add_point(QtCore.QPointF(point[0], point[1]))
            shape.type = "polygon"
            shape.closed = True
            shape.fill_color = "#000000"
            shape.line_color = "#000000"
            shape.line_width = 1
            shape.label = "unknown"
            shape.selected = False
            shapes.append(shape)

        return shapes

    def predict_shapes(self, image):
        """
        Predict shapes from image
        """
        if image is None:
            return []

        shapes = []
        try:
            image = qt_img_to_cv_img(image)
            encoder_inputs = self.pre_process(image)
            image_embedding = self.run_encoder(encoder_inputs)
            masks = self.run_decoder(image_embedding)
            shapes = self.post_process(masks)
        except Exception as e:
            logging.warning("Could not inference model")
            logging.warning(e)
            return []

        return shapes

    def unload(self):
        if self.encoder_session:
            self.encoder_session = None
        if self.decoder_session:
            self.decoder_session = None
