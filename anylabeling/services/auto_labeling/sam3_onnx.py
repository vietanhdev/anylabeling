from typing import Any

import cv2
import numpy as np
import onnxruntime


class SegmentAnything3ONNX:
    """Segmentation model using Segment Anything 3 (SAM3)"""

    def __init__(
        self,
        image_encoder_path,
        decoder_model_path,
        language_encoder_path=None,
    ) -> None:
        self.image_encoder = SAM3ImageEncoder(image_encoder_path)
        self.language_encoder = None
        if language_encoder_path:
            self.language_encoder = SAM3LanguageEncoder(language_encoder_path)
        self.decoder = SAM3ImageDecoder(decoder_model_path)

    def encode(self, cv_image: np.ndarray, text_prompt=None) -> dict[str, Any]:
        """Encode an image (and optional text prompt) into an embedding dict.

        Parameters
        ----------
        cv_image:
            RGB uint8 image as returned by ``qt_img_to_rgb_cv_img``.
        text_prompt:
            Natural-language description of the target object.
            Falls back to ``"visual"`` when *None*.
        """
        original_size = cv_image.shape[:2]
        image_encoder_outputs = self.image_encoder(cv_image)

        embedding: dict[str, Any] = {
            "vision_pos_enc_0": image_encoder_outputs[0],
            "vision_pos_enc_1": image_encoder_outputs[1],
            "vision_pos_enc_2": image_encoder_outputs[2],
            "backbone_fpn_0": image_encoder_outputs[3],
            "backbone_fpn_1": image_encoder_outputs[4],
            "backbone_fpn_2": image_encoder_outputs[5],
            "original_size": original_size,
            # Pre-fill as None; overwritten when a language encoder is available.
            "language_mask": None,
            "language_features": None,
            "language_embeds": None,
        }

        text_prompt = text_prompt or "visual"
        if self.language_encoder is not None:
            lang_outputs = self.language_encoder(text_prompt)
            # lang_outputs indices:
            #   [0] text_attention_mask  – bool  [1, seq_len]
            #   [1] text_memory          – float [seq_len, 1, 256]
            #   [2] text_embeds          – float [seq_len, 1, 1024]
            embedding["language_mask"] = lang_outputs[0]
            embedding["language_features"] = lang_outputs[1]
            embedding["language_embeds"] = lang_outputs[2]

        return embedding

    def predict_masks(
        self,
        embedding: dict[str, Any],
        prompt,
        confidence_threshold: float = 0.5,
    ) -> np.ndarray:
        """Run the decoder for the given geometric prompt.

        Returns
        -------
        Boolean mask array of shape ``(N, 1, H, W)``.  May be empty
        (shape ``(0, 1, H, W)``) when no confident detections are found.
        """
        original_size = embedding["original_size"]
        box_coords = [0.0, 0.0, 0.0, 0.0]
        box_labels = [1]
        # box_masks: True  → no real box (text-only / dummy)
        #            False → a real box is provided
        box_masks = [True]

        for mark in prompt:
            if mark["type"] == "rectangle":
                x1, y1, x2, y2 = mark["data"]
                cx = (x1 + x2) / 2.0 / original_size[1]
                cy = (y1 + y2) / 2.0 / original_size[0]
                w = (x2 - x1) / original_size[1]
                h = (y2 - y1) / original_size[0]
                box_coords = [cx, cy, w, h]
                box_masks = [False]
                break
            elif mark["type"] == "point":
                x, y = mark["data"]
                cx = x / original_size[1]
                cy = y / original_size[0]
                # Point is represented as a very small box (1 % of image).
                box_coords = [cx, cy, 0.01, 0.01]
                box_masks = [False]
                break

        box_coords_np = np.array(box_coords, dtype=np.float32).reshape(1, 1, 4)
        box_labels_np = np.array([box_labels], dtype=np.int64)
        box_masks_np = np.array([box_masks], dtype=np.bool_)

        masks, scores, _boxes = self.decoder(
            original_size,
            embedding["vision_pos_enc_0"],
            embedding["vision_pos_enc_1"],
            embedding["vision_pos_enc_2"],
            embedding["backbone_fpn_0"],
            embedding["backbone_fpn_1"],
            embedding["backbone_fpn_2"],
            embedding.get("language_mask"),
            embedding.get("language_features"),
            embedding.get("language_embeds"),
            box_coords_np,
            box_labels_np,
            box_masks_np,
        )

        # Filter by confidence threshold.
        if len(scores) > 0:
            keep = np.where(scores > confidence_threshold)[0]
            if len(keep) > 0:
                masks = masks[keep]
            else:
                masks = np.zeros((0,) + masks.shape[1:], dtype=masks.dtype)

        return masks

    def update_language(self, embedding: dict, text_prompt: str) -> dict:
        """Return a shallow copy of *embedding* with language features re-encoded.

        The image tensors (vision_pos_enc_*, backbone_fpn_*) are shared by
        reference from the original embedding, so this is inexpensive.
        Use this to re-run just the language encoder for a different class
        term without repeating the costly image encoding step.
        """
        new_embedding = dict(embedding)  # shallow copy; image tensors shared
        new_embedding["language_mask"] = None
        new_embedding["language_features"] = None
        new_embedding["language_embeds"] = None

        if self.language_encoder is not None:
            lang_outputs = self.language_encoder(text_prompt or "visual")
            new_embedding["language_mask"] = lang_outputs[0]
            new_embedding["language_features"] = lang_outputs[1]
            new_embedding["language_embeds"] = lang_outputs[2]

        return new_embedding

    def transform_masks(self, masks, original_size, transform_matrix):
        """No-op: SAM3 already outputs masks in original image resolution."""
        return masks


class SAM3ImageEncoder:
    """Runs the SAM3 image backbone ONNX model.

    Expected model input
    --------------------
    name  : ``"image"``
    shape : ``[3, 1008, 1008]``
    dtype : ``tensor(uint8)`` (the model includes normalisation internally)
            or ``tensor(float)`` for older exports without normalisation.
    """

    def __init__(self, path: str) -> None:
        self.session = onnxruntime.InferenceSession(
            path, providers=onnxruntime.get_available_providers()
        )
        encoder_input = self.session.get_inputs()[0]
        self.input_name: str = encoder_input.name
        self.input_shape = encoder_input.shape
        self.input_type: str = encoder_input.type

        # Determine H/W from the ONNX shape.
        # Current export: [3, H, W]  (no batch dimension)
        # Legacy export:  [1, 3, H, W]
        if len(self.input_shape) == 3:
            self.input_height: int = int(self.input_shape[1]) or 1008
            self.input_width: int = int(self.input_shape[2]) or 1008
        elif len(self.input_shape) >= 4:
            self.input_height = int(self.input_shape[2]) or 1008
            self.input_width = int(self.input_shape[3]) or 1008
        else:
            self.input_height = 1008
            self.input_width = 1008

    def __call__(self, image: np.ndarray) -> list[np.ndarray]:
        input_tensor = self.prepare_input(image)
        return self.session.run(None, {self.input_name: input_tensor})

    def prepare_input(self, image: np.ndarray) -> np.ndarray:
        """Prepare image tensor for the ONNX encoder.

        The anylabeling pipeline passes an RGB image from
        ``qt_img_to_rgb_cv_img``.  Since the SAM3 normalisation uses equal
        mean/std across all channels (0.5, 0.5, 0.5), RGB vs BGR order
        has no effect on the normalised values, so no colour-space
        conversion is required here.
        """
        input_img = cv2.resize(
            image,
            (self.input_width, self.input_height),
            interpolation=cv2.INTER_LINEAR,
        )
        # (H, W, C) → (C, H, W)
        input_img = input_img.transpose(2, 0, 1)

        if self.input_type == "tensor(float)":
            # Older export without normalisation inside the model.
            # Apply (x/255 − 0.5) / 0.5 to map [0,255] → [−1, 1].
            input_tensor = ((input_img / 255.0) - 0.5) / 0.5
            input_tensor = input_tensor.astype(np.float32)
        else:
            # Current export bakes normalisation in – pass raw uint8.
            input_tensor = input_img.astype(np.uint8)

        return input_tensor


class SAM3LanguageEncoder:
    """Runs the SAM3 language-encoder ONNX model.

    Expected model input
    --------------------
    name  : ``"tokens"``
    shape : ``[1, 32]``
    dtype : int64
    """

    def __init__(self, path: str) -> None:
        self.session = onnxruntime.InferenceSession(
            path, providers=onnxruntime.get_available_providers()
        )
        try:
            from osam._models.yoloworld.clip import tokenize

            self._tokenize = tokenize
        except ImportError:
            self._tokenize = self._fallback_tokenize

    def _fallback_tokenize(self, texts, context_length: int = 32) -> np.ndarray:
        """Minimal fallback tokeniser (all zeros = empty sequence).

        Warning: the model will produce near-random language features when
        this fallback is used.  Install ``osam`` for correct tokenisation.
        """
        return np.zeros((len(texts), context_length), dtype=np.int64)

    def __call__(self, text: str) -> list[np.ndarray]:
        tokens = self._tokenize([text], context_length=32)
        if not isinstance(tokens, np.ndarray):
            tokens = np.asarray(tokens, dtype=np.int64)
        return self.session.run(None, {"tokens": tokens})


class SAM3ImageDecoder:
    """Runs the SAM3 decoder ONNX model.

    Expected output order (ONNX export names):
        [0] boxes  – float (N, 4)
        [1] scores – float (N,)
        [2] masks  – bool  (N, 1, H, W)

    ``__call__`` returns ``(masks, scores, boxes)`` for caller convenience.
    """

    def __init__(self, path: str) -> None:
        self.session = onnxruntime.InferenceSession(
            path, providers=onnxruntime.get_available_providers()
        )
        self.input_names: list[str] = [i.name for i in self.session.get_inputs()]

    def __call__(
        self,
        original_size,
        vision_pos_enc_0,
        vision_pos_enc_1,
        vision_pos_enc_2,
        backbone_fpn_0,
        backbone_fpn_1,
        backbone_fpn_2,
        language_mask,
        language_features,
        language_embeds,
        box_coords,
        box_labels,
        box_masks,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        inputs: dict[str, Any] = {
            "original_height": np.array(original_size[0], dtype=np.int64),
            "original_width": np.array(original_size[1], dtype=np.int64),
            "vision_pos_enc_0": vision_pos_enc_0,
            "vision_pos_enc_1": vision_pos_enc_1,
            "vision_pos_enc_2": vision_pos_enc_2,
            "backbone_fpn_0": backbone_fpn_0,
            "backbone_fpn_1": backbone_fpn_1,
            "backbone_fpn_2": backbone_fpn_2,
            "language_mask": language_mask,
            "language_features": language_features,
            "language_embeds": language_embeds,
            "box_coords": box_coords,
            "box_labels": box_labels,
            "box_masks": box_masks,
        }

        # Supply dummy tensors for language inputs when no encoder was used.
        # Shapes match the actual ONNX decoder's expected inputs (verified by
        # inspecting sam3_decoder.onnx with onnxruntime):
        #   language_mask     – bool  [1, 32]
        #   language_features – float [32, 1, 256]
        if "language_mask" in self.input_names and inputs["language_mask"] is None:
            inputs["language_mask"] = np.zeros((1, 32), dtype=np.bool_)
        if (
            "language_features" in self.input_names
            and inputs["language_features"] is None
        ):
            inputs["language_features"] = np.zeros((32, 1, 256), dtype=np.float32)
        if "language_embeds" in self.input_names and inputs["language_embeds"] is None:
            inputs["language_embeds"] = np.zeros((32, 1, 1024), dtype=np.float32)

        # Only forward inputs that the model actually requires – onnxsim may
        # have removed some (e.g. vision_pos_enc_0/1, language_embeds) during
        # simplification.
        model_inputs = {
            k: v for k, v in inputs.items() if k in self.input_names and v is not None
        }
        outputs = self.session.run(None, model_inputs)
        # ONNX export order: [0]=boxes, [1]=scores, [2]=masks
        # Return as (masks, scores, boxes).
        return outputs[2], outputs[1], outputs[0]
