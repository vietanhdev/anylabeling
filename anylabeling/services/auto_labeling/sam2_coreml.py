import os
import cv2
import numpy as np
import coremltools as ct
from pathlib import Path
from PIL import Image


class SegmentAnything2CoreML:
    def __init__(self, model_path: str) -> None:
        print("using CoreML", model_path)
        image_decoder_path = os.path.join(
            model_path, "SAM2_1LargeImageEncoderFLOAT16.mlpackage"
        )
        mask_decoder_path = os.path.join(
            model_path, "SAM2_1LargeMaskDecoderFLOAT16.mlpackage"
        )
        prompt_encoder_path = os.path.join(
            model_path, "SAM2_1LargePromptEncoderFLOAT16.mlpackage"
        )
        self.image_encoder = ct.models.MLModel(image_decoder_path)
        self.mask_decoder = ct.models.MLModel(mask_decoder_path)
        self.prompt_encoder = ct.models.MLModel(prompt_encoder_path)
        self.input_size = (1024, 1024)

    def encode(self, cv_image: np.ndarray) -> dict:
        """Encodes the input image using the image encoder."""
        # Convert OpenCV image to PIL Image
        pil_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))

        # Resize image to input_size
        original_size = pil_image.size
        resized_image = pil_image.resize(self.input_size, Image.Resampling.LANCZOS)

        # Predict image embeddings
        embeddings = self.image_encoder.predict({"image": resized_image})

        return {
            "high_res_feats_0": embeddings["feats_s0"],
            "high_res_feats_1": embeddings["feats_s1"],
            "image_embedding": embeddings["image_embedding"],
            "original_size": original_size,
        }

    def predict_masks(self, embedding: dict, prompt: list) -> list[np.ndarray]:
        """Predicts masks based on image embedding and prompt."""
        points = []
        labels = []
        for mark in prompt:
            if mark["type"] == "point":
                # Scale point coordinates to match the model's input size
                x_scaled = mark["data"][0] * (
                    self.input_size[0] / embedding["original_size"][0]
                )
                y_scaled = mark["data"][1] * (
                    self.input_size[1] / embedding["original_size"][1]
                )
                points.append([x_scaled, y_scaled])
                labels.append(mark["label"])
            elif mark["type"] == "rectangle":
                # Scale rectangle coordinates
                x1_scaled = mark["data"][0] * (
                    self.input_size[0] / embedding["original_size"][0]
                )
                y1_scaled = mark["data"][1] * (
                    self.input_size[1] / embedding["original_size"][1]
                )
                x2_scaled = mark["data"][2] * (
                    self.input_size[0] / embedding["original_size"][0]
                )
                y2_scaled = mark["data"][3] * (
                    self.input_size[1] / embedding["original_size"][1]
                )
                points.append([x1_scaled, y1_scaled])
                points.append([x2_scaled, y2_scaled])
                labels.append(2)  # Label for top-left of box
                labels.append(3)  # Label for bottom-right of box

        points_array = np.array(points, dtype=np.float32).reshape(1, len(points), 2)
        labels_array = np.array(labels, dtype=np.int32).reshape(1, len(labels))

        # Get prompt embeddings
        prompt_embeddings = self.prompt_encoder.predict(
            {"points": points_array, "labels": labels_array}
        )

        # Predict masks
        mask_output = self.mask_decoder.predict(
            {
                "image_embedding": embedding["image_embedding"],
                "sparse_embedding": prompt_embeddings["sparse_embeddings"],
                "dense_embedding": prompt_embeddings["dense_embeddings"],
                "feats_s0": embedding["high_res_feats_0"],
                "feats_s1": embedding["high_res_feats_1"],
            }
        )

        # The model returns low_res_masks, which need to be upscaled and thresholded
        low_res_masks = mask_output["low_res_masks"]

        # Select the best mask based on score
        scores = mask_output["scores"]
        best_mask_idx = np.argmax(scores)
        mask = low_res_masks[0, best_mask_idx]  # Assuming batch size of 1

        # Resize the mask back to the original image size
        original_width, original_height = embedding["original_size"]
        mask = cv2.resize(
            mask, (original_width, original_height), interpolation=cv2.INTER_LINEAR
        )

        # Apply threshold to get a binary mask
        mask = (mask > 0).astype(np.uint8) * 255  # Convert to 0 or 255

        return np.array([mask])  # Return as a list for consistency
