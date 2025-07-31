
import cv2
import numpy as np
import coremltools as ct
from pathlib import Path
from PIL import Image

def find_contour_points(image_path: str):
    # Load the image
    image = cv2.imread(image_path)
    thresh = get_binary_image(image)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # todo skip contour if it primary white
    # Sort contours by area in descending order
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Select the largest contour
    for largest_contour in sorted_contours:
        if countour_content_is_primarily_white(image.copy(), largest_contour):
            continue
        M = cv2.moments(largest_contour)
        if M["m00"] == 0:
            raise ValueError("Countrour centroid issue")
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        centroid_point = (cX, cY)

        mask = np.zeros(image.shape, dtype="uint8")
        cv2.drawContours(mask, [largest_contour], -1, 255, -1)

        # Find non-zero pixels (points inside the contour)
        non_zero_points = np.argwhere(mask == 255)

        # Select a random point
        random_point_index = np.random.randint(0, len(non_zero_points))
        random_point = non_zero_points[random_point_index]
        return (centroid_point, random_point)

class SegmentAnything2CoreML:
    def __init__(self, model_path: str) -> None:
        print("using CoreML", model_path)
        self.image_encoder = ct.models.MLModel(model_path + "/SAM2_1LargeImageEncoderFLOAT16.mlpackage")
        self.mask_decoder = ct.models.MLModel(model_path + "/SAM2_1LargeMaskDecoderFLOAT16.mlpackage")
        self.prompt_encoder = ct.models.MLModel(model_path + "/SAM2_1LargePromptEncoderFLOAT16.mlpackage")
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
                x_scaled = mark["data"][0] * (self.input_size[0] / embedding["original_size"][0])
                y_scaled = mark["data"][1] * (self.input_size[1] / embedding["original_size"][1])
                points.append([x_scaled, y_scaled])
                labels.append(mark["label"])
            elif mark["type"] == "rectangle":
                # Scale rectangle coordinates
                x1_scaled = mark["data"][0] * (self.input_size[0] / embedding["original_size"][0])
                y1_scaled = mark["data"][1] * (self.input_size[1] / embedding["original_size"][1])
                x2_scaled = mark["data"][2] * (self.input_size[0] / embedding["original_size"][0])
                y2_scaled = mark["data"][3] * (self.input_size[1] / embedding["original_size"][1])
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
        mask = low_res_masks[0, best_mask_idx] # Assuming batch size of 1

        # Resize the mask back to the original image size
        original_width, original_height = embedding["original_size"]
        mask = cv2.resize(mask, (original_width, original_height), interpolation=cv2.INTER_LINEAR)
        
        # Apply threshold to get a binary mask
        mask = (mask > 0).astype(np.uint8) * 255 # Convert to 0 or 255

        return np.array([mask]) # Return as a list for consistency

    
    def transform_masks(self, masks, original_size, transform_matrix):
        """Transform the masks back to the original image size."""
        output_masks = []
        for batch in range(masks.shape[0]):
            batch_masks = []
            for mask_id in range(masks.shape[1]):
                mask = masks[batch, mask_id]
                mask = cv2.warpAffine(
                    mask,
                    transform_matrix[:2],
                    (original_size[1], original_size[0]),
                    flags=cv2.INTER_LINEAR,
                )
                batch_masks.append(mask)
            output_masks.append(batch_masks)
        return np.array(output_masks)
