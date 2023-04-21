import logging
import os

import cv2
import numpy as np
from PyQt5 import QtCore

from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.utils.opencv import qt_img_to_cv_img

from .model import Model
from .types import AutoLabelingResult


class YOLOv5_seg(Model):
    """Object detection model using YOLOv5"""

    class Meta:
        required_config_names = [
            "type",
            "name",
            "display_name",
            "model_path",
            "input_width",
            "input_height",
            "score_threshold",
            "nms_threshold",
            "confidence_threshold",
            "mask_threshold",
            "classes",
        ]
        widgets = ["button_run"]
        output_modes = ["polygon", "rectangle"]
        default_output_mode = "polygon"

    def __init__(self, model_config, on_message) -> None:
        # Run the parent class's init method
        super().__init__(model_config, on_message)

        model_abs_path = self.get_model_abs_path(
            self.config["model_path"], self.config["name"]
        )
        if not os.path.isfile(model_abs_path):
            raise Exception(f"Model not found: {model_abs_path}")
            
        self.net = cv2.dnn.readNet(model_abs_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        self.classes = self.config["classes"]

    def pre_process(self, input_image, net):
        """
        Pre-process the input image before feeding it to the network.
        """
        # Create a 4D blob from a frame.
        #input_image= cv2.imread("coco_sample/zidane.jpg", input_image)
        
        blob = cv2.dnn.blobFromImage(
            input_image,
            1 / 255,
            (self.config["input_width"], self.config["input_height"]),
            [0, 0, 0],
            swapRB=False, # Image is already in BGR
            crop=False,
        )

        # Sets the input to the network.
        net.setInput(blob)

        # Runs the forward pass to get output of the output layers.
        output_layers = net.getUnconnectedOutLayersNames()
        outputs = net.forward(output_layers)
        np.save("outputs0.npy", outputs[0])
        np.save("outputs1.npy", outputs[1])

        return outputs
    
    def post_process(self, input_image, outputs):
        """
        Post-process the network's output, to get the bounding boxes and
        their confidence scores.
        """
        
        image_height, image_width = input_image.shape[:2]

        # Resizing factor.
        x_factor = image_width / self.config["input_width"]
        y_factor = image_height / self.config["input_height"]
        
        # Get output dimensions of the model
        _, seg_channels, seg_width, seg_height = outputs[1].shape

        
        # box preprocessing
        outputs[0][0, :, 0] = (outputs[0][0, :, 0] - 0.5 * outputs[0][0, :, 2]) * x_factor
        outputs[0][0, :, 1] = (outputs[0][0, :, 1] - 0.5 * outputs[0][0, :, 3]) * y_factor
        outputs[0][0, :, 2] *= x_factor
        outputs[0][0, :, 3] *= y_factor
        
        # get boxes, conf, score, and mask
        boxes = outputs[0][0, :, :4]
        confidences = outputs[0][0, :, 4]
        scores = confidences.reshape(-1, 1) * outputs[0][0, :, 5 : len(self.classes) + 5]
        masks = outputs[0][0, :, len(self.classes) + 5 :]
        
        # Perform non maximum suppression to eliminate redundant
        # overlapping boxes with lower confidences.
        selected = cv2.dnn.NMSBoxes(
            boxes,
            confidences,
            self.config["confidence_threshold"],
            self.config["nms_threshold"],
        )

        shapes = []  # boxes to draw

        for i in selected:  # loop through selected
            box = boxes[i].round().astype(np.int32)  # to int
            _, score, _, label = cv2.minMaxLoc(scores[i])  # get score and classId

            if score >= self.config["score_threshold"]:  # filtering by score_tresh
            
                if self.output_mode == "polygon":
                    # Transform coordinates
                    left = int(round(box[0] * seg_width / image_width))
                    top = int(round(box[1] * seg_height / image_height))
                    width = int(round(box[2] * seg_width / image_width))
                    height = int(round(box[3] * seg_height / image_height))
    
                    # process protos
                    protos = outputs[1][0, :, top : top + height, left : left + width].reshape(seg_channels, -1)
                    protos = np.expand_dims(masks[i], 0) @ protos  # matmul
                    protos = 1 / (1 + np.exp(-protos))  # sigmoid
                    protos = protos.reshape(height, width)  # reshape
                    mask = cv2.resize(protos, (box[2], box[3]))  # resize mask
                    mask = mask >= self.config["mask_threshold"] # filtering mask by tresh
    
                    # add mask to overlay layer
                    black = np.zeros((image_height, image_width, 3))
                    to_mask = black[box[1] : box[1] + box[3], box[0] : box[0] + box[2]]  # get box roi
                    to_mask[mask] = [255,255,255]  # apply mask

                    gray = cv2.cvtColor(black.astype(np.uint8), cv2.COLOR_BGR2GRAY)
                    contours, hierarchy = cv2.findContours(gray.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    
                    # Refine contours (taken from segment_anything.py)
                    approx_contours = []
                    for contour in contours:
                        # Approximate contour
                        epsilon = 0.001 * cv2.arcLength(contour, True)
                        approx = cv2.approxPolyDP(contour, epsilon, True)
                        approx_contours.append(approx)

                    # Remove small contours (area < 20% of average area)
                    if len(approx_contours) > 1:
                        areas = [cv2.contourArea(contour) for contour in approx_contours]
                        avg_area = np.mean(areas)

                        filtered_approx_contours = [
                            contour
                            for contour, area in zip(approx_contours, areas)
                            if area > avg_area * 0.2
                        ]
                        contours = filtered_approx_contours

                    
                    # Create shape
                    shape = Shape(flags={})
                    
                    for point in contours[0].tolist():
                        point = point[0]
                        point[0] = int(point[0])
                        point[1] = int(point[1])
                        shape.add_point(QtCore.QPointF(point[0], point[1]))
                    
                    shape.shape_type = "polygon"
                    shape.closed = True
                    shape.fill_color = "#000000"
                    shape.line_color = "#000000"
                    shape.line_width = 1
                    shape.label = self.classes[label[1]]
                    shape.selected = False
                    shapes.append(shape)
                    
                elif self.output_mode == "rectangle":
                    # Create shape
                    shape = Shape(flags={})
                    shape.add_point(QtCore.QPointF(box[0], box[1]))
                    shape.add_point(QtCore.QPointF(box[0]+box[2], box[1]+box[3]))
                    shape.shape_type = "rectangle"
                    shape.closed = True
                    shape.fill_color = "#000000"
                    shape.line_color = "#000000"
                    shape.line_width = 1
                    shape.label = self.classes[label[1]]
                    shape.selected = False
                    shapes.append(shape)
                    
        return shapes
        
        
    def predict_shapes(self, image, image_path=None):
        """
        Predict shapes from image
        """

        if image is None:
            return []

        try:
            image = qt_img_to_cv_img(image)
        except Exception as e:  # noqa
            logging.warning("Could not inference model")
            logging.warning(e)
            return []

        outputs = self.pre_process(image, self.net)
        shapes = self.post_process(image, outputs)
        result = AutoLabelingResult(shapes, replace=True)
        return result

    def unload(self):
        del self.net
