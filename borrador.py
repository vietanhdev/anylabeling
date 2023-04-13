#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 10:16:59 2023

@author: henry
"""

import cv2
net = cv2.dnn.readNet("/home/henry/anylabeling_data/models/yolov5/yolov8x.onnx")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
input_image = cv2.imread("coco_sample/000000000009.jpg")

blob = cv2.dnn.blobFromImage(input_image, 1 / 255, (640, 640), swapRB=True, crop=False)

net.setInput(blob)


# Runs the forward pass to get output of the output layers.
output_layers = net.getUnconnectedOutLayersNames()
outputs = net.forward()
outputs = outputs.transpose((0, 2, 1))




# ========================================= post



# Lists to hold respective values while unwrapping.
class_ids = []
confidences = []
boxes = []

# Rows.
rows = outputs[0].shape[0]

image_height, image_width = input_image.shape[:2]

# Resizing factor.
x_factor = image_width / config["input_width"]
y_factor = image_height / config["input_height"]

# Iterate through 25200 detections.
for r in range(rows):
    row = outputs[0][r]
    confidence = row[4]

    # Discard bad detections and continue.
    if confidence >= config["confidence_threshold"]:
        classes_scores = row[5:]

        # Get the index of max class score.
        class_id = np.argmax(classes_scores)

        #  Continue if the class score is above threshold.
        if classes_scores[class_id] > self.config["score_threshold"]:
            confidences.append(confidence)
            class_ids.append(class_id)

            cx, cy, w, h = row[0], row[1], row[2], row[3]

            left = int((cx - w / 2) * x_factor)
            top = int((cy - h / 2) * y_factor)
            width = int(w * x_factor)
            height = int(h * y_factor)

            box = np.array([left, top, width, height])
            boxes.append(box)

# Perform non maximum suppression to eliminate redundant
# overlapping boxes with lower confidences.
indices = cv2.dnn.NMSBoxes(
    boxes,
    confidences,
    config["confidence_threshold"],
    config["nms_threshold"],
)

output_boxes = []
for i in indices:
    box = boxes[i]
    left = box[0]
    top = box[1]
    width = box[2]
    height = box[3]
    label = classes[class_ids[i]]
    score = confidences[i]

    output_box = {
        "x1": left,
        "y1": top,
        "x2": left + width,
        "y2": top + height,
        "label": label,
        "score": score,
    }

    output_boxes.append(output_box)
