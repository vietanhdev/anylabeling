import cv2
import numpy as np
import numpy.typing as npt
from typing import Tuple, Iterable






# Load YOLOv5 segmentation model
net = cv2.dnn.readNet("/home/henry/anylabeling_data/models/yolov5/yolov5x-seg.onnx")

config = {'type': 'yolov8',
          'name': 'yolov8x',
          'display_name':'YOLOv5n Ultralytics',
          'model_path': 'anylabeling_assets/models/yolov5/yolov8x.onnx',
          'input_width': 640,
          'input_height': 640,
          'score_threshold': 0.4,
          'nms_threshold': 0.45,
          'confidence_threshold': 0.45,
          'classes': ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']}
classes = config["classes"]

# Load input image
image = cv2.imread('bus.jpg')
source_padded = image.copy()
overlay = image.copy()  # make overlay mat

# Preprocess image
blob = cv2.dnn.blobFromImage(image, 1/255.0, (640, 640), (0, 0, 0), True, crop=False)

# Set the input to the network
net.setInput(blob)

# Forward pass through the network
output_layers = net.getUnconnectedOutLayersNames()
outputs = net.forward(output_layers)
rows = outputs[0].shape[1]


# Post-process the output
class_ids = []
confidences = []
segmentation_masks = []
boxes = []
# Rows.
rows = outputs[0].shape[1]
h, w = image.shape[:2]
h_source, w_source = image.shape[:2]
# Resizing factor.
x_factor = w / 640
y_factor =  h / 640
# Iterate through detections.

# get boxes, conf, score, and mask
boxes = outputs[0][0, :, :4]
confidences = outputs[0][0, :, 4]
scores = confidences.reshape(-1, 1) * outputs[0][0, :, 5 : len(config["classes"]) + 5]
masks = outputs[0][0, :, len(config["classes"]) + 5 :]


selected = cv2.dnn.NMSBoxes(
    boxes,
    confidences,
    config["confidence_threshold"],
    config["nms_threshold"],
)

boxes_to_draw = []  # boxes to draw

max_size = max(h,w)
_, seg_chanels, seg_width, seg_height = outputs[1].shape
colors= [[0,255,0]]*len(classes)
selected_boxes = []
my_masks=[]
for i in selected:  # loop through selected
    box = boxes[i].round().astype(np.int32)  # to int
    #box = handle_overflow_box(box, [max_size, max_size])  # handle overflow boxes

    _, score, _, label = cv2.minMaxLoc(scores[i])  # get score and classId
    print(label)
    if score >= config["score_threshold"]:  # filtering by score_tresh
        color = colors[label[1]]  # get color

        # save box to draw latter (add mask first)
        boxes_to_draw.append([box, classes[label[1]], score, color])

        # crop mask from proto
        # x = int(round(box[0] ))
        # y = int(round(box[1]  ))
        # w = int(round(box[2]  ))
        # h = int(round(box[3] ))
        
        # crop mask from proto
        x = int(round(box[0] * 160/ max_size))
        y = int(round(box[1] * 160 / max_size))
        w = int(round(box[2] * 160 / max_size))
        h = int(round(box[3] * 160 / max_size))


        # process protos
        protos = outputs[1][0, :, y : y + h, x : x + w].reshape(seg_chanels, -1)
        protos = np.expand_dims(masks[i], 0) @ protos  # matmul
        protos = 1 / (1 + np.exp(-protos))  # sigmoid
        protos = protos.reshape(h, w)  # reshape
        mask = cv2.resize(protos, (w_source, h_source))  # resize mask
        mask = mask >= 0.5 #config["score_threshold"] # mask_tresh  # filtering mask by tresh

        # add mask to overlay layer
        to_mask = overlay[box[1] : box[1] + box[3], box[0] : box[0] + box[2]]  # get box roi
        mask = mask[: to_mask.shape[0], : to_mask.shape[1]]  # crop mask
        to_mask[mask] = color  # apply mask
        
        # boxes
        selected_boxes.append(box)


boxes = selected_boxes

# combine image and overlay
source_padded = cv2.addWeighted(source_padded, 1 - 0.5, overlay, 0.5, 0)
cv2.imwrite("test.jpg", overlay)

for box in selected_boxes:
    x = box[0]
    y = box[1]
    w = box[2]
    h = box[3]
    left = int((x - 0.5 * w) * x_factor)
    top = int((y - 0.5 * h) * y_factor)
    width = int(w * x_factor)
    height = int(h * y_factor)

    output_box = {
        "x1": left,
        "y1": top,
        "x2": left + width,
        "y2": top + height,
        "label": label,
        "score": score,
    }

    cv2.rectangle(image, (left, top), ((left + width), (top + height)), (0,255,0), 3)

cv2.imwrite("test.jpg", image)


