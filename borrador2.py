from typing import Union

import cv2
import numpy as np
import numpy.typing as npt
from typing import Tuple, Iterable




labels = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
colors = np.random.randint(0, 256, size=(len(labels), 3)).tolist()

model = cv2.dnn.readNet("/home/henry/anylabeling_data/models/yolov5/yolov5x-seg.onnx")

source = cv2.imread('bus.jpg')

source_height, source_width, _ = source.shape


## padding image
max_size = max(source_width, source_height)  # get max size
#source_padded = np.zeros((max_size, max_size, 3), dtype=np.uint8)  # initial zeros mat
#source_padded[:source_height, :source_width] = source.copy()  # place original image
source_padded= source.copy()  # place original image
overlay = source_padded.copy()  # make overlay mat

## ratios
x_ratio = source_width / 640
y_ratio = source_height / 640

# run model
input_img = cv2.dnn.blobFromImage(
    source_padded,
    1 / 255.0,
    (640, 640),
    swapRB=True,
    crop=False,
)  # normalize and resize: [h, w, 3] => [1, 3, h, w]

# Set the input to the network
model.setInput(input_img)

# Forward pass through the network
output_layers = model.getUnconnectedOutLayersNames()
result = model.forward(output_layers)


_, seg_chanels, seg_width, seg_height = result[1].shape

# box preprocessing
result[0][0, :, 0] = (result[0][0, :, 0] - 0.5 * result[0][0, :, 2]) * x_ratio
result[0][0, :, 1] = (result[0][0, :, 1] - 0.5 * result[0][0, :, 3]) * y_ratio
result[0][0, :, 2] *= x_ratio
result[0][0, :, 3] *= y_ratio

# get boxes, conf, score, and mask
boxes = result[0][0, :, :4]
confidences = result[0][0, :, 4]
scores = confidences.reshape(-1, 1) * result[0][0, :, 5 : len(labels) + 5]
masks = result[0][0, :, len(labels) + 5 :]

# NMS
selected = cv2.dnn.NMSBoxes(boxes, confidences, 0.45, 0.45, top_k=100)

boxes_to_draw = []  # boxes to draw

for i in selected:  # loop through selected
    box = boxes[i].round().astype(np.int32)  # to int
    _, score, _, label = cv2.minMaxLoc(scores[i])  # get score and classId
    print(label)
    if score >= 0.40:  # filtering by score_tresh
        color = colors[label[1]]  # get color

        # save box to draw latter (add mask first)
        boxes_to_draw.append([box, labels[label[1]], score, color])

        # crop mask from proto
        # x = int(round(box[0] * 160 / max_size))
        # y = int(round(box[1] * 160 / max_size))
        # w = int(round(box[2] * 160 / max_size))
        # h = int(round(box[3] * 160 / max_size))
        
        x = int(round(box[0] * 160 / source_width))
        y = int(round(box[1] * 160 / source_height))
        w = int(round(box[2] * 160 / source_width))
        h = int(round(box[3] * 160 / source_height))

        # process protos
        protos = result[1][0, :, y : y + h, x : x + w].reshape(seg_chanels, -1)
        print(x,x+w)
        protos = np.expand_dims(masks[i], 0) @ protos  # matmul
        protos = 1 / (1 + np.exp(-protos))  # sigmoid
        protos = protos.reshape(h, w)  # reshape
        mask = cv2.resize(protos, (box[2], box[3]))  # resize mask
        mask = mask >= 0.45  # filtering mask by tresh

        # # add mask to overlay layer
        to_mask = overlay[box[1] : box[1] + box[3], box[0] : box[0] + box[2]]  # get box roi
        mask = mask[: to_mask.shape[0], : to_mask.shape[1]]  # crop mask
        to_mask[mask] = color  # apply mask

# combine image and overlay
overlay[1] = overlay[1]+10000
source_padded = cv2.addWeighted(source_padded, 1 - 0.4, overlay, 0.4, 0)


source = source_padded[:source_height, :source_width]  # crop padding

cv2.imwrite("test.jpg",source)

"""

for box in boxes_to_draw:
    box = box[0]
    left = box[0]
    top = box[1]
    width = box[2]
    height = box[3]

    cv2.rectangle(source, (left, top), ((left + width), (top + height)), (0,255,0), 3)

cv2.imwrite("test.jpg", source)
"""