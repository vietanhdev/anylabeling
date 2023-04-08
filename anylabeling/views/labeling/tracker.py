import cv2
from PyQt5 import QtCore

from .utils import opencv as ocvutil


class OCVTracker:
    def __init__(self):
        self.shape = None
        self.prev_image = None
        self.tracker = None

    def init_tracker(self, qimg, shape):
        status = False
        self.shape = shape
        if qimg.isNull() or not shape:
            print("No object initialized")
            return status
        fimg = ocvutil.qt_img_to_cv_img(qimg)
        fimg = cv2.cvtColor(fimg, cv2.COLOR_BGR2GRAY)
        self.prev_image = fimg
        status = True
        return status

    def update_tracker(self, qimg):
        shape = self.shape.copy()
        assert (
            shape and shape.label == self.shape.label
        ), "Invalid tracker state!"
        status = False
        if qimg is None:
            print("No image to update tracker")
            return shape, status

        mimg = ocvutil.qt_img_to_cv_img(qimg)
        mimg = cv2.cvtColor(mimg, cv2.COLOR_BGR2GRAY)

        p1 = (int(self.shape.points[0].x()), int(self.shape.points[0].y()))
        p2 = (int(self.shape.points[1].x()), int(self.shape.points[1].y()))
        prev_box = (p1[0], p1[1], p2[0] - p1[0], p2[1] - p1[1])

        self.tracker = cv2.TrackerKCF_create()
        self.tracker.init(self.prev_image, prev_box)
        success, box = self.tracker.update(mimg)

        if success:
            shape.points = [
                QtCore.QPoint(box[0], box[1]),
                QtCore.QPoint(box[0] + box[2], box[1] + box[3]),
            ]
            status = True
        else:
            print("Tracker failed")

        return shape, status


class Tracker:
    def __init__(self) -> None:
        self.prev_image = None
        self.prev_shapes = None
        self.ocv_tracker = OCVTracker()

    def update(self, prev_shapes, prev_image):
        self.prev_shapes = prev_shapes
        self.prev_image = prev_image

    def get(self, img):
        if self.prev_image is None:
            return []

        new_shapes = []
        for shape in self.prev_shapes:
            if shape.shape_type != "rectangle":
                new_shape.append(shape)
                continue
            self.ocv_tracker.init_tracker(self.prev_image, shape)
            new_shape, _ = self.ocv_tracker.update_tracker(img)
            new_shapes.append(new_shape)

        return new_shapes
