import cv2
import numpy as np
import qimage2ndarray
from PyQt5 import QtGui


def qt_img_to_cv_img(in_image):
    """
    allow to load rgb,gray image to inference
    """
    cv_image = qimage2ndarray.raw_view(in_image)
    # to uint8
    if cv_image.dtype != np.uint8:
        cv2.normalize(cv_image, cv_image, 0, 255, cv2.NORM_MINMAX)
        cv_image = np.array(cv_image, dtype=np.uint8)
    # to rgb
    if len(cv_image.shape) == 2 or cv_image.shape[2] == 1:
        cv_image = cv2.merge([cv_image, cv_image, cv_image])
    return cv_image


def cv_img_to_qt_img(in_mat):
    return QtGui.QImage(qimage2ndarray.array2qimage(in_mat))
