import cv2
import numpy as np
from PyQt5 import QtGui


def qt_img_to_cv_img(in_image):
    in_image = in_image.convertToFormat(13)  # format QImage::Format_RGB888

    width = in_image.width()
    height = in_image.height()

    ptr = in_image.bits()
    ptr.setsize(in_image.byteCount())
    mat = np.array(ptr).reshape(height, width, 3)  # Shape the data

    rgb = cv2.cvtColor(mat, cv2.COLOR_BGR2RGB)
    return rgb


def cv_img_to_qt_img(in_mat):
    assert len(in_mat.shape) == 2 and in_mat.dtype == np.uint8
    arr2 = np.require(in_mat, np.uint8, "C")
    qimg = QtGui.QImage(
        arr2,
        in_mat.shape[1],
        in_mat.shape[0],
        in_mat.shape[1],
        QtGui.QImage.Format_Indexed8,
    )
    return qimg
