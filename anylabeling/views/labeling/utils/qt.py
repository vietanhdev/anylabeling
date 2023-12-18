import os.path as osp
from math import hypot, sqrt

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

here = osp.dirname(osp.abspath(__file__))


def new_icon(icon):
    return QtGui.QIcon(osp.join(f":/images/images/{icon}.png"))


def new_button(text, icon=None, slot=None):
    b = QtWidgets.QPushButton(text)
    if icon is not None:
        b.setIcon(new_icon(icon))
    if slot is not None:
        b.clicked.connect(slot)
    return b


def new_action(
    parent,
    text,
    slot=None,
    shortcut=None,
    icon=None,
    tip=None,
    checkable=False,
    enabled=True,
    checked=False,
):
    """Create a new action and assign callbacks, shortcuts, etc."""
    action = QtWidgets.QAction(text, parent)
    if icon is not None:
        action.setIconText(text.replace(" ", "\n"))
        action.setIcon(new_icon(icon))
    if shortcut is not None:
        if isinstance(shortcut, (list, tuple)):
            action.setShortcuts(shortcut)
        else:
            action.setShortcut(shortcut)
    if tip is not None:
        action.setToolTip(tip)
        action.setStatusTip(tip)
    if slot is not None:
        action.triggered.connect(slot)
    if checkable:
        action.setCheckable(True)
    action.setEnabled(enabled)
    action.setChecked(checked)
    return action


def add_actions(widget, actions):
    for action in actions:
        if action is None:
            widget.addSeparator()
        elif isinstance(action, QtWidgets.QMenu):
            widget.addMenu(action)
        else:
            widget.addAction(action)


def label_validator():
    return QtGui.QRegularExpressionValidator(
        QtCore.QRegularExpression(r"^[^ \t].+"), None
    )


class Struct:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def distance(p):
    return sqrt(p.x() * p.x() + p.y() * p.y())


def distance_to_line(point, line):
    p1, p2 = line
    p1 = np.array([p1.x(), p1.y()])
    p2 = np.array([p2.x(), p2.y()])
    p3 = np.array([point.x(), point.y()])
    if np.dot((p3 - p1), (p2 - p1)) < 0:
        return np.linalg.norm(p3 - p1)
    if np.dot((p3 - p2), (p1 - p2)) < 0:
        return np.linalg.norm(p3 - p2)
    if np.linalg.norm(p2 - p1) == 0:
        return 0
    return np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)


def squared_distance_to_line(point, line):
    """
    Use python math because it is faster than using numpy
    """
    p1, p2 = line
    px, py = point.x(), point.y()
    x1, y1 = p1.x(), p1.y()
    x2, y2 = p2.x(), p2.y()

    dx, dy = x2 - x1, y2 - y1
    if dx == dy == 0:
        return hypot(px - x1, py - y1)

    # Calculate the projection and check if it falls on the line segment
    t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
    if t < 0:
        dx, dy = px - x1, py - y1  # point to p1
    elif t > 1:
        dx, dy = px - x2, py - y2  # point to p2
    else:
        near_x, near_y = x1 + t * dx, y1 + t * dy
        dx, dy = px - near_x, py - near_y

    return hypot(dx, dy)


def fmt_shortcut(text):
    mod, key = text.split("+", 1)
    return f"<b>{mod}</b>+<b>{key}</b>"
