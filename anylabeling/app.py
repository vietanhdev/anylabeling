import os
import sys

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QApplication

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.append(".")

from anylabeling.resources.resources import *  # noqa
from anylabeling.views.mainwindow import MainWindow  # noqa


def main():
    # Enable scaling for high dpi screens
    QtWidgets.QApplication.setAttribute(
        QtCore.Qt.AA_EnableHighDpiScaling, True
    )  # enable highdpi scaling
    QtWidgets.QApplication.setAttribute(
        QtCore.Qt.AA_UseHighDpiPixmaps, True
    )  # use highdpi icons

    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_ShareOpenGLContexts)
    app = QApplication(sys.argv)
    app.processEvents()

    main_win = MainWindow(app)

    main_win.showMaximized()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
