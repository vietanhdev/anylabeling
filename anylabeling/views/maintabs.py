"""Main tab widget for anylabeling app"""

from PyQt5.QtWidgets import QTabWidget

from .labeling.label_wrapper import LabelingWrapper


class MainTabsWidget(QTabWidget):
    """Main tab widget for anylabeling app"""

    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.tabs = []

        self.labeling_tab = LabelingWrapper(self)
        self.tabs.append(self.labeling_tab)
        self.addTab(self.labeling_tab, "Data Labeling")

        self.currentChanged.connect(self.on_change)

    def on_change(self, i):
        """Handle event when the current tab changes"""
        if isinstance(self.tabs[i], LabelingWrapper):
            self.parent.menuBar().show()
        else:
            self.parent.menuBar().hide()
