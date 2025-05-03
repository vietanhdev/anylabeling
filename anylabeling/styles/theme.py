import darkdetect
from PyQt5.QtGui import QPalette, QColor
import os


class AppTheme:
    """
    Theme manager for the application
    Provides consistent styling for both light and dark themes
    """

    # Modern color palette
    PRIMARY_LIGHT = "#2196F3"  # Blue
    PRIMARY_DARK = "#1976D2"  # Darker Blue
    ACCENT_LIGHT = "#FFA000"  # Amber
    ACCENT_DARK = "#FF8F00"  # Darker Amber

    # Light theme colors
    LIGHT = {
        "window": "#FFFFFF",
        "window_text": "#212121",
        "base": "#F5F5F5",
        "alternate_base": "#E0E0E0",
        "text": "#212121",
        "button": "#E0E0E0",
        "button_text": "#212121",
        "bright_text": "#000000",
        "highlight": PRIMARY_LIGHT,
        "highlighted_text": "#FFFFFF",
        "link": PRIMARY_LIGHT,
        "dark": "#455A64",
        "mid": "#9E9E9E",
        "midlight": "#BDBDBD",
        "light": "#F5F5F5",
        # Custom colors
        "border": "#BDBDBD",
        "toolbar_bg": "#FFFFFF",
        "dock_title_bg": "#E0E0E0",
        "dock_title_text": "#212121",
        "success": "#4CAF50",
        "warning": "#FFC107",
        "error": "#F44336",
        "panel_bg": "#FFFFFF",
        "selection": "#BBDEFB",
    }

    # Dark theme colors
    DARK = {
        "window": "#212121",
        "window_text": "#EEEEEE",
        "base": "#303030",
        "alternate_base": "#424242",
        "text": "#EEEEEE",
        "button": "#424242",
        "button_text": "#EEEEEE",
        "bright_text": "#FFFFFF",
        "highlight": PRIMARY_DARK,
        "highlighted_text": "#FFFFFF",
        "link": PRIMARY_LIGHT,
        "dark": "#2D2D2D",
        "mid": "#616161",
        "midlight": "#757575",
        "light": "#424242",
        # Custom colors
        "border": "#616161",
        "toolbar_bg": "#333333",
        "dock_title_bg": "#424242",
        "dock_title_text": "#EEEEEE",
        "success": "#4CAF50",
        "warning": "#FFC107",
        "error": "#F44336",
        "panel_bg": "#303030",
        "selection": "#0D47A1",
    }

    @staticmethod
    def is_dark_mode():
        """Check if system is using dark mode or if it's set via environment variable"""
        # Check environment variable first
        if "DARK_MODE" in os.environ:
            return os.environ["DARK_MODE"] == "1"
        # Fall back to system detection
        return darkdetect.isDark()

    @staticmethod
    def get_color(color_name):
        """Get color based on current theme"""
        is_dark = AppTheme.is_dark_mode()
        colors = AppTheme.DARK if is_dark else AppTheme.LIGHT
        return colors.get(color_name, "#FFFFFF" if not is_dark else "#212121")

    @staticmethod
    def apply_theme(app):
        """Apply theme to entire application"""
        is_dark = AppTheme.is_dark_mode()
        colors = AppTheme.DARK if is_dark else AppTheme.LIGHT

        # Set application style
        app.setStyle("Fusion")

        # Create and apply palette
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(colors["window"]))
        palette.setColor(QPalette.WindowText, QColor(colors["window_text"]))
        palette.setColor(QPalette.Base, QColor(colors["base"]))
        palette.setColor(QPalette.AlternateBase, QColor(colors["alternate_base"]))
        palette.setColor(QPalette.Text, QColor(colors["text"]))
        palette.setColor(QPalette.Button, QColor(colors["button"]))
        palette.setColor(QPalette.ButtonText, QColor(colors["button_text"]))
        palette.setColor(QPalette.BrightText, QColor(colors["bright_text"]))
        palette.setColor(QPalette.Highlight, QColor(colors["highlight"]))
        palette.setColor(QPalette.HighlightedText, QColor(colors["highlighted_text"]))
        palette.setColor(QPalette.Link, QColor(colors["link"]))
        palette.setColor(QPalette.Dark, QColor(colors["dark"]))
        palette.setColor(QPalette.Mid, QColor(colors["mid"]))
        palette.setColor(QPalette.Midlight, QColor(colors["midlight"]))
        palette.setColor(QPalette.Light, QColor(colors["light"]))

        app.setPalette(palette)

        # Apply global stylesheet
        app.setStyleSheet(AppTheme.get_stylesheet())

    @staticmethod
    def get_stylesheet():
        """Get stylesheet for current theme"""
        is_dark = AppTheme.is_dark_mode()
        colors = AppTheme.DARK if is_dark else AppTheme.LIGHT

        return f"""
        /* Main Window */
        QMainWindow {{
            background-color: {colors["window"]};
            color: {colors["window_text"]};
        }}

        /* Menus and Menu Bar */
        QMenuBar {{
            background-color: {colors["window"]};
            color: {colors["window_text"]};
            border-bottom: 1px solid {colors["border"]};
        }}

        QMenuBar::item {{
            background-color: transparent;
            padding: 4px 10px;
        }}

        QMenuBar::item:selected {{
            background-color: {colors["highlight"]};
            color: {colors["highlighted_text"]};
        }}

        QDockWidget::title {{
            text-align: center;
            border-radius: 4px;
            margin-bottom: 2px;
            background-color: {colors["dock_title_bg"]};
            color: {colors["dock_title_text"]};
        }}

        /* Tool Bar */
        QToolBar {{
            background-color: {colors["toolbar_bg"]};
            padding: 2px;
            border: none;
            border-bottom: 1px solid {colors["border"]};
        }}

        QToolButton {{
            background-color: transparent;
            border: 1px solid transparent;
            border-radius: 4px;
            padding: 4px;
            margin: 1px;
        }}

        QToolButton:hover {{
            background-color: {colors["alternate_base"]};
            border: 1px solid {colors["border"]};
        }}

        QToolButton:pressed {{
            background-color: {colors["midlight"]};
        }}

        QToolButton:checked {{
            background-color: {colors["highlight"]};
            color: {colors["highlighted_text"]};
        }}

        /* List Widgets */
        QListWidget {{
            background-color: {colors["base"]};
            color: {colors["text"]};
            border: 1px solid {colors["border"]};
            border-radius: 4px;
        }}

        QListWidget::item:selected {{
            background-color: {colors["highlight"]};
            color: {colors["highlighted_text"]};
        }}

        QListWidget::item:hover:!selected {{
            background-color: {colors["selection"]};
        }}

        /* Scroll Areas and Scroll Bars */
        QScrollArea {{
            background-color: {colors["window"]};
            border: none;
        }}

        QScrollBar:vertical {{
            background-color: {colors["base"]};
            width: 12px;
            margin: 0px;
        }}

        QScrollBar::handle:vertical {{
            background-color: {colors["mid"]};
            min-height: 20px;
            border-radius: 6px;
        }}

        QScrollBar::handle:vertical:hover {{
            background-color: {colors["highlight"]};
        }}

        QScrollBar:horizontal {{
            background-color: {colors["base"]};
            height: 12px;
            margin: 0px;
        }}

        QScrollBar::handle:horizontal {{
            background-color: {colors["mid"]};
            min-width: 20px;
            border-radius: 6px;
        }}

        QScrollBar::handle:horizontal:hover {{
            background-color: {colors["highlight"]};
        }}

        QScrollBar::add-line, QScrollBar::sub-line {{
            width: 0px;
            height: 0px;
        }}

        /* Tab Widget */
        QTabWidget::pane {{
            border: 1px solid {colors["border"]};
            border-radius: 4px;
            top: -1px;
        }}

        QTabBar::tab {{
            background-color: {colors["alternate_base"]};
            color: {colors["text"]};
            border: 1px solid {colors["border"]};
            border-bottom-color: {colors["border"]};
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
            padding: 6px 12px;
            min-width: 80px;
            min-height: 20px;
        }}

        QTabBar::tab:selected {{
            background-color: {colors["window"]};
            border-bottom-color: {colors["window"]};
        }}

        QTabBar::tab:!selected {{
            margin-top: 2px;
        }}

        /* Progress Bar */
        QProgressBar {{
            background-color: {colors["base"]};
            color: {colors["highlighted_text"]};
            border: 1px solid {colors["border"]};
            border-radius: 4px;
            text-align: center;
        }}

        QProgressBar::chunk {{
            background-color: {colors["highlight"]};
            width: 10px;
            margin: 0.5px;
        }}

        /* Status Bar */
        QStatusBar {{
            background-color: {colors["window"]};
            color: {colors["window_text"]};
            border-top: 1px solid {colors["border"]};
        }}

        QStatusBar::item {{
            border: none;
        }}

        /* Specific Widget Styling */
        #zoomWidget QToolButton {{
            margin: 0px 1px;
            padding: 2px;
        }}

        /* Auto Labeling Widget */
        #autoLabelingWidget {{
            background-color: {colors["panel_bg"]};
            border: 1px solid {colors["border"]};
            border-radius: 4px;
        }}

        #autoLabelingWidget QPushButton {{
            min-height: 24px;
        }}
        """
