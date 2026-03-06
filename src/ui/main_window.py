from PySide6.QtWidgets import (
    QMainWindow, QTabWidget, QVBoxLayout, QWidget, QStatusBar
)
from src.ui.validation.main_window import ValidationWidget
from src.ui.generation.generation_widget import GenerationWidget

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Medical Inventory Optimization System (Improved ARIMA)")
        self.resize(1200, 800)
        
        # Central Widget
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        
        # Tab 1: Configuration & Generation
        # "Generate missing dataset" use case
        self.generation_tab = GenerationWidget()
        self.tabs.addTab(self.generation_tab, "Simulation & Generator")
        
        # Tab 2: Model Validation
        # "Discussion based on target.doc" use case
        self.validation_tab = ValidationWidget()
        self.tabs.addTab(self.validation_tab, "Model Validation")
        
        # Styling
        self._apply_styles()
        
        # Status Bar
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("System Ready via .venv")

    def _apply_styles(self):
        # Professional Dark/Light Theme mix
        qss = """
        QMainWindow {
            background-color: #F0F0F0;
        }
        QTabWidget::pane {
            border: 1px solid #C0C0C0;
            background: white;
            border-radius: 4px;
        }
        QTabBar::tab {
            background: #E0E0E0;
            padding: 8px 20px;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
            margin-right: 2px;
        }
        QTabBar::tab:selected {
            background: #007ACC;
            color: white;
            font-weight: bold;
        }
        QGroupBox {
            font-weight: bold;
            border: 1px solid #D0D0D0;
            border-radius: 6px;
            margin-top: 10px;
            padding-top: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 3px;
            color: #333;
        }
        QPushButton {
            background-color: #007ACC;
            color: white;
            border-radius: 4px;
            padding: 6px;
        }
        QPushButton:hover {
            background-color: #005A9E;
        }
        QPushButton:disabled {
            background-color: #A0A0A0;
        }
        """
        self.setStyleSheet(qss)
