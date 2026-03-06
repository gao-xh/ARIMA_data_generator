from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QSizePolicy
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import logging

class MplCanvas(FigureCanvas):
    """
    Standard Matplotlib Canvas for PySide6.
    """
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)
        self.setParent(parent)
        
        # Policy to expand
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()

class PlotWidget(QWidget):
    """
    Widget containing the Canvas and Toolbar.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.canvas = MplCanvas(self, width=8, height=6, dpi=100)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def plot(self, x, y_actual, y_forecast, title="Forecast vs Actual", metrics=None):
        """
        Plots the time series data.
        """
        self.canvas.axes.clear()
        
        self.canvas.axes.plot(x, y_actual, label='Actual Sales', color='#2ecc71', linewidth=2)
        self.canvas.axes.plot(x, y_forecast, label='Forecast (Improved)', color='#e74c3c', linestyle='--', linewidth=2)
        
        self.canvas.axes.set_title(title, fontsize=12, fontweight='bold')
        self.canvas.axes.set_xlabel("Date")
        self.canvas.axes.set_ylabel("Sales Volume")
        self.canvas.axes.legend()
        self.canvas.axes.grid(True, alpha=0.3)
        
        # Add Metrics Text Box
        if metrics:
            textstr = '\n'.join([f"{k}: {v}" for k, v in metrics.items()])
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            self.canvas.axes.text(0.02, 0.95, textstr, transform=self.canvas.axes.transAxes, fontsize=10,
                    verticalalignment='top', bbox=props)
        
        self.canvas.fig.tight_layout()
        self.canvas.draw()
