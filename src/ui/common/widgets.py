from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QSizePolicy, QHBoxLayout, QLabel, QSlider, QLineEdit, QDoubleSpinBox, QTableWidget, QTableWidgetItem, QHeaderView
)
from PySide6.QtCore import Qt, Signal
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import logging

class SliderInputWidget(QWidget):
    """
    A widget that combines a Label, a Slider, and a SpinBox for synchronized input.
    """
    valueChanged = Signal(float)

    def __init__(self, label_text, min_val=0.0, max_val=100.0, initial_val=50.0, 
                 step=1.0, decimals=2, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 1. Label
        self.label = QLabel(label_text)
        self.label.setMinimumWidth(100)
        layout.addWidget(self.label)
        
        # Factor to handle float values in integer slider
        self.factor = 10 ** decimals
        
        # 2. Slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(int(min_val * self.factor), int(max_val * self.factor))
        self.slider.setValue(int(initial_val * self.factor))
        self.slider.setSingleStep(int(step * self.factor))
        layout.addWidget(self.slider)
        
        # 3. SpinBox
        self.spinbox = QDoubleSpinBox()
        self.spinbox.setRange(min_val, max_val)
        self.spinbox.setValue(initial_val)
        self.spinbox.setSingleStep(step)
        self.spinbox.setDecimals(decimals)
        layout.addWidget(self.spinbox)
        
        # Connect signals
        self.slider.valueChanged.connect(self._on_slider_changed)
        self.spinbox.valueChanged.connect(self._on_spinbox_changed)

    def _on_slider_changed(self, value):
        float_val = value / self.factor
        self.spinbox.setValue(float_val)
        self.valueChanged.emit(float_val)

    def _on_spinbox_changed(self, value):
        slider_val = int(value * self.factor)
        if self.slider.value() != slider_val:
            self.slider.setValue(slider_val)
        self.valueChanged.emit(value)
        
    def value(self):
        return self.spinbox.value()

class MetricsTableWidget(QTableWidget):
    """
    A simple table to display key-value metrics like Variance, MAPE, RMSE.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setColumnCount(2)
        self.setHorizontalHeaderLabels(["Metric", "Value"])
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.verticalHeader().setVisible(False)
        self.setAlternatingRowColors(True)
        
    def update_metrics(self, metrics_dict):
        """
        Updates the table with a dictionary of metrics.
        """
        self.setRowCount(len(metrics_dict))
        for row, (key, value) in enumerate(metrics_dict.items()):
            key_item = QTableWidgetItem(str(key))
            key_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable) # Read-only
            
            if isinstance(value, float):
                val_str = f"{value:.4f}"
            else:
                val_str = str(value)
                
            val_item = QTableWidgetItem(val_str)
            val_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            
            self.setItem(row, 0, key_item)
            self.setItem(row, 1, val_item)

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
