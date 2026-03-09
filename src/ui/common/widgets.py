from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QSizePolicy, QHBoxLayout, QLabel, QSlider, QDoubleSpinBox, 
    QTableWidget, QTableWidgetItem, QHeaderView
)
from PySide6.QtCore import Qt, Signal
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.dates as mdates

class SliderInputWidget(QWidget):
    valueChanged = Signal(float)

    def __init__(self, label_text, min_val=0.0, max_val=100.0, initial_val=50.0, 
                 step=1.0, decimals=2, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.label = QLabel(label_text)
        self.label.setMinimumWidth(100)
        layout.addWidget(self.label)
        
        self.factor = 10 ** decimals
        
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(int(min_val * self.factor), int(max_val * self.factor))
        self.slider.setValue(int(initial_val * self.factor))
        self.slider.setSingleStep(int(step * self.factor))
        layout.addWidget(self.slider)
        
        self.spinbox = QDoubleSpinBox()
        self.spinbox.setRange(min_val, max_val)
        self.spinbox.setValue(initial_val)
        self.spinbox.setSingleStep(step)
        self.spinbox.setDecimals(decimals)
        layout.addWidget(self.spinbox)
        
        self.slider.valueChanged.connect(self._on_slider_changed)
        self.spinbox.valueChanged.connect(self._on_spinbox_changed)

    def _on_slider_changed(self, value):
        float_val = value / self.factor
        if abs(self.spinbox.value() - float_val) > 1e-5:
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
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setColumnCount(2)
        self.setHorizontalHeaderLabels(['Metric', 'Value'])
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.verticalHeader().setVisible(False)
        self.setAlternatingRowColors(True)
        
    def update_metrics(self, metrics_dict):
        self.setRowCount(len(metrics_dict))
        for row, (key, value) in enumerate(metrics_dict.items()):
            key_item = QTableWidgetItem(str(key))
            key_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            
            if isinstance(value, float):
                val_str = f'{value:.4f}'
            else:
                val_str = str(value)
                
            val_item = QTableWidgetItem(val_str)
            val_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            
            self.setItem(row, 0, key_item)
            self.setItem(row, 1, val_item)

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()

class PlotWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.canvas = MplCanvas(self, width=8, height=6, dpi=100)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def plot(self, x, y_actual, y_forecast, title='Forecast vs Actual', metrics=None):
        self.canvas.axes.clear()
        
        try:
             # Basic Plot
            self.canvas.axes.plot(x, y_actual, label='Actual Sales', color='#2ecc71', linewidth=2, alpha=0.8)
            self.canvas.axes.plot(x, y_forecast, label='Forecast (Improved)', color='#e74c3c', linestyle='--', linewidth=2)
            
            # Formatting
            self.canvas.axes.set_title(title, fontsize=12, fontweight='bold')
            self.canvas.axes.set_xlabel('Date')
            self.canvas.axes.set_ylabel('Sales Volume')
            self.canvas.axes.legend()
            self.canvas.axes.grid(True, alpha=0.3)
            
            # Date Formatting
            self.canvas.axes.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            self.canvas.axes.xaxis.set_major_locator(mdates.AutoDateLocator())
            self.canvas.fig.autofmt_xdate()
            
            if metrics:
                textstr = '\n'.join([f'{k}: {v}' for k, v in metrics.items()])
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                self.canvas.axes.text(0.02, 0.95, textstr, transform=self.canvas.axes.transAxes, fontsize=9,
                        verticalalignment='top', bbox=props)
            
            self.canvas.fig.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            print(f'Plotting error: {e}')
