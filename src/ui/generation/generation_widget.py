from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, 
    QGroupBox, QPushButton, QProgressBar, QLabel, QMessageBox, QComboBox
)
from PySide6.QtCore import Qt, QThread, Signal
import pandas as pd
import numpy as np
import logging
from datetime import datetime

from src.core.generator import DataGenerator
from src.core import constants as C
from src.ui.common.widgets import SliderInputWidget, MetricsTableWidget, PlotWidget

logger = logging.getLogger(__name__)

class GenerationWorker(QThread):
    finished = Signal(object, dict) # DataFrame, Metrics
    progress = Signal(int)
    error = Signal(str)

    def __init__(self, params):
        super().__init__()
        self.params = params

    def run(self):
        try:
            # Load base data (Mocking loading for now, should use real loader)
            # In a real app, we should inject the data loader
            drug_df = pd.read_csv(C.FILE_DRUG_INFO)
            ext_df = pd.read_csv(C.FILE_EXTERNAL_FACTORS)
            
            # Filter by date if needed (e.g., self.params['start_year'])
            # Keeping it simple for demo
            
            generator = DataGenerator(drug_df, ext_df, params=self.params)
            
            # Only simulate ONE clinic for responsiveness in UI preview
            dataset = generator.generate_full_dataset(clinics=['Clinic_A']) 
            
            # Calculate metrics
            total_sales = dataset[C.COL_SALES].sum()
            stockouts = dataset[C.COL_STOCKOUT].sum()
            stockout_rate = (stockouts / len(dataset)) * 100
            
            metrics = {
                "Total Sales (Units)": total_sales,
                "Stockout Events": stockouts,
                "Stockout Rate (%)": round(stockout_rate, 2),
                "Simulated Days": len(dataset)
            }
            
            self.finished.emit(dataset, metrics)
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            self.error.emit(str(e))

class GenerationWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        
        # Main Layout: Splitter
        splitter = QSplitter(Qt.Horizontal)
        self.layout.addWidget(splitter)
        
        # --- Left Panel: Controls ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setAlignment(Qt.AlignTop)
        
        # Group 1: Environmental Factors
        env_group = QGroupBox("Environmental Parameters")
        env_layout = QVBoxLayout()
        
        self.flu_threshold = SliderInputWidget("Flu Threshold (%)", 0.0, 20.0, 5.0, 0.5)
        self.flu_impact = SliderInputWidget("Flu Impact Factor", 0.0, 2.0, 0.2, 0.1)
        self.temp_threshold = SliderInputWidget("Cold Temp Threshold (°C)", -10.0, 30.0, 5.0, 1.0)
        
        env_layout.addWidget(self.flu_threshold)
        env_layout.addWidget(self.flu_impact)
        env_layout.addWidget(self.temp_threshold)
        env_group.setLayout(env_layout)
        left_layout.addWidget(env_group)
        
        # Group 2: Simulation Settings
        sim_group = QGroupBox("Simulation Settings")
        sim_layout = QVBoxLayout()
        
        self.replenishment_days = SliderInputWidget("Replenishment Cycle (Days)", 1, 30, 14, 1)
        sim_layout.addWidget(self.replenishment_days)
        sim_group.setLayout(sim_layout)
        left_layout.addWidget(sim_group)
        
        # Action Button
        self.btn_generate = QPushButton("Run Simulation")
        self.btn_generate.setStyleSheet("background-color: #007ACC; color: white; padding: 8px; font-weight: bold;")
        self.btn_generate.clicked.connect(self.start_generation)
        left_layout.addWidget(self.btn_generate)
        
        # Metrics Table (Output)
        self.metrics_table = MetricsTableWidget()
        left_layout.addWidget(QLabel("Simulation Results:"))
        left_layout.addWidget(self.metrics_table)
        
        # Progress Bar
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        left_layout.addWidget(self.progress)

        splitter.addWidget(left_panel)
        
        # --- Right Panel: Visualization ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        self.plot_widget = PlotWidget()
        right_layout.addWidget(self.plot_widget)
        
        splitter.addWidget(right_panel)
        
        # Set splitter proportions (30% Left, 70% Right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 7)

    def start_generation(self):
        params = {
            'flu_threshold': self.flu_threshold.value(),
            'flu_impact_factor': self.flu_impact.value(),
            'temp_threshold': self.temp_threshold.value(),
            'base_replenishment_days': self.replenishment_days.value()
        }
        
        self.btn_generate.setEnabled(False)
        self.progress.setRange(0, 0) # Indeterminate
        self.progress.setVisible(True)
        
        # Start Worker
        self.worker = GenerationWorker(params)
        self.worker.finished.connect(self.on_generation_finished)
        self.worker.error.connect(self.on_generation_error)
        self.worker.start()

    def on_generation_finished(self, dataset, metrics):
        self.btn_generate.setEnabled(True)
        self.progress.setVisible(False)
        
        # Update Table
        self.metrics_table.update_metrics(metrics)
        
        # Update Plot
        self._update_plot(dataset)
        
        QMessageBox.information(self, "Success", f"Generated {len(dataset)} records.")

    def on_generation_error(self, error_msg):
        self.btn_generate.setEnabled(True)
        self.progress.setVisible(False)
        QMessageBox.critical(self, "Error", f"Simulation failed: {error_msg}")

    def _update_plot(self, df):
        # Plot Sales vs Temperature for the first drug found
        if df.empty:
            return
            
        first_drug = df[C.COL_DRUG_CODE].unique()[0]
        drug_data = df[df[C.COL_DRUG_CODE] == first_drug].copy()
        drug_data[C.COL_DATE] = pd.to_datetime(drug_data[C.COL_DATE])
        drug_data = drug_data.sort_values(C.COL_DATE)
        
        ax = self.plot_widget.canvas.axes
        ax.clear()
        
        # Primary Axis: Sales
        line1, = ax.plot(drug_data[C.COL_DATE], drug_data[C.COL_SALES], 'b-', label='Daily Sales', alpha=0.6)
        ax.set_ylabel('Sales (Units)', color='b')
        ax.tick_params(axis='y', labelcolor='b')
        
        # Secondary Axis: Temperature (if available in merged data, currently mocked)
        # Note: In real generation, we fetch temp again or merge it. 
        # For this demo, let's just show Sales & Inventory
        
        ax2 = ax.twinx()
        line2, = ax2.plot(drug_data[C.COL_DATE], drug_data[C.COL_INV_END], 'g--', label='Inventory Level', alpha=0.4)
        ax2.set_ylabel('Inventory (Units)', color='g')
        ax2.tick_params(axis='y', labelcolor='g')
        
        # Title
        ax.set_title(f"Simulation Preview: Drug {first_drug}")
        
        # Legend (Combined)
        lines = [line1, line2]
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper left')
        
        self.plot_widget.canvas.draw()
