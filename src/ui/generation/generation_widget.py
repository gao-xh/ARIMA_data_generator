import sys
import traceback
import datetime
import pandas as pd
import numpy as np
from pathlib import Path
from PySide6.QtWidgets import (
     QWidget, QVBoxLayout, QHBoxLayout, 
     QGroupBox, QPushButton, QLabel, 
     QTextEdit, QProgressBar, QComboBox, QDoubleSpinBox, QSpinBox,
     QSplitter, QSizePolicy, QFormLayout
)
from PySide6.QtCore import Qt, QThread, Signal

# Matplotlib Integration
try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
except ImportError:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    
from matplotlib.figure import Figure
import matplotlib.dates as mdates

# Core Logic
from src.core.tools.simulation_tuner import SimulationTuner
from src.core.simulation_config import SimulationConfig
from src.config import DRUG_INFO, EXTERNAL_FACTORS_FILE

class SimulationWorker(QThread):
    """
    Background worker to run simulation without freezing UI.
    """
    finished = Signal(pd.DataFrame)
    error = Signal(str)
    
    def __init__(self, config: SimulationConfig, drug_info: dict, external_data: pd.DataFrame, duration_days: int):
        super().__init__()
        self.config = config
        self.drug_info = drug_info
        self.external_data = external_data
        self.duration_days = duration_days
        
    def run(self):
        try:
            # Initialize Tuner
            tuner = SimulationTuner(
                self.config, 
                self.drug_info, 
                self.external_data
            )
            # Run
            df = tuner.run_simulation_only(total_days=self.duration_days)
            self.finished.emit(df)
            
        except Exception as e:
            err_msg = f"Simulation Failed: {str(e)}\n{traceback.format_exc()}"
            self.error.emit(err_msg)

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()

class GenerationWidget(QWidget):
    """
    Enhanced Simulation UI with Parameter Control and Visualization.
    """
    def __init__(self):
        super().__init__()
        self.drug_df = None
        self.ext_df = None
        self._init_data()
        self._init_ui()
        
    def _init_data(self):
        try:
            # Load External Factors (Once)
            if Path(EXTERNAL_FACTORS_FILE).exists():
                self.ext_df = pd.read_csv(EXTERNAL_FACTORS_FILE)
                # Ensure date parsing
                date_col = next((c for c in self.ext_df.columns if 'date' in c.lower() or '日期' in c), None)
                if date_col:
                    self.ext_df['date'] = pd.to_datetime(self.ext_df[date_col])
            else:
                # Mock External Data if missing
                dates = pd.date_range(start='2023-01-01', end='2025-12-31')
                self.ext_df = pd.DataFrame({
                    'date': dates, 
                    '平均气温': np.random.normal(20, 5, len(dates)),
                    'ILI%': np.random.uniform(0, 0.05, len(dates))
                })
        except Exception as e:
            print(f"Error loading external data: {e}")

    def _init_ui(self):
        main_layout = QHBoxLayout(self)
        
        # --- Left Panel: Controls ---
        control_panel = QGroupBox("Configuration")
        control_layout = QVBoxLayout()
        
        # 1. Drug Selection
        control_layout.addWidget(QLabel("Select Drug:"))
        self.combo_drug = QComboBox()
        self.combo_drug.currentIndexChanged.connect(self._on_drug_selected)
        control_layout.addWidget(self.combo_drug)
        
        # 2. Parameters Form
        param_group = QGroupBox("Simulation Parameters")
        form_layout = QFormLayout()
        
        self.spin_safety = QDoubleSpinBox()
        self.spin_safety.setRange(0.5, 5.0)
        self.spin_safety.setSingleStep(0.1)
        self.spin_safety.setValue(1.96)
        self.spin_safety.setToolTip("Safety Stock Factor (Z-Score)")
        
        self.spin_flu_sens = QDoubleSpinBox()
        self.spin_flu_sens.setRange(0.0, 5.0)
        self.spin_flu_sens.setSingleStep(0.1)
        self.spin_flu_sens.setValue(1.0)
        self.spin_flu_sens.setToolTip("Sensitivity to Flu Outbreaks")

        self.spin_replenish = QSpinBox()
        self.spin_replenish.setRange(1, 90)
        self.spin_replenish.setValue(30)
        self.spin_replenish.setSuffix(" Days")
        self.spin_replenish.setToolTip("Replenishment Cycle (Review Period R)")

        self.spin_lead_time = QSpinBox()
        self.spin_lead_time.setRange(0, 30)
        self.spin_lead_time.setValue(3)
        self.spin_lead_time.setSuffix(" Days")
        self.spin_lead_time.setToolTip("Lead Time (Delivery Delay L)")
        
        self.spin_validity = QDoubleSpinBox()
        self.spin_validity.setRange(30, 2000)
        self.spin_validity.setSingleStep(30)
        self.spin_validity.setValue(365)
        self.spin_validity.setSuffix(" Days")
        
        self.spin_duration = QDoubleSpinBox()
        self.spin_duration.setRange(30, 1095)
        self.spin_duration.setDecimals(0)
        self.spin_duration.setValue(730)
        self.spin_duration.setSuffix(" Days")
        
        form_layout.addRow("Safety Factor:", self.spin_safety)
        form_layout.addRow("Flu Sensitivity:", self.spin_flu_sens)
        form_layout.addRow("Cycle (R):", self.spin_replenish)
        form_layout.addRow("Lead Time (L):", self.spin_lead_time)
        form_layout.addRow("Shelf Life:", self.spin_validity)
        form_layout.addRow("Duration:", self.spin_duration)
        param_group.setLayout(form_layout)
        
        control_layout.addWidget(param_group)
        
        # 3. Actions
        self.btn_load = QPushButton("Reload Drugs")
        self.btn_load.clicked.connect(self.load_drugs_list)
        
        self.btn_run = QPushButton("Run Simulation")
        self.btn_run.setStyleSheet("background-color: #28a745; color: white; font-weight: bold; padding: 10px;")
        self.btn_run.clicked.connect(self.start_simulation)
        
        control_layout.addWidget(self.btn_load)
        control_layout.addStretch()
        control_layout.addWidget(self.btn_run)
        
        control_panel.setLayout(control_layout)
        control_panel.setFixedWidth(300)
        
        # --- Right Panel: Visualization ---
        viz_panel = QGroupBox("Results Visualization")
        viz_layout = QVBoxLayout()
        
        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        self.log_console = QTextEdit()
        self.log_console.setReadOnly(True)
        self.log_console.setMaximumHeight(150)
        
        viz_layout.addWidget(self.canvas)
        viz_layout.addWidget(QLabel("Simulation Log:"))
        viz_layout.addWidget(self.log_console)
        
        viz_panel.setLayout(viz_layout)
        
        # Add to Main Layout
        main_layout.addWidget(control_panel)
        main_layout.addWidget(viz_panel)
        
        # Initial Load
        self.load_drugs_list()

    def log(self, msg):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_console.append(f"[{timestamp}] {msg}")

    def load_drugs_list(self):
        try:
            if Path(DRUG_INFO).exists():
                try:
                    self.drug_df = pd.read_csv(DRUG_INFO, encoding='utf-8')
                except UnicodeDecodeError:
                    self.drug_df = pd.read_csv(DRUG_INFO, encoding='gb18030')
                
                self.combo_drug.clear()
                items = []
                for idx, row in self.drug_df.iterrows():
                    name = str(row.get('药品名称', 'Unknown'))
                    cat = str(row.get('药品品类', 'Misc'))
                    items.append(f"{name} ({cat})")
                self.combo_drug.addItems(items)
                self.log(f"Loaded {len(items)} drugs from metadata.")
                
                # Trigger selection change to update params
                self._on_drug_selected(0)
            else:
                self.log("Drug Info file not found.")
        except Exception as e:
            self.log(f"Error loading drugs: {e}")

    def _on_drug_selected(self, index):
        if self.drug_df is None or index < 0 or index >= len(self.drug_df):
            return
            
        row = self.drug_df.iloc[index]
        # Auto-fill suggested validity
        try:
            months = float(row.get('效期（月）', 12))
            self.spin_validity.setValue(months * 30)
        except:
            pass

    def start_simulation(self):
        idx = self.combo_drug.currentIndex()
        if self.drug_df is None or idx < 0:
            self.log("Please select a drug first.")
            return
            
        # Prepare Config
        config = SimulationConfig(
            start_date=pd.Timestamp('2023-01-01'),
            end_date=pd.Timestamp('2025-12-31'), # Max range
            safety_stock_factor=self.spin_safety.value(),
            flu_sensitivity=self.spin_flu_sens.value(),
            replenishment_days=int(self.spin_replenish.value())
        )
        
        row = self.drug_df.iloc[idx]
        drug_info = row.to_dict()
        # Override params from UI
        drug_info['有效期'] = int(self.spin_validity.value())
        drug_info['补货提前期'] = int(self.spin_lead_time.value())
        drug_info['药品ID'] = str(row.get('药品编号', f'DRUG_{idx}'))
        drug_info['药品名称'] = str(row.get('药品名称', 'Unknown'))
        drug_info['单价'] = float(row.get('零售价', 35.0))
        
        duration = int(self.spin_duration.value())
        
        self.btn_run.setEnabled(False)
        self.log(f"Starting simulation for {drug_info['药品名称']}...")
        
        self.worker = SimulationWorker(config, drug_info, self.ext_df, duration)
        self.worker.finished.connect(self.on_simulation_finished)
        self.worker.error.connect(self.on_simulation_error)
        self.worker.start()

    def on_simulation_finished(self, df: pd.DataFrame):
        self.btn_run.setEnabled(True)
        self.log(f"Simulation completed. Generated {len(df)} records.")
        self.plot_results(df)

    def on_simulation_error(self, msg):
        self.btn_run.setEnabled(True)
        self.log(f"Error: {msg}")

    def plot_results(self, df: pd.DataFrame):
        self.canvas.axes.clear()
        
        # Scenario Split
        # Fallback if scenario column missing (legacy run)
        if 'scenario' not in df.columns:
            df['scenario'] = 'Optimized' 
            
        df_base = df[df['scenario'] == 'Baseline']
        df_opt = df[df['scenario'] == 'Optimized']
        
        # Use date checks for plotting
        if not df_base.empty:
            x_base = df_base['date'] if 'date' in df_base.columns else range(len(df_base))
            self.canvas.axes.plot(x_base, df_base['inventory'], color='#808080', linestyle='--', alpha=0.7, label='Baseline (Manual)')
            
            # Add Baseline Stockouts
            base_out = df_base[df_base['stockout_flag'] > 0]
            if not base_out.empty:
                self.canvas.axes.scatter(base_out['date'], [-2]*len(base_out), color='black', marker='o', s=15, label='Baseline Stockout')

        # KPI Report
        self.log("\n=== 📊 Simulation Report ===")
        for name, sub_df in [('Baseline', df_base), ('Optimized', df_opt)]:
            if sub_df.empty: continue
            
            avg_inv = sub_df['inventory'].mean()
            stockout_days = sub_df['stockout_flag'].sum()
            total_sales = sub_df['sales'].sum()
            total_days = (sub_df['date'].max() - sub_df['date'].min()).days + 1
            
            turnover = (avg_inv / (total_sales / total_days)) if total_sales > 0 else 0
            service_level = 1.0 - (stockout_days / total_days)
            
            self.log(f"🔹 {name} Strategy:")
            self.log(f"   • Avg Inventory: {avg_inv:.1f} units")
            self.log(f"   • Turnover Days: {turnover:.1f} days")
            self.log(f"   • Service Level: {service_level*100:.1f}% ({stockout_days} stockouts)")

        self.canvas.axes.set_title("Strategy Comparison: Empirical vs ARIMA-Optimized")
        self.canvas.axes.set_xlabel("Date")
        self.canvas.axes.set_ylabel("Inventory Level")
        self.canvas.axes.legend()
        self.canvas.axes.grid(True, alpha=0.3)
        
        # Rotate dates
        self.canvas.fig.autofmt_xdate()
        self.canvas.draw()
