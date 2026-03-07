import sys
from pathlib import Path
from PySide6.QtWidgets import (
     QWidget, QVBoxLayout, QHBoxLayout, 
     QGroupBox, QPushButton, QLabel, 
     QTextEdit, QProgressBar, QComboBox, QSpinBox, 
     QTableWidget, QTableWidgetItem, QSplitter, QScrollArea
)
from PySide6.QtCore import Qt, QThread, Signal, Slot
import pandas as pd
import numpy as np
import datetime
import traceback

# Imports
from src.core.tools.simulation_tuner import SimulationTuner
from src.core.simulation_config import SimulationConfig
from src.core.thesis_params import ThesisParams

class GenerationWorker(QThread):
    """
    Background worker to run SimulationTuner without freezing UI.
    """
    progress = Signal(dict) # Emits the entire progress payload
    finished = Signal(pd.DataFrame)
    error = Signal(str)
    
    def __init__(self, config: SimulationConfig, drug_info: dict, external_data: pd.DataFrame):
        super().__init__()
        self.config = config
        self.drug_info = drug_info
        self.external_data = external_data
        
    def run(self):
        try:
            # Create Tuner with our signal emitter as callback
            tuner = SimulationTuner(
                self.config, 
                self.drug_info, 
                self.external_data,
                progress_callback=self.emit_progress
            )
            
            # Run
            delta = self.config.end_date - self.config.start_date
            total_days = delta.days
            final_df = tuner.run_adaptive_simulation(total_days=total_days)
            self.finished.emit(final_df)
            
        except Exception as e:
            err_msg = f"{str(e)}\n{traceback.format_exc()}"
            self.error.emit(err_msg)
            
    def emit_progress(self, data: dict):
        self.progress.emit(data)

class GenerationWidget(QWidget):
    """
    Main UI for Dataset Generation & Validation.
    """
    def __init__(self):
        super().__init__()
        self._init_ui()
        self.worker = None
        
    def _init_ui(self):
        layout = QVBoxLayout(self)
        
        # 1. Config Panel
        config_group = QGroupBox("Simulation Configuration")
        config_layout = QHBoxLayout()
        
        self.btn_load_meta = QPushButton("Load Drug Metadata")
        self.lbl_drug_status = QLabel("No Drug Selected")
        
        self.spin_days = QSpinBox()
        self.spin_days.setRange(30, 730)
        self.spin_days.setValue(365)
        self.spin_days.setSuffix(" Days")
        
        self.btn_start = QPushButton("Start Adaptive Simulation")
        self.btn_start.setEnabled(False)
        self.btn_start.clicked.connect(self.start_simulation)
        
        config_layout.addWidget(self.btn_load_meta)
        config_layout.addWidget(self.lbl_drug_status)
        config_layout.addWidget(QLabel("Duration:"))
        config_layout.addWidget(self.spin_days)
        config_layout.addWidget(self.btn_start)
        config_group.setLayout(config_layout)
        
        # 2. Progress Visualization
        viz_group = QGroupBox("Adaptive Process Visualization")
        viz_layout = QVBoxLayout()
        
        # Progress Bar for Overall Timeline
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.lbl_progress = QLabel("Ready")
        
        # Log Console
        self.log_console = QTextEdit()
        self.log_console.setReadOnly(True)
        self.log_console.setStyleSheet("font-family: Consolas; font-size: 10pt;")
        
        viz_layout.addWidget(self.lbl_progress)
        viz_layout.addWidget(self.progress_bar)
        viz_layout.addWidget(self.log_console)
        viz_group.setLayout(viz_layout)
        
        # 3. Main Layout Assembly
        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(config_group)
        splitter.addWidget(viz_group)
        
        layout.addWidget(splitter)
        
        # Mock Data Loading (Replace with real logic)
        self.btn_load_meta.clicked.connect(self.load_mock_drug)

    def load_mock_drug(self):
        # In real app, load from CSV defined in config
        from src.config import DRUG_INFO
        import pandas as pd
        
        try:
            # Load CSV (drug_info.csv)
            # Encoding might be GBK or UTF-8. Try UTF-8 first, fallback to gb18030
            try:
                df = pd.read_csv(DRUG_INFO, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(DRUG_INFO, encoding='gb18030')
                
            # Available Columns based on file preview:
            # 药品编号, 药品名称, 含量规格, 零售单位, 效期（月）, 药品剂型, 零售价, 药品品类, 波动区间分类
            
            # Map to internal keys
            # Pick the FIRST row for demo, or handle selection in a real list
            row = df.iloc[0]
            
            validity_months = row.get('效期（月）', 12)
            try:
                validity_days = int(validity_months) * 30
            except:
                validity_days = 365
                
            self.current_drug = {
                '药品ID': str(row.get('药品编号', 'UNKNOWN')),
                '药品名称': str(row.get('药品名称', 'Unknown Drug')),
                '有效期': validity_days,
                '单价': float(row.get('零售价', 35.0)),
                '波动分类': str(row.get('波动区间分类', '中波动'))
            }
            self.log(f"Loaded real drug: {self.current_drug['药品名称']} (ID: {self.current_drug['药品ID']})")
            self.log(f"  Price: {self.current_drug['单价']}, Validity: {self.current_drug['有效期']} days")
            
        except Exception as e:
            self.log(f"Metadata file error ({str(e)}), using mock.")
            self.current_drug = {
                '药品ID': 'DRUG_001',
                '药品名称': 'Test Antibiotic',
                '有效期': 365,
                '单价': 25.0 
            }
        
        self.lbl_drug_status.setText(f"Selected: {self.current_drug.get('药品名称', 'Unknown')}")
        self.btn_start.setEnabled(True)

    def log(self, message: str):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_console.append(f"[{ts}] {message}")
        # Auto scroll
        sb = self.log_console.verticalScrollBar()
        sb.setValue(sb.maximum())

    def start_simulation(self):
        if not hasattr(self, 'current_drug'):
            return
            
        # Create Config
        config = SimulationConfig()
        days_to_sim = self.spin_days.value()
        config.end_date = config.start_date + datetime.timedelta(days=days_to_sim)
        
        # Mock External Data
        # Assume external_data is passed or loaded. Creating dummy for now.
        dates = pd.date_range(config.start_date, config.end_date)
        ext_df = pd.DataFrame(index=dates)
        # Create sinusoidal temperature
        day_indices = np.arange(len(dates))
        temp_cycle = 20 + 10 * np.cos((day_indices - 30) / 365 * 2 * np.pi) # Coldest around Jan (day 0 or 365)
        ext_df['平均气温'] = temp_cycle
        # Flu seasonality (Peak in Dec/Jan)
        ili_cycle = 0.02 + 0.08 * np.exp(-((day_indices % 365) - 15)**2 / (2 * 30**2))
        ext_df['ILI%'] = ili_cycle
        
        # Start Worker
        self.worker = GenerationWorker(config, self.current_drug, ext_df)
        self.worker.progress.connect(self.on_progress)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        
        self.worker.start()
        self.btn_start.setEnabled(False)
        self.log_console.clear()
        self.log("Starting Adaptive Simulation...")
        self.log(f"Target Period: {config.start_date.date()} to {config.end_date.date()}")
        
    @Slot(dict)
    def on_progress(self, data: dict):
        event = data.get('event')
        
        if event == 'start':
            self.total_days = data.get('total_days', 365)
            self.progress_bar.setMaximum(self.total_days)
            self.log(f"Process Started: Target {self.total_days} days.")
            
        elif event == 'segment_start':
            start = data.get('start_day')
            end = data.get('end_day')
            params = data.get('current_params')
            self.log(f"--- Segment [{start} - {end}] ---")
            self.log(f"Params: SafetyFactor={params.get('safety_factor'):.2f}, Validity={params.get('validity_days')}")
            
        elif event == 'iteration':
            attempt = data.get('attempt')
            error = data.get('error')
            feedback = data.get('feedback')
            msg = f"  Attempt {attempt}: Error={error:.4f}"
            if feedback:
                msg += f" Feedback: {feedback}"
            self.log(msg)
            
        elif event == 'segment_committed':
            final_err = data.get('final_error')
            end_day = data.get('day_range', (0,0))[1]
            self.progress_bar.setValue(end_day)
            self.log(f"Segment Committed. Final Error: {final_err:.4f}")
            
        elif event == 'complete':
            total = data.get('total_records')
            self.log(f"Simulation Complete. Generated {total} records.")

    @Slot(pd.DataFrame)
    def on_finished(self, df: pd.DataFrame):
        self.btn_start.setEnabled(True)
        self.log("Task Finished Successfully.")
        # TODO: Save or display dataframe
        
    @Slot(str)
    def on_error(self, err: str):
        self.log(f"ERROR: {err}")
        self.btn_start.setEnabled(True)
