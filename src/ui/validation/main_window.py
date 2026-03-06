import sys
from pathlib import Path
from PySide6.QtWidgets import (
     QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
     QPushButton, QFileDialog, QComboBox, QDoubleSpinBox, 
     QLabel, QGroupBox, QSplitter, QProgressBar, QMessageBox, 
     QTextEdit, QTableWidget, QTableWidgetItem
)
from PySide6.QtCore import Qt, QThread, Signal, Slot
import pandas as pd
import numpy as np

# Adjust path for local imports
sys.path.append(str(Path(__file__).resolve().parents[3]))

from src.models import ImprovedARIMA
from src.ui.common.widgets import PlotWidget, MetricsTableWidget
from src.core import constants as C
from src.evaluation import calculate_mape, calculate_rmse
from sklearn.metrics import explained_variance_score, r2_score

class ModelWorker(QThread):
    finished = Signal(object, object, object, dict) # dates, actuals, forecast, metrics
    error = Signal(str)
    
    def __init__(self, df: pd.DataFrame, drug_info: pd.Series, params: dict):
        super().__init__()
        self.df = df
        self.drug_info = drug_info
        self.params = params
        
    def run(self):
        try:
            # Prepare Data
            train_size = int(len(self.df) * 0.8)
            train_df = self.df.iloc[:train_size].copy()
            test_df = self.df.iloc[train_size:].copy()
            
            # Apply Parameter Overrides (for "What-If" Analysis)
            # Override Fluctuation Class
            override_cls = self.params.get('fluctuation_class', None)
            if override_cls:
                self.drug_info['波动区间分类'] = override_cls
                
            # Init Model
            model = ImprovedARIMA(drug_info=self.drug_info)
            
            # Param Updates (Order Override if needed - currently not exposed in __init__ but maybe in future)
            # Here we trust the class logic based on the override_cls above.
            
            # Train
            # Rename columns to match model expectations if needed
            # Model expects: '平均气温', '流感ILI%'
            rename_map = {'Temp': C.EXT_TEMP, 'Flu': C.EXT_FLU, 'Sales': C.COL_SALES, 'Date': C.COL_DATE}
            train_ready = train_df.rename(columns=rename_map)
            test_ready = test_df.rename(columns=rename_map)
            
            model.train(train_ready)
            
            # Predict
            future_exog = test_ready[[C.COL_DATE, C.EXT_TEMP, '流感ILI%']]
            if '平局降水量' not in future_exog.columns:
                 future_exog['平均降水量'] = 0 # Dummy if missing
                 
            # Apply Decay Params
            days_valid = self.params.get('validity_days', 365)
            
            forecast = model.predict(steps=len(test_ready), future_exog_df=future_exog, current_stock_validity_days=days_valid)
            
            # Calculate Metrics
            actuals = test_ready[C.COL_SALES].values
            mape = calculate_mape(actuals, forecast)
            rmse = calculate_rmse(actuals, forecast)
            r2 = r2_score(actuals, forecast) if len(actuals) > 1 else 0
            variance_explained = explained_variance_score(actuals, forecast) if len(actuals) > 1 else 0
            
            metrics = {
                'MAPE': f"{mape:.2f}%", 
                'RMSE': f"{rmse:.2f}",
                'R² Score': f"{r2:.4f}",
                'Explained Variance': f"{variance_explained:.4f}"
            }
            
            self.finished.emit(test_ready[C.COL_DATE], actuals, forecast, metrics)

            
        except Exception as e:
            self.error.emit(str(e))

class ValidationWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        # self.setWindowTitle("Improved ARIMA - Forward Validation Tool") # Managed by Parent
        # self.resize(1200, 800)
        
        self.df_full = None
        self.current_drug_info = None
        
        # UI Setup
        # central_widget = QWidget() # Removed for QWidget inheritance
        # self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(self) # Directly on self
        
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # --- Left Panel: Controls ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        # left_panel.setFixedWidth(300) # Remove fixed width to allow resizing
        
        # 1. Data Loading
        grp_data = QGroupBox("Data Source")
        layout_data = QVBoxLayout()
        self.btn_load = QPushButton("Load Dataset (CSV)")
        self.btn_load.clicked.connect(self.load_file)
        self.lbl_file = QLabel("No file loaded")
        layout_data.addWidget(self.btn_load)
        layout_data.addWidget(self.lbl_file)
        grp_data.setLayout(layout_data)
        
        # 2. Selection
        grp_select = QGroupBox("Target Selection")
        layout_select = QVBoxLayout()
        self.combo_clinic = QComboBox()
        self.combo_drug = QComboBox()
        layout_select.addWidget(QLabel("Select Clinic:"))
        layout_select.addWidget(self.combo_clinic)
        layout_select.addWidget(QLabel("Select Drug:"))
        layout_select.addWidget(self.combo_drug)
        grp_select.setLayout(layout_select)
        
        # 3. Model Parameters (Tunable)
        grp_params = QGroupBox("Model Configuration")
        layout_params = QVBoxLayout()
        
        # Validity Days Override
        layout_params.addWidget(QLabel("Remaining Shelf Life (Days):"))
        self.spin_validity = QDoubleSpinBox()
        self.spin_validity.setRange(0, 730)
        self.spin_validity.setValue(365)
        layout_params.addWidget(self.spin_validity)
        
        # Fluctuation Class Override (Force Model Mode)
        layout_params.addWidget(QLabel("Force Volatility Class:"))
        self.combo_class = QComboBox()
        self.combo_class.addItems(["Auto (Default)", C.FLUC_LOW, C.FLUC_MED, C.FLUC_HIGH])
        layout_params.addWidget(self.combo_class)
        
        grp_params.setLayout(layout_params)
        
        # Run Button
        self.btn_run = QPushButton("Run Validation")
        self.btn_run.setStyleSheet("background-color: #3498db; color: white; font-weight: bold; padding: 8px;")
        self.btn_run.clicked.connect(self.run_model)
        
        left_layout.addWidget(grp_data)
        left_layout.addWidget(grp_select)
        left_layout.addWidget(grp_params)
        left_layout.addWidget(self.btn_run)
        
        # Result Table (as requested for Variance/Fit stats)
        self.metrics_table = MetricsTableWidget()
        left_layout.addWidget(QLabel("Validation Results:"))
        left_layout.addWidget(self.metrics_table)
        
        left_layout.addStretch()
        
        # --- Right Panel: Visualization ---
        right_panel = QSplitter(Qt.Vertical)
        
        self.plot_widget = PlotWidget()
        right_panel.addWidget(self.plot_widget)
        
        # Logs/Output
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        # self.log_output.setMaximumHeight(150) # Allow resizing
        right_panel.addWidget(self.log_output)
        
        # Set stretch: Plot takes most space
        right_panel.setStretchFactor(0, 4)
        right_panel.setStretchFactor(1, 1)

        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 7)
        
        self.log("System Ready. Please load a dataset.")

    def log(self, msg):
        self.log_output.append(msg)

    def load_file(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Open Dataset", "", "CSV Files (*.csv)")
        if fname:
            try:
                self.df_full = pd.read_csv(fname)
                self.lbl_file.setText(Path(fname).name)
                self.log(f"Loaded {len(self.df_full)} records from {fname}")
                
                # Populate Clinics
                clinics = sorted(self.df_full['所属诊所'].unique().tolist())
                self.combo_clinic.clear()
                self.combo_clinic.addItems(clinics)
                
                # Populate Drugs (Triggered by clinic change usually, but for now unique all)
                self.update_drug_list()
                self.combo_clinic.currentTextChanged.connect(self.update_drug_list)
                
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))

    def update_drug_list(self):
        if self.df_full is None: return
        clinic = self.combo_clinic.currentText()
        drugs = sorted(self.df_full[self.df_full['所属诊所'] == clinic]['药品编码'].unique().astype(str).tolist())
        self.combo_drug.clear()
        self.combo_drug.addItems(drugs)
        self.log(f"Found {len(drugs)} drugs for {clinic}")

    def run_model(self):
        if self.df_full is None:
            QMessageBox.warning(self, "Warning", "No data loaded.")
            return

        clinic = self.combo_clinic.currentText()
        drug = self.combo_drug.currentText()
        
        # Filter Data
        # Ensure type match for filtering
        df_filtered = self.df_full[
            (self.df_full['所属诊所'] == clinic) & 
            (self.df_full['药品编码'].astype(str) == drug)
        ].copy()
        
        if df_filtered.empty:
            self.log("No data found for selection.")
            return
            
        # Prepare Params
        params = {
            'validity_days': self.spin_validity.value(),
            'fluctuation_class': None if self.combo_class.currentText() == "Auto (Default)" else self.combo_class.currentText()
        }
        
        # Mock Drug Info (In real app, join with drug_info.xls)
        # Here we construct a series for the model init
        drug_info = pd.Series({
            '波动区间分类': C.FLUC_MED, # Default, will be overridden by params if set
            '效期（月）': 12,
            'CV': 0.5 
        })
        
        self.worker = ModelWorker(df_filtered, drug_info, params)
        self.worker.finished.connect(self.on_model_finished)
        self.worker.error.connect(lambda e: self.log(f"Error: {e}"))
        
        self.log(f"Starting Validation for {drug}...")
        self.worker.start()
        self.btn_run.setEnabled(False)

    def on_model_finished(self, dates, actuals, forecast, metrics):
        self.btn_run.setEnabled(True)
        self.log(f"Validation Complete using Improved ARIMA.")
        self.log(f"Metrics: {metrics}")
        
        # New: Update Metrics Table
        self.metrics_table.update_metrics(metrics)
        
        self.plot_widget.plot(
            dates, actuals, forecast, 
            title=f"Forecast Validation: {self.combo_drug.currentText()}",
            metrics=metrics
        )
