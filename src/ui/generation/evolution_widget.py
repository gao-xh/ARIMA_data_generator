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
     QSplitter, QSizePolicy, QFormLayout, QTabWidget,
     QTableWidget, QTableWidgetItem, QHeaderView, QScrollArea,
     QDateEdit, QCheckBox
)
from PySide6.QtGui import QColor, QFont
from PySide6.QtCore import Qt, QThread, Signal, QDate
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score

# Matplotlib Integration
try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
except ImportError:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
    
from matplotlib.figure import Figure
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

# FIX: Set Font for Chinese Characters Support in Matplotlib
# SimHei is standard for Windows, ensure it falls back gracefully
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False # Fix minus sign display

# Core Logic
from src.core.tools.simulation_tuner import SimulationTuner
from src.core.simulation_config import SimulationConfig
from src.config import DRUG_INFO, EXTERNAL_FACTORS_FILE
from src.core import constants as C # Import constants
from src.ui.common.widgets import PlotWidget

class EvolutionWorker(QThread):
    """
    Background worker to run evolution (two-stage) simulation without freezing UI.
    """
    finished = Signal(pd.DataFrame)
    error = Signal(str)
    
    def __init__(self, config: SimulationConfig, drug_info: dict, external_data: pd.DataFrame, duration_days: int, split_date: str, seed: int = None):
        super().__init__()
        self.config = config
        self.drug_info = drug_info
        self.external_data = external_data
        self.duration_days = duration_days
        self.split_date = split_date
        self.seed = seed
        
    def run(self):
        try:
            # Initialize Tuner
            tuner = SimulationTuner(
                self.config, 
                self.drug_info, 
                self.external_data
            )
            # Run in Evolution Mode
            df = tuner.run_simulation_only(
                total_days=self.duration_days,
                evolution_mode=True,
                split_date=self.split_date,
                seed_value=self.seed
            )
            self.finished.emit(df)
            
        except Exception as e:
            err_msg = f"Evolution Simulation Failed: {str(e)}\n{traceback.format_exc()}"
            self.error.emit(err_msg)

class EvolutionWidget(QWidget):
    """
    Two-Stage Evolution Simulation UI.
    Stage 1: Manual Strategy (Baseline)
    Stage 2: AI Strategy (Optimized) starting from Split Date.
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
                    if date_col != C.COL_DATE:
                        self.ext_df = self.ext_df.rename(columns={date_col: C.COL_DATE})
                    
                    self.ext_df[C.COL_DATE] = pd.to_datetime(self.ext_df[C.COL_DATE])
                    self.ext_df = self.ext_df.set_index(C.COL_DATE, drop=False)
            else:
                # Mock External Data if missing
                dates = pd.date_range(start='2024-01-01', end='2025-12-31')
                self.ext_df = pd.DataFrame({
                    C.COL_DATE: dates, 
                    '平均气温': np.random.normal(20, 5, len(dates)),
                    'ILI%': np.random.uniform(0, 0.05, len(dates))
                })
                self.ext_df = self.ext_df.set_index(C.COL_DATE, drop=False)
        except Exception as e:
            print(f"Error loading external data: {e}")

    def _init_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Splitter to allow resizing
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # --- Left Panel: Controls (in ScrollArea) ---
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        control_layout.setSpacing(10)
        control_layout.setContentsMargins(10, 10, 10, 10)
        
        # 0. Study Context
        context_group = QGroupBox("Evolution Context")
        context_layout = QVBoxLayout()
        
        info_label = QLabel(
            "<b>Concept:</b> Two-Stage Evolution<br>"
            "1. <span style='color:gray'><b>Manual Stage:</b></span> Before Split Date (Historical)<br>"
            "2. <span style='color:blue'><b>AI Stage:</b></span> After Split Date (Optimized)<br>"
        )
        info_label.setStyleSheet("color: #333; font-size: 11px;")
        context_layout.addWidget(info_label)
        context_group.setLayout(context_layout)
        control_layout.addWidget(context_group)

        # 1. Drug Selection (Same)
        drug_group = QGroupBox("Target Drug")
        drug_layout = QVBoxLayout()
        self.combo_drug = QComboBox()
        self.combo_drug.setMinimumHeight(30)
        self.combo_drug.currentIndexChanged.connect(self._on_drug_selected)
        drug_layout.addWidget(self.combo_drug)
        drug_group.setLayout(drug_layout)
        control_layout.addWidget(drug_group)
        
        # 2. Evolution Settings
        policy_group = QGroupBox("Evolution Strategy")
        policy_layout = QFormLayout()
        policy_layout.setSpacing(8)
        
        # Split Date Control
        self.date_split = QDateEdit()
        self.date_split.setDisplayFormat("yyyy-MM-dd")
        self.date_split.setDate(QDate(2025, 9, 1)) # Default 2025-09-01
        self.date_split.setCalendarPopup(True)

        self.spin_initial_stock = QSpinBox()
        self.spin_initial_stock.setRange(0, 9999)
        self.spin_initial_stock.setValue(14)
        self.spin_initial_stock.setSuffix(" Days")

        self.spin_replenish = QSpinBox()
        self.spin_replenish.setRange(1, 365)
        self.spin_replenish.setValue(30)
        self.spin_replenish.setSuffix(" Days")
        
        # For optimized stage
        self.combo_service_level = QComboBox()
        self.combo_service_level.addItems(["95% (Low Vol)", "98% (Med Vol)", "99% (High Vol)", "Custom"])
        self.combo_service_level.currentIndexChanged.connect(self._on_service_level_changed)
        
        self.spin_safety = QDoubleSpinBox()
        self.spin_safety.setRange(0.1, 10.0)
        self.spin_safety.setSingleStep(0.1)
        self.spin_safety.setValue(1.96)
        self.spin_safety.setEnabled(False)

        policy_layout.addRow("<b>Split Date:</b>", self.date_split)
        policy_layout.addRow("Initial Stock:", self.spin_initial_stock)
        policy_layout.addRow("Review Period (R):", self.spin_replenish)
        policy_layout.addRow("Target Service (AI):", self.combo_service_level)
        policy_layout.addRow("Safety Factor (Z):", self.spin_safety)
        policy_group.setLayout(policy_layout)
        
        control_layout.addWidget(policy_group)

        # 3. Environment Factors
        env_group = QGroupBox("Environment Factors")
        env_layout = QFormLayout()
        env_layout.setSpacing(8)

        self.spin_flu_sens = QDoubleSpinBox()
        self.spin_flu_sens.setRange(0.0, 10.0)
        self.spin_flu_sens.setSingleStep(0.1)
        self.spin_flu_sens.setValue(1.2)

        self.spin_temp_sens = QDoubleSpinBox()
        self.spin_temp_sens.setRange(0.0, 10.0)
        self.spin_temp_sens.setSingleStep(0.1) 
        self.spin_temp_sens.setValue(1.0)
        
        self.spin_rain_sens = QDoubleSpinBox()
        self.spin_rain_sens.setRange(0.0, 10.0)
        self.spin_rain_sens.setSingleStep(0.1)
        self.spin_rain_sens.setValue(0.0)

        env_layout.addRow("Flu Sensitivity:", self.spin_flu_sens)
        env_layout.addRow("Temp Sensitivity:", self.spin_temp_sens)
        env_layout.addRow("Rain Sensitivity:", self.spin_rain_sens)
        
        # Random Seed Control
        seed_layout = QHBoxLayout()
        self.chk_random_seed = QCheckBox("Random Seed")
        self.chk_random_seed.setChecked(True)
        self.chk_random_seed.stateChanged.connect(self._on_random_chk_changed)
        
        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(0, 999999)
        self.spin_seed.setValue(42)
        self.spin_seed.setEnabled(False) # Default disabled because random is checked
        self.spin_seed.setToolTip("Fixed seed for reproducibility")
        
        seed_layout.addWidget(self.chk_random_seed)
        seed_layout.addWidget(self.spin_seed)
        env_layout.addRow("Seed Control:", seed_layout)

        env_group.setLayout(env_layout)

        control_layout.addWidget(env_group)
        
        # 5. Actions
        action_layout = QHBoxLayout()
        self.btn_reset = QPushButton("Reset")
        self.btn_reset.clicked.connect(self._reset_params)
        
        self.btn_run = QPushButton("Run Evolution")
        self.btn_run.setStyleSheet("background-color: #28a745; color: white; font-weight: bold; padding: 10px; border-radius: 4px;")
        self.btn_run.setCursor(Qt.PointingHandCursor)
        self.btn_run.clicked.connect(self.start_simulation)
        
        action_layout.addWidget(self.btn_reset)
        action_layout.addWidget(self.btn_run)
        control_layout.addLayout(action_layout)
        
        control_layout.addStretch()
        
        # Set Control Panel Widget to Scroll Area
        scroll_area.setWidget(control_panel)
        scroll_area.setMinimumWidth(340) 

        splitter.addWidget(scroll_area)
        
        # --- Right Panel: Visualization ---
        viz_panel = QWidget()
        viz_layout = QVBoxLayout(viz_panel)
        viz_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create Vertical Splitter
        viz_splitter = QSplitter(Qt.Vertical)
        
        # KPI Table
        self.kpi_table = QTableWidget()
        self.kpi_table.setColumnCount(4)
        self.kpi_table.setHorizontalHeaderLabels(["Metric", "Manual Only", "Evolution (Manual->AI)", "Impact"])
        self.kpi_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.kpi_table.verticalHeader().setVisible(False)
        self.kpi_table.setAlternatingRowColors(True)
        self.kpi_table.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.kpi_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.kpi_table.setMinimumHeight(100)
        
        viz_splitter.addWidget(self.kpi_table)
        
        # Charts Area (Tabbed)
        self.viz_tabs = QTabWidget()
        self.viz_tabs.setMinimumHeight(300)
        
        # Tab 1: Overview
        self.plot_overview = PlotWidget()
        self.viz_tabs.addTab(self.plot_overview, "Evolution Overview")
        
        # Tab 2: Inventory Details
        self.plot_inventory = PlotWidget()
        self.viz_tabs.addTab(self.plot_inventory, "Inventory Details")
        
        # Tab 3: Sales Analysis
        self.plot_sales = PlotWidget()
        self.viz_tabs.addTab(self.plot_sales, "Sales Analysis (Full)")
        
        # Tab 4: 2024 Zoom (Baseline)
        self.plot_sales_2024 = PlotWidget()
        self.viz_tabs.addTab(self.plot_sales_2024, "Sales Analysis (2024)")
        
        # Tab 5: ARIMAX Fit Overlay
        self.plot_arimax = PlotWidget()
        self.viz_tabs.addTab(self.plot_arimax, "Model Fit Overlay")
        
        # Tab 6: Method Comparison (Enhanced vs Traditional) - NEW
        self.plot_comparison = PlotWidget()
        self.viz_tabs.addTab(self.plot_comparison, "Method Comparison")
        
        viz_splitter.addWidget(self.viz_tabs)
        
        # Logs
        log_group = QGroupBox("Simulation Log")
        log_layout = QVBoxLayout()
        self.log_console = QTextEdit()
        self.log_console.setReadOnly(True)
        log_layout.addWidget(self.log_console)
        log_group.setLayout(log_layout)
        
        viz_splitter.addWidget(log_group)
        
        # Set initial stretch factors for Viz Splitter
        viz_splitter.setStretchFactor(0, 1)
        viz_splitter.setStretchFactor(1, 10)
        viz_splitter.setStretchFactor(2, 2)
        
        viz_layout.addWidget(viz_splitter)
        
        splitter.addWidget(viz_panel)
        
        # Set initial stretch factors
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 7)

    def _on_random_chk_changed(self, state):
        self.spin_seed.setEnabled(state != 2) # If checked (2), disabled. If unchecked (0), enabled.

    def _reset_params(self):
        self.spin_initial_stock.setValue(14)
        self.spin_replenish.setValue(30)
        self.combo_service_level.setCurrentIndex(1)
        self.spin_flu_sens.setValue(1.0)
        self.spin_temp_sens.setValue(1.0)
        self.spin_rain_sens.setValue(0.0)
        self.date_split.setDate(QDate(2025, 9, 1))
        self.chk_random_seed.setChecked(True)
        self.log_console.append("Parameters reset.")

    def log(self, msg):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_console.append(f"[{timestamp}] {msg}")

    def load_drugs_list(self):
        try:
            items = []
            if self.ext_df is None: self._init_data()

            if Path(DRUG_INFO).exists():
                try:
                    self.drug_df = pd.read_csv(DRUG_INFO, encoding='utf-8')
                except UnicodeDecodeError:
                    self.drug_df = pd.read_csv(DRUG_INFO, encoding='gb18030')
                
                self.combo_drug.clear()
                
                for idx, row in self.drug_df.iterrows():
                    name = str(row.get('药品名称', 'Unknown'))
                    vol_raw = str(row.get('波动区间分类', '中波动'))
                    if '低' in vol_raw: vol_cat = 'Low'
                    elif '高' in vol_raw: vol_cat = 'High'
                    else: vol_cat = 'Medium'
                    items.append(f"{name} | {vol_cat}")
                
                self.combo_drug.addItems(items)
                self._on_drug_selected(0)
            else:
                self.log_console.append("Drug Info file not found.")
        except Exception as e:
            self.log_console.append(f"Error loading drugs: {e}")
            print(traceback.format_exc())

    def _on_drug_selected(self, index):
        if self.drug_df is None or index < 0 or index >= len(self.drug_df):
            return
            
        row = self.drug_df.iloc[index]
        vol_raw = str(row.get('波动区间分类', '中波动'))
        if '高' in vol_raw: self.spin_safety.setValue(2.33)
        elif '低' in vol_raw: self.spin_safety.setValue(1.65)
        else: self.spin_safety.setValue(1.96)

        cat_str = str(row.get('药品品类', '')).upper()
        name_str = str(row.get('药品名称', '')).upper()
        combined = cat_str + " " + name_str
        
        if any(x in combined for x in ['感冒', '流感', '病毒', '清热', '解热']):
            self.spin_flu_sens.setValue(2.5) 
        elif any(x in combined for x in ['呼吸', '咳', '肺', '炎', '头孢', '抗生素']):
            self.spin_flu_sens.setValue(1.5)
        elif any(x in combined for x in ['慢病', '心脑', '血压', '糖', '脂']):
            self.spin_flu_sens.setValue(0.0)
        else:
            self.spin_flu_sens.setValue(0.5)

    def _on_service_level_changed(self, index):
        self.spin_safety.setEnabled(False)
        if index == 0: self.spin_safety.setValue(1.65)
        elif index == 1: self.spin_safety.setValue(1.96)
        elif index == 2: self.spin_safety.setValue(2.33)
        else: self.spin_safety.setEnabled(True)

    def start_simulation(self):
        idx = self.combo_drug.currentIndex()
        if self.drug_df is None:
            self.log_console.append("No drug data loaded.")
            self.load_drugs_list()
            if self.drug_df is None: return

        if idx < 0: idx = 0
        duration = 366 + 365 
        
        config = SimulationConfig(
            start_date=pd.Timestamp('2024-01-01'),
            end_date=pd.Timestamp('2025-12-31'),
            replenishment_days=int(self.spin_replenish.value()),
            active_clinic_scale=1.0
        )
        
        config.safety_stock_factor = self.spin_safety.value()
        config.flu_sensitivity = self.spin_flu_sens.value()
        config.temp_sensitivity = self.spin_temp_sens.value()
        config.rain_sensitivity = self.spin_rain_sens.value()
        config.initial_stock_days = int(self.spin_initial_stock.value())
        config.random_noise_sigma = 0.2
        
        split_date_str = self.date_split.date().toString("yyyy-MM-dd")
        
        row = self.drug_df.iloc[idx]
        drug_info = row.to_dict()
        try:
             v_months = float(row.get('效期（月）', 12))
             config.validity_days = int(v_months * 30)
        except:
             config.validity_days = 365

        # Store for dashboard use
        self.current_drug_info = drug_info
        
        drug_info['有效期'] = config.validity_days
        drug_info['补货提前期'] = 3 # Fixed for now or add control if needed, existing widget has it
        drug_info['药品ID'] = str(row.get('药品编号', f'DRUG_{idx}'))
        drug_info['药品名称'] = str(row.get('药品名称', 'Unknown'))
        drug_info['单价'] = float(row.get('零售价', 35.0))
        drug_info['药品品类'] = str(row.get('药品品类', 'Misc'))
        
        # Determine Seed
        final_seed = None
        if not self.chk_random_seed.isChecked():
            final_seed = self.spin_seed.value()
        else:
            final_seed = np.random.randint(0, 100000)
            self.spin_seed.blockSignals(True)
            self.spin_seed.setValue(final_seed) # Show the random seed used
            self.spin_seed.blockSignals(False)

        self.log_console.append(f"Starting Evolution Simulation for {drug_info['药品名称']}...")
        self.log_console.append(f"Split Date: {split_date_str} | Seed: {final_seed}")
        
        self.worker = EvolutionWorker(config, drug_info, self.ext_df, duration, split_date_str, seed=final_seed)
        self.worker.finished.connect(self.on_simulation_finished)
        self.worker.error.connect(self.on_simulation_error)
        self.worker.start()

    def on_simulation_finished(self, df: pd.DataFrame):
        self.btn_run.setEnabled(True)
        self.btn_run.setText("Run Evolution")
        self.log_console.append(f"Data generated. Rows: {len(df)}")
        self.update_dashboard(df)

    def on_simulation_error(self, msg):
        self.btn_run.setEnabled(True)
        self.btn_run.setText("Run Evolution")
        self.log_console.append(f"Error: {msg}")

    def update_dashboard(self, df: pd.DataFrame):
        try:
            if '日期' in df.columns or 'date' in df.columns:
                 col_date = '日期' if '日期' in df.columns else 'date'
                 # Ensure datetime
                 dates_series = pd.to_datetime(df[col_date])
                 # FIX: Set index to DatetimeIndex and update 'dates' variable to be the index
                 df = df.set_index(dates_series)
                 dates = df.index 
            else:
                 return

            # Baseline_Inventory = Pure Manual
            # Optimized_Inventory = Evolution (Manual -> AI)
            stock_base = df.get('Baseline_Inventory', pd.Series(0, index=df.index))
            stock_opt = df.get('Optimized_Inventory', pd.Series(0, index=df.index))
            sales_base = df.get('Baseline_Sales', pd.Series(0, index=df.index))
            sales_opt = df.get('Optimized_Sales', pd.Series(0, index=df.index))
            
            stockout_base_flag = df.get('Baseline_Stockout_Flag', pd.Series(0, index=df.index)) > 0
            stockout_opt_flag = df.get('Optimized_Stockout_Flag', pd.Series(0, index=df.index)) > 0
            
            # Split Date Line
            split_date_ts = pd.Timestamp(self.date_split.date().toPython())

            # --- 1. Overview Tab (3 Subplots) ---
            fig = self.plot_overview.canvas.fig
            fig.clear()
            ax1 = fig.add_subplot(211) # Reduced to 2 plots for clarity
            ax2 = fig.add_subplot(212, sharex=ax1) 
            
            # Inventory
            ax1.plot(dates, stock_base, label='Manual Only (Baseline)', color='gray', alpha=0.5, linestyle='--')
            ax1.plot(dates, stock_opt, label='Evolution (Manual -> AI)', color='blue', linewidth=1.5)
            
            # Vertical Split Line
            ax1.axvline(x=split_date_ts, color='red', linestyle='--', linewidth=1.5, label='Test Period Start')
            
            drug_name = self.current_drug_info.get('药品名称', 'Target Drug')
            ax1.set_title(f'Evolution Simulation: Inventory Levels - {drug_name}')
            ax1.legend(loc='upper right', fontsize='x-small')
            ax1.grid(True, alpha=0.3)
            
            # Cumulative Stockout
            cum_base = stockout_base_flag.astype(int).cumsum()
            cum_opt = stockout_opt_flag.astype(int).cumsum()
            ax2.plot(dates, cum_base, label='Manual Cum Stockouts', color='red', alpha=0.6)
            ax2.plot(dates, cum_opt, label='Evolution Cum Stockouts', color='green', linewidth=2)
            ax2.axvline(x=split_date_ts, color='red', linestyle='--', linewidth=1)
            
            ax2.set_title('Cumulative Stockout Impact')
            ax2.legend(loc='upper left', fontsize='x-small')
            ax2.grid(True, alpha=0.3)
            
            try:
                ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
                fig.autofmt_xdate()
            except: pass
            fig.tight_layout()
            self.plot_overview.canvas.draw()
            
            # --- 2. Inventory Detail Tab ---
            fig_inv = self.plot_inventory.canvas.fig
            fig_inv.clear()
            ax_inv = fig_inv.add_subplot(111)
            ax_inv.plot(dates, stock_base, label='Manual', color='gray', alpha=0.5, linestyle=':')
            ax_inv.plot(dates, stock_opt, label='Evolution Strategy', color='blue', linewidth=2)
            
            # Highlight AI Region
            # ax_inv.axvspan(split_date_ts, dates.max(), color='green', alpha=0.05, label='AI Active Region')
            ax_inv.axvline(x=split_date_ts, color='red', linestyle='--', linewidth=2, label='Test Period Start')

            ax_inv.set_title(f'Detailed Inventory: Before vs After - {self.current_drug_info.get("药品名称")}')
            ax_inv.set_ylabel('Stock Quantity')
            ax_inv.legend()
            ax_inv.grid(True, alpha=0.3)
            try:
                ax_inv.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                fig_inv.autofmt_xdate()
            except: pass
            fig_inv.tight_layout()
            self.plot_inventory.canvas.draw()

            # --- 3. Sales Analysis Tab ---
            fig_sales = self.plot_sales.canvas.fig
            fig_sales.clear()
            
            # Single plot as requested: Sales Line + Trend + Stockout Markers
            ax_s1 = fig_sales.add_subplot(111)
            
            # Use 'Optimized' series as the main view
            demand_opt = df.get('Optimized_Demand', pd.Series(0, index=df.index))
            sales_opt = df.get('Optimized_Sales', pd.Series(0, index=df.index))
            
            # 1. Sales Line
            ax_s1.plot(dates, sales_opt, label='Actual Sales', color='green', alpha=0.4, linewidth=1)
            
            # 2. Trend Curve (7-Day Rolling Mean - De-noised)
            sales_trend = sales_opt.rolling(window=7, min_periods=1, center=True).mean()
            ax_s1.plot(dates, sales_trend, label='Sales Trend (7d Avg)', color='orange', linewidth=2.5)
            
            # 3. Stockout Markers (Red X at y=0)
            # Filter dates where stockout happened
            stockout_mask = stockout_opt_flag.astype(bool)
            stockout_dates = dates[stockout_mask]
            
            if len(stockout_dates) > 0:
                # Plot X markers at y=0 on the stockout dates
                ax_s1.scatter(stockout_dates, [0] * len(stockout_dates), 
                              color='red', marker='x', s=60, label='Stockout Event', zorder=5)

            # Switch Date Line
            ax_s1.axvline(x=split_date_ts, color='blue', linestyle='--', linewidth=1.5, label='Experiment Start (Optimized)')

            ax_s1.set_title(f'Sales Analysis: Trend & Stockouts - {self.current_drug_info.get("药品名称")}')
            ax_s1.legend(loc='upper right', fontsize='x-small')
            ax_s1.grid(True, alpha=0.3)
            
            try:
                ax_s1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                fig_sales.autofmt_xdate()
            except: pass
            
            fig_sales.tight_layout()
            self.plot_sales.canvas.draw()
            
            # --- 4. Sales Analysis (2024 Zoom) Tab ---
            fig_sales_24 = self.plot_sales_2024.canvas.fig
            fig_sales_24.clear()
            
            ax_s24 = fig_sales_24.add_subplot(111)
            
            # Filter Data for 2024
            mask_24 = (dates >= pd.Timestamp('2024-01-01')) & (dates < pd.Timestamp('2025-01-01'))
            dates_24 = dates[mask_24]
            sales_24 = df.loc[mask_24, 'Baseline_Sales'] # Use Baseline for 2024
            
            if not dates_24.empty:
                # Actual Sales
                ax_s24.plot(dates_24, sales_24, label='Baseline Sales (2024)', color='blue', alpha=0.6, linewidth=1)
                
                # Trend
                trend_24 = sales_24.rolling(window=7, min_periods=1, center=True).mean()
                ax_s24.plot(dates_24, trend_24, label='Trend (7d Avg)', color='orange', linewidth=2)
                
                # Stockouts (Baseline)
                stockout_24 = df.loc[mask_24, 'Baseline_Stockout_Flag']
                so_dates_24 = dates_24[stockout_24 > 0]
                if not so_dates_24.empty:
                    ax_s24.scatter(so_dates_24, [0]*len(so_dates_24), color='red', marker='x', s=60, label='Stockout', zorder=5)

                ax_s24.set_title(f'Baseline Performance: 2024 Full Year - {self.current_drug_info.get("药品名称")}')
                ax_s24.legend()
                ax_s24.grid(True, alpha=0.3)
                try:
                    ax_s24.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                    fig_sales_24.autofmt_xdate()
                except: pass
            
            fig_sales_24.tight_layout()
            self.plot_sales_2024.canvas.draw()
            
            # --- 5. ARIMAX Fit Overlay Tab ---
            fig_arimax = self.plot_arimax.canvas.fig
            fig_arimax.clear()
            
            ax_fit = fig_arimax.add_subplot(111)
            
            # Data from SimulationTuner (Pure ARIMAX columns)
            # Need columns: 'Optimized_Demand' (Real), 'Pure_ARIMAX_Fitted', 'Pure_ARIMAX_Forecast'
            
            real_demand = df.get('Optimized_Demand', pd.Series(0, index=df.index))
            fit_vals = df.get('Pure_ARIMAX_Fitted', pd.Series(np.nan, index=df.index))
            cast_vals = df.get('Pure_ARIMAX_Forecast', pd.Series(np.nan, index=df.index))
            
            # --- Visualization Logic: Training & Test Split ---
            # 1. Real Demand (Ground Truth) - Plot Full History
            ax_fit.plot(dates, real_demand, label='Real Demand (Ground Truth)', color='lightgray', alpha=0.6, linewidth=1.5)
            
            # 2. Real Trend (7-Day Moving Avg) - Smoother comparison line
            real_trend = real_demand.rolling(window=7, min_periods=1, center=True).mean()
            ax_fit.plot(dates, real_trend, label='Real Trend (7d Avg)', color='green', alpha=0.5, linewidth=1.0, linestyle='-')

            # 3. Fitted Values (Training Phase) - approx first 20 months
            # Only plot where not NaN (ARIMAX fitted values)
            mask_fit = ~fit_vals.isna()
            
            # Burn-in Truncation: Hide first 30 days to avoid initialization transients (Cold Start)
            if mask_fit.any():
                # Find indices where fit exists
                fit_indices = np.where(mask_fit)[0]
                if len(fit_indices) > 30:
                    # Set mask to False for the first 30 valid points
                    mask_fit.iloc[fit_indices[:30]] = False

            if mask_fit.any():
                ax_fit.plot(dates[mask_fit], fit_vals[mask_fit], label='ARIMAX Fit (Training)', color='blue', linewidth=1.5, linestyle='-')
                
            # 4. Forecast Values (Test Phase) - approx last 4 months
            # Only plot where not NaN (ARIMAX forecast values)
            mask_cast = ~cast_vals.isna()
            # Filter out 0s if they were filled (just in case)
            mask_cast = mask_cast & (cast_vals != 0)
            
            if mask_cast.any():
                ax_fit.plot(dates[mask_cast], cast_vals[mask_cast], label='ARIMAX Forecast (Test)', color='red', linewidth=2.0, linestyle='--')
            
            # 5. Split Line (Training / Test Boundary)
            ax_fit.axvline(x=split_date_ts, color='black', linestyle=':', label='Train/Test Split', linewidth=1.0)
            
            ax_fit.set_title(f'Model Validation: Training Fit (20mo) vs Test Forecast (4mo) - {self.current_drug_info.get("药品名称")}')
            ax_fit.legend(loc='upper left')
            ax_fit.grid(True, alpha=0.3)
            try:
                ax_fit.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                fig_arimax.autofmt_xdate()
            except: pass
            
            fig_arimax.tight_layout()
            self.plot_arimax.canvas.draw()
            
            # --- 6. Method Comparison Tab (New) ---
            # Compare Enhanced ARIMAX vs Traditional ARIMA vs Real Demand
            fig_comp = self.plot_comparison.canvas.fig
            fig_comp.clear()
            ax_comp = fig_comp.add_subplot(111)
            
            
            # --- Re-define Training Slice Variables for Comparison ---
            mask_train = dates <= split_date_ts
            dates_train = dates[mask_train]
            demand_train = real_demand[mask_train]
            fit_train = fit_vals[mask_train]

            # Get data for comparison
            # Need columns: 'Optimized_Demand', 'Pure_ARIMAX_Fitted', 'Traditional_ARIMA_Fitted'
            trad_fit = df.get('Traditional_ARIMA_Fitted', pd.Series(np.nan, index=df.index))
            # Just focus on the TRAINING part for comparison of fit quality
            # Or the whole range? Let's show whole range of fit to see reaction.
            
            # 1. Real Demand (Background)
            ax_comp.plot(dates_train, demand_train, label='Real Demand', color='lightgray', alpha=0.5, linewidth=1.0)
            
            # 2. Real Trend (7d Avg)
            # Slice trend to match training dates
            real_trend_train = real_trend[mask_train]
            ax_comp.plot(dates_train, real_trend_train, label='Real Trend (7d Avg)', color='green', alpha=0.6, linewidth=1.5, linestyle='-')
            
            # 3. Enhanced ARIMAX Fit (Blue)
            # Use mask_fit which already has burn-in removed, but slice it for training set
            mask_fit_train_slice = mask_fit[mask_train]
            if mask_fit_train_slice.any():
                ax_comp.plot(dates_train[mask_fit_train_slice], fit_train[mask_fit_train_slice], label='Enhanced ARIMAX (Proposed)', color='blue', linewidth=1.5)
                
            # 4. Traditional ARIMA Fit (Orange/Dashey)
            # Filter for Traditional Fit on Training set
            trad_train = trad_fit[mask_train]
            mask_trad = ~trad_train.isna()
            
            # Burn-in for Traditional too
            if mask_trad.any():
                idx_trad = np.where(mask_trad)[0]
                if len(idx_trad) > 30:
                     mask_trad.iloc[idx_trad[:30]] = False
                     
            if mask_trad.any():
                ax_comp.plot(dates_train[mask_trad], trad_train[mask_trad], label='Traditional ARIMA (Baseline)', color='orange', linewidth=1.5, linestyle='--')

            ax_comp.set_title(f'Method Comparison: Enhanced ARIMAX vs Traditional - {self.current_drug_info.get("药品名称")}')
            ax_comp.legend(loc='upper right')
            ax_comp.grid(True, alpha=0.3)
            
            try:
                ax_comp.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                fig_comp.autofmt_xdate()
            except: pass
            
            fig_comp.tight_layout()
            self.plot_comparison.canvas.draw()
            
            # --- KPI Table ---
            # Updated to Compare 2024 Q4 (Baseline) vs 2025 Q4 (Optimized)
            self.kpi_table.setRowCount(0)
            self.kpi_table.setHorizontalHeaderLabels([
                "Metric", 
                "2024 Q4 (Baseline)", 
                "2025 Q4 (Optimized)", 
                "Change"
            ])
            
            # Helper to filter dates
            def get_period_stats(df_full, start_date, end_date, prefix='Baseline'):
                mask = (df_full.index >= start_date) & (df_full.index <= end_date)
                sub_df = df_full.loc[mask]
                
                if sub_df.empty: return None
                
                # Extract Columns
                sales = sub_df.get(f'{prefix}_Sales', pd.Series(0, index=sub_df.index))
                inv = sub_df.get(f'{prefix}_Inventory', pd.Series(0, index=sub_df.index))
                loss = sub_df.get(f'{prefix}_Loss', pd.Series(0, index=sub_df.index))
                stockout_flag = sub_df.get(f'{prefix}_Stockout_Flag', pd.Series(0, index=sub_df.index))
                
                # 1. Total Sales
                total_sales = sales.sum()
                
                # 2. Loss Rate
                total_loss = loss.sum()
                loss_rate = (total_loss / (total_sales + total_loss + 1e-6)) * 100
                
                # 3. Stockout Rate
                stockout_days = (stockout_flag > 0).sum()
                stockout_rate = (stockout_days / len(sub_df)) * 100
                
                # 4. Turnover Days (Avg Inv / Avg Daily Sales)
                avg_inv = inv.mean()
                avg_sales = sales.mean()
                turnover = avg_inv / (avg_sales + 1e-6)
                
                # 5. Funds Occupied
                unit_price = self.current_drug_info.get('单价', 35.0)
                funds = avg_inv * unit_price
                
                # 6. Backlog Rate (Here using Avg Inventory Level as proxy for backlog/overstock pressure)
                # Or defined as: Inventory / (Daily Demand * Validity)? Let's use Normalized Inventory
                backlog_rate = avg_inv # Display as raw quantity for now
                
                # 7. MAPE (If available)
                mape = 0.0
                r2 = -999.0
                rmse = -1.0
                
                if f'{prefix}_Forecast' in sub_df.columns:
                    # Ensure we use Demand as Ground Truth, not Sales
                    y_true = sub_df[f'{prefix}_Demand'].values
                    y_pred = sub_df[f'{prefix}_Forecast'].values
                    
                    # Filter for non-zero demand to avoid div-by-zero in MAPE, 
                    # but R2/RMSE should use all points ideally. 
                    # For MAPE:
                    mask_nz = y_true > 0.1
                    if mask_nz.any():
                        mape = np.mean(np.abs((y_true[mask_nz] - y_pred[mask_nz]) / y_true[mask_nz])) * 100
                        
                    # For R2/RMSE (Use all valid points)
                    try:
                        # R2 can be computed even if y_true is constant (will be 0 or neg)
                        # Ensure no NaNs
                        mask_valid = ~np.isnan(y_true) & ~np.isnan(y_pred)
                        if mask_valid.any():
                            r2 = r2_score(y_true[mask_valid], y_pred[mask_valid])
                            rmse = np.sqrt(mean_squared_error(y_true[mask_valid], y_pred[mask_valid]))
                    except Exception:
                        pass
                
                return {
                    'loss_rate': loss_rate,
                    'stockout_rate': stockout_rate,
                    'turnover': turnover,
                    'funds': funds,
                    'backlog': backlog_rate,
                    'mape': mape,
                    'r2': r2,
                    'rmse': rmse
                }
            
            # Define Periods
            # 2024 Sep-Dec (Baseline)
            stats_24 = get_period_stats(df, '2024-09-01', '2024-12-31', 'Baseline')
            # 2025 Sep-Dec (Optimized)
            stats_25 = get_period_stats(df, '2025-09-01', '2025-12-31', 'Optimized')
            
            if stats_24 and stats_25:
                # 1. Loss Rate
                self._add_kpi_row("Loss Rate (损耗率)", 
                                  f"{stats_24['loss_rate']:.2f}%", 
                                  f"{stats_25['loss_rate']:.2f}%", 
                                  stats_25['loss_rate'] - stats_24['loss_rate'], inverse=True)
                
                # 2. Stockout Rate
                self._add_kpi_row("Stockout Rate (缺货率)", 
                                  f"{stats_24['stockout_rate']:.2f}%", 
                                  f"{stats_25['stockout_rate']:.2f}%", 
                                  stats_25['stockout_rate'] - stats_24['stockout_rate'], inverse=True)
                
                # 3. Backlog Rate (积压率 - using Avg Inv)
                # Displaying as Avg Units for clarity, or rename if ratio strictly needed
                self._add_kpi_row("Backlog/Avg Inv (积压)", 
                                  f"{stats_24['backlog']:.1f}", 
                                  f"{stats_25['backlog']:.1f}", 
                                  stats_25['backlog'] - stats_24['backlog'], inverse=True)

                # 4. Turnover Days
                self._add_kpi_row("Turnover Days (周转)", 
                                  f"{stats_24['turnover']:.1f} d", 
                                  f"{stats_25['turnover']:.1f} d", 
                                  stats_25['turnover'] - stats_24['turnover'], inverse=True)
                
                # 5. Funds Occupied
                self._add_kpi_row("Funds Occupied (资金)", 
                                  f"¥{stats_24['funds']:,.0f}", 
                                  f"¥{stats_25['funds']:,.0f}", 
                                  stats_25['funds'] - stats_24['funds'], inverse=True)
                
                # 6. Model MAPE (Only valid if Forecast exists)
                # Lower MAPE is better
                self._add_kpi_row("Model MAPE (预测误差)", 
                                  f"{stats_24['mape']:.1f}%", 
                                  f"{stats_25['mape']:.1f}%", 
                                  stats_25['mape'] - stats_24['mape'], inverse=True)
                
                # 7. Model Fit on Test Data (Prediction vs Actual)
                metrics = getattr(df, 'attrs', {}).get('model_metrics', {})
                
                # Preferred: Use "Pure" Test Metrics if available (calculated out of loop)
                r2_test = metrics.get('r2_test', -999.0)
                rmse_test = metrics.get('rmse_test', -1.0)
                
                # Fallback: Use Simulation DataFrame columns (Optimized Forecast)
                # But stats_25 dictionary already contains R2/RMSE computed from period data
                if r2_test == -999.0:
                    r2_test = stats_25.get('r2', -999.0) 
                if rmse_test == -1.0:
                    rmse_test = stats_25.get('rmse', -1.0)


                self._add_kpi_row("Forecast R² (测试集拟合)",
                                  "N/A", # Baseline doesn't have Forecast
                                  f"{r2_test:.3f}" if r2_test != -999.0 else "N/A",
                                  0, 
                                  inverse=False)
                                  
                self._add_kpi_row("Forecast RMSE (均方根误差)",
                                  "N/A",
                                  f"{rmse_test:.2f}" if rmse_test != -1.0 else "N/A",
                                  0, 
                                  inverse=True)

                # 8. Model Diagnostics (Using stored metrics - Training Fit)
                if metrics:
                    # R2 (Training) - Show in FIRST column (Baseline/Training History)
                    r2_train = metrics.get('r2_train', metrics.get('r2', 0))
                    if r2_train is None: r2_train = 0
                    self._add_kpi_row("Train R² (训练集拟合)", 
                                      f"{r2_train:.3f}", 
                                      "N/A", 
                                      0, 
                                      inverse=False)

                    # AIC - Show in FIRST column
                    aic_val = metrics.get('aic', 0)
                    if aic_val is None: aic_val = 0
                    self._add_kpi_row("Model AIC (信息准则)", 
                                      f"{aic_val:.1f}", 
                                      "N/A", 
                                      0, 
                                      inverse=True) 
                    
                    # Order - Show in FIRST column
                    order = metrics.get('order', None)
                    if order:
                        row = self.kpi_table.rowCount()
                        self.kpi_table.insertRow(row)
                        self.kpi_table.setItem(row, 0, QTableWidgetItem("Best Params (p,d,q)"))
                        self.kpi_table.setItem(row, 1, QTableWidgetItem(str(order))) # Put in Manual Column
                        self.kpi_table.setItem(row, 2, QTableWidgetItem("N/A"))
                        self.kpi_table.setItem(row, 3, QTableWidgetItem(""))

            else:
                self.kpi_table.setRowCount(0)
                item = QTableWidgetItem("Insufficient Data for Q4 Comparison")

                self.kpi_table.setItem(0, 0, item)
            
        except Exception as e:
            self.log_console.append(f"Visualization Error: {e}")
            print(traceback.format_exc())

    def _add_kpi_row(self, metric, val_base, val_opt, diff, inverse=False):
        row = self.kpi_table.rowCount()
        self.kpi_table.insertRow(row)
        
        self.kpi_table.setItem(row, 0, QTableWidgetItem(metric))
        self.kpi_table.setItem(row, 1, QTableWidgetItem(val_base))
        self.kpi_table.setItem(row, 2, QTableWidgetItem(val_opt))
        
        # Improvement Color
        item_diff = QTableWidgetItem(f"{diff:+.1f}")
        is_good = (diff > 0 and not inverse) or (diff < 0 and inverse)
        
        if diff == 0:
            item_diff.setForeground(QColor("black"))
        elif is_good:
            item_diff.setForeground(QColor("green"))
            item_diff.setText(f"{diff:+.1f} (Good)")
        else:
            item_diff.setForeground(QColor("red"))
            
        self.kpi_table.setItem(row, 3, item_diff)
