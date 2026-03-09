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
     QDateEdit
)
from PySide6.QtGui import QColor, QFont
from PySide6.QtCore import Qt, QThread, Signal, QDate
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

# Matplotlib Integration
try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
except ImportError:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
    
from matplotlib.figure import Figure
import matplotlib.dates as mdates

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
    
    def __init__(self, config: SimulationConfig, drug_info: dict, external_data: pd.DataFrame, duration_days: int, split_date: str):
        super().__init__()
        self.config = config
        self.drug_info = drug_info
        self.external_data = external_data
        self.duration_days = duration_days
        self.split_date = split_date
        
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
                split_date=self.split_date
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
                dates = pd.date_range(start='2023-01-01', end='2025-12-31')
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
        self.viz_tabs.addTab(self.plot_sales, "Sales Analysis") # Same sales
        
        viz_splitter.addWidget(self.viz_tabs)
        
        # Logs
        log_group = QGroupBox("Simulation Log")
        log_layout = QVBoxLayout()
        self.log_console = QTextEdit()
        self.log_console.setReadOnly(True)
        log_layout.addWidget(self.log_console)
        log_group.setLayout(log_layout)
        
        viz_splitter.addWidget(log_group)
        
        # Set initial stretch factors
        viz_splitter.setStretchFactor(0, 1)
        viz_splitter.setStretchFactor(1, 10)
        viz_splitter.setStretchFactor(2, 2)
        
        viz_layout.addWidget(viz_splitter)
        
        splitter.addWidget(viz_panel)
        
        # Set initial stretch factors
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 7)

    def _reset_params(self):
        self.spin_initial_stock.setValue(14)
        self.spin_replenish.setValue(30)
        self.combo_service_level.setCurrentIndex(1)
        self.spin_flu_sens.setValue(1.0)
        self.spin_temp_sens.setValue(1.0)
        self.spin_rain_sens.setValue(0.0)
        self.date_split.setDate(QDate(2025, 9, 1))
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
        duration = 365 + 366 
        
        config = SimulationConfig(
            start_date=pd.Timestamp('2023-01-01'),
            end_date=pd.Timestamp('2024-12-31'),
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

        drug_info['有效期'] = config.validity_days
        drug_info['补货提前期'] = 3 # Fixed for now or add control if needed, existing widget has it
        drug_info['药品ID'] = str(row.get('药品编号', f'DRUG_{idx}'))
        drug_info['药品名称'] = str(row.get('药品名称', 'Unknown'))
        drug_info['单价'] = float(row.get('零售价', 35.0))
        drug_info['药品品类'] = str(row.get('药品品类', 'Misc'))
        drug_info['波动区间分类'] = str(row.get('波动区间分类', '中波动'))
        
        self.btn_run.setEnabled(False)
        self.btn_run.setText("Running...")
        self.log_console.append(f"Starting Evolution Simulation for {drug_info['药品名称']}...")
        self.log_console.append(f"Split Date: {split_date_str}")
        
        self.worker = EvolutionWorker(config, drug_info, self.ext_df, duration, split_date_str)
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
                 dates = pd.to_datetime(df[col_date])
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
            ax1.axvline(x=split_date_ts, color='red', linestyle='--', linewidth=1.5, label='Switch Date')
            
            ax1.set_title('Evolution Simulation: Inventory Levels')
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
            ax_inv.axvline(x=split_date_ts, color='red', linestyle='--', linewidth=2, label='Switch Date')

            ax_inv.set_title('Detailed Inventory: Before vs After Switch')
            ax_inv.set_ylabel('Stock Quantity')
            ax_inv.legend()
            ax_inv.grid(True, alpha=0.3)
            try:
                ax_inv.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                fig_inv.autofmt_xdate()
            except: pass
            fig_inv.tight_layout()
            self.plot_inventory.canvas.draw()
            
            # --- KPI Table ---
            # Calculate metrics
            self.kpi_table.setRowCount(0)
            
            # 1. Total Sales
            total_sales_base = sales_base.sum()
            total_sales_opt = sales_opt.sum()
            self._add_kpi_row("Total Sales (Units)", f"{total_sales_base:,.0f}", f"{total_sales_opt:,.0f}", total_sales_opt - total_sales_base)
            
            # 2. Total Stockouts
            total_out_base = stockout_base_flag.sum()
            total_out_opt = stockout_opt_flag.sum()
            self._add_kpi_row("Stockout Days", f"{total_out_base}", f"{total_out_opt}", total_out_base - total_out_opt, inverse=True)
            
            # 3. Avg Inventory
            avg_inv_base = stock_base.mean()
            avg_inv_opt = stock_opt.mean()
            self._add_kpi_row("Avg Inventory", f"{avg_inv_base:.1f}", f"{avg_inv_opt:.1f}", avg_inv_base - avg_inv_opt, inverse=True)
            
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
