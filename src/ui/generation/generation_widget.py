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
     QTableWidget, QTableWidgetItem, QHeaderView, QScrollArea
)
from PySide6.QtGui import QColor, QFont
from PySide6.QtCore import Qt, QThread, Signal
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
        context_group = QGroupBox("Research Object")
        context_layout = QVBoxLayout()
        
        info_label = QLabel(
            "<b>Managed Clinics:</b> 7 Total (Abstracted)<br>"
            "<b>Date Range:</b> 2023-01-01 to 2024-12-31<br>"
        )
        info_label.setStyleSheet("color: #555; font-size: 11px;")
        context_layout.addWidget(info_label)
        
        legend_layout = QFormLayout()
        
        lbl_low = QLabel("Low (CV < 0.2):")
        lbl_low.setStyleSheet("color: green; font-weight: bold;")
        self.val_low = QLabel("41 SKUs")
        
        lbl_med = QLabel("Medium (0.2-0.5):")
        lbl_med.setStyleSheet("color: orange; font-weight: bold;")
        self.val_med = QLabel("63 SKUs")
        
        lbl_high = QLabel("High (CV > 0.5):")
        lbl_high.setStyleSheet("color: red; font-weight: bold;")
        self.val_high = QLabel("24 SKUs")
        
        legend_layout.addRow(lbl_low, self.val_low)
        legend_layout.addRow(lbl_med, self.val_med)
        legend_layout.addRow(lbl_high, self.val_high)
        
        context_layout.addLayout(legend_layout)
        context_group.setLayout(context_layout)
        control_layout.addWidget(context_group)

        # 1. Drug Selection
        drug_group = QGroupBox("Target Drug")
        drug_layout = QVBoxLayout()
        self.combo_drug = QComboBox()
        self.combo_drug.setMinimumHeight(30)
        self.combo_drug.currentIndexChanged.connect(self._on_drug_selected)
        drug_layout.addWidget(self.combo_drug)
        drug_group.setLayout(drug_layout)
        control_layout.addWidget(drug_group)
        
        # 2. Inventory Policy
        policy_group = QGroupBox("Inventory Policy")
        policy_layout = QFormLayout()
        policy_layout.setSpacing(8)

        self.spin_initial_stock = QSpinBox()
        self.spin_initial_stock.setRange(0, 9999) # Relaxed limit
        self.spin_initial_stock.setValue(14)
        self.spin_initial_stock.setSuffix(" Days")

        self.spin_replenish = QSpinBox()
        self.spin_replenish.setRange(1, 365) # Relaxed limit
        self.spin_replenish.setValue(30)
        self.spin_replenish.setSuffix(" Days")

        self.spin_lead_time = QSpinBox()
        self.spin_lead_time.setRange(0, 100) # Relaxed limit
        self.spin_lead_time.setValue(3)
        self.spin_lead_time.setSuffix(" Days")
        
        self.combo_service_level = QComboBox()
        self.combo_service_level.addItems(["95% (Low Vol)", "98% (Med Vol)", "99% (High Vol)", "Custom"])
        self.combo_service_level.currentIndexChanged.connect(self._on_service_level_changed)
        
        self.spin_safety = QDoubleSpinBox()
        self.spin_safety.setRange(0.1, 10.0) # Relaxed limit
        self.spin_safety.setSingleStep(0.1)
        self.spin_safety.setValue(1.96)
        self.spin_safety.setEnabled(False)

        policy_layout.addRow("Initial Stock:", self.spin_initial_stock)
        policy_layout.addRow("Review Period (R):", self.spin_replenish)
        policy_layout.addRow("Lead Time (L):", self.spin_lead_time)
        policy_layout.addRow("Target Service Level:", self.combo_service_level)
        policy_layout.addRow("Safety Factor (Z):", self.spin_safety)
        policy_group.setLayout(policy_layout)
        
        control_layout.addWidget(policy_group)

        # 3. Environment Factors
        env_group = QGroupBox("Environment Factors")
        env_layout = QFormLayout()
        env_layout.setSpacing(8)

        self.spin_flu_sens = QDoubleSpinBox()
        self.spin_flu_sens.setRange(0.0, 10.0) # Relaxed limit
        self.spin_flu_sens.setSingleStep(0.1)
        self.spin_flu_sens.setValue(1.2)

        self.spin_temp_sens = QDoubleSpinBox()
        self.spin_temp_sens.setRange(0.0, 10.0) # Relaxed limit
        self.spin_temp_sens.setSingleStep(0.1) 
        self.spin_temp_sens.setValue(1.0)

        self.spin_rain_sens = QDoubleSpinBox()
        self.spin_rain_sens.setRange(0.0, 10.0) # Relaxed limit
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
        
        self.btn_run = QPushButton("Run Simulation")
        self.btn_run.setStyleSheet("background-color: #007ACC; color: white; font-weight: bold; padding: 10px; border-radius: 4px;")
        self.btn_run.setCursor(Qt.PointingHandCursor)
        self.btn_run.clicked.connect(self.start_simulation)
        
        action_layout.addWidget(self.btn_reset)
        action_layout.addWidget(self.btn_run)
        control_layout.addLayout(action_layout)
        
        control_layout.addStretch()
        
        # Set Control Panel Widget to Scroll Area
        scroll_area.setWidget(control_panel)
        # Set minimum width for the scroll area so controls aren't squashed
        scroll_area.setMinimumWidth(340) 

        splitter.addWidget(scroll_area)
        
        # --- Right Panel: Visualization ---
        viz_panel = QWidget()
        viz_layout = QVBoxLayout(viz_panel)
        viz_layout.setContentsMargins(0, 0, 0, 0)
        
        # KPI Table
        self.kpi_table = QTableWidget()
        self.kpi_table.setColumnCount(4)
        self.kpi_table.setHorizontalHeaderLabels(["Metric", "Baseline", "Optimized", "Improvement"])
        self.kpi_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.kpi_table.verticalHeader().setVisible(False)
        self.kpi_table.setAlternatingRowColors(True)
        self.kpi_table.setMaximumHeight(200)
        viz_layout.addWidget(self.kpi_table)
        
        # Charts Area (Tabbed)
        self.viz_tabs = QTabWidget()
        
        # Tab 1: Overview
        self.plot_overview = PlotWidget()
        self.viz_tabs.addTab(self.plot_overview, "Overview")
        
        # Tab 2: Inventory Details
        self.plot_inventory = PlotWidget()
        self.viz_tabs.addTab(self.plot_inventory, "Inventory Details")
        
        # Tab 3: Sales Analysis
        self.plot_sales = PlotWidget()
        self.viz_tabs.addTab(self.plot_sales, "Sales Analysis")
        
        # Tab 4: Loss Trend
        self.plot_loss = PlotWidget()
        self.viz_tabs.addTab(self.plot_loss, "Loss Trend")
        
        viz_layout.addWidget(self.viz_tabs)
        
        # Logs
        log_group = QGroupBox("System Log")
        log_layout = QVBoxLayout()
        self.log_console = QTextEdit()
        self.log_console.setReadOnly(True)
        self.log_console.setMaximumHeight(100)
        log_layout.addWidget(self.log_console)
        log_group.setLayout(log_layout)
        
        # Use splitter for vertical adjustment in right panel too if needed
        # But for now just simple layout
        viz_layout.addWidget(log_group)
        
        splitter.addWidget(viz_panel)
        
        # Set initial stretch factors
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 7)

    def _reset_params(self):
        self.spin_initial_stock.setValue(14)
        self.spin_replenish.setValue(30)
        self.spin_lead_time.setValue(3)
        self.combo_service_level.setCurrentIndex(1)
        self.spin_flu_sens.setValue(1.0)
        self.spin_temp_sens.setValue(1.0)
        self.spin_rain_sens.setValue(0.0)
        self.log_console.append("Parameters reset.")

    def log(self, msg):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_console.append(f"[{timestamp}] {msg}")

    def load_drugs_list(self):
        try:
            volatility_counts = {'LOW': 0, 'MEDIUM': 0, 'HIGH': 0}
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
                    
                    if '低' in vol_raw:
                        vol_cat = 'Low'
                        volatility_counts['LOW'] += 1
                    elif '高' in vol_raw:
                        vol_cat = 'High'
                        volatility_counts['HIGH'] += 1
                    else:
                        vol_cat = 'Medium'
                        volatility_counts['MEDIUM'] += 1
                        
                    items.append(f"{name} | {vol_cat}")
                
                self.combo_drug.addItems(items)
                self.val_low.setText(f"{volatility_counts['LOW']} SKUs")
                self.val_med.setText(f"{volatility_counts['MEDIUM']} SKUs")
                self.val_high.setText(f"{volatility_counts['HIGH']} SKUs")
                
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
        
        row = self.drug_df.iloc[idx]
        drug_info = row.to_dict()
        try:
             v_months = float(row.get('效期（月）', 12))
             config.validity_days = int(v_months * 30)
        except:
             config.validity_days = 365

        drug_info['有效期'] = config.validity_days
        drug_info['补货提前期'] = int(self.spin_lead_time.value())
        drug_info['药品ID'] = str(row.get('药品编号', f'DRUG_{idx}'))
        drug_info['药品名称'] = str(row.get('药品名称', 'Unknown'))
        drug_info['单价'] = float(row.get('零售价', 35.0))
        drug_info['药品品类'] = str(row.get('药品品类', 'Misc'))
        drug_info['波动区间分类'] = str(row.get('波动区间分类', '中波动'))
        
        self.btn_run.setEnabled(False)
        self.btn_run.setText("Running...")
        self.log_console.append(f"Starting simulation for {drug_info['药品名称']}...")
        
        self.worker = SimulationWorker(config, drug_info, self.ext_df, duration)
        self.worker.finished.connect(self.on_simulation_finished)
        self.worker.error.connect(self.on_simulation_error)
        self.worker.start()

    def on_simulation_finished(self, df: pd.DataFrame):
        self.btn_run.setEnabled(True)
        self.btn_run.setText("Run Simulation")
        self.log_console.append(f"Data generated. Rows: {len(df)}")
        self.update_dashboard(df)

    def on_simulation_error(self, msg):
        self.btn_run.setEnabled(True)
        self.btn_run.setText("Run Simulation")
        self.log_console.append(f"Error: {msg}")

    def update_dashboard(self, df: pd.DataFrame):
        try:
            if '日期' in df.columns or 'date' in df.columns:
                 col_date = '日期' if '日期' in df.columns else 'date'
                 dates = pd.to_datetime(df[col_date])
            else:
                 return

            stock_base = df.get('Baseline_Inventory', pd.Series(0, index=df.index))
            stock_opt = df.get('Optimized_Inventory', pd.Series(0, index=df.index))
            sales_base = df.get('Baseline_Sales', pd.Series(0, index=df.index))
            sales_opt = df.get('Optimized_Sales', pd.Series(0, index=df.index))
            loss_base = df.get('Baseline_Loss', pd.Series(0, index=df.index))
            loss_opt = df.get('Optimized_Loss', pd.Series(0, index=df.index))
            
            stockout_base_flag = df.get('Baseline_Stockout_Flag', pd.Series(0, index=df.index)) > 0
            stockout_opt_flag = df.get('Optimized_Stockout_Flag', pd.Series(0, index=df.index)) > 0
            
            stockout_base_indices = df.index[stockout_base_flag]
            stockout_opt_indices = df.index[stockout_opt_flag]

            # --- 1. Overview Tab (3 Subplots) ---
            fig = self.plot_overview.canvas.fig
            fig.clear()
            ax1 = fig.add_subplot(311)
            ax2 = fig.add_subplot(312, sharex=ax1) 
            ax3 = fig.add_subplot(313, sharex=ax1)
            
            # Inventory
            ax1.plot(dates, stock_base, label='Baseline', color='gray', alpha=0.6)
            ax1.plot(dates, stock_opt, label='Optimized', color='blue', linewidth=1.5)
            if not stockout_base_indices.empty:
                 d = dates.loc[stockout_base_indices]
                 ax1.scatter(d, [0]*len(d), color='red', marker='x', s=20, zorder=5)
            if not stockout_opt_indices.empty:
                 d = dates.loc[stockout_opt_indices]
                 ax1.scatter(d, [0]*len(d), color='orange', marker='^', s=20, zorder=5)
            ax1.set_title('Inventory Level & Stockouts')
            ax1.legend(loc='upper right', fontsize='x-small')
            ax1.grid(True, alpha=0.3)
            
            # Sales
            ax2.plot(dates, sales_base, label='Baseline', color='orange', alpha=0.6, linestyle='--')
            ax2.plot(dates, sales_opt, label='Optimized', color='green', alpha=0.8)
            ax2.set_title('Sales')
            ax2.legend(loc='upper right', fontsize='x-small')
            ax2.grid(True, alpha=0.3)
            
            # Cumulative Stockout
            cum_base = stockout_base_flag.astype(int).cumsum()
            cum_opt = stockout_opt_flag.astype(int).cumsum()
            ax3.plot(dates, cum_base, label='Baseline Cum Stockouts', color='red')
            ax3.plot(dates, cum_opt, label='Optimized Cum Stockouts', color='green')
            ax3.fill_between(dates, cum_base, cum_opt, color='green', alpha=0.1)
            ax3.set_title('Cumulative Stockout Days')
            ax3.legend(loc='upper left', fontsize='x-small')
            ax3.grid(True, alpha=0.3)
            
            try:
                ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                ax3.xaxis.set_major_locator(mdates.AutoDateLocator())
                fig.autofmt_xdate()
            except: pass
            fig.tight_layout()
            self.plot_overview.canvas.draw()
            
            # --- 2. Inventory Detail Tab ---
            fig_inv = self.plot_inventory.canvas.fig
            fig_inv.clear()
            ax_inv = fig_inv.add_subplot(111)
            ax_inv.plot(dates, stock_base, label='Baseline', color='gray', alpha=0.7)
            ax_inv.plot(dates, stock_opt, label='Optimized', color='blue', linewidth=2)
            if not stockout_base_indices.empty:
                d = dates.loc[stockout_base_indices]
                ax_inv.scatter(d, [0]*len(d), color='red', marker='x', s=50, label='Baseline Stockout', zorder=5)
            if not stockout_opt_indices.empty:
                d = dates.loc[stockout_opt_indices]
                ax_inv.scatter(d, [0]*len(d), color='orange', marker='^', s=50, label='Optimized Stockout', zorder=5)
            ax_inv.set_title('Detailed Inventory Comparison')
            ax_inv.set_ylabel('Stock Quantity')
            ax_inv.legend()
            ax_inv.grid(True, alpha=0.3)
            try:
                ax_inv.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                ax_inv.xaxis.set_major_locator(mdates.AutoDateLocator())
                fig_inv.autofmt_xdate()
            except: pass
            fig_inv.tight_layout()
            self.plot_inventory.canvas.draw()
            
            # --- 3. Sales Detail Tab ---
            fig_sales = self.plot_sales.canvas.fig
            fig_sales.clear()
            ax_sales = fig_sales.add_subplot(111)
            ax_sales.plot(dates, sales_base, label='Baseline', color='orange', linestyle='--')
            ax_sales.plot(dates, sales_opt, label='Optimized', color='green', linewidth=1.5)
            ax_sales.set_title('Daily Sales Comparison')
            ax_sales.legend()
            ax_sales.grid(True, alpha=0.3)
            try:
                ax_sales.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                ax_sales.xaxis.set_major_locator(mdates.AutoDateLocator())
                fig_sales.autofmt_xdate()
            except: pass
            fig_sales.tight_layout()
            self.plot_sales.canvas.draw()

            # --- 4. Loss Tab ---
            fig_loss = self.plot_loss.canvas.fig
            fig_loss.clear()
            ax_l1 = fig_loss.add_subplot(211)
            ax_l2 = fig_loss.add_subplot(212, sharex=ax_l1)
            
            ax_l1.bar(dates, loss_base, label='Baseline Expiry', color='red', alpha=0.5, width=2)
            ax_l1.bar(dates, loss_opt, label='Optimized Expiry', color='purple', alpha=0.5, width=2)
            ax_l1.set_title('Daily Expiration Loss')
            ax_l1.set_ylabel('Qty Expired')
            ax_l1.legend()
            ax_l1.grid(True, alpha=0.3)
            
            cum_loss_base = loss_base.cumsum()
            cum_loss_opt = loss_opt.cumsum()
            ax_l2.plot(dates, cum_loss_base, label='Baseline Cumulative', color='darkred', linewidth=2)
            ax_l2.plot(dates, cum_loss_opt, label='Optimized Cumulative', color='indigo', linewidth=2)
            ax_l2.fill_between(dates, cum_loss_base, cum_loss_opt, color='indigo', alpha=0.1, label='Loss Reduction')
            ax_l2.set_title('Cumulative Expiration Trend')
            ax_l2.set_ylabel('Total Qty Lost')
            ax_l2.legend()
            ax_l2.grid(True, alpha=0.3)
            
            try:
                ax_l2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                ax_l2.xaxis.set_major_locator(mdates.AutoDateLocator())
                fig_loss.autofmt_xdate()
            except: pass
            fig_loss.tight_layout()
            self.plot_loss.canvas.draw()

            # KPI Table
            base_loss = df.get('Baseline_Loss', pd.Series([0])).sum()
            opt_loss = df.get('Optimized_Loss', pd.Series([0])).sum()
            loss_imp = (base_loss - opt_loss) / base_loss * 100 if base_loss > 0 else 0
            
            base_stockout = (df.get('Baseline_Stockout_Flag', pd.Series([0])) > 0).sum()
            opt_stockout = (df.get('Optimized_Stockout_Flag', pd.Series([0])) > 0).sum()
            stock_imp = (base_stockout - opt_stockout) / base_stockout * 100 if base_stockout > 0 else 0
            
            metrics = [
                ("Total Loss", f"{base_loss:.0f}", f"{opt_loss:.0f}", f"{loss_imp:.1f}%"),
                ("Stockout Days", f"{base_stockout}", f"{opt_stockout}", f"{stock_imp:.1f}%")
            ]
            
            self.kpi_table.setRowCount(len(metrics))
            for i, (m, b, o, imp) in enumerate(metrics):
                self.kpi_table.setItem(i, 0, QTableWidgetItem(m))
                self.kpi_table.setItem(i, 1, QTableWidgetItem(b))
                self.kpi_table.setItem(i, 2, QTableWidgetItem(o))
                
                item_imp = QTableWidgetItem(imp)
                if float(imp.strip('%')) > 0:
                    item_imp.setForeground(QColor('green'))
                else:
                    item_imp.setForeground(QColor('red'))
                self.kpi_table.setItem(i, 3, item_imp)
                
        except Exception as e:
            # self.log_console.append(str(e))
            print(e)
            print(traceback.format_exc())
