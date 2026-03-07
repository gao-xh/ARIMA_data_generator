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
     QSplitter, QSizePolicy, QFormLayout, QTabWidget
)
from PySide6.QtCore import Qt, QThread, Signal
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

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
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(15)

        # --- Left Panel: Controls ---
        control_panel = QGroupBox("Configuration")
        control_layout = QVBoxLayout()
        control_layout.setSpacing(10)
        
        # 0. Study Context
        context_group = QGroupBox("Research Object (Thesis Context)")
        context_layout = QVBoxLayout()
        
        info_label = QLabel(
            "<b>Managed Clinics:</b> 7 Total (Abstracted as Single Entity)<br>"
            "<b>Date Range:</b> 2023-01-01 to 2024-12-31 (Fixed)<br>"
            "<b>Included Drugs:</b> 128 SKUs (Categorized)"
        )
        info_label.setStyleSheet("color: #555; font-size: 11px;")
        context_layout.addWidget(info_label)
        
        # Volatility Classification Legend
        legend_layout = QFormLayout()
        legend_layout.setContentsMargins(0, 5, 0, 0)
        
        lbl_low = QLabel("Low (CV < 0.2):")
        lbl_low.setStyleSheet("color: green; font-weight: bold;")
        self.val_low = QLabel("41 SKUs")
        
        lbl_med = QLabel("Medium (0.2 ≤ CV ≤ 0.5):")
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
        
        # 2. Inventory Policy (Control) Group
        policy_group = QGroupBox("Inventory Policy (H2 - Strategy)")
        policy_layout = QFormLayout()
        policy_layout.setSpacing(8)

        self.spin_initial_stock = QSpinBox()
        self.spin_initial_stock.setRange(0, 90)
        self.spin_initial_stock.setValue(14)
        self.spin_initial_stock.setSuffix(" Days")
        self.spin_initial_stock.setToolTip("Initial Inventory Level (Days of Demand)")

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
        
        self.combo_service_level = QComboBox()
        self.combo_service_level.addItems(["95% (Low Vol / Z=1.65)", "98% (Medium Vol / Z=1.96)", "99% (High Vol / Z=2.33)", "Custom"])
        self.combo_service_level.currentIndexChanged.connect(self._on_service_level_changed)
        
        self.spin_safety = QDoubleSpinBox()
        self.spin_safety.setRange(0.5, 5.0)
        self.spin_safety.setSingleStep(0.1)
        self.spin_safety.setValue(1.96)
        self.spin_safety.setToolTip("Safety Stock Factor (Z-Score)")
        self.spin_safety.setEnabled(False) # Default to auto/locked unless Custom

        policy_layout.addRow("Initial Stock:", self.spin_initial_stock)
        policy_layout.addRow("Review Period (R):", self.spin_replenish)
        policy_layout.addRow("Lead Time (L):", self.spin_lead_time)
        policy_layout.addRow("Target Service Level:", self.combo_service_level)
        policy_layout.addRow("Safety Factor (Z):", self.spin_safety)
        policy_group.setLayout(policy_layout)
        
        control_layout.addWidget(policy_group)

        # 3. External Factors (Environment) Group
        env_group = QGroupBox("Environment Factors (H1 - ARIMAX)")
        env_layout = QFormLayout()
        env_layout.setSpacing(8)

        self.spin_flu_sens = QDoubleSpinBox()
        self.spin_flu_sens.setRange(0.0, 5.0)
        self.spin_flu_sens.setSingleStep(0.1)
        self.spin_flu_sens.setValue(1.2)
        self.spin_flu_sens.setToolTip("Sensitivity to Flu Outbreaks (Logic: D = Base * FluFactor * Sens)")

        self.spin_temp_sens = QDoubleSpinBox()
        self.spin_temp_sens.setRange(0.0, 3.0)
        self.spin_temp_sens.setSingleStep(0.1) 
        self.spin_temp_sens.setValue(1.0)
        self.spin_temp_sens.setToolTip("Sensitivity to Cold Weather (Respiratory/Chronic)")

        self.spin_rain_sens = QDoubleSpinBox()
        self.spin_rain_sens.setRange(0.0, 2.0)
        self.spin_rain_sens.setSingleStep(0.1)
        self.spin_rain_sens.setValue(0.0)
        self.spin_rain_sens.setToolTip("Sensitivity to Rainfall (Log-Rainfall Model)")

        env_layout.addRow("Flu Sensitivity:", self.spin_flu_sens)
        env_layout.addRow("Temp Sensitivity:", self.spin_temp_sens)
        env_layout.addRow("Rain Sensitivity:", self.spin_rain_sens)
        env_group.setLayout(env_layout)

        control_layout.addWidget(env_group)
        
        # 5. Actions
        action_layout = QHBoxLayout()
        self.btn_reset = QPushButton("Reset Default")
        self.btn_reset.setToolTip("Reset to Thesis Defaults")
        self.btn_reset.clicked.connect(self._reset_params)
        
        self.btn_run = QPushButton("Run Simulation")
        self.btn_run.setStyleSheet("background-color: #007ACC; color: white; font-weight: bold; padding: 10px; border-radius: 4px;")
        self.btn_run.setCursor(Qt.PointingHandCursor)
        self.btn_run.clicked.connect(self.start_simulation)
        
        action_layout.addWidget(self.btn_reset)
        action_layout.addWidget(self.btn_run)
        control_layout.addLayout(action_layout)
        
        control_layout.addStretch()
        control_panel.setLayout(control_layout)
        control_panel.setFixedWidth(320)
        
        # --- Right Panel: Visualization ---
        viz_panel = QWidget()
        viz_layout = QVBoxLayout(viz_panel)
        viz_layout.setContentsMargins(0, 0, 0, 0)
        
        # KPI Dashboard
        kpi_row = QHBoxLayout()
        self.kpi_sales = self._create_kpi_card("Total Demand", "0", "#007ACC")
        self.kpi_stockout = self._create_kpi_card("Stockout Rate", "0.0%", "#d9534f")
        self.kpi_loss = self._create_kpi_card("Loss Rate", "0.0%", "#f0ad4e")
        self.kpi_turnover = self._create_kpi_card("Turnover Days", "0.0", "#5cb85c")
        
        kpi_row.addWidget(self.kpi_sales)
        kpi_row.addWidget(self.kpi_stockout)
        kpi_row.addWidget(self.kpi_loss)
        kpi_row.addWidget(self.kpi_turnover)
        viz_layout.addLayout(kpi_row)
        
        # Charts Area
        self.viz_tabs = QTabWidget()
        self.viz_tabs.setStyleSheet("QTabWidget::pane { border: 1px solid #ddd; }")
        
        # Tab 1: Comprehensive Dashboard
        self.tab_dashboard = QWidget()
        dash_lay = QVBoxLayout()
        self.canvas_dash = MplCanvas(self, width=8, height=6, dpi=100)
        dash_lay.addWidget(self.canvas_dash)
        self.tab_dashboard.setLayout(dash_lay)
        self.viz_tabs.addTab(self.tab_dashboard, "Overview Dashboard")
        
        # Tab 2: Inventory Detail
        self.tab_inv = QWidget()
        inv_lay = QVBoxLayout()
        self.canvas_inv = MplCanvas(self, width=8, height=6, dpi=100)
        inv_lay.addWidget(self.canvas_inv)
        self.tab_inv.setLayout(inv_lay)
        self.viz_tabs.addTab(self.tab_inv, "Inventory Flow")
        
        viz_layout.addWidget(self.viz_tabs)
        
        # Logs
        log_group = QGroupBox("System Log")
        log_layout = QVBoxLayout()
        self.log_console = QTextEdit()
        self.log_console.setReadOnly(True)
        self.log_console.setMaximumHeight(120)
        self.log_console.setStyleSheet("background: #f8f9fa; border: none; font-family: Consolas; font-size: 10pt;")
        log_layout.addWidget(self.log_console)
        log_group.setLayout(log_layout)
        viz_layout.addWidget(log_group)
        
        # Add to Main Layout
        main_layout.addWidget(control_panel)
        main_layout.addWidget(viz_panel)
        
    def _create_kpi_card(self, title, value, color):
        card = QGroupBox()
        card.setStyleSheet(f"QGroupBox {{ border: 1px solid #ddd; border-radius: 6px; background: white; border-left: 5px solid {color}; }}")
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 5)
        
        lbl_title = QLabel(title)
        lbl_title.setStyleSheet("color: #666; font-size: 11px; text-transform: uppercase;")
        
        lbl_val = QLabel(value)
        lbl_val.setStyleSheet(f"color: {color}; font-size: 18px; font-weight: bold;")
        lbl_val.setAlignment(Qt.AlignRight)
        
        layout.addWidget(lbl_title)
        layout.addWidget(lbl_val)
        card.setLayout(layout)
        card.card_value_label = lbl_val # Store reference
        return card

    def _reset_params(self):
        """Reset parameters to Thesis defaults"""
        self.spin_initial_stock.setValue(14)
        self.spin_replenish.setValue(30)
        self.spin_lead_time.setValue(3)
        self.combo_service_level.setCurrentIndex(1) # Medium
        self.spin_flu_sens.setValue(1.0)
        self.spin_temp_sens.setValue(1.0)
        self.spin_rain_sens.setValue(0.0)
        self.log_console.append("Parameters reset to default.")

    def update_kpi(self, sales, stockout_rate, loss_rate, turnover):
        self.kpi_sales.card_value_label.setText(f"{int(sales):,}")
        self.kpi_stockout.card_value_label.setText(f"{stockout_rate:.2%}")
        self.kpi_loss.card_value_label.setText(f"{loss_rate:.2%}")
        self.kpi_turnover.card_value_label.setText(f"{turnover:.1f}")

    def log(self, msg):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_console.append(f"[{timestamp}] {msg}")

    def load_drugs_list(self):
        try:
            # Thesis Logic for Classification
            volatility_counts = {'LOW': 0, 'MEDIUM': 0, 'HIGH': 0}
            items = []
            
            if self.ext_df is None:
                self._init_data()

            if Path(DRUG_INFO).exists():
                try:
                    self.drug_df = pd.read_csv(DRUG_INFO, encoding='utf-8')
                except UnicodeDecodeError:
                    self.drug_df = pd.read_csv(DRUG_INFO, encoding='gb18030')
                
                self.combo_drug.clear()
                
                for idx, row in self.drug_df.iterrows():
                    name = str(row.get('药品名称', 'Unknown'))
                    cat = str(row.get('药品品类', 'Misc'))
                    # Calculate or Infer Volatility
                    vol_raw = str(row.get('波动区间分类', '中波动'))
                    
                    if '低' in vol_raw:
                        vol_cat = 'Low'
                        volatility_counts['LOW'] += 1
                        cv_display = "<0.2"
                    elif '高' in vol_raw:
                        vol_cat = 'High'
                        volatility_counts['HIGH'] += 1
                        cv_display = ">0.5"
                    else:
                        vol_cat = 'Medium'
                        volatility_counts['MEDIUM'] += 1
                        cv_display = "0.2-0.5"
                        
                    items.append(f"{name} | {vol_cat} ({cv_display})")
                
                self.combo_drug.addItems(items)
                self.log_console.append(f"Loaded {len(items)} drugs from metadata.")
                
                # Update UI
                self.val_low.setText(f"{volatility_counts['LOW']} SKUs")
                self.val_med.setText(f"{volatility_counts['MEDIUM']} SKUs")
                self.val_high.setText(f"{volatility_counts['HIGH']} SKUs")
                
                # Trigger selection change to update params
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
        # Auto-set Safety Factor based on Volatility (Thesis Logic)
        vol_raw = str(row.get('波动区间分类', '中波动'))
        if '高' in vol_raw:
            self.spin_safety.setValue(2.33) # 99% SL
        elif '低' in vol_raw:
            self.spin_safety.setValue(1.65) # 95% SL
        else:
            self.spin_safety.setValue(1.96) # 97.5% SL

        # Auto-set Flu Sensitivity based on Drug Category/Name Keyword
        # Logic: Respiratory/Cold meds are highly sensitive. Chronic meds are not.
        cat_str = str(row.get('药品品类', '')).upper()
        name_str = str(row.get('药品名称', '')).upper()
        combined = cat_str + " " + name_str
        
        if any(x in combined for x in ['感冒', '流感', '病毒', '清热', '解热']):
            # Direct Flu meds -> High Sensitivity
            self.spin_flu_sens.setValue(2.5) 
        elif any(x in combined for x in ['呼吸', '咳', '肺', '炎', '头孢', '阿莫西林', '抗生素']):
            # Secondary Respiratory/Antibiotics -> Medium High
            self.spin_flu_sens.setValue(1.5)
        elif any(x in combined for x in ['慢病', '心脑', '血压', '糖', '脂', '维', '钙']):
            # Chronic / Maintenance -> Zero Sensitivity
            self.spin_flu_sens.setValue(0.0)
        else:
            # General -> Low default
            self.spin_flu_sens.setValue(0.5)

    def _on_service_level_changed(self, index):
        """Update Safety Factor (Z) based on Service Level preset"""
        self.spin_safety.setEnabled(False)
        if index == 0:   # 95% (Low Vol)
            self.spin_safety.setValue(1.65)
        elif index == 1: # 98% (Medium Vol)
            self.spin_safety.setValue(1.96)
        elif index == 2: # 99% (High Vol)
            self.spin_safety.setValue(2.33)
        else:            # Custom
            self.spin_safety.setEnabled(True)

    def start_simulation(self):
        idx = self.combo_drug.currentIndex()
        # If no selection, allow running for first drug or internal test
        if self.drug_df is None:
            self.log_console.append("No drug data loaded. Attempting to load default...")
            self.load_drugs_list()
            if self.drug_df is None: return

        if idx < 0: idx = 0
        
        # Fixed Duration: 2023-2024 inclusive (Leap year 2024 has 366 days)
        duration = 365 + 366 
        
        # Abstracted Clinic Scale (1.0 = Representative Entity)
        clinic_scale = 1.0
            
        # Prepare Config
        config = SimulationConfig(
            start_date=pd.Timestamp('2023-01-01'),
            end_date=pd.Timestamp('2024-12-31'), # 2 Years
            replenishment_days=int(self.spin_replenish.value()),
            active_clinic_scale=clinic_scale
        )
        
        # UI overrides
        config.safety_stock_factor = self.spin_safety.value()
        config.flu_sensitivity = self.spin_flu_sens.value()
        config.temp_sensitivity = self.spin_temp_sens.value()
        config.rain_sensitivity = self.spin_rain_sens.value()
        config.initial_stock_days = int(self.spin_initial_stock.value())
        config.random_noise_sigma = 0.2
        
        row = self.drug_df.iloc[idx]
        drug_info = row.to_dict()
        
        # Recalculate validity in days
        try:
             # Default to 12 months if missing
             v_months = float(row.get('效期（月）', 12))
             config.validity_days = int(v_months * 30)
        except:
             config.validity_days = 365

        # Override params passed to Tuner
        drug_info['有效期'] = config.validity_days # Already converted or passed as is? Tuner expects days now?
        # Actually Tuner re-reads '有效期' from drug_info generally. Let's ensure it's days.
        # But wait, Tuner.__init__ does: config.validity_days = int(self.drug_info['有效期'])
        # If drug_info has MONTHS (from CSV), Tuner might set validity_days to 12 days!
        # Fix: We pass the calculated days in drug_info
        
        drug_info['补货提前期'] = int(self.spin_lead_time.value())
        
        # Ensure required fields exist
        drug_info['药品ID'] = str(row.get('药品编号', f'DRUG_{idx}'))
        drug_info['药品名称'] = str(row.get('药品名称', 'Unknown'))
        drug_info['单价'] = float(row.get('零售价', 35.0))
        drug_info['药品品类'] = str(row.get('药品品类', 'Misc'))
        drug_info['波动区间分类'] = str(row.get('波动区间分类', '中波动'))
        
        self.btn_run.setEnabled(False)
        self.btn_run.setText("Running...")
        self.log_console.append(f"Starting simulation for {drug_info['药品名称']} (Scale: {clinic_scale}x)...")
        
        # Start Worker
        self.worker = SimulationWorker(config, drug_info, self.ext_df, duration)
        self.worker.finished.connect(self.on_simulation_finished)
        self.worker.error.connect(self.on_simulation_error)
        self.worker.start()

    def on_simulation_finished(self, df: pd.DataFrame):
        self.btn_run.setEnabled(True)
        self.btn_run.setText("Run Simulation")
        self.log_console.append(f"Simulation completed. Generated {len(df)} records.")
        self.update_dashboard(df)

    def on_simulation_error(self, msg):
        self.btn_run.setEnabled(True)
        self.btn_run.setText("Run Simulation")
        self.log_console.append(f"Error: {msg}")

    def update_dashboard(self, df: pd.DataFrame):
        # 1. Update KPI Cards (Based on 'Optimized' scenario or blended?)
        if 'scenario' in df.columns:
            df_opt = df[df['scenario'] == 'Optimized']
            if df_opt.empty: df_opt = df
        else:
            df_opt = df

        total_sales = df_opt['sales'].sum()
        stockout_days = len(df_opt[df_opt['stockout_flag'] == 1])
        loss_qty = df_opt['loss'].sum()
        
        stockout_rate = stockout_days / len(df_opt) if len(df_opt) > 0 else 0
        loss_rate = loss_qty / (total_sales + loss_qty) if (total_sales + loss_qty) > 0 else 0
        
        # Turnover Calculation: Avg Inventory / Avg Daily Sales
        avg_inv = df_opt['inventory'].mean()
        avg_sales = df_opt['sales'].mean()
        turnover = avg_inv / avg_sales if avg_sales > 0 else 0
        
        self.update_kpi(total_sales, stockout_rate, loss_rate, turnover)

        # 2. Update Charts
        self._plot_dashboard(df)
        self._plot_inventory_detail(df)

    def _plot_dashboard(self, df: pd.DataFrame):
        self.canvas_dash.figure.clear()
        
        # Grid Spec: Top for Sales/Demand, Bottom for Issues
        gs = self.canvas_dash.figure.add_gridspec(2, 1, hspace=0.3)
        ax1 = self.canvas_dash.figure.add_subplot(gs[0, 0])
        ax2 = self.canvas_dash.figure.add_subplot(gs[1, 0])
        
        if 'scenario' not in df.columns:
            df['scenario'] = 'Optimized'
            
        scenarios = df['scenario'].unique()
        colors = {'Baseline': '#999999', 'Optimized': '#007ACC'}
        
        # Top Chart: Demand vs Sales
        for sc in scenarios:
            d = df[df['scenario'] == sc]
            if sc == 'Baseline': continue # Too cluttered to show demand for both?
            
            # Plot Weekly Rolling Mean to reduce noise
            roll_sales = d['sales'].rolling(7).mean()
            ax1.plot(d['date'], roll_sales, label=f'Sales ({sc})', color=colors.get(sc, 'blue'))
            
        ax1.set_title("Weekly Average Sales Trend")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Bottom Chart: Stockout Events & Loss
        # Stacked Bar? Or Scatter?
        # Scatter is better for events
        for sc in scenarios:
            d = df[df['scenario'] == sc]
            out = d[d['stockout_flag'] > 0]
            loss = d[d['loss'] > 0]
            
            offset = 0 if sc == 'Optimized' else 1
            
            if not out.empty:
                ax2.scatter(out['date'], [1+offset]*len(out), marker='x', color='red', label=f'Stockout ({sc})', alpha=0.7)
            
            if not loss.empty:
                # Plot Loss Magnitude?
                ax2.bar(d['date'], d['loss'], alpha=0.3, label=f'Expiry Loss ({sc})', color='orange')

        ax2.set_title("Risk Events: Stockouts (Markers) & Expiry Loss (Bars)")
        ax2.legend()
        
        self.canvas_dash.draw()

    def _plot_inventory_detail(self, df: pd.DataFrame):
        self.canvas_inv.axes.clear()
        
        if 'scenario' not in df.columns:
            df['scenario'] = 'Optimized'

        scenarios = df['scenario'].unique()
        colors = {'Baseline': 'gray', 'Optimized': '#28a745'}
        
        for sc in scenarios:
            d = df[df['scenario'] == sc]
            label_text = f'Inventory ({sc})'
            self.canvas_inv.axes.plot(d['date'], d['inventory'], label=label_text, color=colors.get(sc, 'blue'), alpha=0.8)
            
            # Stockouts
            out = d[d['stockout_flag'] > 0]
            if not out.empty:
                 self.canvas_inv.axes.scatter(out['date'], [0]*len(out), color='red', marker='x', s=20, label=f'Stockout ({sc})')

        self.canvas_inv.axes.set_title("Strategy Comparison: Empirical vs ARIMA-Optimized")
        self.canvas_inv.axes.set_xlabel("Date")
        self.canvas_inv.axes.set_ylabel("Inventory Level")
        self.canvas_inv.axes.legend()
        self.canvas_inv.axes.grid(True, alpha=0.3)
        self.canvas_inv.fig.autofmt_xdate()
        self.canvas_inv.draw()

        # KPI Report
        self.log("\n=== Simulation Report ===")
        for sc in scenarios:
            sub_df = df[df['scenario'] == sc]
            if sub_df.empty: continue
            
            avg_inv = sub_df['inventory'].mean()
            stockout_days = sub_df['stockout_flag'].sum()
            total_sales = sub_df['sales'].sum()
            total_days = max(1, (sub_df['date'].max() - sub_df['date'].min()).days + 1)
            
            turnover = (avg_inv / (total_sales / total_days)) if total_sales > 0 else 0
            service_level = 1.0 - (stockout_days / total_days)
            
            self.log(f"> {sc} Strategy:")
            self.log(f"   - Avg Inventory: {avg_inv:.1f} units")
            self.log(f"   - Turnover Days: {turnover:.1f} days")
            self.log(f"   - Service Level: {service_level*100:.1f}% ({stockout_days} stockouts)")

    def analyze_model_fit(self, df: pd.DataFrame):
        """
        Run ARIMA on the simulated 'Demand' data to visualize fit quality.
        This validates that the demand pattern is indeed predictable by ARIMA.
        """
        self.canvas_mod.axes.clear()
        
        # 1. Extract Time Series (Use Baseline Demand as Ground Truth)
        df_base = df[df['scenario'] == 'Baseline'].copy()
        if df_base.empty: df_base = df.copy()
            
        if 'date' not in df_base.columns or 'demand' not in df_base.columns:
            return

        ts = df_base.set_index('date')['demand'].asfreq('D').fillna(0)
        
        # 2. Fit ARIMA Model
        try:
            order = (5, 1, 0) # Weekly AR logic
            model = ARIMA(ts, order=order)
            res = model.fit()
            
            fitted = res.fittedvalues
            
            # 4. Plot
            self.canvas_mod.axes.plot(ts.index, ts, label='Actual Demand', color='#1f77b4', alpha=0.5)
            self.canvas_mod.axes.plot(fitted.index, fitted, label=f'ARIMA{order} Fit', color='#ff7f0e', linestyle='--')
            
            # 5. Metrics
            aic = res.aic
            bic = res.bic
            mse = mean_squared_error(ts, fitted)
            rmse = np.sqrt(mse)
            mape = mean_absolute_percentage_error(ts[ts>0], fitted[ts>0])
            
            self.log(f"\n=== Model Fit Diagnostics ===")
            self.log(f"   - Model Order: ARIMA{order}")
            self.log(f"   - AIC: {aic:.2f} | BIC: {bic:.2f}")
            self.log(f"   - RMSE: {rmse:.4f}")
            self.log(f"   - MAPE: {mape:.2%}")
            
            self.canvas_mod.axes.set_title(f"ARIMA Model Fit (RMSE={rmse:.2f}, MAPE={mape:.1%})")
            self.canvas_mod.axes.legend()
            self.canvas_mod.axes.grid(True, alpha=0.3)
            self.canvas_mod.fig.autofmt_xdate()
            self.canvas_mod.draw()
            
        except Exception as e:
            self.log(f"ARIMA Analysis Failed: {str(e)}")
