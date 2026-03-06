import math
import random
import csv
import logging
from datetime import datetime, timedelta
from pathlib import Path
import xlrd

try:
    from src import config
except ImportError:
    pass # Try local lookup or relative

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataSimulator:
    """
    Generates synthetic sales and inventory data based on real external factors and drug info.
    Simulates demand driven by seasonality, weather, and flu trends.
    """

    def __init__(self):
        self.external_factors = [] # List of dicts (dates sorted)
        self.drugs = [] # List of dicts
        self.clinics = ["Clinic_A", "Clinic_B", "Clinic_C", "Clinic_D", "Clinic_E", "Clinic_F", "Clinic_G"]
        self.simulation_data = []

    def load_base_data(self):
        """Loads the external factors (time backbone) and drug info (products)."""
        logger.info("Loading Base Data for Simulation...")
        
        # Resolve paths
        try:
            from src import config
            ext_path = config.EXTERNAL_FACTORS_FILE
            drug_path = config.DRUG_INFO
        except ImportError:
            ext_path = Path("data lib/外部影响因子表.xls")
            drug_path = Path("data lib/药品基础信息表.xls")

        # 1. Load External Factors (Time Backbone)
        if not Path(ext_path).exists():
            raise FileNotFoundError(f"External factors file not found: {ext_path}")
            
        wb = xlrd.open_workbook(ext_path)
        ws = wb.sheet_by_index(0)
        headers = [str(ws.cell_value(0, i)).strip() for i in range(ws.ncols)]
        
        for r in range(1, ws.nrows):
            row = {}
            for c in range(ws.ncols):
                row[headers[c]] = ws.cell_value(r, c)
            
            # Parse Date
            try:
                d_val = row.get("日期(UTC)")
                if isinstance(d_val, float): # Excel serial
                    dt = xlrd.xldate_as_datetime(d_val, wb.datemode)
                    row['Date'] = dt
                elif isinstance(d_val, str) and len(d_val) == 8:
                    row['Date'] = datetime.strptime(d_val, "%Y%m%d")
                else:
                    # Try other formats
                    row['Date'] = datetime.strptime(str(d_val)[:10], "%Y-%m-%d")
            except Exception as e:
                logger.warning(f"Skipping row {r} due to date error: {e}")
                continue
                
            self.external_factors.append(row)
        
        # Sort by date
        self.external_factors.sort(key=lambda x: x['Date'])
        logger.info(f"Loaded {len(self.external_factors)} days of external factors.")

        # 2. Load Drug Info
        if not Path(drug_path).exists():
            raise FileNotFoundError(f"Drug info file not found: {drug_path}")

        wb_d = xlrd.open_workbook(drug_path)
        ws_d = wb_d.sheet_by_index(0)
        h_d = [str(ws_d.cell_value(0, i)).strip() for i in range(ws_d.ncols)]
        
        for r in range(1, ws_d.nrows):
            # Check if row is empty
            if not any(ws_d.cell_value(r, c) for c in range(ws_d.ncols)):
                continue
                
            drug = {}
            for c in range(ws_d.ncols):
                drug[h_d[c]] = ws_d.cell_value(r, c)
            
            # Clean Code
            code = str(drug.get('药品编号', '')).strip()
            if code.endswith('.0'): code = code[:-2]
            drug['Code'] = code
            
            # Determine fluctuation class for simulation parameters
            cls = drug.get('波动区间分类', '中波动')
            if '高' in cls:
                drug['Sim_Lambda'] = random.uniform(5, 15) # High volume
                drug['Sim_Var'] = 0.5 # High variance
            elif '低' in cls:
                drug['Sim_Lambda'] = random.uniform(0.5, 2) # Low volume
                drug['Sim_Var'] = 0.1
            else:
                drug['Sim_Lambda'] = random.uniform(2, 6) # Mid volume
                drug['Sim_Var'] = 0.3
            
            # Sensitivity to Flu (External Factor)
            # Heuristic: If "感冒" or "抗炎" or "解热" in Name/Category -> Sensitive
            name = str(drug.get('药品名称', ''))
            cat = str(drug.get('药品品类', ''))
            if any(x in name or x in cat for x in ['感冒', '抗炎', '解热', '止咳', '抗病毒']):
                drug['Flu_Sensitivity'] = random.uniform(0.5, 2.0)
            else:
                drug['Flu_Sensitivity'] = 0.0

            self.drugs.append(drug)
            
        logger.info(f"Loaded {len(self.drugs)} drugs.")

    def simulate(self):
        """Run the simulation loop."""
        if not self.external_factors or not self.drugs:
            raise ValueError("Base data not loaded.")

        logger.info("Starting Simulation...")
        self.simulation_data = []
        
        # Pre-calculate seasonality/flu curve normalization
        # Average ILI is likely small (0.xx). We want to scale impact.
        
        for clinic in self.clinics:
            # Different clinics might have different base demand multipliers
            clinic_mult = random.uniform(0.8, 1.2)
            
            # Create inventory state for each drug in this clinic
            # Inv = Target Stock + Random
            inventory_state = {
                d['Code']: int(d['Sim_Lambda'] * 30 * clinic_mult) for d in self.drugs
            }

            for day_idx, day_data in enumerate(self.external_factors):
                current_date = day_data['Date']
                date_str = current_date.strftime("%Y-%m-%d")
                
                # Factors
                ili = float(day_data.get('ILI%', 0))
                temp = float(day_data.get('平均气温2m(℃)', 0))
                rain = float(day_data.get('平均降水量(mm)', 0))
                flu_rate = float(day_data.get('流感发病率', 0))
                
                # Normalized Temp factor: Cold (below 10) increases demand for some drugs?
                # Simplified: Use Flu sensitivity directly.
                
                is_holiday = float(day_data.get('节假日标记（1=节假日/周末，0=工作日）', 0))
                # Holiday might decrease clinic visits (closed?) or increase?
                # Assumption: Clinics open, demand slightly higher on weekends? Or lower?
                # Let's assume lower on holidays/weekends for routine, higher for acute?
                # Random factor.
                
                for drug in self.drugs:
                    base_lambda = drug['Sim_Lambda'] * clinic_mult
                    
                    # 1. Check Flu Impact
                    flu_effect = 1.0 + (drug['Flu_Sensitivity'] * ili * 2) # Scale ILI impact
                    
                    # 2. Check Seasonality (Temperature)
                    # Cold (<5C) -> +20% for cold meds
                    temp_effect = 1.0
                    if temp < 5 and drug['Flu_Sensitivity'] > 0:
                        temp_effect = 1.2
                    
                    # 3. Calculate Demand (Poisson)
                    mean_demand = base_lambda * flu_effect * temp_effect
                    
                    # Add noise
                    variance_factor = drug['Sim_Var']
                    actual_demand = max(0, int(random.gauss(mean_demand, mean_demand * variance_factor)))
                    
                    # 4. Inventory Logic
                    curr_inv = inventory_state[drug['Code']]
                    
                    # Sales cannot exceed inventory? Or represent lost sales?
                    # Generally: Sales = min(Demand, Inventory)
                    # But if we want to track "Demand" (stockout is separate), let's say Sales = Demand
                    # and we mark Stockout.
                    # The schema has '缺货标记'.
                    
                    sales = min(actual_demand, curr_inv)
                    lost_sales = actual_demand - sales
                    stockout_flag = 1 if lost_sales > 0 else 0
                    
                    # Update Inventory
                    next_inv = curr_inv - sales
                    
                    # Restock policy (Simple (s, S))
                    # s (reorder point) = 7 days demand
                    # S (target) = 30 days demand
                    s = int(mean_demand * 7)
                    S = int(mean_demand * 30)
                    
                    restock = 0
                    if next_inv < s:
                        restock = S - next_inv
                        next_inv += restock # Assume instant replenishment for simplicity, or lag
                        # Schema has '当日补货量'.
                    
                    inventory_state[drug['Code']] = next_inv
                    
                    # 5. Record
                    record = {
                        "日期": date_str,
                        "所属诊所": clinic,
                        "药品编码": drug['Code'],
                        "药品名称": drug.get('药品名称', ''),
                        "当日销量（单位）": sales,
                        "当日期初库存": curr_inv,
                        "当日期末库存": next_inv,
                        "当日补货量": restock,
                        "缺货标记": stockout_flag,
                        "损耗数量": 0 if random.random() > 0.01 else 1, # 1% chance of loss
                        # Merge External Factors too for ML convenience
                        "平均气温": temp,
                        "平均降水量": rain,
                        "流感发病率": flu_rate,
                        "流感ILI%": ili,
                        "节假日": is_holiday
                    }
                    self.simulation_data.append(record)
                    
            logger.info(f"Simulated {len(self.external_factors)} days for {clinic}.")

    def save(self, path=None):
        if not self.simulation_data:
            return
            
        p = Path(path) if path else Path("data/processed/synthetic_sales.csv")
        p.parent.mkdir(parents=True, exist_ok=True)
        
        keys = list(self.simulation_data[0].keys())
        
        with open(p, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(self.simulation_data)
        logger.info(f"Saved synthetic dataset to {p}")

    def run(self):
        self.load_base_data()
        self.simulate()
        try:
            from src import config
            out = config.MERGED_DATASET_CSV
        except:
            out = None
        self.save(out)
        return self.simulation_data

if __name__ == "__main__":
    sim = DataSimulator()
    sim.run()
