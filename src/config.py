from pathlib import Path
import os

# --- Project Base ---
# Try to be robust for relative paths
try:
    BASE_DIR = Path(os.getcwd())
except:
    BASE_DIR = Path(".")

# --- Raw Data Paths ---
DATA_LIB_DIR = BASE_DIR / "data lib"
SALES_TEMPLATE = DATA_LIB_DIR / "销量库存业务表.xlsx"
EXTERNAL_FACTORS_FILE = DATA_LIB_DIR / "外部影响因子表.xls"
DRUG_INFO = DATA_LIB_DIR / "药品基础信息表.xls"

# --- Output Paths ---
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
MERGED_DATASET_CSV = PROCESSED_DATA_DIR / "synthetic_dataset.csv"

# --- Column Mappings ---
COL_DATE = "日期"
COL_DRUG_CODE = "药品编码"
COL_CLINIC = "所属诊所"

# --- Simulation Settings ---
# Defaults if not extracted
SIM_START_DATE = "2024-01-01" 
SIM_END_DATE = "2025-12-31" 
# 7 Clinics mentioned in doc
CLINICS = ["诊所A", "诊所B", "诊所C", "诊所D", "诊所E", "诊所F", "诊所G"]
