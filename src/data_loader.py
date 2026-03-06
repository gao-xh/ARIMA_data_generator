import csv
import logging
from pathlib import Path
from datetime import datetime
import xlrd
import openpyxl

try:
    from src import config
except ImportError:
    pass # config might not be importable if not running from root

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Fallback config if import failed
DATA_DIR = Path("data lib")
SALES_FILE = DATA_DIR / "销量库存业务表.xlsx"
EXT_FILE = DATA_DIR / "外部影响因子表.xls"
DRUG_FILE = DATA_DIR / "药品基础信息表.xls"
OUT_FILE = Path("data/processed/merged_dataset.csv")

class DatasetGenerator:
    """
    Handles loading, cleaning, and merging of raw data files into a single dataset for modeling.
    (Pure Python implementation to avoid NumPy dependencies on Python 3.13)
    """

    def __init__(self):
        self.sales_data = [] # List of dicts
        self.external_data = {} # Dict keyed by date
        self.drug_info = {} # Dict keyed by drug code

    def _read_xlsx(self, path):
        """Reads XLSX file into a list of dicts using openpyxl."""
        if not Path(path).exists():
            logger.error(f"File not found: {path}")
            return []
            
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        ws = wb.active
        rows = list(ws.iter_rows(values_only=True))
        if not rows:
            return []
        
        headers = [str(h).strip() if h is not None else f"col_{i}" for i, h in enumerate(rows[0])]
        data = []
        for row in rows[1:]:
            record = {}
            for i, val in enumerate(row):
                if i < len(headers):
                    record[headers[i]] = val
            data.append(record)
        return data

    def _read_xls(self, path):
        """Reads XLS file into a list of dicts using xlrd."""
        if not Path(path).exists():
            logger.error(f"File not found: {path}")
            return []
            
        wb = xlrd.open_workbook(path)
        ws = wb.sheet_by_index(0)
        
        headers = [str(ws.cell_value(0, i)).strip() for i in range(ws.ncols)]
        data = []
        for r in range(1, ws.nrows):
            record = {}
            for c in range(ws.ncols):
                val = ws.cell_value(r, c)
                record[headers[c]] = val
            data.append(record)
        return data

    def _parse_date(self, val):
        """Parses date from various formats."""
        if val is None:
            return None
        if isinstance(val, datetime):
            return val.strftime('%Y-%m-%d')
        
        val_str = str(val).strip()
        
        # Try YYYYMMDD (e.g., '20240101')
        if len(val_str) == 8 and val_str.isdigit():
            try:
                return f"{val_str[:4]}-{val_str[4:6]}-{val_str[6:]}"
            except:
                pass
        
        # Try YYYY-MM-DD
        try:
            dt = datetime.strptime(val_str[:10], '%Y-%m-%d')
            return dt.strftime('%Y-%m-%d')
        except:
            pass
            
        return val_str 

    def load_raw_data(self):
        """Loads the three main Excel files."""
        logger.info("Loading Raw Data Files (Pure Python Mode)...")
        
        # Use config if available, else local constants
        try:
            from src import config
            sA = config.SALES_FILE
            eF = config.EXTERNAL_FACTORS_FILE
            dF = config.DRUG_INFO_FILE
        except ImportError:
            sA, eF, dF = SALES_FILE, EXT_FILE, DRUG_FILE

        try:
            # 1. Load Sales Data (XLSX)
            logger.info(f"Reading Sales Data: {sA}")
            self.sales_data = self._read_xlsx(sA)
            if not self.sales_data:
                logger.warning("Sales data is empty!")
            logger.info(f"Sales Records: {len(self.sales_data)}")

            # 2. Load External Factors (XLS)
            logger.info(f"Reading External Factors: {eF}")
            ext_list = self._read_xls(eF)
            
            # Index by Date for lookup
            self.external_data = {}
            for item in ext_list:
                # Key usually '日期(UTC)'
                date_val = item.get("日期(UTC)")
                iso_date = self._parse_date(date_val)
                if iso_date:
                    self.external_data[iso_date] = item

            # 3. Load Drug Info (XLS)
            logger.info(f"Reading Drug Info: {dF}")
            drug_list = self._read_xls(dF)
            
            # Index by Drug Code
            self.drug_info = {}
            for item in drug_list:
                # Key: '药品编号'
                code = item.get('药品编号')
                if code is not None:
                    # Clean the code (remove .0 if float)
                    str_code = str(code).strip()
                    if str_code.endswith('.0'): 
                        str_code = str_code[:-2]
                    self.drug_info[str_code] = item
            
        except Exception as e:
            logger.error(f"Error loading files: {e}")
            raise

    def merge_datasets(self):
        """Merges Sales, Drug Info, and External Factors."""
        logger.info("Merging Datasets...")

        if not self.sales_data:
            logger.warning("No sales data to merge.")
            return []

        merged_list = []
        
        for sale in self.sales_data:
            merged_record = sale.copy()

            # 1. Join Drug Info
            raw_code = sale.get('药品编码', '')
            if raw_code is None: 
                raw_code = ''
            drug_code = str(raw_code).strip()
            if drug_code.endswith('.0'): 
                drug_code = drug_code[:-2]
            
            drug_data = self.drug_info.get(drug_code, {})
            # Merge drug info
            for k, v in drug_data.items():
                if k != '药品编号': 
                    merged_record[k] = v

            # 2. Join External Factors
            date_val = sale.get('日期')
            iso_date = self._parse_date(date_val)
            
            if iso_date:
                merged_record['Date_ISO'] = iso_date
                ext_data = self.external_data.get(iso_date, {})
                for k, v in ext_data.items():
                    if k != '日期(UTC)': 
                        merged_record[k] = v
            
            merged_list.append(merged_record)

        self.merged_data = merged_list
        logger.info(f"Merged {len(self.merged_data)} records.")
        return self.merged_data

    def save_dataset(self, output_path=None):
        """Saves the merged dataset to CSV."""
        if not hasattr(self, 'merged_data') or not self.merged_data:
            logger.warning("No merged data to save.")
            return

        p = Path(output_path) if output_path else OUT_FILE
        if not p.is_absolute():
            # try to resolve relative to cwd if needed
            p = Path(".").resolve() / p
            
        p.parent.mkdir(parents=True, exist_ok=True)
        
        # Collect all unique headers
        keys = set()
        for item in self.merged_data:
            keys.update(item.keys())
        header = sorted(list(keys))
        
        # Clean header (remove None keys if any)
        header = [k for k in header if k is not None]

        # Prioritize columns
        priority = ['Date_ISO', '日期', '所属诊所', '药品编码', '药品名称', '当日销量（单位）']
        
        # Sort headers with priority first
        sorted_header = []
        for c in priority:
            if c in header:
                sorted_header.append(c)
                header.remove(c)
        sorted_header.extend(header)

        try:
            with open(p, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.DictWriter(f, fieldnames=sorted_header)
                writer.writeheader()
                for row in self.merged_data:
                    # Filter row to matching keys
                    clean_row = {k: row.get(k) for k in sorted_header}
                    writer.writerow(clean_row)
            logger.info(f"Dataset saved to: {p}")
        except Exception as e:
            logger.error(f"Failed to save CSV: {e}")

    def run(self):
        self.load_raw_data()
        self.merge_datasets()
        self.save_dataset()
        return self.merged_data
