# Core configuration and types
from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd
import numpy as np

# Column Names for clarity and reusability
COL_DATE = '日期'
COL_CLINIC = '所属诊所'
COL_DRUG_CODE = '药品编码'
COL_SALES = '当日销量（单位）'
COL_INV_START = '当日期初库存'
COL_INV_END = '当日期末库存'
COL_REPLENISH = '当日补货量'
COL_STOCKOUT = '缺货标记'
COL_LOSS = '损耗数量'

# External Factor Columns
EXT_TEMP = '平均气温'
EXT_FLU = '流感ILI%'
EXT_HOLIDAY = '节假日'

# Fluctuation Categories
FLUC_HIGH = '高波动'
FLUC_MED = '中波动'
FLUC_LOW = '低波动'

# File Paths (Relative to project root)
# Using forward slashes for cross-platform compatibility
import os
DEFAULT_DATA_LIB = "data lib"
DEFAULT_PROCESSED_DATA = "data/processed"

FILE_DRUG_INFO = os.path.join(DEFAULT_DATA_LIB, "drug_info.csv")
FILE_EXTERNAL_FACTORS = os.path.join(DEFAULT_DATA_LIB, "external_factors.csv")
FILE_SYNTHETIC_SALES = os.path.join(DEFAULT_PROCESSED_DATA, "synthetic_sales.csv")

# Excel Source options (for fallback)
FILE_DRUG_INFO_XLS = os.path.join(DEFAULT_DATA_LIB, "药品基础信息表.xls")
FILE_EXTERNAL_FACTORS_XLS = os.path.join(DEFAULT_DATA_LIB, "外部影响因子表.xls")
