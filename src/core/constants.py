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
