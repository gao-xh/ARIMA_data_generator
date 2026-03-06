#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main script to generate the synthetic dataset.
Uses configuration from src.core.constants.
"""

import sys
import logging
import pandas as pd
from pathlib import Path

# Add src to path just in case
sys.path.append(str(Path(__file__).parent))

from src.core import constants as C
from src.core.generator import DataGenerator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data():
    """Load base data from configured paths."""
    # 1. Load External Factors
    ext_path_csv = Path(C.FILE_EXTERNAL_FACTORS)
    ext_path_xls = Path(C.FILE_EXTERNAL_FACTORS_XLS)
    
    if ext_path_csv.exists():
        logger.info(f"Loading External Factors from CSV: {ext_path_csv}")
        df_factors = pd.read_csv(ext_path_csv)
    elif ext_path_xls.exists():
        logger.info(f"Loading External Factors from Excel: {ext_path_xls}")
        df_factors = pd.read_excel(ext_path_xls)
    else:
        raise FileNotFoundError(f"External factors file not found at {ext_path_csv} or {ext_path_xls}")

    # 2. Load Drug Info
    drug_path_csv = Path(C.FILE_DRUG_INFO)
    drug_path_xls = Path(C.FILE_DRUG_INFO_XLS)

    if drug_path_csv.exists():
        logger.info(f"Loading Drug Info from CSV: {drug_path_csv}")
        df_drugs = pd.read_csv(drug_path_csv)
    elif drug_path_xls.exists():
        logger.info(f"Loading Drug Info from Excel: {drug_path_xls}")
        df_drugs = pd.read_excel(drug_path_xls)
    else:
        raise FileNotFoundError(f"Drug info file not found at {drug_path_csv} or {drug_path_xls}")
    
    return df_factors, df_drugs

def main():
    try:
        # Load inputs
        df_factors, df_drugs = load_data()
        
        # Initialize Core Generator
        generator = DataGenerator(drug_df=df_drugs, external_factors_df=df_factors)
        
        # Generate Data for All Clinics
        # Default clinics: Clinic_A to Clinic_G (7 clinics)
        clinics = [f"Clinic_{chr(65+i)}" for i in range(7)]
        logger.info(f"Generating data for clinics: {clinics}")
        
        full_dataset = generator.generate_full_dataset(clinics=clinics)
        
        # Save Result
        output_path = Path(C.FILE_SYNTHETIC_SALES)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        full_dataset.to_csv(output_path, index=False, encoding='utf-8-sig')
        logger.info(f"Successfully generated dataset at: {output_path}")
        logger.info(f"Total Records: {len(full_dataset)}")
        
    except Exception as e:
        logger.error(f"Failed to generate dataset: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
