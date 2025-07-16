"""Data utilities for evaluation of medical concept matching.

This module provides functions to load and sample medical vocabulary data
for evaluation purposes.
"""
import os
import csv
import random
from pydantic import BaseModel


DATA_ROOT = os.path.join(os.path.dirname(__file__), '..', '..', 'backend', 'data')
TEST_DATA_ROOT = os.path.join(os.path.dirname(__file__), '..', 'test_data')


class Data(BaseModel):
    """Base model for medical code data.
    
    Attributes:
        code: The medical code identifier.
        text: The human-readable description of the code.
    """
    code: str
    text: str

def get_icd_data() -> list[Data]:
    """Load ICD-10 diagnosis codes from file.
    
    Returns:
        List of Data objects containing ICD codes and descriptions.
    """
    file_path = os.path.join(DATA_ROOT, 'icd10cm_codes_2026.txt')
    
    data = []
    codes = set()
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                parts = line.split(None, 1)
                if len(parts) == 2:
                    code, text = parts
                    if code not in codes:
                        data.append(Data(text=text, code=code))
                        codes.add(code)
    
    return data

def get_atc_data() -> list[Data]:
    """Load ATC drug classification codes from CSV file.
    
    Returns:
        List of Data objects containing ATC codes and descriptions.
    """
    file_path = os.path.join(DATA_ROOT, 'WHO ATC-DDD 2024-07-31.csv')
    
    data = []
    codes = set()
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            code = row['atc_code']
            text = row['atc_name']
            if code and text and code not in codes:
                data.append(Data(text=text, code=code))
                codes.add(code)
    
    return data

def get_cpt_data() -> list[Data]:
    """Load CPT procedure codes from file.
    
    Returns:
        List of Data objects containing CPT codes and descriptions.
    """
    file_path = os.path.join(DATA_ROOT, '2025_DHS_Code_List_Addendum_11_26_2024.txt')
    
    data = []
    codes = set()
    with open(file_path, 'r', encoding='latin-1') as file:
        for line in file:
            line = line.strip()
            if line and not line.startswith('"') and '\t' in line:
                parts = line.split('\t', 1)
                if len(parts) == 2:
                    code, text = parts
                    code = code.strip()
                    text = text.strip()
                    if code and text and code not in codes:
                        data.append(Data(text=text, code=code))
                        codes.add(code)
    
    return data
    
def get_loinc_data() -> list[Data]:
    """Load LOINC measurement codes from CSV file.
    
    Returns:
        List of Data objects containing LOINC codes and descriptions.
    """
    file_path = os.path.join(DATA_ROOT, 'LoincTableCore.csv')
    
    data = []
    codes = set()
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            code = row['LOINC_NUM']
            text = row['LONG_COMMON_NAME']
            if code and text and code not in codes:
                data.append(Data(text=text, code=code))
                codes.add(code)
    
    return data


def sample_data(data: list[Data]) -> list[Data]:
    """Sample a subset of data for testing.
    
    Args:
        data: List of Data objects to sample from.
        
    Returns:
        List of 100 randomly sampled Data objects.
    """
    sample = random.sample(data, 100)
    return sample


def prepare_test_data() -> None:
    """Prepare test data by sampling from all medical vocabularies.
    
    Creates CSV files with 100 samples from each vocabulary type
    (ICD, ATC, LOINC, CPT) in the test_data directory.
    """
    icd_data = get_icd_data()
    atc_data = get_atc_data()
    loinc_data = get_loinc_data()
    cpt_data = get_cpt_data()

    icd_data_sample = sample_data(icd_data)
    atc_data_sample = sample_data(atc_data)
    loinc_data_sample = sample_data(loinc_data)
    cpt_data_sample = sample_data(cpt_data)

    # Create test_data directory if it doesn't exist
    os.makedirs(TEST_DATA_ROOT, exist_ok=True)
    
    # Save samples to CSV files
    datasets = [
        (icd_data_sample, 'icd_test.csv'),
        (atc_data_sample, 'atc_test.csv'),
        (loinc_data_sample, 'loinc_test.csv'),
        (cpt_data_sample, 'cpt_test.csv')
    ]
    
    for data_sample, filename in datasets:
        file_path = os.path.join(TEST_DATA_ROOT, filename)
        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['code', 'text'])  # Header
            for item in data_sample:
                writer.writerow([item.code, item.text])


if __name__ == '__main__':
    prepare_test_data()




