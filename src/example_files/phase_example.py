#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
This example file shows how to use the Phase class to process data.
Uses the same tutorial_1.csv synthetic dataset as the chemometrics example.
'''

# Written by
# Alfred Worrad <worrada@udel.edu>,

__author__ = "Alfred Worrad"
__version__ = "1.2.1"
__maintainer__ = "Alfred Worrad"
__email__ = "worrada@udel.edu"
__status__ = "Development"
__project__ = "MES wavey"
__created__ = "December 18, 2023"
__updated__ = "September 11, 2025"

# built-in modules
from typing import List
import os

# third-party modules
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# project modules
from src.core_functionality.data_processing import DataProcessingCSV
from src.run_configs.phase_run_config import PhaseRunConfig
from src.config_files.config_dictionaries import (
    SPECTRUM_DIRECTORY, DATA_PER_PULSE, OUTPUT_DIRECTORY, STARTING_FRAME, 
    ENDING_FRAME, FILE_TYPE, ANALYSIS_TYPE, CSV, ENDING_ROW_NUMBER, STARTING_ROW_NUMBER, 
    FREQUENCY_COLUMN, INTENSITY_COLUMN, BASELINE_CORRECTION_METHOD, LIST_OF_HARMONICS,
    SAVE_FFT_DATA, SAVE_IFFT_DATA, SAVE_PHASE_DATA, NONE_DICT_CONFIG, SPECTRAL_COORDINATE_START, SPECTRAL_COORDINATE_END,
    PROJECT_NAME
)
from src.config_files.dict_config_utils import GENERAL_PAGE, ANALYSIS_PAGE, CURRENT_STATE, DictConfig



script_dir = os.path.dirname(os.path.abspath(__file__))

# Use tutorial_1 dataset for phase analysis  
spectrum_directory = os.path.join(script_dir, "example_synthetic_data")
file_type = "csv"
csv_frequency_column = "frequencies"  # Updated for tutorial_1.csv
csv_intensity_column = "intensities"  # Updated for tutorial_1.csv
csv_row_num_start = 1
csv_row_num_end = -1
spectral_coordinate_start = 0
spectral_coordinate_end = -1

# Set project name to match the file name by default
project_name = "phase_tutorial_1"

starting_frame = 0
ending_frame = -1  # Process all frames like tutorial_1
data_per_pulse = 800  # Match tutorial_1: 800 collected spectra
output_directory = os.path.join(script_dir, 'phase_outputs')

# Ensure output directory exists
os.makedirs(output_directory, exist_ok=True)

dict_config_general_1 = {
    SPECTRUM_DIRECTORY: DictConfig(parent_key=SPECTRUM_DIRECTORY, type_val=str, current_state=spectrum_directory, is_required=True, select_location=True, tooltip="contains full path of folder / directory with the files").get_config(),
    DATA_PER_PULSE: DictConfig(parent_key=DATA_PER_PULSE, type_val=int, current_state=data_per_pulse, is_required=True, select_location=False, tooltip="number of time points collected.").get_config(),
    OUTPUT_DIRECTORY: DictConfig(parent_key=OUTPUT_DIRECTORY, type_val=str, current_state=output_directory, is_required=True, select_location=True, tooltip="output directory where the output CSV files will be created. Cannot be the same as spectrum_dir.").get_config(),
    PROJECT_NAME: DictConfig(parent_key=PROJECT_NAME, type_val=str, current_state=project_name, is_required=True, select_location=False, tooltip="The name of the project save file.").get_config(),
    STARTING_FRAME: DictConfig(parent_key=STARTING_FRAME, type_val=int, current_state=starting_frame, is_required=True, select_location=False, tooltip="start frame").get_config(),
    ENDING_FRAME: DictConfig(parent_key=ENDING_FRAME, type_val=int, current_state=ending_frame, is_required=True, select_location=False, tooltip="last frame, -1 take all frames into account").get_config(),
    SPECTRAL_COORDINATE_START: DictConfig(parent_key=SPECTRAL_COORDINATE_START, type_val=int, current_state=spectral_coordinate_start, is_required=True, select_location=False, tooltip="start spectral coordinate").get_config(),
    SPECTRAL_COORDINATE_END: DictConfig(parent_key=SPECTRAL_COORDINATE_END, type_val=int, current_state=spectral_coordinate_end, is_required=True, select_location=False, tooltip="end spectral coordinate").get_config(),
    FILE_TYPE: DictConfig(parent_key=FILE_TYPE, type_val=dict, current_state={CSV: DictConfig(parent_key=CSV, type_val=dict, current_state={
            FREQUENCY_COLUMN: DictConfig(parent_key=FREQUENCY_COLUMN, type_val=str, current_state=csv_frequency_column, tooltip="frequency column name").get_config(),
            INTENSITY_COLUMN: DictConfig(parent_key=INTENSITY_COLUMN, type_val=str, current_state=csv_intensity_column, tooltip="intensity column name").get_config(),
            STARTING_ROW_NUMBER: DictConfig(parent_key=STARTING_ROW_NUMBER, type_val=int, current_state=csv_row_num_start, tooltip="row number to start reading data").get_config(),
            ENDING_ROW_NUMBER: DictConfig(parent_key=ENDING_ROW_NUMBER, type_val=int, current_state=csv_row_num_end, tooltip="row number to end reading data").get_config()
        }).get_config()}, is_required=True, select_location=False, tooltip="file type of files, csv or txt", has_options=True, options=[
        {"None": NONE_DICT_CONFIG.get_config()},
        {CSV: DictConfig(parent_key=CSV, type_val=dict, current_state={
            FREQUENCY_COLUMN: DictConfig(parent_key=FREQUENCY_COLUMN, type_val=str, current_state='frequencies', tooltip="frequency column name").get_config(),
            INTENSITY_COLUMN: DictConfig(parent_key=INTENSITY_COLUMN, type_val=str, current_state='intensities', tooltip="intensity column name").get_config(),
            STARTING_ROW_NUMBER: DictConfig(parent_key=STARTING_ROW_NUMBER, type_val=int, current_state=1, tooltip="row number to start reading data").get_config(),
            ENDING_ROW_NUMBER: DictConfig(parent_key=ENDING_ROW_NUMBER, type_val=int, current_state=-1, tooltip="row number to end reading data").get_config()
        }).get_config()},
        {"txt": DictConfig(parent_key="txt", type_val=dict, current_state=dict()).get_config()}
    ]).get_config(),
    BASELINE_CORRECTION_METHOD: DictConfig(parent_key=BASELINE_CORRECTION_METHOD, type_val=dict, current_state={"None": NONE_DICT_CONFIG.get_config()}, is_required=True, select_location=False, tooltip="baseline correction method to use, available options: 'arpls'", has_options=True, options=[
        {"None": NONE_DICT_CONFIG.get_config()},
        {"arpls": DictConfig(parent_key="arpls", type_val=dict, current_state={
            'lambda': DictConfig(parent_key="lambda", type_val=int, current_state=100000, tooltip="smoothness parameter (higher values give smoother baselines)").get_config(),
            'stop ratio': DictConfig(parent_key="stop ratio", type_val=float, current_state=.000001, tooltip="convergence criterion").get_config()
        }).get_config()}
    ]).get_config()
}

dict_config_phase_1 = {
    ANALYSIS_TYPE: DictConfig(parent_key=ANALYSIS_TYPE, type_val=str, current_state="Phase", is_required=True, select_location=False, tooltip="analysis type", editable=False).get_config(),
    LIST_OF_HARMONICS: DictConfig(parent_key=LIST_OF_HARMONICS, type_val=List[int], current_state=[1], is_required=False, select_location=False, tooltip="list of harmonics").get_config(),
    SAVE_PHASE_DATA: DictConfig(parent_key=SAVE_PHASE_DATA, type_val=bool, current_state=True, is_required=True, select_location=False, tooltip="save phase data").get_config(),
    SAVE_FFT_DATA: DictConfig(parent_key=SAVE_FFT_DATA, type_val=bool, current_state=True, is_required=True, select_location=False, tooltip="save fft data").get_config(),
    SAVE_IFFT_DATA: DictConfig(parent_key=SAVE_IFFT_DATA, type_val=bool, current_state=True, is_required=True, select_location=False, tooltip="save ifft data").get_config()
}

complete_dict_config = {
    GENERAL_PAGE : dict_config_general_1,
    ANALYSIS_PAGE : dict_config_phase_1,
}

print("=== Phase Analysis Example using Tutorial 1 Dataset ===")
print(f"Data directory: {spectrum_directory}")
print(f"Output directory: {output_directory}")

run_config_instance = PhaseRunConfig(complete_dict_config)

# =============================================================================
# GENERATE CONFIGURATION FILES
# =============================================================================

# Create example_yamls directory if it doesn't exist (it should already exist)
yaml_dir = os.path.join(script_dir, 'example_yamls')
os.makedirs(yaml_dir, exist_ok=True)

print("\n=== Generating Configuration Files ===")

# Generate YAML configuration file
yaml_path = os.path.join(yaml_dir, 'phase_tutorial_1_config.yaml')
run_config_instance.write_yaml_config(yaml_path)
print(f"Generated YAML config: {yaml_path}")

# Generate clean JSON configuration file
json_clean_path = os.path.join(yaml_dir, 'phase_tutorial_1_config_clean.json')
run_config_instance.write_json_clean(json_clean_path)
print(f"Generated clean JSON config: {json_clean_path}")

# Generate full JSON configuration file (GUI compatible)
json_full_path = os.path.join(yaml_dir, 'phase_tutorial_1_config_full.json')
run_config_instance.write_json_config(json_full_path)
print(f"Generated full JSON config: {json_full_path}")

# =============================================================================
# TEST CONFIGURATION LOADING
# =============================================================================

print(f"\n=== Testing Configuration File Loading ===")

# Test YAML loading
try:
    yaml_config = PhaseRunConfig.from_yaml(yaml_path)
    print(f"YAML loading successful")
    print(f"Analysis type: {yaml_config.analysis_page_config[ANALYSIS_TYPE][CURRENT_STATE]}")
    print(f"Harmonics: {yaml_config.analysis_page_config[LIST_OF_HARMONICS][CURRENT_STATE]}")
except Exception as e:
    print(f"YAML loading failed: {e}")

# Test clean JSON loading
try:
    json_clean_config = PhaseRunConfig.from_json(json_clean_path)
    print(f"Clean JSON loading successful")
    print(f"Analysis type: {json_clean_config.analysis_page_config[ANALYSIS_TYPE][CURRENT_STATE]}")
    print(f"Harmonics: {json_clean_config.analysis_page_config[LIST_OF_HARMONICS][CURRENT_STATE]}")
except Exception as e:
    print(f"Clean JSON loading failed: {e}")

# Test full JSON loading
try:
    json_full_config = PhaseRunConfig.from_json(json_full_path)
    print(f"Full JSON loading successful")
    print(f"Analysis type: {json_full_config.analysis_page_config[ANALYSIS_TYPE][CURRENT_STATE]}")
    print(f"Harmonics: {json_full_config.analysis_page_config[LIST_OF_HARMONICS][CURRENT_STATE]}")
except Exception as e:
    print(f"Full JSON loading failed: {e}")

# =============================================================================
# FILE SIZE COMPARISON
# =============================================================================

print(f"\n=== Configuration File Sizes ===")
try:
    yaml_size = os.path.getsize(yaml_path)
    json_clean_size = os.path.getsize(json_clean_path)
    json_full_size = os.path.getsize(json_full_path)
    
    print(f"YAML config: {yaml_size:,} bytes")
    print(f"JSON clean config: {json_clean_size:,} bytes")
    print(f"JSON full config: {json_full_size:,} bytes")
    print(f"Full JSON is {json_full_size/yaml_size:.1f}x larger than YAML")
except FileNotFoundError as e:
    print(f"Could not get file sizes: {e}")

# =============================================================================
# RUN ANALYSIS WITH TUTORIAL DATA
# =============================================================================

print(f"\n=== Running Phase Analysis ===")
print(f"Loading tutorial_1.csv data and performing phase analysis...")

# Comment out the actual analysis to avoid long processing time
# Uncomment these lines to actually run the analysis:
run_config_instance.load_data()
run_config_instance.perform_analysis()

print(f"Phase analysis configuration ready!")
print(f"To run the actual analysis, uncomment the load_data() and perform_analysis() calls")

# If you want to run the analysis, uncomment the following:
phase = run_config_instance.phase
spectra = run_config_instance.spectra

# Generate some example plots
phase.plot_sinusodial_PD_spectra(10)
phase.plot_phase_as_sine(71)
phase.plot_phase_as_sine(26)
plt.show()

print(f"\nPhase example completed successfully.")
print(f"Configuration files saved to: {yaml_dir}")

