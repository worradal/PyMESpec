#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
This example file shows how to use the Chemometrics class to process data that would match 
the inputs provided to the GUI. This example uses the same synthetic dataset as tutorial_1.ipynb.

Dataset details:
This is a synthetic dataset made to mimic data from an A<->B<->C system where A, B, and C are 
different oxidation states of a catalyst (+5 -> +4 -> +3).

The dataset has 800 collected spectra, taken every 0.25 seconds.
A reducing agent is flowed over the catalyst for the first 100 seconds (first 400 spectra), 
then an oxidizing agent is flowed over the catalyst for the next 100 seconds (last 400 spectra).
'''

# Written by
# Alfred Worrad <worrada@udel.edu>,

__author__ = "Afred Worrad"
__version__ = "1.2.1"
__maintainer__ = "Alfred Worrad"
__email__ = "worrada@udel.edu"
__status__ = "Development"
__project__ = "MES wavey"
__date__ = "September 11, 2025"

# built-in modules
from pathlib import Path
from typing import List
import os
import sys

# Add the project root to sys.path to allow imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src'))

# third-party modules
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# project modules
from core_functionality.data_processing import DataProcessingCSV
from core_functionality.chemometrics import Chemometrics, FIT_PARAMETERS, CONFIDENCE_INTERVALS
from config_files.config_dictionaries import  DictConfig, NONE_DICT_CONFIG
from config_files.config_dictionaries import (
    SPECTRUM_DIRECTORY, DATA_PER_PULSE, OUTPUT_DIRECTORY, STARTING_FRAME, 
    ENDING_FRAME, FILE_TYPE, ANALYSIS_TYPE, CSV, ENDING_ROW_NUMBER, STARTING_ROW_NUMBER, 
    FREQUENCY_COLUMN, INTENSITY_COLUMN, BASELINE_CORRECTION_METHOD, CHEMOMETRICS_DIRECTORY,
    CHEMOMETRICS_FILE_TYPE, CHEMOMETRICS_SAMPLE_STANDARDS_FILE_LIST, CONFIDENCE_INTERVAL,
    CONSTRAINED_FITTING_CONFIGS, NORMALIZE_FIT, MAX_ITER, EPSILON, CONSTRAINED_FIT, 
    SPECTRAL_COORDINATE_END, SPECTRAL_COORDINATE_START, PROJECT_NAME  
    
)
from config_files.dict_config_utils import GENERAL_PAGE, ANALYSIS_PAGE, CURRENT_STATE, OPTIONS
from run_configs.chemometrics_run_config import ChemometricsRunConfig

# Get the current script directory and project root
script_dir = os.path.dirname(os.path.abspath(__file__))

# Data paths using the tutorial_1 synthetic data
spectrum_directory = os.path.join(script_dir, "example_synthetic_data")
file_type = "csv"
csv_frequency_column = "frequencies"
csv_intensity_column = "intensities"
csv_row_num_start = 1
csv_row_num_end = -1
spectral_coordinate_start = 0
spectral_coordinate_end = -1

# Parameters for chemometrics - reference spectra from tutorial_1
chemometric_path = os.path.join(script_dir, "example_synthetic_data", "example_baselines")
sample_standards_file_list = ["reference_spectrum_0.csv", "reference_spectrum_1.csv", 
                             "reference_spectrum_2.csv", "reference_spectrum_3.csv"]
chemometric_file_type = "csv"
csv_frequency_column_chem = "frequencies"
csv_intensity_column_chem = "normalized_intensities"  # Note: tutorial uses normalized_intensities
csv_row_num_start_chem = 1
csv_row_num_end_chem = -1

# Analysis parameters matching tutorial_1
confidence_interval = 0.05  # 95% confidence (alpha = 0.05)
alpha = confidence_interval

# Constrained fit prevents amount of deconvoluted spectra from being negative
constrained_fit = True
epsilon = 1e-5  # convergence criteria for constrained fit
max_iters = 10000  # maximum number of iterations for constrained fit
normalize_fit = False  # Match tutorial_1.ipynb: normalize_fit=False
starting_frame = 0
ending_frame = -1
data_per_pulse = 800  # Match tutorial_1: 800 collected spectra
output_directory = os.path.join(script_dir, "chemometric_outputs")

# Set project name to match the file name by default
project_name = "chemometrics_tutorial_1"

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

dict_config_chemometrics_1 = {
    ANALYSIS_TYPE: DictConfig(parent_key=ANALYSIS_TYPE, type_val=str, current_state="Chemometrics", is_required=True, select_location=False, tooltip="analysis type", editable=False).get_config(),
    CHEMOMETRICS_DIRECTORY: DictConfig(parent_key=CHEMOMETRICS_DIRECTORY, type_val=str, current_state=chemometric_path, is_required=True, select_location=True, tooltip="contains full path of folder / directory with the files").get_config(),
    CHEMOMETRICS_SAMPLE_STANDARDS_FILE_LIST: DictConfig(parent_key=CHEMOMETRICS_SAMPLE_STANDARDS_FILE_LIST, type_val=List[str], current_state=sample_standards_file_list, is_required=True, select_location=False, tooltip="contains list of files to be used as sample standards").get_config(),
    CHEMOMETRICS_FILE_TYPE: DictConfig(parent_key=CHEMOMETRICS_FILE_TYPE, type_val=dict, current_state={"csv": DictConfig(parent_key="csv", type_val=dict, current_state={
            FREQUENCY_COLUMN: DictConfig(parent_key=FREQUENCY_COLUMN, type_val=str, current_state=csv_frequency_column_chem, tooltip="frequency column name").get_config(),
            INTENSITY_COLUMN: DictConfig(parent_key=INTENSITY_COLUMN, type_val=str, current_state=csv_intensity_column_chem, tooltip="intensity column name").get_config(),
            STARTING_ROW_NUMBER: DictConfig(parent_key=STARTING_ROW_NUMBER, type_val=int, current_state=csv_row_num_start_chem, tooltip="row number to start reading data").get_config(),
            ENDING_ROW_NUMBER: DictConfig(parent_key=ENDING_ROW_NUMBER, type_val=int, current_state=csv_row_num_end_chem, tooltip="row number to end reading data").get_config()
        }).get_config()}, is_required=True, select_location=False, tooltip="file type of files, csv or txt", has_options=True, options=[
        {"None": NONE_DICT_CONFIG.get_config()},
        {"csv": DictConfig(parent_key="csv", type_val=dict, current_state={
            FREQUENCY_COLUMN: DictConfig(parent_key=FREQUENCY_COLUMN, type_val=str, current_state='frequencies', tooltip="frequency column name").get_config(),
            INTENSITY_COLUMN: DictConfig(parent_key=INTENSITY_COLUMN, type_val=str, current_state='normalized_intensities', tooltip="intensity column name").get_config(),
            STARTING_ROW_NUMBER: DictConfig(parent_key=STARTING_ROW_NUMBER, type_val=int, current_state=1, tooltip="row number to start reading data").get_config(),
            ENDING_ROW_NUMBER: DictConfig(parent_key=ENDING_ROW_NUMBER, type_val=int, current_state=-1, tooltip="row number to end reading data").get_config()
        }).get_config()},
        {"txt": DictConfig(parent_key="txt", type_val=dict, current_state=dict()).get_config()}
    ]).get_config(),
    CONFIDENCE_INTERVAL: DictConfig(parent_key=CONFIDENCE_INTERVAL, type_val=float, current_state=confidence_interval, is_required=True, select_location=False, tooltip="confidence interval for chemometrics").get_config(),
    CONSTRAINED_FITTING_CONFIGS: DictConfig(parent_key=CONSTRAINED_FITTING_CONFIGS, type_val=dict, current_state={
        CONSTRAINED_FIT: DictConfig(parent_key=CONSTRAINED_FIT, type_val=bool, current_state=constrained_fit, tooltip="constrained fit").get_config(),
        EPSILON: DictConfig(parent_key=EPSILON, type_val=float, current_state=epsilon, tooltip="epsilon").get_config(),
        MAX_ITER: DictConfig(parent_key=MAX_ITER, type_val=int, current_state=max_iters, tooltip="max iters for constraint").get_config()
    }, is_required=True, select_location=False, tooltip="configs for constrained fitting").get_config(),
    NORMALIZE_FIT: DictConfig(parent_key=NORMALIZE_FIT, type_val=bool, current_state=normalize_fit, is_required=True, select_location=False, tooltip="normalize fit").get_config()
}

complete_dict_config = {
    GENERAL_PAGE : dict_config_general_1,
    ANALYSIS_PAGE : dict_config_chemometrics_1,
}

# Create output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Run the chemometrics analysis
run_config_instance = ChemometricsRunConfig(complete_dict_config)
run_config_instance.load_data()
fit_arr, confidence_intervals = run_config_instance.perform_analysis()  

# Extract species concentrations (4 species: A, B, C, Spectator)
species_a_conc = fit_arr[:, 0]  # Species A (+5 oxidation state)
species_a_ci_low = confidence_intervals[:, 0, 0]
species_a_ci_high = confidence_intervals[:, 0, 1]

species_b_conc = fit_arr[:, 1]  # Species B (+4 oxidation state)
species_b_ci_low = confidence_intervals[:, 1, 0]
species_b_ci_high = confidence_intervals[:, 1, 1]

species_c_conc = fit_arr[:, 2]  # Species C (+3 oxidation state)
species_c_ci_low = confidence_intervals[:, 2, 0]
species_c_ci_high = confidence_intervals[:, 2, 1]

species_s_conc = fit_arr[:, 3]  # Spectator species
species_s_ci_low = confidence_intervals[:, 3, 0]
species_s_ci_high = confidence_intervals[:, 3, 1]

# Create time array (800 spectra, 0.25 second intervals)
collection_dt = 0.25
t = np.linspace(0, data_per_pulse * collection_dt, data_per_pulse)

# Plot the results similar to tutorial_1
plt.figure(figsize=(10, 6))
plt.plot(t, species_a_conc, label="Species A (+5)", linestyle='-', linewidth=2)
plt.plot(t, species_b_conc, label="Species B (+4)", linestyle='-', linewidth=2)
plt.plot(t, species_c_conc, label="Species C (+3)", linestyle='-', linewidth=2)
plt.plot(t, species_s_conc, label="Spectator", linestyle='-', linewidth=2)

plt.axvline(x=100, color='red', linestyle='--', alpha=0.7, label='Switch point')
plt.xlabel("Time (s)")
plt.ylabel("Concentration (a.u.)")
plt.title("Chemometrics Fitting Results: A→B→C System")
plt.legend()
plt.tight_layout()

# Save the plot
plt.savefig(os.path.join(output_directory, "chemometrics_results.png"), dpi=300, bbox_inches='tight')
plt.show()

print(f"Analysis complete!")
print(f"Number of spectra analyzed: {len(species_a_conc)}")
print(f"Data collection time: {data_per_pulse * collection_dt} seconds")
print(f"Results saved to: {output_directory}")
print(f"Species A concentration range: {species_a_conc.min():.3f} to {species_a_conc.max():.3f}")
print(f"Species B concentration range: {species_b_conc.min():.3f} to {species_b_conc.max():.3f}")
print(f"Species C concentration range: {species_c_conc.min():.3f} to {species_c_conc.max():.3f}")
print(f"Spectator concentration range: {species_s_conc.min():.3f} to {species_s_conc.max():.3f}")

# Write the configuration to both YAML and JSON files in the example_yamls directory
yaml_output_path = os.path.join(script_dir, "example_yamls", "chemometrics_tutorial_1_config.yaml")
json_output_path = os.path.join(script_dir, "example_yamls", "chemometrics_tutorial_1_config.json")
json_clean_path = os.path.join(script_dir, "example_yamls", "chemometrics_tutorial_1_config_clean.json")

# Write YAML format (always clean)
run_config_instance.write_yaml_config(yaml_output_path)
print(f"YAML configuration saved to: {yaml_output_path}")

# Write JSON format (full DictConfig structure - GUI compatible)
run_config_instance.write_json_config(json_output_path)
print(f"JSON configuration (GUI format) saved to: {json_output_path}")

# Write clean JSON format (for comparison)
run_config_instance.write_json_clean(json_clean_path)
print(f"JSON configuration (clean format) saved to: {json_clean_path}")

print("="*80)
print("PART 2: Demonstrating YAML and JSON configuration compatibility")
print("="*80)

# Test both formats
print(f"Testing YAML configuration:")
with open(yaml_output_path, 'r') as f:
    yaml_content = f.read()
    print(f"YAML file size: {len(yaml_content)} characters")
    print("YAML content preview:")
    print(yaml_content[:300] + "..." if len(yaml_content) > 300 else yaml_content)

print(f"\nTesting JSON configuration (GUI format):")
with open(json_output_path, 'r') as f:
    json_content = f.read()
    print(f"JSON file size: {len(json_content)} characters")
    print("JSON content preview:")
    print(json_content[:300] + "..." if len(json_content) > 300 else json_content)

print(f"\nTesting JSON configuration (clean format):")
with open(json_clean_path, 'r') as f:
    json_clean_content = f.read()
    print(f"JSON clean file size: {len(json_clean_content)} characters")
    print("JSON clean content preview:")
    print(json_clean_content[:300] + "..." if len(json_clean_content) > 300 else json_clean_content)

print("\nConfiguration files generated:")
print("• YAML format - for human editing and run config loading") 
print("• JSON format (GUI) - fully compatible with existing GUI workflows")
print("• JSON format (clean) - similar to YAML, for lightweight use")
print("\nThe GUI can now load both YAML and JSON files!")

# For demonstration, test loading from both formats
print(f"\nTesting configuration round-trip...")

try:
    # Test YAML round-trip
    yaml_run_config = ChemometricsRunConfig.from_file(yaml_output_path)
    print("YAML configuration loaded successfully")
    
    # Test JSON round-trip
    json_run_config = ChemometricsRunConfig.from_file(json_output_path)
    print("JSON configuration loaded successfully")
    
    print("Both formats work perfectly with the run config system!")
    
except Exception as e:
    print(f"Error testing configuration loading: {e}")
    import traceback
    traceback.print_exc()

# For demonstration, create a duplicate analysis using the original configuration
print(f"\nRunning duplicate analysis for verification...")
duplicate_fit_arr, duplicate_confidence_intervals = run_config_instance.perform_analysis()

# Verify the results are identical
differences = np.abs(duplicate_fit_arr - fit_arr)
max_difference = np.max(differences)
print(f"Maximum difference between runs: {max_difference:.10f}")
print(f"Results are {'IDENTICAL' if max_difference < 1e-10 else 'DIFFERENT'}")

if max_difference < 1e-10:
    print("SUCCESS: Configuration system produces consistent results!")
else:
    print("WARNING: Results differ between runs")

# Extract species concentrations from duplicate analysis
duplicate_species_a = duplicate_fit_arr[:, 0]
duplicate_species_b = duplicate_fit_arr[:, 1] 
duplicate_species_c = duplicate_fit_arr[:, 2]
duplicate_species_s = duplicate_fit_arr[:, 3]

# Create comparison plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot results from original analysis
ax1.plot(t, species_a_conc, label="Species A (+5)", linestyle='-', linewidth=2)
ax1.plot(t, species_b_conc, label="Species B (+4)", linestyle='-', linewidth=2)
ax1.plot(t, species_c_conc, label="Species C (+3)", linestyle='-', linewidth=2)
ax1.plot(t, species_s_conc, label="Spectator", linestyle='-', linewidth=2)
ax1.axvline(x=100, color='red', linestyle='--', alpha=0.7, label='Switch point')
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Signal (a.u.)")
ax1.set_title("Original Analysis Results")
ax1.legend()

# Plot results from duplicate analysis 
ax2.plot(t, duplicate_species_a, label="Species A (+5)", linestyle='-', linewidth=2)
ax2.plot(t, duplicate_species_b, label="Species B (+4)", linestyle='-', linewidth=2)
ax2.plot(t, duplicate_species_c, label="Species C (+3)", linestyle='-', linewidth=2)
ax2.plot(t, duplicate_species_s, label="Spectator", linestyle='-', linewidth=2)
ax2.axvline(x=100, color='red', linestyle='--', alpha=0.7, label='Switch point')
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Signal (a.u.)")
ax2.set_title("Duplicate Analysis Results")
ax2.legend()

plt.tight_layout()
plt.savefig(os.path.join(output_directory, "chemometrics_comparison.png"), dpi=300, bbox_inches='tight')
plt.show()

# Verify the results are identical (already done above)
print(f"\nWorkflow complete! Configuration and results saved to: {output_directory}")
print(f"YAML configuration saved to: {yaml_output_path}")
print(f"JSON configuration (GUI format) saved to: {json_output_path}")
print(f"JSON configuration (clean format) saved to: {json_clean_path}")
print(f"Configuration files are now compatible with:")
print(f"The GUI: Load the main JSON file ({os.path.basename(json_output_path)}) or YAML file")
print(f"Run configs: Use any of the three files with .from_file()")
print(f"Human editing: YAML and clean JSON are most readable")

