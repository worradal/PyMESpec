#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
This is the file that handles running the Chemometrics analysis based on a set of configurations.
'''

# Written by
# Alfred Worrad <worrada@udel.edu>,

__author__ = "Afred Worrad"
__version__ = "1.2.1"
__maintainer__ = "Alfred Worrad"
__email__ = "worrada@udel.edu"
__status__ = "Development"
__project__ = "MES wavey"
__date__ = "October 1, 2024"

# built-in modules
from abc import ABC, abstractmethod
from typing import (Any, Callable, Dict, Iterable, List, Optional, Set, Tuple,
                    Union)
from itertools import count
import sys
import os
# Add the project root to sys.path to allow local imports when running this file directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# third-party modules
import matplotlib.pyplot as plt
import numpy as np

# project modules
from src.core_functionality.data_processing import DataProcessingCSV
from src.core_functionality.chemometrics import Chemometrics, FIT_PARAMETERS, CONFIDENCE_INTERVALS
from src.run_configs.base_run_config import BaseRunConfig
from src.config_files.config_dictionaries import  DictConfig, NONE_DICT_CONFIG
from src.config_files.config_dictionaries import (
    SPECTRUM_DIRECTORY, DATA_PER_PULSE, OUTPUT_DIRECTORY, STARTING_FRAME, SPECTRAL_COORDINATE_START, SPECTRAL_COORDINATE_END,
    ENDING_FRAME, FILE_TYPE, ANALYSIS_TYPE, CSV, ENDING_ROW_NUMBER, STARTING_ROW_NUMBER, 
    FREQUENCY_COLUMN, INTENSITY_COLUMN, BASELINE_CORRECTION_METHOD, CHEMOMETRICS_DIRECTORY,
    CHEMOMETRICS_FILE_TYPE, CHEMOMETRICS_SAMPLE_STANDARDS_FILE_LIST, CONFIDENCE_INTERVAL, 
    CONSTRAINED_FITTING_CONFIGS, NORMALIZE_FIT, MAX_ITER, EPSILON, CONSTRAINED_FIT
    
)
from config_files.dict_config_utils import GENERAL_PAGE, ANALYSIS_PAGE, CURRENT_STATE, OPTIONS


class ChemometricsRunConfig(BaseRunConfig):
    def __init__(self, dict_config) -> None:
        super().__init__(dict_config)
        
        
        self.chemometrics_file_type = None
        self.chemometrics_path = None
        self.sample_standards_file_list = None 
        self.csv_frequency_column_chem = None
        self.csv_intensity_column_chem = None
        self.chemometrics_spectral_coordinate_start = None
        self.chemometrics_spectral_coordinate_end = None
        self.csv_row_num_start_chem = None
        self.csv_row_num_end_chem = None
        self.spectra = None
        self.ref_spectra = None
        self.output_directory = None
        self.data_per_pulse = None
        self.starting_frame = None
        self.ending_frame = None
        self.alpha = None

    def load_data(self):
        
        file_type = list(self.general_page_config[FILE_TYPE][CURRENT_STATE].keys())[0]
        self.output_directory = self.general_page_config[OUTPUT_DIRECTORY][CURRENT_STATE]
        self.data_per_pulse = self.general_page_config[DATA_PER_PULSE][CURRENT_STATE]
        self.starting_frame = self.general_page_config[STARTING_FRAME][CURRENT_STATE]
        self.ending_frame = self.general_page_config[ENDING_FRAME][CURRENT_STATE]
        
        self.chemometrics_spectral_coordinate_start = self.general_page_config[SPECTRAL_COORDINATE_START][CURRENT_STATE]
        self.chemometrics_spectral_coordinate_end = self.general_page_config[SPECTRAL_COORDINATE_END][CURRENT_STATE]

        
        if file_type == CSV:
            csv_dict = self.general_page_config[FILE_TYPE][CURRENT_STATE][CSV][CURRENT_STATE]
            print('DEBUG: csv_dict keys:', list(csv_dict.keys()))
            if FREQUENCY_COLUMN not in csv_dict:
                print('ERROR: FREQUENCY_COLUMN not in csv_dict! Available keys:', list(csv_dict.keys()))
                raise KeyError(FREQUENCY_COLUMN)

            data_processing = DataProcessingCSV(
                in_dir=self.general_page_config[SPECTRUM_DIRECTORY][CURRENT_STATE], 
                csv_frequency_column=csv_dict[FREQUENCY_COLUMN][CURRENT_STATE],
                csv_intensity_column=csv_dict[INTENSITY_COLUMN][CURRENT_STATE], 
                csv_row_num_start=csv_dict[STARTING_ROW_NUMBER][CURRENT_STATE], 
                csv_row_num_end=csv_dict[ENDING_ROW_NUMBER][CURRENT_STATE]
            )
            spectra = data_processing.get_spectra()
            if self.chemometrics_spectral_coordinate_end  == -1:
                self.chemometrics_spectral_coordinate_end  = spectra[0].frequencies[-1]
            self.spectra = spectra.isolate_spectra_sections(
                frequency_sections=[(self.chemometrics_spectral_coordinate_start, 
                                   self.chemometrics_spectral_coordinate_end) ]
            )

            # phase = Phase(spectra, data_per_pulse, starting_frame, ending_frame)
        else:
            raise Exception(f"The file type '{file_type}' does not have an data processing implementation yet.")
        
        self.chemometrics_file_type = list(self.analysis_page_config[CHEMOMETRICS_FILE_TYPE][CURRENT_STATE].keys())[0]
        
        if self.chemometrics_file_type == CSV:
            self.sample_standards_file_list = self.analysis_page_config[CHEMOMETRICS_SAMPLE_STANDARDS_FILE_LIST][CURRENT_STATE]
            self.chemometrics_path = self.analysis_page_config[CHEMOMETRICS_DIRECTORY][CURRENT_STATE]
            chemometrics_csv_dict = self.analysis_page_config[CHEMOMETRICS_FILE_TYPE][CURRENT_STATE][CSV][CURRENT_STATE]
            self.csv_frequency_column_chem = chemometrics_csv_dict[FREQUENCY_COLUMN][CURRENT_STATE]
            self.csv_intensity_column_chem = chemometrics_csv_dict[INTENSITY_COLUMN][CURRENT_STATE]
            self.csv_row_num_start_chem = chemometrics_csv_dict[STARTING_ROW_NUMBER][CURRENT_STATE]
            self.csv_row_num_end_chem = chemometrics_csv_dict[ENDING_ROW_NUMBER][CURRENT_STATE]

            ref_file_data_processing = DataProcessingCSV(
            in_dir=self.chemometrics_path, specific_files=self.sample_standards_file_list, csv_frequency_column=self.csv_frequency_column_chem,
            csv_intensity_column=self.csv_intensity_column_chem, csv_row_num_start=self.csv_row_num_start_chem, csv_row_num_end=self.csv_row_num_end_chem)
            files_in_order = ref_file_data_processing.all_files
            self.ref_spectra = ref_file_data_processing.get_spectra()
            if self.chemometrics_spectral_coordinate_end == -1:
                self.chemometrics_spectral_coordinate_end = self.ref_spectra[0].frequencies[-1]
            self.ref_spectra = self.ref_spectra.isolate_spectra_sections(
                frequency_sections=[(self.chemometrics_spectral_coordinate_start, 
                                   self.chemometrics_spectral_coordinate_end) ]
            )
            # return self.spectra, self.ref_spectra
        else:
            raise Exception(f"The file type '{self.chemometrics_file_type}' does not have an data processing implementation yet.")


    def perform_analysis(self):
        
        alpha = 1 - self.analysis_page_config[CONFIDENCE_INTERVAL][CURRENT_STATE]
        constrained_fit_configs = self.analysis_page_config[CONSTRAINED_FITTING_CONFIGS][CURRENT_STATE]
        constrained_fit = constrained_fit_configs[CONSTRAINED_FIT][CURRENT_STATE]
        max_iters = constrained_fit_configs[MAX_ITER][CURRENT_STATE]
        epsilon = constrained_fit_configs[EPSILON][CURRENT_STATE]
        normalize_fit = self.analysis_page_config[NORMALIZE_FIT][CURRENT_STATE]
        fitting_file = os.path.join(self.output_directory, 'fitting_list.csv')

        # We may need to include this in the DictConfig
        if alpha != 0:
            save_file_include_confidence_intervals = True
        # save_file_include_confidence_intervals = self.analysis_page_config[][CURRENT_STATE]

        # phase = Phase(self.spectra, self.data_per_pulse, self.starting_frame, self.ending_frame)

        # FFT
        chemometrics = Chemometrics(reference_spectra=self.ref_spectra)
        fitting_list = chemometrics.compute_spectra_fitting(
            self.spectra, 
            alpha=alpha, 
            constrained_fit=constrained_fit, 
            epsilon=epsilon, 
            max_iter=max_iters, 
            normalize_fit=normalize_fit,
            save_csv_file=fitting_file, 
            save_file_include_confidence_intervals=save_file_include_confidence_intervals
        )

        
        # Extract fit parameters and confidence intervals from the new dictionary format
        fit_arr = np.array([result[FIT_PARAMETERS] for result in fitting_list])
        confidence_intervals = np.array([result[CONFIDENCE_INTERVALS] for result in fitting_list])
        

        return fit_arr, confidence_intervals



        


if __name__ == "__main__":
    pass