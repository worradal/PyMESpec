#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
This is the file that handles running the Baseline correction based on a set of configurations.
'''

# Written by
# Alfred Worrad <worrada@udel.edu>,

__author__ = "Afred Worrad"
__version__ = "1.2.1"
__maintainer__ = "Alfred Worrad"
__email__ = "worrada@udel.edu"
__status__ = "Development"
__project__ = "MES wavey"
__date__ = "December 18, 2023"

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
from run_configs.base_run_config import BaseRunConfig
from core_functionality.data_processing import DataProcessingCSV
from core_functionality.baseline_correction import BaselineCorrection, ARPLS
from config_files.config_dictionaries import DictConfig, NONE_DICT_CONFIG
from config_files.config_dictionaries import (
    SPECTRUM_DIRECTORY, DATA_PER_PULSE, OUTPUT_DIRECTORY, STARTING_FRAME, SPECTRAL_COORDINATE_START, SPECTRAL_COORDINATE_END,
    ENDING_FRAME, FILE_TYPE, ANALYSIS_TYPE, CSV, ENDING_ROW_NUMBER, STARTING_ROW_NUMBER, 
    FREQUENCY_COLUMN, INTENSITY_COLUMN, BASELINE_CORRECTION_METHOD,
    SAVE_FFT_DATA, SAVE_IFFT_DATA, SAVE_PHASE_DATA, ARPLS_TAG, LAMBDA_TAG, STOP_RATIO, MAX_ITER
)
from config_files.dict_config_utils import GENERAL_PAGE, ANALYSIS_PAGE, CURRENT_STATE, OPTIONS


class BaselineRunConfig(BaseRunConfig):
    def __init__(self, dict_config) -> None:
        super().__init__(dict_config)
        
        self.phase = None
        self.spectra = None
        self.output_directory = None
        self.data_per_pulse = None
        self.starting_frame = None
        self.ending_frame = None
        self.spectral_coordinate_start = None
        self.spectral_coordinate_end = None


    def load_data(self):
        
        file_type = list(self.general_page_config[FILE_TYPE][CURRENT_STATE].keys())[0]
        self.output_directory = self.general_page_config[OUTPUT_DIRECTORY][CURRENT_STATE]
        self.data_per_pulse = self.general_page_config[DATA_PER_PULSE][CURRENT_STATE]
        self.starting_frame = self.general_page_config[STARTING_FRAME][CURRENT_STATE]
        self.ending_frame = self.general_page_config[ENDING_FRAME][CURRENT_STATE]
        self.spectral_coordinate_start = self.general_page_config[SPECTRAL_COORDINATE_START][CURRENT_STATE]
        self.spectral_coordinate_end = self.general_page_config[SPECTRAL_COORDINATE_END][CURRENT_STATE]
        if file_type == CSV:
            csv_dict = self.general_page_config[FILE_TYPE][CURRENT_STATE][CSV][CURRENT_STATE]

            data_processing = DataProcessingCSV(
                in_dir=self.general_page_config[SPECTRUM_DIRECTORY][CURRENT_STATE], 
                csv_frequency_column=csv_dict[FREQUENCY_COLUMN][CURRENT_STATE],
                csv_intensity_column=csv_dict[INTENSITY_COLUMN][CURRENT_STATE], 
                csv_row_num_start=csv_dict[STARTING_ROW_NUMBER][CURRENT_STATE], 
                csv_row_num_end=csv_dict[ENDING_ROW_NUMBER][CURRENT_STATE]
            )
            spectra = data_processing.get_spectra()
            if self.spectral_coordinate_end  == -1:
                self.spectral_coordinate_end = spectra[0].frequencies[-1]
            self.spectra = spectra.isolate_spectra_sections([(self.spectral_coordinate_start, self.spectral_coordinate_end)])
            # phase = Phase(spectra, data_per_pulse, starting_frame, ending_frame)
        else:
            raise Exception(f"The file type '{file_type}' does not have an data processing implementation yet.")
        
        

    def run_analysis(self):
        
        
        baseline_corrections_configs = self.general_page_config[BASELINE_CORRECTION_METHOD][CURRENT_STATE]
        baseline_method = list(baseline_corrections_configs.keys())[0]

        if baseline_method == ARPLS_TAG:
            arpls_instance = ARPLS()
            arpls_config = baseline_corrections_configs[baseline_method][CURRENT_STATE]
            lambda_parameter = arpls_config[LAMBDA_TAG][CURRENT_STATE]
            stop_ratio = arpls_config[STOP_RATIO][CURRENT_STATE]
            max_iters = arpls_config[MAX_ITER][CURRENT_STATE]

            #TODO: need to figure out what to do about the full output options, if it is necessary or should be removed
            full_output = False
            spectra_new = arpls_instance.baseline_corrected_spectra(
                self.spectra, lambda_parameter, stop_ratio, max_iters, full_output)

            # Plotting
            plt.plot(self.spectra[0].frequencies, self.spectra[0].intensities)
            plt.plot(spectra_new[0].frequencies, spectra_new[0].intensities)
            # plt.plot(self.spectra_target[0].frequencies, self.spectra_target[0].intensities)
            plt.legend(["Uncorrected", "Baseline Corrected"]) #, "Target"])
            plt.show()
        

       


if __name__ == "__main__":
    pass