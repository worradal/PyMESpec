#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
This is the file that handles running the Rate Date analysis based on a set of configurations.
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

# third-party modules
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# project modules
from src.run_configs.read_yaml_config import ReadYAMLConfig
from src.run_configs.base_run_config import BaseRunConfig
from src.core_functionality.data_processing import DataProcessingCSV
from src.core_functionality.rate_data import RateData, fit_exp
from src.config_files.config_dictionaries import DictConfig, NONE_DICT_CONFIG
from src.config_files.config_dictionaries import (
    SPECTRUM_DIRECTORY, DATA_PER_PULSE, OUTPUT_DIRECTORY, STARTING_FRAME, SPECTRAL_COORDINATE_START, SPECTRAL_COORDINATE_END,
    ENDING_FRAME, FILE_TYPE, ANALYSIS_TYPE, CSV, ENDING_ROW_NUMBER, STARTING_ROW_NUMBER, 
    FREQUENCY_COLUMN, INTENSITY_COLUMN, BASELINE_CORRECTION_METHOD, AVERAGE_RATE,
    FITTING_FUNCTION, DESIRED_FREQUENCY_FOR_RATE_DATA
)
from config_files.dict_config_utils import GENERAL_PAGE, ANALYSIS_PAGE, CURRENT_STATE


class RateDataRunConfig(BaseRunConfig):
    def __init__(self, dict_config) -> None:
        super().__init__(dict_config)

        self.spectra = None
        self.output_directory = None
        self.data_per_pulse = None
        self.starting_frame = None
        self.ending_frame = None
        self.rate_data = None
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
            if self.spectral_coordinate_end == -1:
                self.spectral_coordinate_end = spectra[0].frequencies[-1]
            self.spectra = spectra.isolate_spectra_sections([(self.spectral_coordinate_start, self.spectral_coordinate_end)])

            # phase = Phase(spectra, data_per_pulse, starting_frame, ending_frame)
        else:
            raise Exception(f"The file type '{file_type}' does not have an data processing implementation yet.")
        
        
    def perform_analysis(self):
        
        average_rate = self.analysis_page_config[AVERAGE_RATE][CURRENT_STATE]
        fitting_function = self.analysis_page_config[FITTING_FUNCTION][CURRENT_STATE]
        desired_frequency = self.analysis_page_config[DESIRED_FREQUENCY_FOR_RATE_DATA][CURRENT_STATE]
        if fitting_function == "exponential":
            fit = fit_exp
        else:
            raise Exception(f"The fitting function '{fitting_function}' is not implemented. Available options are: 'exponential'")
        
        self.rate_data = RateData(self.spectra)

        if desired_frequency == -1:
            all_rates = self.rate_data.compute_all_rate_data_all_freq(
                self.data_per_pulse, self.starting_frame, self.ending_frame, fitting_function=fit
            )
            print("All exponential rate constants extracted from the data: ", all_rates)

        elif desired_frequency != -1:
            rates = self.rate_data.compute_average_rate_data(
                desired_frequency, self.data_per_pulse, self.starting_frame, self.ending_frame, 
                fitting_function=fit,
                )
            all_rates = self.rate_data.compute_all_rate_data_single_freq(
                desired_frequency, self.data_per_pulse, self.starting_frame, self.ending_frame,
                  fitting_function=fit, 
                  )
            # this should be output to a file
            print("Average exponential rate constant extracted from the data: ", rates)
            print("All exponential rate constants extracted from the data: ", all_rates)

        
        collected_rates_csv = os.path.join(self.output_directory, 'collected_rates.csv')


        rate_df = pd.DataFrame(
            {
                "Rates": all_rates
            }
        )
        rate_df.to_csv(collected_rates_csv)
       


if __name__ == "__main__":
    pass
