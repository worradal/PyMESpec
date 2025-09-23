#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
This is the file that handles running the Phase analysis based on a set of configurations.
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
from src.run_configs.read_yaml_config import ReadYAMLConfig
from src.run_configs.base_run_config import BaseRunConfig
from src.core_functionality.data_processing import DataProcessingCSV
from src.core_functionality.phase import Phase
from src.config_files.config_dictionaries import DictConfig, NONE_DICT_CONFIG
from src.config_files.config_dictionaries import (
    SPECTRUM_DIRECTORY, DATA_PER_PULSE, OUTPUT_DIRECTORY, STARTING_FRAME, SPECTRAL_COORDINATE_START, SPECTRAL_COORDINATE_END,
    ENDING_FRAME, FILE_TYPE, ANALYSIS_TYPE, CSV, ENDING_ROW_NUMBER, STARTING_ROW_NUMBER, 
    FREQUENCY_COLUMN, INTENSITY_COLUMN, BASELINE_CORRECTION_METHOD, LIST_OF_HARMONICS,
    SAVE_FFT_DATA, SAVE_IFFT_DATA, SAVE_PHASE_DATA
)
from src.config_files.dict_config_utils import GENERAL_PAGE, ANALYSIS_PAGE, CURRENT_STATE


#TODO: Typing and get methods
class PhaseRunConfig(BaseRunConfig):
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
            if self.spectral_coordinate_end == -1:
                self.spectral_coordinate_end = spectra[0].frequencies[-1]
            self.spectra = spectra.isolate_spectra_sections([(self.spectral_coordinate_start, self.spectral_coordinate_end)])
            # phase = Phase(spectra, data_per_pulse, starting_frame, ending_frame)
        else:
            raise Exception(f"The file type '{file_type}' does not have a data processing implementation yet. Make sure you use the approved file type 'csv'")
        
        # return spectra

    def perform_analysis(self):
        
        
        list_of_harmonics = self.analysis_page_config[LIST_OF_HARMONICS][CURRENT_STATE]
        to_save_phase = self.analysis_page_config[SAVE_PHASE_DATA][CURRENT_STATE]
        to_save_fft = self.analysis_page_config[SAVE_FFT_DATA][CURRENT_STATE]
        to_save_ifft = self.analysis_page_config[SAVE_IFFT_DATA][CURRENT_STATE]

        self.phase = Phase(self.spectra, self.data_per_pulse, self.starting_frame, self.ending_frame)

        # FFT
        self.phase.fourier_transform_on_avg_data()
        self.phase.weight(list_of_harmonics)

        # Saves the data in real space after weighting in Fourier space
        if to_save_ifft:
            iff_file = os.path.join(self.output_directory, 'ifft.csv')
            self.phase.save_ifft_to(iff_file)

        # Saves the data in Fourier space
        if to_save_fft:
            fft_file = os.path.join(self.output_directory, 'fft.csv')
            self.phase.save_fft_to(fft_file)

        # Saves phase shift data
        if to_save_phase:
            phase_file = os.path.join(self.output_directory, 'phase.csv')
            self.phase.save_phase_to(phase_file)
        # return phase


if __name__ == "__main__":
    pass