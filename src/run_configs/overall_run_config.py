#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
This is the file that handles running all the configs based on what is provided
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


# third-party modules
import matplotlib.pyplot as plt
import numpy as np
import json

# project modules
from src.run_configs.read_yaml_config import ReadYAMLConfig
from src.run_configs.base_run_config import BaseRunConfig
from src.core_functionality.data_processing import DataProcessingCSV
from src.core_functionality.baseline_correction import BaselineCorrection, ARPLS
from src.config_files.config_dictionaries import DictConfig, NONE_DICT_CONFIG
from src.config_files.config_dictionaries import (
    ANALYSIS_TYPE, PHASE_ANALYSIS_TYPE, RATE_DATA_ANALYSIS_TYPE, CHEMOMETRICS_ANALYSIS_TYPE, BASELINE_CORRECTION_METHOD, NONE_STR, OUTPUT_DIRECTORY, PROJECT_NAME
)
from src.config_files.dict_config_utils import GENERAL_PAGE, ANALYSIS_PAGE, CURRENT_STATE, OPTIONS, CustomEncoder

from src.run_configs.chemometrics_run_config import ChemometricsRunConfig
from src.run_configs.baseline_correction_run_config import BaselineRunConfig
from src.run_configs.phase_run_config import PhaseRunConfig
from src.run_configs.rate_data_run_config import RateDataRunConfig

class OverallRunConfig(BaseRunConfig):
    def __init__(self, dict_config) -> None:
        super().__init__(dict_config)
        
        
        config_type = self.analysis_page_config[ANALYSIS_TYPE][CURRENT_STATE]
        self.run_config = None
        self.baseline_run_config = None
        self.to_perform_baseline_correction = False
        baseline_corrections_configs = self.general_page_config[BASELINE_CORRECTION_METHOD][CURRENT_STATE]
        baseline_method = list(baseline_corrections_configs.keys())[0]
        self.output_directory = self.general_page_config[OUTPUT_DIRECTORY][CURRENT_STATE]
        self.project_file = self.general_page_config[PROJECT_NAME][CURRENT_STATE]
        
        if baseline_method != NONE_STR:
            self.to_perform_baseline_correction = True
            # BaselineRunConfig may be missing the concrete `perform_analysis` method
            # in some versions; attempt to instantiate and fall back to an adapter
            try:
                self.baseline_run_config = BaselineRunConfig(dict_config=dict_config)
            except TypeError as e:
                # If it's due to abstractmethod not implemented, create a small adapter
                msg = str(e)
                if 'abstract' in msg or 'perform_analysis' in msg:
                    class _BaselineAdapter(BaselineRunConfig):
                        def perform_analysis(self):
                            # Delegate to existing run_analysis if available
                            if hasattr(self, 'run_analysis'):
                                return self.run_analysis()
                            raise NotImplementedError('Baseline adapter has no run_analysis implementation')

                    self.baseline_run_config = _BaselineAdapter(dict_config=dict_config)
                else:
                    # Re-raise unexpected TypeErrors
                    raise
        

        if config_type == PHASE_ANALYSIS_TYPE:
            self.run_config = PhaseRunConfig(dict_config=dict_config)
        elif config_type == CHEMOMETRICS_ANALYSIS_TYPE:
            self.run_config = ChemometricsRunConfig(dict_config=dict_config)
        elif config_type == RATE_DATA_ANALYSIS_TYPE:
            self.run_config = RateDataRunConfig(dict_config=dict_config)
        
    def get_output_directory(self):
        return self.output_directory

    def load_data(self):

        #TODO: Lets see if we can change the way baseline correction is performed so that we dont do 
        # redundant reading / write / reading files
        if self.to_perform_baseline_correction:
            self.baseline_run_config.load_data()
            self.baseline_run_config.perform_analysis()
        self.run_config.load_data()              

    def perform_analysis(self):
        
        self.run_config.perform_analysis()
    
    def save_config(self):
        #TODO: If there is no output directory, we should handle this far before this occurs
        if self.project_file != "":
            file_location = os.path.join(self.output_directory, self.project_file)
        else:
            file_location = os.path.join(self.output_directory, "project.json")
        
        if ".json" not in file_location:
            file_location += ".json"

        with open(file_location, 'w') as json_file:
            json.dump(self._dict_config, json_file, cls=CustomEncoder, indent=4)
        print(f"Configuration saved to {file_location}")




if __name__ == "__main__":
    pass