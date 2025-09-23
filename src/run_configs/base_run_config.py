#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
This is the file that handles holds the parent class for all run configs
'''

# Written by
# Alfred Worrad <worrada@udel.edu>,

__author__ = "Afred Worrad"
__version__ = "1.2.1"
__maintainer__ = "Alfred Worrad"
__email__ = "worrada@udel.edu"
__status__ = "Development"
__project__ = "MES wavey"
__date__ = "September 10, 2025"

# built-in modules
from abc import ABC, abstractmethod
from typing import (Any, Callable, Dict, Iterable, List, Optional, Set, Tuple,
                    Union)
import os
import yaml
import json

from config_files.dict_config_utils import GENERAL_PAGE, ANALYSIS_PAGE


class BaseRunConfig(ABC):
    def __init__(self, dict_config) -> None:
        self._dict_config = dict_config
        self.general_page_config = dict_config[GENERAL_PAGE]
        self.analysis_page_config = dict_config[ANALYSIS_PAGE]

    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def perform_analysis(self):
        pass

    def _extract_current_state(self, config_dict: dict) -> dict:
        """
        Recursively extract the 'current_state' values from DictConfig objects
        to create a clean dictionary suitable for YAML serialization.
        """
        from config_files.dict_config_utils import CURRENT_STATE
        
        clean_dict = {}
        for key, value in config_dict.items():
            if isinstance(value, dict):
                if CURRENT_STATE in value:
                    # This is a DictConfig object, extract current_state
                    current_state = value[CURRENT_STATE]
                    if isinstance(current_state, dict):
                        # Recursively process nested dictionaries
                        clean_dict[key] = self._extract_current_state(current_state)
                    else:
                        clean_dict[key] = current_state
                else:
                    # Regular dictionary, process recursively
                    clean_dict[key] = self._extract_current_state(value)
            else:
                clean_dict[key] = value
        return clean_dict

    def write_yaml_config(self, output_path: str) -> None:
        """
        Writes the current configuration to a YAML file.

        Parameters:
        output_path (str): The path where the YAML file will be saved.
        """
        # Extract clean configuration values for YAML serialization
        clean_config = self._extract_current_state(self._dict_config)
        
        with open(output_path, 'w') as file:
            yaml.dump(clean_config, file, default_flow_style=False, sort_keys=False)

    def write_json_config(self, output_path: str) -> None:
        """
        Writes the current configuration to a JSON file.
        This preserves the full DictConfig structure for GUI compatibility.

        Parameters:
        output_path (str): The path where the JSON file will be saved.
        """
        from src.config_files.dict_config_utils import CustomEncoder
        
        with open(output_path, 'w') as file:
            json.dump(self._dict_config, file, indent=2, ensure_ascii=False, cls=CustomEncoder)

    def write_json_clean(self, output_path: str) -> None:
        """
        Writes a clean JSON configuration (similar to YAML format).

        Parameters:
        output_path (str): The path where the clean JSON file will be saved.
        """
        # Extract clean configuration values for JSON serialization
        clean_config = self._extract_current_state(self._dict_config)
        
        with open(output_path, 'w') as file:
            json.dump(clean_config, file, indent=2, ensure_ascii=False)

    def write_config(self, output_path: str) -> None:
        """
        Writes the current configuration to a file. Format is determined by file extension.

        Parameters:
        output_path (str): The path where the config file will be saved (.yaml/.yml or .json).
        """
        file_ext = os.path.splitext(output_path)[1].lower()
        
        if file_ext in ['.yaml', '.yml']:
            self.write_yaml_config(output_path)
        elif file_ext == '.json':
            self.write_json_config(output_path)
        else:
            raise ValueError(f"Unsupported file extension: {file_ext}. Use .yaml, .yml, or .json")

    @classmethod  
    def from_yaml(cls, yaml_path: str):
        """
        Creates a run config instance from a YAML configuration file.
        This method reconstructs a compatible configuration structure from clean YAML data.

        Parameters:
        yaml_path (str): The path to the YAML configuration file.

        Returns:
        BaseRunConfig: An instance of the appropriate run config class.
        """
        return cls.from_file(yaml_path)

    @classmethod
    def from_json(cls, json_path: str):
        """
        Creates a run config instance from a JSON configuration file.

        Parameters:
        json_path (str): The path to the JSON configuration file.

        Returns:
        BaseRunConfig: An instance of the appropriate run config class.
        """
        return cls.from_file(json_path)

    @classmethod  
    def from_file(cls, file_path: str):
        """
        Creates a run config instance from a configuration file (YAML or JSON).
        Format is automatically detected by file extension.

        Parameters:
        file_path (str): The path to the configuration file (.yaml/.yml or .json).

        Returns:
        BaseRunConfig: An instance of the appropriate run config class.
        """
        from src.config_files.dict_config_utils import CURRENT_STATE, custom_decoder
        
        file_ext = os.path.splitext(file_path)[1].lower()
        
        # Load the file based on extension
        if file_ext in ['.yaml', '.yml']:
            with open(file_path, 'r') as file:
                clean_config = yaml.safe_load(file)
        elif file_ext == '.json':
            with open(file_path, 'r') as file:
                clean_config = json.load(file, object_hook=custom_decoder)
        else:
            raise ValueError(f"Unsupported file extension: {file_ext}. Use .yaml, .yml, or .json")
        
        # For JSON files loaded with custom_decoder, the structure might already be correct
        if file_ext == '.json' and isinstance(clean_config, dict):
            # Check if this looks like it already has the DictConfig structure
            if any(isinstance(v, dict) and CURRENT_STATE in v for page in clean_config.values() if isinstance(page, dict) for v in page.values()):
                # Already has the right structure
                return cls(clean_config)
        
        # Reconstruct the DictConfig structure from the clean data
        def create_dict_config_structure(config_data):
            reconstructed = {}
            for key, value in config_data.items():
                if isinstance(value, dict):
                    # For complex nested structures, we need to add current_state at the right level
                    # Check if this dict contains only simple values or has nested dicts
                    has_nested_dicts = any(isinstance(v, dict) for v in value.values())
                    
                    if has_nested_dicts:
                        # This is a complex structure like "File Type" with nested configs
                        # We need to create the structure: key -> current_state -> nested_dict
                        nested_structure = {}
                        for sub_key, sub_value in value.items():
                            if isinstance(sub_value, dict):
                                # Add current_state wrapper for each nested dict
                                nested_structure[sub_key] = {
                                    CURRENT_STATE: sub_value
                                }
                            else:
                                nested_structure[sub_key] = sub_value
                        
                        reconstructed[key] = {
                            CURRENT_STATE: nested_structure
                        }
                    else:
                        # Simple dict, treat the whole thing as current_state
                        reconstructed[key] = {
                            CURRENT_STATE: value
                        }
                else:
                    # Simple value
                    reconstructed[key] = {
                        CURRENT_STATE: value
                    }
            return reconstructed
        
        reconstructed_config = create_dict_config_structure(clean_config)
        return cls(reconstructed_config)
