# Utility: Convert a clean config (plain dict) to GUI format (nested dicts with CURRENT_STATE keys)
def convert_clean_config_to_gui_format(config):
    """
    Recursively wrap all leaf values in a dict with the CURRENT_STATE key.
    If a value is already a dict and has CURRENT_STATE, leave it as is.
    """
    # Import constants for mapping
    from src.config_files.config_dictionaries import (
        SPECTRUM_DIRECTORY, DATA_PER_PULSE, OUTPUT_DIRECTORY, PROJECT_NAME, STARTING_FRAME, ENDING_FRAME,
        SPECTRAL_COORDINATE_START, SPECTRAL_COORDINATE_END, FILE_TYPE, BASELINE_CORRECTION_METHOD,
        ANALYSIS_TYPE, CHEMOMETRICS_DIRECTORY, CHEMOMETRICS_SAMPLE_STANDARDS_FILE_LIST, CHEMOMETRICS_FILE_TYPE,
        CONFIDENCE_INTERVAL, CONSTRAINED_FITTING_CONFIGS, NORMALIZE_FIT
    )
    # Mapping from user-friendly keys to code constant values
    key_map = {
        # General_Page
        "Spectrum Directory": SPECTRUM_DIRECTORY,
        "Data per Pulse": DATA_PER_PULSE,
        "Output Directory": OUTPUT_DIRECTORY,
        "Project Name": PROJECT_NAME,
        "Starting Frame": STARTING_FRAME,
        "Ending Frame": ENDING_FRAME,
        "Spectral coordinate start": SPECTRAL_COORDINATE_START,
        "Spectral coordinate end": SPECTRAL_COORDINATE_END,
        "File Type": FILE_TYPE,
        "Baseline Correction Method": BASELINE_CORRECTION_METHOD,
        # Analysis_Page
        "Analysis type": ANALYSIS_TYPE,
        "Chemometrics Directory": CHEMOMETRICS_DIRECTORY,
        "Chemometrics sample standards file list": CHEMOMETRICS_SAMPLE_STANDARDS_FILE_LIST,
        "Chemometrics file type": CHEMOMETRICS_FILE_TYPE,
        "Confidence interval": CONFIDENCE_INTERVAL,
        "Constrained fitting configs": CONSTRAINED_FITTING_CONFIGS,
        "normalize fit": NORMALIZE_FIT,
    }

    def _convert(obj, is_top_level=False, parent_key=None):
        print(f"\nDEBUG: Converting object: {obj}")
        print(f"DEBUG: is_top_level={is_top_level}, parent_key={parent_key}")
        if isinstance(obj, dict):
            # If this is already a config dict with CURRENT_STATE, return as is
            if set(obj.keys()) == {CURRENT_STATE}:
                print(f"DEBUG: Already has CURRENT_STATE, returning as is: {obj}")
                return obj
            # If this is an option dict (all keys are strings, all values are dicts or empty dicts), wrap in CURRENT_STATE
            if obj and all(isinstance(v, dict) for v in obj.values()):
                # Option dict (e.g. File Type -> {csv: {...}}).
                # Convert each option value without adding an extra CURRENT_STATE wrapper
                # around the already-converted value to avoid double-nesting.
                wrapped = {key_map.get(k, k): _convert(v, parent_key=k) for k, v in obj.items()}
                if not is_top_level:
                    return {CURRENT_STATE: wrapped}
                else:
                    return wrapped
            # If this is the top-level General_Page or Analysis_Page, do not wrap, just map keys and convert contents
            if parent_key in ("General_Page", "Analysis_Page") or is_top_level:
                return {key_map.get(k, k): _convert(v, parent_key=k) for k, v in obj.items()}
            # For all other dicts (not leaves, not option dicts), wrap in CURRENT_STATE and map keys recursively
            wrapped = {key_map.get(k, k): _convert(v, parent_key=k) for k, v in obj.items()}
            return {CURRENT_STATE: wrapped}
        elif isinstance(obj, list):
            # If this is a list of primitives, wrap the whole list in CURRENT_STATE
            if all(isinstance(item, (str, int, float, bool, type(None))) for item in obj):
                return {CURRENT_STATE: obj}
            # Otherwise, recursively convert each item
            return [_convert(item) for item in obj]
        else:
            # Leaf value: wrap in CURRENT_STATE
            return {CURRENT_STATE: obj}
    # Map the top-level General_Page and Analysis_Page keys, but not their names
    return {k: _convert(v, is_top_level=True, parent_key=k) for k, v in config.items()}
from typing import List, Dict, Any, Union, get_origin, get_args, _GenericAlias
import json
import sys
import os
# Add the project root to sys.path to allow local imports when running this file directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ANALYSIS_PAGE = "Analysis_Page"
GENERAL_PAGE = "General_Page"

PARENT_KEY = "parent_key"
TYPE_VALUE = "type_val"
CURRENT_STATE = "current_state"
HAS_OPTIONS = "has_options"
OPTIONS = "options"
IS_REQUIRED = "is_required"
SELECT_LOCATION = "select_location"
TOOLTIP = "tooltip"
EDITABLE = "editable"


class DictConfig():
    def __init__(
            self,
            parent_key: str,
            type_val: type = None,
            current_state=None,
            has_options: bool = False,
            options: list = [],
            is_required: bool = False,
            select_location: bool = False,
            tooltip: str = "",
            editable: bool = True,
    ):
        if type_val is not None:
            if get_origin(type_val) == list and get_args(type_val) == (str,):
                if not isinstance(current_state, list) or not all(isinstance(item, str) for item in current_state):
                    raise ValueError(f"Current value {current_state} is not of type List[str]")
            elif get_origin(type_val) == list and get_args(type_val) == (int,):
                if not isinstance(current_state, list) or not all(isinstance(item, int) for item in current_state):
                    raise ValueError(f"Current value {current_state} is not of type List[str]")

            elif type(current_state) != type_val:
                raise ValueError(
                    f"Current value {current_state} is not of type {type_val}")
        self._dictionary_config = {
            PARENT_KEY: parent_key,
            TYPE_VALUE: type_val,
            CURRENT_STATE: current_state,
            HAS_OPTIONS: has_options,
            OPTIONS: options,
            IS_REQUIRED: is_required,
            SELECT_LOCATION: select_location,
            TOOLTIP: tooltip,
            EDITABLE: editable,
        }

    def get_config(self):
        return self._dictionary_config
    
    # def current_state_of_key(self, key):
    #     return self._dictionary_config[key]["current_state"]
    
def current_state_of_key(dictionary:dict, key:str):
    return dictionary[key]["current_"]

class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, type):
            # print(obj.__name__)
            return {'__type__': obj.__name__}  # Store the type's name
        if isinstance(obj, _GenericAlias):  # Handle typing generics
            return {'__type__': str(obj)}   # Convert the type to a string representation

        return super().default(obj)

# Define a safe mapping of recognized base types
type_mapping = {
    'int': int,
    'float': float,
    'str': str,
    'list': list,
    'dict': dict,
    'bool': bool,
    'NoneType': type(None),
}

# Define a safe mapping for generics
generic_type_mapping = {
    'List': List,
    'Dict': Dict,
    # Add more generic types as needed
}

def custom_decoder(obj):
    if '__type__' in obj:
        type_name = obj['__type__']
        
        # Check if it matches a basic type
        if type_name in type_mapping:
            return type_mapping[type_name]
        
        # Check if it matches a generic type (e.g., List[str])
        if type_name.startswith('List['):  # Handle List generics
            # Extract the inner type (e.g., str in List[str])
            inner_type_name = type_name[type_name.index('[') + 1:type_name.index(']')]
            
            # Reconstruct the List type with the inner type
            if inner_type_name in type_mapping:
                inner_type = type_mapping[inner_type_name]
                return List[inner_type]  # Return the reconstructed List type
            
        elif type_name.startswith('Dict['):  # Handle Dict generics
            # Extract the key and value types for Dict
            key_value_type = type_name[type_name.index('[') + 1:type_name.index(']')]
            key_type_name, value_type_name = key_value_type.split(', ')
            
            # Reconstruct the Dict type with the key and value types
            if key_type_name in type_mapping and value_type_name in type_mapping:
                key_type = type_mapping[key_type_name]
                value_type = type_mapping[value_type_name]
                return Dict[key_type, value_type]  # Return the reconstructed Dict type
        
        # If the type is not recognized, return the object unchanged
    return obj


def convert_to_dictconfig(saved_config: Dict[str, Any]) -> Dict[str, DictConfig]:
    def recursive_convert(config: Dict[str, Any]) -> Union[DictConfig, Dict[str, DictConfig]]:
        parent_key = config.get('parent_key')
        type_val = config.get('type_val')
        current_state = config.get('current_state')
        has_options = config.get('has_options', False)
        options = config.get('options', [])
        is_required = config.get('is_required', False)
        select_location = config.get('select_location', False)
        tooltip = config.get('tooltip', '')
        editable = config.get('editable', True)

        if has_options and isinstance(options, list):
            converted_options = []
            for option in options:
                converted_option = {k: recursive_convert(v) for k, v in option.items()}
                converted_options.append(converted_option)
            options = converted_options

        return DictConfig(
            parent_key=parent_key,
            type_val=type_val,
            current_state=current_state,
            has_options=has_options,
            options=options,
            is_required=is_required,
            select_location=select_location,
            tooltip=tooltip,
            editable=editable
        )

    return {key: recursive_convert(value) for key, value in saved_config.items()}

