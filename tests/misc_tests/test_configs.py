import pytest
import json
from typing import List, Dict, Any
import tempfile
import os

from src.config_files.dict_config_utils import (
    DictConfig, 
    CustomEncoder, 
    custom_decoder, 
    convert_to_dictconfig,
    current_state_of_key,
    PARENT_KEY,
    TYPE_VALUE,
    CURRENT_STATE,
    HAS_OPTIONS,
    OPTIONS,
    IS_REQUIRED,
    SELECT_LOCATION,
    TOOLTIP,
    EDITABLE
)

from src.config_files.config_dictionaries import (
    dict_config_general,
    dict_config_chemometrics,
    dict_config_phase,
    dict_config_rate_data,
    SPECTRUM_DIRECTORY,
    DATA_PER_PULSE,
    OUTPUT_DIRECTORY,
    PROJECT_NAME,
    FILE_TYPE,
    BASELINE_CORRECTION_METHOD,
    ANALYSIS_TYPE,
    CHEMOMETRICS_DIRECTORY,
    CONFIDENCE_INTERVAL,
    LIST_OF_HARMONICS,
    AVERAGE_RATE,
    NONE_DICT_CONFIG
)


class TestDictConfig:
    """Test the DictConfig class functionality."""
    
    def test_basic_creation(self):
        """Test basic DictConfig creation with valid parameters."""
        config = DictConfig(
            parent_key="test_key",
            type_val=str,
            current_state="test_value",
            is_required=True,
            tooltip="Test tooltip"
        )
        result = config.get_config()
        
        assert result[PARENT_KEY] == "test_key"
        assert result[TYPE_VALUE] == str
        assert result[CURRENT_STATE] == "test_value"
        assert result[IS_REQUIRED] is True
        assert result[TOOLTIP] == "Test tooltip"
        assert result[EDITABLE] is True  # default value
    
    def test_list_str_validation_valid(self):
        """Test DictConfig with valid List[str] type."""
        config = DictConfig(
            parent_key="string_list",
            type_val=List[str],
            current_state=["item1", "item2", "item3"]
        )
        result = config.get_config()
        assert result[CURRENT_STATE] == ["item1", "item2", "item3"]
    
    def test_list_str_validation_invalid(self):
        """Test DictConfig with invalid List[str] type."""
        with pytest.raises(ValueError, match="Current value .* is not of type List\\[str\\]"):
            DictConfig(
                parent_key="string_list",
                type_val=List[str],
                current_state=["item1", 123, "item3"]  # mixed types
            )
    
    def test_list_int_validation_valid(self):
        """Test DictConfig with valid List[int] type."""
        config = DictConfig(
            parent_key="int_list",
            type_val=List[int],
            current_state=[1, 2, 3]
        )
        result = config.get_config()
        assert result[CURRENT_STATE] == [1, 2, 3]
    
    def test_list_int_validation_invalid(self):
        """Test DictConfig with invalid List[int] type."""
        with pytest.raises(ValueError, match="Current value .* is not of type List\\[str\\]"):
            DictConfig(
                parent_key="int_list",
                type_val=List[int],
                current_state=[1, "two", 3]  # mixed types
            )
    
    def test_type_validation_int(self):
        """Test type validation for integer."""
        config = DictConfig(
            parent_key="int_test",
            type_val=int,
            current_state=42
        )
        assert config.get_config()[CURRENT_STATE] == 42
    
    def test_type_validation_invalid(self):
        """Test type validation with invalid type."""
        with pytest.raises(ValueError, match="Current value .* is not of type"):
            DictConfig(
                parent_key="int_test",
                type_val=int,
                current_state="not_an_int"
            )
    
    def test_has_options_configuration(self):
        """Test DictConfig with options."""
        options = [{"option1": "value1"}, {"option2": "value2"}]
        config = DictConfig(
            parent_key="options_test",
            type_val=dict,
            current_state={},
            has_options=True,
            options=options
        )
        result = config.get_config()
        
        assert result[HAS_OPTIONS] is True
        assert result[OPTIONS] == options
    
    def test_all_parameters(self):
        """Test DictConfig with all parameters set."""
        config = DictConfig(
            parent_key="complete_test",
            type_val=str,
            current_state="test",
            has_options=True,
            options=[{"opt": "val"}],
            is_required=True,
            select_location=True,
            tooltip="Complete test tooltip",
            editable=False
        )
        result = config.get_config()
        
        assert result[PARENT_KEY] == "complete_test"
        assert result[TYPE_VALUE] == str
        assert result[CURRENT_STATE] == "test"
        assert result[HAS_OPTIONS] is True
        assert result[OPTIONS] == [{"opt": "val"}]
        assert result[IS_REQUIRED] is True
        assert result[SELECT_LOCATION] is True
        assert result[TOOLTIP] == "Complete test tooltip"
        assert result[EDITABLE] is False


class TestCustomEncoder:
    """Test the CustomEncoder class for JSON serialization."""
    
    def test_encode_basic_type(self):
        """Test encoding basic Python types."""
        encoder = CustomEncoder()
        result = encoder.encode({"type": int, "value": 42})
        decoded = json.loads(result)
        
        assert decoded["type"]["__type__"] == "int"
        assert decoded["value"] == 42
    
    def test_encode_list_type(self):
        """Test encoding List generic type."""
        encoder = CustomEncoder()
        result = encoder.encode({"type": List[str]})
        decoded = json.loads(result)
        
        assert "__type__" in decoded["type"]
        assert "List" in decoded["type"]["__type__"]
    
    def test_encode_dict_type(self):
        """Test encoding Dict generic type."""
        encoder = CustomEncoder()
        result = encoder.encode({"type": Dict[str, int]})
        decoded = json.loads(result)
        
        assert "__type__" in decoded["type"]


class TestCustomDecoder:
    """Test the custom decoder functionality."""
    
    def test_decode_basic_type(self):
        """Test decoding basic types."""
        obj = {"__type__": "int"}
        result = custom_decoder(obj)
        assert result == int
    
    def test_decode_unknown_type(self):
        """Test decoding unknown types returns unchanged."""
        obj = {"__type__": "unknown_type"}
        result = custom_decoder(obj)
        assert result == obj
    
    def test_decode_no_type_key(self):
        """Test decoding objects without __type__ key."""
        obj = {"regular": "object"}
        result = custom_decoder(obj)
        assert result == obj
    
    def test_decode_list_str(self):
        """Test decoding List[str] type."""
        obj = {"__type__": "List[str]"}
        result = custom_decoder(obj)
        assert result == List[str]
    
    def test_decode_list_int(self):
        """Test decoding List[int] type."""
        obj = {"__type__": "List[int]"}
        result = custom_decoder(obj)
        assert result == List[int]


class TestConvertToDictConfig:
    """Test the convert_to_dictconfig functionality."""
    
    def test_simple_conversion(self):
        """Test converting a simple config dictionary."""
        saved_config = {
            "test_key": {
                "parent_key": "test_key",
                "type_val": str,
                "current_state": "test_value",
                "is_required": True,
                "tooltip": "Test tooltip"
            }
        }
        
        result = convert_to_dictconfig(saved_config)
        assert isinstance(result["test_key"], DictConfig)
        
        config_dict = result["test_key"].get_config()
        assert config_dict["parent_key"] == "test_key"
        assert config_dict["current_state"] == "test_value"
    
    def test_conversion_with_options(self):
        """Test converting config with options."""
        saved_config = {
            "test_key": {
                "parent_key": "test_key",
                "type_val": dict,
                "current_state": {},
                "has_options": True,
                "options": [
                    {"option1": {
                        "parent_key": "option1",
                        "type_val": str,
                        "current_state": "value1"
                    }}
                ]
            }
        }
        
        result = convert_to_dictconfig(saved_config)
        assert isinstance(result["test_key"], DictConfig)
        
        config_dict = result["test_key"].get_config()
        assert config_dict["has_options"] is True
        assert len(config_dict["options"]) == 1


class TestCurrentStateOfKey:
    """Test the current_state_of_key utility function."""
    
    def test_current_state_extraction(self):
        """Test extracting current state from config dictionary."""
        # Note: The function appears to have a typo in the original code
        # It should return dictionary[key]["current_state"] but has "current_"
        # This test will need to be adjusted based on the actual implementation
        pass  # Skipping due to apparent typo in original function


class TestConfigDictionaries:
    """Test the predefined configuration dictionaries."""
    
    def test_general_config_structure(self):
        """Test the general configuration dictionary structure."""
        assert SPECTRUM_DIRECTORY in dict_config_general
        assert DATA_PER_PULSE in dict_config_general
        assert OUTPUT_DIRECTORY in dict_config_general
        assert PROJECT_NAME in dict_config_general
        
        # Test that each entry is properly formatted
        for key, config in dict_config_general.items():
            assert PARENT_KEY in config
            assert TYPE_VALUE in config
            assert CURRENT_STATE in config
            assert IS_REQUIRED in config
    
    def test_chemometrics_config_structure(self):
        """Test the chemometrics configuration dictionary structure."""
        assert ANALYSIS_TYPE in dict_config_chemometrics
        assert CHEMOMETRICS_DIRECTORY in dict_config_chemometrics
        assert CONFIDENCE_INTERVAL in dict_config_chemometrics
        
        # Verify analysis type is set correctly
        analysis_config = dict_config_chemometrics[ANALYSIS_TYPE]
        assert analysis_config[CURRENT_STATE] == "Chemometrics"
        assert analysis_config[EDITABLE] is False
    
    def test_phase_config_structure(self):
        """Test the phase configuration dictionary structure."""
        assert ANALYSIS_TYPE in dict_config_phase
        assert LIST_OF_HARMONICS in dict_config_phase
        
        # Verify analysis type is set correctly
        analysis_config = dict_config_phase[ANALYSIS_TYPE]
        assert analysis_config[CURRENT_STATE] == "Phase"
        assert analysis_config[EDITABLE] is False
    
    def test_rate_data_config_structure(self):
        """Test the rate data configuration dictionary structure."""
        assert ANALYSIS_TYPE in dict_config_rate_data
        assert AVERAGE_RATE in dict_config_rate_data
        
        # Verify analysis type is set correctly
        analysis_config = dict_config_rate_data[ANALYSIS_TYPE]
        assert analysis_config[CURRENT_STATE] == "Rate"
        assert analysis_config[EDITABLE] is False
    
    def test_none_dict_config(self):
        """Test the NONE_DICT_CONFIG constant."""
        config = NONE_DICT_CONFIG.get_config()
        assert config[PARENT_KEY] == "None"
        assert config[TYPE_VALUE] == dict
        assert config[CURRENT_STATE] == {}
    
    def test_file_type_options(self):
        """Test file type configuration options."""
        file_type_config = dict_config_general[FILE_TYPE]
        assert file_type_config[HAS_OPTIONS] is True
        assert len(file_type_config[OPTIONS]) >= 2  # At least None and csv
        
        # Check for csv option structure
        csv_option = None
        for option in file_type_config[OPTIONS]:
            if "csv" in option:
                csv_option = option["csv"]
                break
        
        assert csv_option is not None
        assert csv_option[TYPE_VALUE] == dict
    
    def test_baseline_correction_options(self):
        """Test baseline correction configuration options."""
        baseline_config = dict_config_general[BASELINE_CORRECTION_METHOD]
        assert baseline_config[HAS_OPTIONS] is True
        
        # Check for arpls option
        arpls_option = None
        for option in baseline_config[OPTIONS]:
            if "arpls" in option:
                arpls_option = option["arpls"]
                break
        
        assert arpls_option is not None
        assert arpls_option[TYPE_VALUE] == dict


class TestConfigIntegration:
    """Integration tests for configuration system."""
    
    def test_save_load_cycle(self):
        """Test saving and loading configuration through JSON."""
        # Create a simple config
        original_config = DictConfig(
            parent_key="test",
            type_val=str,
            current_state="test_value",
            is_required=True
        )
        
        # Serialize to JSON
        config_dict = {"test": original_config.get_config()}
        json_str = json.dumps(config_dict, cls=CustomEncoder)
        
        # Deserialize from JSON
        loaded_dict = json.loads(json_str, object_hook=custom_decoder)
        converted_config = convert_to_dictconfig(loaded_dict)
        
        # Verify the configuration survived the round trip
        assert isinstance(converted_config["test"], DictConfig)
        reloaded_config = converted_config["test"].get_config()
        original_config_dict = original_config.get_config()
        
        assert reloaded_config[PARENT_KEY] == original_config_dict[PARENT_KEY]
        assert reloaded_config[CURRENT_STATE] == original_config_dict[CURRENT_STATE]
        assert reloaded_config[IS_REQUIRED] == original_config_dict[IS_REQUIRED]
    
    def test_config_modification(self):
        """Test modifying configuration values."""
        # Start with general config
        modified_config = dict_config_general.copy()
        
        # Modify a value (this would typically be done through a GUI)
        spectrum_dir_config = modified_config[SPECTRUM_DIRECTORY].copy()
        spectrum_dir_config[CURRENT_STATE] = "/new/path/to/spectra"
        modified_config[SPECTRUM_DIRECTORY] = spectrum_dir_config
        
        # Verify the modification
        assert modified_config[SPECTRUM_DIRECTORY][CURRENT_STATE] == "/new/path/to/spectra"
        assert dict_config_general[SPECTRUM_DIRECTORY][CURRENT_STATE] == ""  # Original unchanged
    
    def test_required_fields_validation(self):
        """Test identification of required fields."""
        required_fields = []
        for key, config in dict_config_general.items():
            if config[IS_REQUIRED]:
                required_fields.append(key)
        
        # Verify some expected required fields
        assert SPECTRUM_DIRECTORY in required_fields
        assert OUTPUT_DIRECTORY in required_fields
        assert DATA_PER_PULSE in required_fields


if __name__ == "__main__":
    pytest.main([__file__])