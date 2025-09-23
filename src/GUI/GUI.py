import sys
import os
from PyQt5.QtCore import Qt
from typing import Any, Dict, List, Optional, Tuple, Union, get_origin
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QFormLayout, QLabel, QLineEdit, QCheckBox,
    QSpinBox, QDoubleSpinBox, QComboBox, QPushButton, QFileDialog, QStackedWidget, QHBoxLayout, QStackedWidget, QMessageBox
)
import sys
from itertools import count
import copy
import json
import yaml

from src.config_files.config_dictionaries import dict_config_general, dict_config_chemometrics, dict_config_phase, dict_config_rate_data, DictConfig, CHEMOMETRICS_ANALYSIS_TYPE, RATE_DATA_ANALYSIS_TYPE, PHASE_ANALYSIS_TYPE, ANALYSIS_TYPE

from src.config_files.dict_config_utils import custom_decoder, GENERAL_PAGE, ANALYSIS_PAGE, CURRENT_STATE, convert_clean_config_to_gui_format

from src.run_configs.overall_run_config import OverallRunConfig

class MainMenuWidget(QWidget):
    def __init__(self, parent_dict: Dict[str, Any], config_stack: 'ConfigStack', parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.menu_parent_dict = parent_dict  # Store data here
        self.config_stack = config_stack  # Reference to ConfigStack
        self.layout = QVBoxLayout(self)

        # Create the main menu buttons
        self.new_analysis_button = QPushButton("Begin New Analysis", self)
        self.new_analysis_button.clicked.connect(self.begin_new_analysis)
        self.load_previous_button = QPushButton("Load Previous", self)
        self.load_previous_button.clicked.connect(self.load_previous)

        self.layout.addWidget(self.new_analysis_button)
        self.layout.addWidget(self.load_previous_button)

    def begin_new_analysis(self):
        # Directly access the config_stack to change the page
        print("Navigating to general configuration page...")
        self.config_stack.stack.setCurrentIndex(1)  # Move to general config page

    def load_previous(self):
        # Load the previous analysis from a YAML or JSON file
        # Open a file dialog to select a configuration file
        file_name, _ = QFileDialog.getOpenFileName(
            self, 
            "Open Configuration File", 
            "", 
            "Configuration Files (*.yaml *.yml *.json);;YAML Files (*.yaml *.yml);;JSON Files (*.json);;All Files (*)"
        )
        
        if file_name:
            # Load the selected configuration file
            try:
                file_ext = os.path.splitext(file_name)[1].lower()
                if file_ext in ['.yaml', '.yml']:
                    with open(file_name, 'r') as yaml_file:
                        loaded_dict = yaml.safe_load(yaml_file)
                        # Always convert to GUI format (nested dicts, not DictConfig)
                        loaded_dict = self._convert_clean_config_to_gui_format(loaded_dict)
                elif file_ext == '.json':
                    with open(file_name, 'r') as json_file:
                        loaded_dict = json.load(json_file)
                        # Always convert to GUI format (nested dicts, not DictConfig)
                        if self._is_clean_format(loaded_dict):
                            loaded_dict = self._convert_clean_config_to_gui_format(loaded_dict)
                else:
                    QMessageBox.warning(self, "Error", f"Unsupported file format: {file_ext}")
                    return
                # Overwrite operational_dict with loaded config (do not merge DictConfig etc)
                self.config_stack.operational_dict.clear()
                self.config_stack.operational_dict.update(loaded_dict)
                # Update the UI after successfully loading the data
                print(f"DEBUG: Operational dict keys after loading: {list(self.config_stack.operational_dict.keys())}")
                if GENERAL_PAGE in self.config_stack.operational_dict:
                    print(f"DEBUG: General page keys: {list(self.config_stack.operational_dict[GENERAL_PAGE].keys())}")
                if ANALYSIS_PAGE in self.config_stack.operational_dict:
                    print(f"DEBUG: Analysis page keys: {list(self.config_stack.operational_dict[ANALYSIS_PAGE].keys())}")
                    if ANALYSIS_TYPE in self.config_stack.operational_dict[ANALYSIS_PAGE]:
                        analysis_type_config = self.config_stack.operational_dict[ANALYSIS_PAGE][ANALYSIS_TYPE]
                        print(f"DEBUG: Analysis type config: {analysis_type_config}")
                        if CURRENT_STATE in analysis_type_config:
                            loaded_analysis_type = analysis_type_config[CURRENT_STATE]
                            print(f"DEBUG: Current analysis type: {loaded_analysis_type}")
                            
                            # Find and switch to the tab that matches the loaded analysis type
                            for i in range(self.config_stack.stack.count()):
                                widget = self.config_stack.stack.widget(i)
                                if hasattr(widget, 'widget_name') and widget.widget_name == loaded_analysis_type:
                                    print(f"DEBUG: Switching to tab {i} for analysis type '{loaded_analysis_type}'")
                                    self.config_stack.stack.setCurrentIndex(i)
                                    break
                            else:
                                print(f"DEBUG: No tab found for analysis type '{loaded_analysis_type}', staying on general page")
                
                self.config_stack.update_pages()
                for i in range(self.config_stack.stack.count()):
                    widget = self.config_stack.stack.widget(i)
                    if isinstance(widget, ConfigWidget):
                        print(f"Page {widget.widget_name} - Config:")
                        print(widget.config)
                QMessageBox.information(self, "Success", f"Configuration loaded successfully from {os.path.basename(file_name)}.")
            except FileNotFoundError:
                QMessageBox.warning(self, "Error", "Configuration file not found.")
            except (json.JSONDecodeError, yaml.YAMLError) as e:
                QMessageBox.warning(self, "Error", f"Error parsing the configuration file: {str(e)}")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Unexpected error loading configuration: {str(e)}")
        

    def _is_clean_format(self, config_dict):
        """
        Check if this is a clean format (from YAML/run config) vs full DictConfig format (from GUI).
        Clean format has direct values, full format has 'current_state' keys.
        """
        if not isinstance(config_dict, dict):
            return True
            
        # Look for the structure of a full DictConfig format
        for page_key, page_config in config_dict.items():
            if isinstance(page_config, dict):
                for config_key, config_value in page_config.items():
                    if isinstance(config_value, dict):
                        # If we find 'current_state' keys, it's full format
                        if CURRENT_STATE in config_value:
                            return False
                        # If we find DictConfig metadata keys, it's full format  
                        if any(key in config_value for key in ['parent_key', 'type_val', 'tooltip']):
                            return False
        
        # No DictConfig structure found, assume it's clean format
        return True

    def _convert_clean_config_to_gui_format(self, clean_config):
        """
        Convert clean YAML/JSON config format to GUI-compatible DictConfig format.
        This merges the clean values back into the original GUI structure.
        """
        import copy
        # Use the shared converter to normalize keys and wrap leaves
        normalized = convert_clean_config_to_gui_format(clean_config)

        # Start with the original GUI templates to preserve metadata
        gui_config = {
            GENERAL_PAGE: copy.deepcopy(dict_config_general),
            ANALYSIS_PAGE: None
        }

        # Determine analysis page template selection (keep previous heuristic)
        if ANALYSIS_PAGE in normalized and ANALYSIS_TYPE in normalized[ANALYSIS_PAGE]:
            # normalized stores values wrapped with CURRENT_STATE
            analysis_type_val = normalized[ANALYSIS_PAGE][ANALYSIS_TYPE].get(CURRENT_STATE)
            # Unwrap if necessary
            if isinstance(analysis_type_val, dict):
                analysis_type = list(analysis_type_val.values())[0]
            else:
                analysis_type = analysis_type_val

            if analysis_type == CHEMOMETRICS_ANALYSIS_TYPE:
                gui_config[ANALYSIS_PAGE] = copy.deepcopy(dict_config_chemometrics)
            elif analysis_type == PHASE_ANALYSIS_TYPE:
                gui_config[ANALYSIS_PAGE] = copy.deepcopy(dict_config_phase)
            elif analysis_type == RATE_DATA_ANALYSIS_TYPE:
                gui_config[ANALYSIS_PAGE] = copy.deepcopy(dict_config_rate_data)
            else:
                gui_config[ANALYSIS_PAGE] = copy.deepcopy(dict_config_chemometrics)
        else:
            gui_config[ANALYSIS_PAGE] = copy.deepcopy(dict_config_chemometrics)

        # Merge normalized CURRENT_STATE values into the GUI templates
        for page_key, page_val in normalized.items():
            if page_key not in gui_config:
                continue
            if not isinstance(page_val, dict):
                continue
            for cfg_key, cfg_val in page_val.items():
                # cfg_val should be a {CURRENT_STATE: ...} wrapper
                if cfg_key in gui_config[page_key] and isinstance(gui_config[page_key][cfg_key], dict) and CURRENT_STATE in gui_config[page_key][cfg_key]:
                    # If normalized value is the wrapper, set the inner value directly
                    if isinstance(cfg_val, dict) and CURRENT_STATE in cfg_val:
                        gui_config[page_key][cfg_key][CURRENT_STATE] = cfg_val[CURRENT_STATE]
                        # If the template has an 'options' list, also merge children into those option entries
                        try:
                            template = gui_config[page_key][cfg_key]
                            if 'options' in template and isinstance(template['options'], list):
                                # cfg_val[CURRENT_STATE] is mapping option_name -> {CURRENT_STATE: {...}}
                                option_map = cfg_val[CURRENT_STATE]
                                if isinstance(option_map, dict):
                                    for opt_entry in template['options']:
                                        if not isinstance(opt_entry, dict):
                                            continue
                                        for opt_name, opt_val in list(opt_entry.items()):
                                            # If user provided values for this option, merge them
                                            if opt_name in option_map:
                                                user_opt = option_map[opt_name]
                                                # user_opt may be {CURRENT_STATE: {child_key: {CURRENT_STATE: val}}}
                                                if isinstance(user_opt, dict) and CURRENT_STATE in user_opt:
                                                    user_children = user_opt[CURRENT_STATE]
                                                else:
                                                    user_children = user_opt
                                                if isinstance(opt_val, dict) and CURRENT_STATE in opt_val:
                                                    template_children = opt_val[CURRENT_STATE]
                                                else:
                                                    template_children = opt_val
                                                # Merge each child value into the template child's CURRENT_STATE
                                                if isinstance(user_children, dict) and isinstance(template_children, dict):
                                                    for child_k, child_v in user_children.items():
                                                        # child_v may be {CURRENT_STATE: val} or primitive
                                                        if isinstance(child_v, dict) and CURRENT_STATE in child_v:
                                                            val_to_set = child_v[CURRENT_STATE]
                                                        else:
                                                            val_to_set = child_v
                                                        # Only set if the template has the child key
                                                        if child_k in template_children and isinstance(template_children[child_k], dict) and CURRENT_STATE in template_children[child_k]:
                                                            template_children[child_k][CURRENT_STATE] = val_to_set
                        except Exception:
                            # Be defensive; if merging fails, leave template as-is
                            pass
                    else:
                        # fallback: assign whatever was provided
                        gui_config[page_key][cfg_key][CURRENT_STATE] = cfg_val
                else:
                    # Unknown key or mismatched structure - skip
                    continue

        return gui_config


class ConfigStack(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.config_stack_parent_dict = {
            GENERAL_PAGE : dict_config_general,
            "Phase_Page" : dict_config_phase,
            "Chemometrics_Page" : dict_config_chemometrics,
            "Rate_Data_Page" : dict_config_rate_data
        }
        self.operational_dict = {
            GENERAL_PAGE : dict_config_general,
            ANALYSIS_PAGE : None
        }

        self.layout = QVBoxLayout(self)
        self.stack = QStackedWidget(self)

        # Main menu page
        main_menu_page = MainMenuWidget(self.config_stack_parent_dict, self)  # Pass self to reference ConfigStack
        self.stack.addWidget(main_menu_page)

        # General config page
        general_page = ConfigWidget(dict_config_general, self.config_stack_parent_dict, self, is_general_page=True, name=GENERAL_PAGE)  # Pass True to indicate the general page
        self.stack.addWidget(general_page)

        # Chemometrics, Phase, and Rate Data pages with "Run" and "Back" buttons
        chemometrics_page = ConfigWidget(dict_config_chemometrics, self.config_stack_parent_dict, self, is_general_page=False, name=CHEMOMETRICS_ANALYSIS_TYPE)  # Pass False for the rest
        phase_page = ConfigWidget(dict_config_phase, self.config_stack_parent_dict, self, is_general_page=False, name=PHASE_ANALYSIS_TYPE)
        rate_data_page = ConfigWidget(dict_config_rate_data, self.config_stack_parent_dict, self, is_general_page=False, name=RATE_DATA_ANALYSIS_TYPE)

        self.stack.addWidget(chemometrics_page)
        self.stack.addWidget(phase_page)
        self.stack.addWidget(rate_data_page)

        self.layout.addWidget(self.stack)
        self.setLayout(self.layout)

        # Ensure the first page is the main menu
        self.stack.setCurrentIndex(0)
    
    def update_pages(self):
        for i in range(self.stack.count()):
            widget = self.stack.widget(i)
            if isinstance(widget, ConfigWidget):
                print(widget.widget_name)
                widget.refresh_with_loaded_data()



class ConfigWidget(QWidget):
    def __init__(self, config: Dict[str, Dict[str, Any]], parent_dict: Dict[str, Any], config_stack: 'ConfigStack', is_general_page: bool = False, parent: Optional[QWidget] = None, name:str="None") -> None:
        super().__init__(parent)
        self.config_stack = config_stack  # Reference to ConfigStack
        self.configwidget_parent_dict = parent_dict
        self.is_general_page = is_general_page  # New argument to track if this is the general page
        self.setup_configs(config=config)
        self.widget_name = name

        
        if self.is_general_page:
            self.widget_name = GENERAL_PAGE
            # Add navigation buttons for Chemometrics, Phase, and Rate Data only on the General config page
            
            self.button_layout = QHBoxLayout()
            self.chemometrics_button = QPushButton("Chemometrics", self)
            self.chemometrics_button.clicked.connect(self.navigate_to_chemometrics)

            self.phase_button = QPushButton("Phase", self)
            self.phase_button.clicked.connect(self.navigate_to_phase)

            self.rate_data_button = QPushButton("Rate Data", self)
            self.rate_data_button.clicked.connect(self.navigate_to_rate_data)

            self.button_layout.addWidget(self.chemometrics_button)
            self.button_layout.addWidget(self.phase_button)
            self.button_layout.addWidget(self.rate_data_button)

            self.layout.addLayout(self.button_layout)
        else:
            # On Chemometrics, Phase, and Rate pages, only show "Run" and "Back" buttons
            self.button_layout = QHBoxLayout()
            self.back_button = QPushButton("Back", self)
            self.back_button.clicked.connect(self.on_back_clicked)

            self.run_button = QPushButton("Run", self)
            self.run_button.clicked.connect(self.on_run_clicked)

            self.button_layout.addWidget(self.back_button)
            self.button_layout.addWidget(self.run_button)
            
            self.layout.addLayout(self.button_layout)

        self.layout.addLayout(self.form_layout)

    def setup_configs(self, config):
        self.config = config
        
        self.layout = QVBoxLayout(self)
        self.form_layout = QFormLayout()
        self._recreation_dict = {
            GENERAL_PAGE : dict_config_general,
            ANALYSIS_PAGE : None,
        }

        self.init_ui(config)

    def update_recreation_dict(self, all_configs):
        self._recreation_dict = {
            GENERAL_PAGE : all_configs[GENERAL_PAGE],
            ANALYSIS_PAGE : all_configs[ANALYSIS_PAGE],
        }

    def _is_clean_format(self, config_dict):
        """
        Check if this is a clean format (from YAML/run config) vs full DictConfig format (from GUI).
        Clean format has direct values, full format has 'current_state' keys.
        This mirrors the helper in MainMenuWidget so both widgets can detect clean configs.
        """
        if not isinstance(config_dict, dict):
            return True

        for page_key, page_config in config_dict.items():
            if isinstance(page_config, dict):
                for config_key, config_value in page_config.items():
                    if isinstance(config_value, dict):
                        if CURRENT_STATE in config_value:
                            return False
                        if any(key in config_value for key in ['parent_key', 'type_val', 'tooltip']):
                            return False

        return True


    def on_back_clicked(self):
        current_index = self.config_stack.stack.currentIndex()
        # Ensure not to go back from the first page (Main Menu) or invalid page index
        if current_index > 0:
            self.config_stack.stack.setCurrentIndex(1)

    def refresh_with_loaded_data(self):
        print(f"DEBUG: Refreshing widget '{self.widget_name}'")
        
        # Clear the form layout
        while self.form_layout.count():
            item = self.form_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()  # Ensure old widgets are removed properly

        # Determine which config to use
        if self.widget_name == GENERAL_PAGE:
            print(f"DEBUG: Loading general page config")
            if GENERAL_PAGE in self.config_stack.operational_dict:
                self.config = self.config_stack.operational_dict[GENERAL_PAGE]
                print(f"DEBUG: General config keys: {list(self.config.keys()) if self.config else 'None'}")
            else:
                print(f"DEBUG: No general page config found!")
                return
        else:
            # This is an analysis page - try to find the right config
            print(f"DEBUG: Loading analysis page config for widget '{self.widget_name}'")
            if ANALYSIS_PAGE in self.config_stack.operational_dict:
                self.config = self.config_stack.operational_dict[ANALYSIS_PAGE]
                print(f"DEBUG: Analysis config keys: {list(self.config.keys()) if self.config else 'None'}")
                
                # Check if the analysis type matches what we expect
                if ANALYSIS_TYPE in self.config_stack.operational_dict[ANALYSIS_PAGE]:
                    current_analysis_type = self.config_stack.operational_dict[ANALYSIS_PAGE][ANALYSIS_TYPE].get(CURRENT_STATE, "Unknown")
                    print(f"DEBUG: Current analysis type: '{current_analysis_type}', Widget name: '{self.widget_name}'")
                    
                    if current_analysis_type != self.widget_name:
                        print(f"DEBUG: Analysis type mismatch! Expected '{self.widget_name}', got '{current_analysis_type}'")
                        # Still use the config, but warn about the mismatch
                else:
                    print(f"DEBUG: No analysis type found in config")
            else:
                print(f"DEBUG: No analysis page config found!")
                return

        if self.config:
            print(f"DEBUG: About to call populate_form with config having {len(self.config)} keys")
            print(f"DEBUG: Sample config item: {list(self.config.items())[:2] if self.config else 'None'}")
            self.update_recreation_dict(all_configs=self.config_stack.operational_dict)
            
            # Populate the form without adding it to the layout again
            self.populate_form(self.config)
            self.update()
            print(f"DEBUG: Finished refreshing {self.widget_name}")
        else:
            print(f"DEBUG: No config available for {self.widget_name}")
    
    def populate_form(self, config: Dict[str, Dict[str, Any]]) -> None:
        """Populate the form layout with widgets based on config, without modifying the main layout."""
        combo_boxes_to_initialize = []
        
        for key, value in config.items():
            widget = self.create_widget(value)
            if widget:
                label = QLabel(key)
                label.setToolTip(value.get('tooltip', ''))  # Set tooltip on the label only
                self.form_layout.addRow(label, widget)
                
                # Collect QComboBox widgets to initialize after full setup
                if isinstance(widget, QComboBox):
                    combo_boxes_to_initialize.append((key, widget, value['options']))

        # Initialize QComboBoxes after full setup
        for key, widget, options in combo_boxes_to_initialize:
            current_state_key = widget.currentText()
            self.on_option_change(current_state_key, key, options)


    def navigate_to_chemometrics(self):
        general_page_dict = self.update_config_with_user_input(self.config, self.form_layout)
        self._recreation_dict[GENERAL_PAGE] = general_page_dict
        self.config_stack.operational_dict[GENERAL_PAGE] = general_page_dict
        self.save_and_navigate(2)

    def navigate_to_phase(self):
        general_page_dict = self.update_config_with_user_input(self.config, self.form_layout)
        self._recreation_dict[GENERAL_PAGE] = general_page_dict
        self.config_stack.operational_dict[GENERAL_PAGE] = general_page_dict
        self.save_and_navigate(3)

    def navigate_to_rate_data(self):
        general_page_dict = self.update_config_with_user_input(self.config, self.form_layout)
        self._recreation_dict[GENERAL_PAGE] = general_page_dict
        self.config_stack.operational_dict[GENERAL_PAGE] = general_page_dict
        self.save_and_navigate(4)

    def get_updated_analysis_config(self):
        analysis_config = self.update_config_with_user_input(self.config, self.form_layout)
        self._recreation_dict[ANALYSIS_PAGE] = analysis_config
        self.config_stack.operational_dict[ANALYSIS_PAGE] = analysis_config
        # return analysis_config

    def on_run_clicked(self):
        print("Running analysis with the following configuration:")
        self.print_updated_config()  # Print the updated config when running analysis

        self.get_updated_analysis_config()

        try:
            
            # Build runtime config from GUI metadata so GUI can run the same analysis as CLI
            # If the operational dict is a 'clean' config (from file), convert it first (same fix applied in pymespec.py)
            operational = self.config_stack.operational_dict
            # Only convert when the operational dict appears to be a clean config (to avoid double-conversion)
            if self._is_clean_format(operational):
                print("DEBUG: Operational dict detected as clean format - converting to GUI metadata format before building runtime config")
                metadata_source = convert_clean_config_to_gui_format(operational)
            else:
                metadata_source = operational

            runtime_config = self._build_runtime_config_from_metadata(metadata_source)
            # Normalize runtime just before handing to OverallRunConfig to ensure exact shape
            runtime_config = self._normalize_runtime_config(runtime_config)

            # Debug print resolved CSV column names to help trace YAML mapping issues
            try:
                self._debug_print_csv_columns(runtime_config)
            except Exception as _:
                print("DEBUG: Could not print CSV columns (unexpected structure)")
            # Probe actual CSV files for header names and auto-fix mapping if possible
            try:
                self._auto_fix_csv_column_names(runtime_config)
            except Exception as _:
                print("DEBUG: Could not auto-fix CSV column names")
            # Diagnostic: probe numeric lengths for frequency/intensity columns to catch parsing issues
            try:
                self._probe_csv_column_lengths(runtime_config)
            except Exception as _:
                print("DEBUG: Could not probe CSV column lengths")
            overall_run_config = OverallRunConfig(runtime_config)
            overall_run_config.save_config()
            overall_run_config.load_data()
            overall_run_config.perform_analysis()

            
            QMessageBox.information(self, "Success", "Analysis completed successfully.")
        except Exception as e:
            
            if "Permission denied" in str(e):
                e = str(e) + ". Please ensure the file is not open in another program."
            QMessageBox.information(self, "Failure", f"The analysis failed with error: {e}")
        # print("decoded_complete")

    def print_updated_config(self):
        updated_config = self.update_config_with_user_input(self.config, self.form_layout)
        print("Updated Configuration:")
        print(repr(self.configwidget_parent_dict))  # Print using repr()

    
    def save_and_navigate(self, index: int):
        
        updated_config = self.update_config_with_user_input(self.config, self.form_layout)
        self.configwidget_parent_dict.update(updated_config)
        
        # Print the updated configuration to the console using repr()
        print("Updated Configuration:")
        print(repr(self.configwidget_parent_dict))  # Maintain Python structure

        # Navigate to the next page
        self.config_stack.stack.setCurrentIndex(index)

    def _build_runtime_config_from_metadata(self, gui_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert the GUI metadata structure (dict of DictConfig.get_config() dicts)
        into the runtime config structure expected by OverallRunConfig.
        """
        from src.config_files.dict_config_utils import CURRENT_STATE

        def extract_meta(meta):
            # If this is a DictConfig metadata dict
            if isinstance(meta, dict) and 'parent_key' in meta:
                cur = meta.get(CURRENT_STATE)
                # Option dict (e.g. {'csv': { ...metadata... }})
                if isinstance(cur, dict):
                    options_runtime = {}
                    for opt_name, opt_val in cur.items():
                        # opt_val may be metadata dict for the option
                        if isinstance(opt_val, dict) and 'parent_key' in opt_val:
                            # opt_val['current_state'] contains children mapping
                            children = {}
                            for child_k, child_v in opt_val.get(CURRENT_STATE, {}).items():
                                children[child_k] = extract_meta(child_v)
                            options_runtime[opt_name] = {CURRENT_STATE: children}
                        elif isinstance(opt_val, dict) and CURRENT_STATE in opt_val:
                            # already in runtime form
                            options_runtime[opt_name] = opt_val
                        else:
                            # primitive or plain value
                            options_runtime[opt_name] = {CURRENT_STATE: opt_val}
                    return {CURRENT_STATE: options_runtime}
                # Primitive current_state
                return {CURRENT_STATE: cur}

            # If already runtime form
            if isinstance(meta, dict) and CURRENT_STATE in meta:
                return meta

            # If plain dict (not metadata), recurse
            if isinstance(meta, dict):
                return {k: extract_meta(v) for k, v in meta.items()}

            # Primitive
            return {CURRENT_STATE: meta}

        runtime = {}
        for page_key, page_val in gui_metadata.items():
            if isinstance(page_val, dict):
                runtime[page_key] = {k: extract_meta(v) for k, v in page_val.items()}
            else:
                runtime[page_key] = page_val
        return runtime


    def _normalize_runtime_config(self, runtime: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensure the runtime config strictly uses the CURRENT_STATE wrapper for all leaves and
        that options are represented as {CURRENT_STATE: {option_name: {CURRENT_STATE: {...}}}}
        This is a defensive normalization step applied only in the GUI before running.
        """
        from src.config_files.dict_config_utils import CURRENT_STATE

        def norm_node(node):
            # If node is already a CURRENT_STATE wrapper mapping
            if isinstance(node, dict) and CURRENT_STATE in node:
                val = node[CURRENT_STATE]
                # If the wrapped value is a dict of options (option_name -> {...})
                if isinstance(val, dict):
                    new_opts = {}
                    for opt_name, opt_val in val.items():
                        # opt_val might be a nested runtime mapping or primitive
                        if isinstance(opt_val, dict) and CURRENT_STATE in opt_val:
                            # already shaped correctly, recursively normalize children
                            child = opt_val[CURRENT_STATE]
                            if isinstance(child, dict):
                                new_opts[opt_name] = {CURRENT_STATE: {k: norm_node(v) for k, v in child.items()}}
                            else:
                                new_opts[opt_name] = {CURRENT_STATE: child}
                        else:
                            # primitive or plain dict -> ensure it's wrapped
                            if isinstance(opt_val, dict):
                                # treat as mapping of children
                                new_opts[opt_name] = {CURRENT_STATE: {k: norm_node(v) for k, v in opt_val.items()}}
                            else:
                                new_opts[opt_name] = {CURRENT_STATE: opt_val}
                    return {CURRENT_STATE: new_opts}
                else:
                    # Primitive wrapped by CURRENT_STATE
                    return {CURRENT_STATE: val}

            # If node is a plain dict (not wrapped), normalize all its children
            if isinstance(node, dict):
                return {k: norm_node(v) for k, v in node.items()}

            # Primitive -> wrap
            return {CURRENT_STATE: node}

        return {k: norm_node(v) for k, v in runtime.items()}

    def _debug_print_csv_columns(self, runtime: Dict[str, Any]) -> None:
        """
        Diagnostic helper: print available keys for CSV frequency/intensity columns
        from both General_Page and Analysis_Page so we can see what the GUI will pass in.
        """
        from src.config_files.dict_config_utils import CURRENT_STATE
        from src.config_files.config_dictionaries import FILE_TYPE, CSV, FREQUENCY_COLUMN, INTENSITY_COLUMN, ANALYSIS_PAGE, GENERAL_PAGE

        print("DEBUG: Beginning CSV column diagnostics")
        # General page
        if GENERAL_PAGE in runtime:
            gen = runtime[GENERAL_PAGE]
            if FILE_TYPE in gen and CURRENT_STATE in gen[FILE_TYPE]:
                ft = gen[FILE_TYPE][CURRENT_STATE]
                print(f"DEBUG: General Page FILE_TYPE keys: {list(ft.keys())}")
                if CSV in ft:
                    csv_map = ft[CSV]
                    print(f"DEBUG: General CSV mapping keys: {list(csv_map.keys())}")
                    freq = csv_map.get(FREQUENCY_COLUMN, {}).get(CURRENT_STATE)
                    inten = csv_map.get(INTENSITY_COLUMN, {}).get(CURRENT_STATE)
                    print(f"DEBUG: Resolved General frequency column: {freq}")
                    print(f"DEBUG: Resolved General intensity column: {inten}")

        # Analysis page (chemometrics)
        if ANALYSIS_PAGE in runtime:
            ap = runtime[ANALYSIS_PAGE]
            from src.config_files.config_dictionaries import CHEMOMETRICS_FILE_TYPE, CHEMOMETRICS_DIRECTORY, CHEMOMETRICS_SAMPLE_STANDARDS_FILE_LIST
            if CHEMOMETRICS_FILE_TYPE in ap and CURRENT_STATE in ap[CHEMOMETRICS_FILE_TYPE]:
                cft = ap[CHEMOMETRICS_FILE_TYPE][CURRENT_STATE]
                print(f"DEBUG: Analysis Page CHEMOMETRICS_FILE_TYPE keys: {list(cft.keys())}")
                if CSV in cft:
                    chem_csv_map = cft[CSV]
                    print(f"DEBUG: Chem CSV mapping keys: {list(chem_csv_map.keys())}")
                    freq_c = chem_csv_map.get(FREQUENCY_COLUMN, {}).get(CURRENT_STATE)
                    int_c = chem_csv_map.get(INTENSITY_COLUMN, {}).get(CURRENT_STATE)
                    print(f"DEBUG: Resolved Chem frequency column: {freq_c}")
                    print(f"DEBUG: Resolved Chem intensity column: {int_c}")
        print("DEBUG: Finished CSV column diagnostics")

    def _probe_csv_headers(self, directory: str, sample_files: int = 1) -> Dict[str, list]:
        """
        Read up to `sample_files` CSV files from `directory` and return a mapping of
        filename -> list of header column names. If files have no header, returns an empty list.
        """
        from pathlib import Path
        import pandas as pd

        p = Path(directory)
        headers = {}
        if not p.exists() or not p.is_dir():
            return headers
        csvs = list(p.glob('*.csv'))[:sample_files]
        for f in csvs:
            try:
                df = pd.read_csv(f, nrows=1)
                headers[f.name] = list(df.columns)
            except Exception:
                # Try reading without headers
                try:
                    df = pd.read_csv(f, header=None, nrows=1)
                    headers[f.name] = []
                except Exception:
                    headers[f.name] = []
        return headers

    def _auto_fix_csv_column_names(self, runtime: Dict[str, Any]) -> None:
        """
        Look for mismatches between requested CSV column names in the runtime config and
        actual headers in the sample CSV files. If a simple mapping can be inferred
        (e.g., 'frequency' -> 'frequencies'), update the runtime in-place to use the
        actual header names so the DataProcessingCSV will succeed.
        This is a GUI-only proactive fix and does not change core data processing.
        """
        from src.config_files.dict_config_utils import CURRENT_STATE
        from src.config_files.config_dictionaries import FILE_TYPE, CSV, FREQUENCY_COLUMN, INTENSITY_COLUMN, GENERAL_PAGE, SPECTRUM_DIRECTORY
        # Probe general page files
        if GENERAL_PAGE not in runtime:
            return
        try:
            gen = runtime[GENERAL_PAGE]
            if FILE_TYPE in gen and CURRENT_STATE in gen[FILE_TYPE]:
                ft = gen[FILE_TYPE][CURRENT_STATE]
                if CSV in ft:
                    csv_map = ft[CSV]
                    # requested names (may be primitives or CURRENT_STATE wrappers)
                    req_freq = csv_map.get(FREQUENCY_COLUMN, {}).get(CURRENT_STATE)
                    req_int = csv_map.get(INTENSITY_COLUMN, {}).get(CURRENT_STATE)
                    # Discover sample headers
                    data_dir = gen.get(SPECTRUM_DIRECTORY, {}).get(CURRENT_STATE)
                    if isinstance(data_dir, dict):
                        data_dir = data_dir.get(CURRENT_STATE)
                    if not data_dir:
                        return
                    headers = self._probe_csv_headers(data_dir, sample_files=2)
                    # Gather candidate header names across files
                    all_headers = set()
                    for h in headers.values():
                        all_headers.update(h)

                    if not all_headers:
                        return

                    def find_best_match(requested, candidates):
                        if not requested or not candidates:
                            return None
                        requested_str = str(requested).lower()
                        for c in candidates:
                            if c.lower() == requested_str:
                                return c
                        # fuzzy: plural/singular
                        for c in candidates:
                            if c.lower().rstrip('s') == requested_str or requested_str.rstrip('s') == c.lower():
                                return c
                        # substring match
                        for c in candidates:
                            if requested_str in c.lower() or c.lower() in requested_str:
                                return c
                        return None

                    best_freq = find_best_match(req_freq, all_headers)
                    best_int = find_best_match(req_int, all_headers)

                    updated = False
                    if best_freq and best_freq != req_freq:
                        print(f"DEBUG: Auto-mapping frequency column '{req_freq}' -> '{best_freq}'")
                        csv_map[FREQUENCY_COLUMN] = {CURRENT_STATE: best_freq}
                        updated = True
                    if best_int and best_int != req_int:
                        print(f"DEBUG: Auto-mapping intensity column '{req_int}' -> '{best_int}'")
                        csv_map[INTENSITY_COLUMN] = {CURRENT_STATE: best_int}
                        updated = True

                    if updated:
                        # write back into runtime (mutating in-place)
                        runtime[GENERAL_PAGE][FILE_TYPE] = {CURRENT_STATE: ft}
        except Exception as e:
            print(f"DEBUG: CSV auto-fix exception: {e}")

    def _probe_csv_column_lengths(self, runtime: Dict[str, Any], sample_files: int = 2) -> None:
        """
        For sample CSV files in the General page spectrum directory, attempt to read the
        frequency/intensity columns using the runtime mapping and print the numeric
        lengths and a small preview. This mirrors DataProcessingCSV numeric conversion
        and row slicing to surface parsing issues.
        """
        from src.config_files.dict_config_utils import CURRENT_STATE
        from src.config_files.config_dictionaries import FILE_TYPE, CSV, FREQUENCY_COLUMN, INTENSITY_COLUMN, GENERAL_PAGE, SPECTRUM_DIRECTORY
        import pandas as pd
        from pathlib import Path

        if GENERAL_PAGE not in runtime:
            return
        gen = runtime[GENERAL_PAGE]
        if FILE_TYPE not in gen or CURRENT_STATE not in gen[FILE_TYPE]:
            return
        ft = gen[FILE_TYPE][CURRENT_STATE]
        if CSV not in ft:
            return
        csv_map = ft[CSV]
        freq_key = csv_map.get(FREQUENCY_COLUMN, {}).get(CURRENT_STATE)
        int_key = csv_map.get(INTENSITY_COLUMN, {}).get(CURRENT_STATE)
        data_dir = gen.get(SPECTRUM_DIRECTORY, {}).get(CURRENT_STATE)
        if isinstance(data_dir, dict):
            data_dir = data_dir.get(CURRENT_STATE)
        if not data_dir:
            print("DEBUG: No spectrum directory set; cannot probe CSV lengths")
            return

        p = Path(data_dir)
        csvs = list(p.glob('*.csv'))[:sample_files]
        if not csvs:
            print(f"DEBUG: No CSV files found in {data_dir}")
            return

        for f in csvs:
            try:
                if freq_key is None or int_key is None:
                    # read without header assumption
                    df = pd.read_csv(f, header=None)
                    print(f"DEBUG: {f.name} read without header, shape={df.shape}")
                    # show first two rows
                    print(df.head(2))
                    continue
                # read with header
                df = pd.read_csv(f)
                print(f"DEBUG: {f.name} columns: {list(df.columns)}")
                missing = [c for c in (freq_key, int_key) if c not in df.columns]
                if missing:
                    print(f"DEBUG: {f.name} missing requested columns: {missing}")
                    # show available columns and sample rows
                    print(f"DEBUG: Available columns: {list(df.columns)}")
                    print(df.head(3))
                    continue
                # apply row slicing rules similar to DataProcessingCSV: use dtype coercion and drop NA
                x = pd.to_numeric(df[freq_key], errors='coerce')
                y = pd.to_numeric(df[int_key], errors='coerce')
                mask = x.notna() & y.notna()
                xvals = x[mask].to_numpy()
                yvals = y[mask].to_numpy()
                print(f"DEBUG: {f.name} numeric lengths -> freq:{len(xvals)}, intensity:{len(yvals)}")
                print(f"DEBUG: freq preview: {xvals[:5]}\nDEBUG: int preview: {yvals[:5]}")
            except Exception as e:
                print(f"DEBUG: Error probing {f.name}: {e}")



    def init_ui(self, config: Dict[str, Dict[str, Any]]) -> None:
        self.populate_form(config)
        self.layout.addLayout(self.form_layout)

    def create_widget(self, config: Dict[str, Any]) -> Optional[QWidget]:
        config_type = config.get('type_val')
        editable = config.get('editable', True)  # Default to True if editable key is not present

        if not editable:
            # If the field is not editable, display it as a label
            return QLabel(str(config.get('current_state', ''))) 

        if config_type == str:
            widget = QLineEdit(self)
            widget.setText(config.get('current_state', ''))
        
        elif config_type == bool:
            widget = QCheckBox(self)
            
            # Set the current state of the checkbox based on the config's current value
            widget.setChecked(config.get('current_state', False))

            # Set the state because it not automcatically included in the dictionary
            current_state = config.get('current_state', False)
            self.configwidget_parent_dict[config['parent_key']] = current_state

            # Connect signal to handle state changes
            widget.stateChanged.connect(lambda state: self.on_value_change(config['parent_key'], state == Qt.Checked))





        elif get_origin(config_type) == list and isinstance(config.get('current_state', []), list) and all(isinstance(i, str) for i in config['current_state']):
            # Create a QLineEdit that displays the list of files as a comma-separated string
            widget = QLineEdit(self)
            widget.setText(", ".join(config.get('current_state', [])))  # Show the current list of files
            
            # Add a browse button to allow users to select multiple files
            widget = self.add_browse_button(widget, config['parent_key'], multiple_files=True)

        elif config_type == int:
            widget = QSpinBox(self)
            widget.setRange(-10**9, 10**9)  # Set range for int values
            widget.setValue(config.get('current_state', 0))

        elif config_type == float:
            widget = QDoubleSpinBox(self)
            widget.setRange(-10**10, 10**10)  # Set range for float values
            widget.setDecimals(10)  # Set precision for float values
            widget.setValue(config.get('current_state', 0.0))

        elif config_type == dict:
            if 'has_options' in config and config['has_options']:
                widget = QComboBox(self)
                for option in config['options']:
                    for option_key in option:
                        widget.addItem(option_key)

                # Set the current index based on the current state
                current_state_key = list(config['current_state'].keys())[0]
                index = widget.findText(current_state_key)
                if index >= 0:
                    widget.setCurrentIndex(index)

                # Connect the signal to handle changes
                widget.currentIndexChanged.connect(
                    lambda idx, opt=widget, key=config['parent_key']: self.on_option_change(
                        opt.currentText(), key, config['options']
                    )
                )
                    
                return widget

            else:
                widget = self.create_sub_form(config)

        else:
            widget = None

        # Ensure that file and directory fields have their browse button
        if config.get('select_location', False):
            widget = self.add_browse_button(widget, config['parent_key'], multiple_files=(get_origin(config_type) == list))

        return widget


    def populate_sub_form_recursively(self, config: Dict[str, Any]) -> None:
        """
        Recursively populate sub-forms for the given configuration.
        This ensures that all nested configurations are expanded based on the current state.
        """
        for key, value in config['current_state'].items():
            if isinstance(value, dict) and value.get('has_options', False):
                current_state_key = list(value['current_state'].keys())[0]

                # Trigger on_option_change for the nested QComboBox
                self.on_option_change(current_state_key, key, value['options'])

                # Recursively populate any further nested sub-forms
                for option in value['options']:
                    if current_state_key in option:
                        self.populate_sub_form_recursively(option[current_state_key])


    
    def create_sub_form(self, sub_config: Dict[str, Any]) -> QWidget:
        sub_form = QWidget(self)
        sub_form_layout = QFormLayout(sub_form)

        for sub_key, sub_value in sub_config.get('current_state', {}).items():
            if isinstance(sub_value, dict):
                sub_widget = self.create_widget(sub_value)
                if sub_widget:
                    label = QLabel(sub_key)
                    label.setToolTip(sub_value.get('tooltip', ''))
                    sub_form_layout.addRow(label, sub_widget)

        sub_form.setLayout(sub_form_layout)
        return sub_form

    def add_browse_button(self, widget: QWidget, key: str, multiple_files: bool = False) -> QWidget:
        container = QWidget(self)
        layout = QHBoxLayout(container)
        layout.addWidget(widget)

        browse_btn = QPushButton("Browse", self)
        if 'file' in key:
            browse_btn.clicked.connect(lambda: self.browse_file(widget, multiple_files))
        elif 'dir' in key.lower():
            browse_btn.clicked.connect(lambda: self.browse_directory(widget))
        layout.addWidget(browse_btn)

        container.setLayout(layout)
        return container

    def on_option_change(self, selected_option: str, key: str, options: Any) -> None:
        row_index = -1
        for i in range(self.form_layout.rowCount()):
            label_item = self.form_layout.itemAt(i, QFormLayout.LabelRole)
            if label_item is not None:
                label = label_item.widget()
                if label and label.text() == key:
                    row_index = i
                    break

        if row_index == -1:
            print(f"Error: The key '{key}' was not found in the form layout.")
            return

        # Remove existing sub-forms related to this option
        while True:
            next_item = self.form_layout.itemAt(row_index + 1, QFormLayout.SpanningRole)
            if not next_item or not next_item.widget():
                break
            self.form_layout.removeRow(row_index + 1)
        
        # Find and insert the correct sub-form based on the selected option
        for option in options:
            if selected_option in option:
                sub_form = self.create_sub_form(option[selected_option])
                if sub_form:
                    self.form_layout.insertRow(row_index + 1, sub_form)
                break

    def on_back_clicked(self):
        # Navigate back to the previous page
        self.config_stack.stack.setCurrentIndex(1)

    def on_next_clicked(self):
        # Save the current page's config to the parent_dict
        updated_config = self.update_config_with_user_input(self.config, self.form_layout)
        self.configwidget_parent_dict.update(updated_config)
        
        # Navigate to the next page
        parent_widget = self.parentWidget()
        if isinstance(parent_widget, ConfigStack):
            parent_widget.stack.setCurrentIndex(parent_widget.stack.currentIndex() + 1)

    def on_value_change(self, key: str, value: bool) -> None:
        """
        Updates the parent dictionary when the checkbox value is changed.
        """
        # Update the parent dictionary with the new value
        self.configwidget_parent_dict[key] = value
        print(f"Key: {key}, New Value: {value}")



    def update_config_with_user_input(self, config: Dict[str, Any], layout: QFormLayout) -> Dict[str, Any]:
        updated_config = copy.deepcopy(config)  # Deepcopy the original config to preserve the structure

        for i in range(layout.rowCount()):
            label_item = layout.itemAt(i, QFormLayout.LabelRole)
            field_item = layout.itemAt(i, QFormLayout.FieldRole)
            
            if label_item and field_item:
                key = label_item.widget().text()  # Get the key (label)
                field = field_item.widget()  # Get the widget (QLineEdit, QSpinBox, etc.)

                # aid debug
                # if key == "Weight file":
                #     pass

                # Handle QLineEdit input (for strings, files, directories, or file lists)
                if isinstance(field, QLineEdit):
                    current_state = config[key]['current_state']
                    
                    # Check if it's a directory selection (use 'dir' in key to differentiate directories)
                    if 'dir' in key:
                        updated_config[key]['current_state'] = field.text()  # Save directory as a string
                    # Check if the field is for a list of files (List[str])
                    elif get_origin(config[key]['type_val']) == list and isinstance(current_state, list):
                        # Split the comma-separated file list back into a list of strings
                        updated_config[key]['current_state'] = field.text().split(', ') if field.text() else []
                    else:
                        # For regular strings (including single file paths)
                        updated_config[key]['current_state'] = field.text()

                # Handle QSpinBox input
                elif isinstance(field, QSpinBox):
                    updated_config[key]['current_state'] = field.value()

                # Handle QDoubleSpinBox input
                elif isinstance(field, QDoubleSpinBox):
                    updated_config[key]['current_state'] = field.value()

                # Handle QComboBox input with possible nested sub-widgets
                elif isinstance(field, QComboBox):
                    selected_value = field.currentText()
                    option_list = updated_config[key]["options"]

                    for option, option_index in zip(option_list, count()):
                        if selected_value in option.keys():
                            updated_config[key]['current_state'] = copy.deepcopy(option)

                            # Check the next row in the layout to see if it's part of the nested configuration
                            if i + 1 < layout.rowCount():
                                next_label_item = layout.itemAt(i + 1, QFormLayout.LabelRole)
                                if not next_label_item:  # If there's no label, it's likely a sub-form
                                    sub_form_widget = layout.itemAt(i + 1, QFormLayout.SpanningRole).widget()
                                    if isinstance(sub_form_widget, QWidget) and isinstance(sub_form_widget.layout(), QFormLayout):
                                        # Ensure the structure is a mapping before indexing
                                        curr = updated_config[key].get('current_state')
                                        if not isinstance(curr, dict):
                                            # If the current_state was a primitive or list (from a malformed conversion),
                                            # replace it with the expected option mapping structure.
                                            updated_config[key]['current_state'] = { selected_value: { "current_state": {} } }

                                        # If the selected option entry is missing or not a dict, create it
                                        if selected_value not in updated_config[key]['current_state'] or not isinstance(updated_config[key]['current_state'][selected_value], dict):
                                            updated_config[key]['current_state'][selected_value] = { "current_state": {} }

                                        # Ensure there is a nested dict to update
                                        nested = updated_config[key]['current_state'][selected_value].get("current_state")
                                        if not isinstance(nested, dict):
                                            updated_config[key]['current_state'][selected_value]["current_state"] = {}

                                        # Recursively update the nested configuration
                                        updated_config[key]['current_state'][selected_value]["current_state"] = self.update_config_with_user_input(
                                            updated_config[key]['current_state'][selected_value]["current_state"], sub_form_widget.layout()
                                        )
                                        # should this be a deepcopy?
                                        updated_config[key]["options"][option_index] = updated_config[key]['current_state']
                            break

                # Handle cases where the widget is a container (QWidget) with a layout
                elif isinstance(field, QWidget) and field.layout() is not None:
                    sub_layout = field.layout()

                    # Check if the type is List (e.g., List[str])
                    if get_origin(config[key]['type_val']) == list:
                        # Handle the case where the QLineEdit contains a comma-separated list of file paths (List[str])
                        for j in range(sub_layout.count()):
                            child_item = sub_layout.itemAt(j)
                            child_widget = child_item.widget()

                            if isinstance(child_widget, QLineEdit):
                                # Get the comma-separated text from the QLineEdit
                                file_list = child_widget.text().split(', ') if child_widget.text() else []
                                # Update the current state with the list of files
                                updated_config[key]['current_state'] = file_list
                                break
                    
                    #TODO: I need to update and change elements of this to remove legacy code.
                    elif "dir" in key.lower() or "file" in key.lower():
                        # Handle the case where the QLineEdit contains a comma-separated list of file paths (List[str])
                        for j in range(sub_layout.count()):
                            child_item = sub_layout.itemAt(j)
                            child_widget = child_item.widget()

                            if isinstance(child_widget, QLineEdit):
                                # Get the comma-separated text from the QLineEdit
                                text = child_widget.text()
                                # Update the current state with the list of files
                                updated_config[key]['current_state'] = text
                                break

                    elif isinstance(sub_layout, QFormLayout):
                        # Recursively update the nested form layout as before
                        updated_config[key]['current_state'] = self.update_config_with_user_input(
                            updated_config[key]['current_state'], sub_layout
                        )
                # else:
                #     print(f"Had an issue with {key}")

                
        return updated_config


    def find_sub_form_widget(self, widget: QWidget) -> Optional[QWidget]:
        """
        Utility method to locate the sub-form widget within a layout.
        This helps in locating nested layouts inside option-based widgets.
        """
        if widget is not None and widget.layout() is not None:
            for i in range(widget.layout().count()):
                item = widget.layout().itemAt(i)
                if isinstance(item.widget(), QWidget):
                    return item.widget()
        return None



    def browse_file(self, line_edit: QLineEdit, multiple_files: bool = False) -> None:
        if multiple_files:
            files, _ = QFileDialog.getOpenFileNames(self, "Select Files")
            if files:
                line_edit.setText(", ".join(files))
        else:
            file, _ = QFileDialog.getOpenFileName(self, "Select File")
            if file:
                line_edit.setText(file)

    def browse_directory(self, line_edit: QLineEdit) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Select Directory")
        if directory:
            line_edit.setText(directory)  # Ensure directory is set in QLineEdit



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ConfigStack()  # Launch the application with the stack widget
    window.show()
    sys.exit(app.exec_())



