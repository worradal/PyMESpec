import sys
import os
import json
import yaml
from src.config_files.dict_config_utils import convert_clean_config_to_gui_format, CURRENT_STATE
from src.run_configs.overall_run_config import OverallRunConfig

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pymespec.py <config_file.yaml|json>")
        sys.exit(1)

    config_path = sys.argv[1]
    ext = os.path.splitext(config_path)[1].lower()
    with open(config_path, 'r') as f:
        if ext in ['.yaml', '.yml']:
            config = yaml.safe_load(f)
        elif ext == '.json':
            config = json.load(f)
        else:
            print(f"Unsupported config file extension: {ext}")
            sys.exit(1)

    def is_clean_config(cfg):
        """Return True if cfg looks like a clean YAML/run-config (primitive leaves),
        False if it already contains GUI DictConfig metadata (contains CURRENT_STATE or parent_key entries).
        """
        if not isinstance(cfg, dict):
            return True
        for page_key, page_config in cfg.items():
            if isinstance(page_config, dict):
                for config_key, config_value in page_config.items():
                    if isinstance(config_value, dict):
                        # presence of CURRENT_STATE or DictConfig metadata indicates GUI format
                        if CURRENT_STATE in config_value:
                            return False
                        if any(k in config_value for k in ['parent_key', 'type_val', 'tooltip']):
                            return False
        return True

    print("DEBUG: Raw config before possible conversion:", config)
    if is_clean_config(config):
        config = convert_clean_config_to_gui_format(config)
        print("\nDEBUG: Config after conversion:", config)
    else:
        print("DEBUG: Detected GUI-formatted config; skipping conversion")

    print("\nDEBUG: Top-level config keys after conversion:", list(config.keys()))
    print("DEBUG: Analysis_Page keys after conversion:", list(config.get('Analysis_Page', {}).keys()))
    if 'General_Page' not in config:
        print("ERROR: 'General_Page' not found in config! Top-level keys:", list(config.keys()))
        print("Config object:", config)
        sys.exit(1)

    run_config = OverallRunConfig(config)
    run_config.load_data()
    run_config.perform_analysis()
    print("Analysis complete.")
