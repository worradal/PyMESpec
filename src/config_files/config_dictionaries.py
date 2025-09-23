from typing import List
import sys
import os

# Add the project root to sys.path to allow local imports when running this file directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_files.dict_config_utils import DictConfig


# Constants for dictionary keys
SPECTRUM_DIRECTORY = "Spectrum Directory"
DATA_PER_PULSE = "Data per Pulse"
OUTPUT_DIRECTORY = "Output Directory"
STARTING_FRAME = "Starting Frame"
ENDING_FRAME = "Ending Frame"
PROJECT_NAME = "Project Name"
PROJECT_DEFAULT = "Project"
FILE_TYPE = "File Type"
CSV = "csv"
FREQUENCY_COLUMN = "Frequency column"
INTENSITY_COLUMN = "Intensity column"
STARTING_ROW_NUMBER = "Starting row number"
ENDING_ROW_NUMBER = "Ending row number"
ANALYSIS_TYPE = "Analysis type"
NONE_STR = "None"
SPECTRAL_COORDINATE_START = "Spectral coordinate start"
SPECTRAL_COORDINATE_END = "Spectral coordinate end"
# Constants for Baseline correction
BASELINE_CORRECTION_METHOD = "Baseline Correction Method"
ARPLS_TAG = "arpls"
LAMBDA_TAG = 'lambda'
STOP_RATIO = 'stop ratio'

# Constants for Chemometrics
CHEMOMETRICS_ANALYSIS_TYPE = "Chemometrics"
CHEMOMETRICS_DIRECTORY = "Chemometrics Directory"
CHEMOMETRICS_SAMPLE_STANDARDS_FILE_LIST = "Chemometrics sample standards file list"
CHEMOMETRICS_FILE_TYPE = "Chemometrics file type"
CONFIDENCE_INTERVAL = "Confidence interval"
CONSTRAINED_FITTING_CONFIGS = "Constrained fitting configs"
NORMALIZE_FIT = "normalize fit"
CONSTRAINED_FIT = "Constrained fit"
EPSILON = "epsilon"
MAX_ITER = 'max iterations for constraint'

# Constants for Phase
PHASE_ANALYSIS_TYPE = "Phase"
LIST_OF_HARMONICS = "List of harmonics"
SAVE_PHASE_DATA = "Save phase data"
SAVE_FFT_DATA = "Save fft data"
SAVE_IFFT_DATA = "Save ifft data"

# Constants for Rate Data
RATE_DATA_ANALYSIS_TYPE = "Rate"
AVERAGE_RATE = "Average rate"
PLOT_DATA = "Plot data"
FITTING_FUNCTION = "Fitting function"
DESIRED_FREQUENCY_FOR_RATE_DATA = "Desired frequency for rate data"

NONE_DICT_CONFIG = DictConfig(parent_key="None", type_val=dict, current_state=dict())

dict_config_general = {
    SPECTRUM_DIRECTORY: DictConfig(parent_key=SPECTRUM_DIRECTORY, type_val=str, current_state="", is_required=True, select_location=True, tooltip="contains full path of folder / directory with the files").get_config(),
    DATA_PER_PULSE: DictConfig(parent_key=DATA_PER_PULSE, type_val=int, current_state=10, is_required=True, select_location=False, tooltip="number of time points collected.").get_config(),
    OUTPUT_DIRECTORY: DictConfig(parent_key=OUTPUT_DIRECTORY, type_val=str, current_state="", is_required=True, select_location=True, tooltip="output directory where the output CSV files will be created. Cannot be the same as spectrum_dir.").get_config(),
    PROJECT_NAME: DictConfig(parent_key=PROJECT_NAME, type_val=str, current_state=PROJECT_DEFAULT, is_required=True, select_location=False, tooltip="The name of the project save file.").get_config(),
    STARTING_FRAME: DictConfig(parent_key=STARTING_FRAME, type_val=int, current_state=0, is_required=True, select_location=False, tooltip="start frame").get_config(),
    ENDING_FRAME: DictConfig(parent_key=ENDING_FRAME, type_val=int, current_state=-1, is_required=True, select_location=False, tooltip="last frame, -1 take all frames into account").get_config(),
    SPECTRAL_COORDINATE_START: DictConfig(parent_key=SPECTRAL_COORDINATE_START, type_val=int, current_state=0, is_required=True, select_location=False, tooltip="start of the spectral coordinate").get_config(),
    SPECTRAL_COORDINATE_END: DictConfig(parent_key=SPECTRAL_COORDINATE_END, type_val=int, current_state=-1, is_required=True, select_location=False, tooltip="end of the spectral coordinate").get_config(),
    FILE_TYPE: DictConfig(parent_key=FILE_TYPE, type_val=dict, current_state={"None": NONE_DICT_CONFIG.get_config()}, is_required=True, select_location=False, tooltip="file type of files, csv or txt", has_options=True, options=[
        {"None": NONE_DICT_CONFIG.get_config()},
        {CSV: DictConfig(parent_key=CSV, type_val=dict, current_state={
            FREQUENCY_COLUMN: DictConfig(parent_key=FREQUENCY_COLUMN, type_val=str, current_state='frequency', tooltip="frequency column name").get_config(),
            INTENSITY_COLUMN: DictConfig(parent_key=INTENSITY_COLUMN, type_val=str, current_state='intensity', tooltip="intensity column name").get_config(),
            STARTING_ROW_NUMBER: DictConfig(parent_key=STARTING_ROW_NUMBER, type_val=int, current_state=1, tooltip="row number to start reading data").get_config(),
            ENDING_ROW_NUMBER: DictConfig(parent_key=ENDING_ROW_NUMBER, type_val=int, current_state=-1, tooltip="row number to end reading data").get_config()
        }).get_config()},
        {"txt": DictConfig(parent_key="txt", type_val=dict, current_state=dict()).get_config()}
    ]).get_config(),
    BASELINE_CORRECTION_METHOD: DictConfig(parent_key=BASELINE_CORRECTION_METHOD, type_val=dict, current_state={"None": NONE_DICT_CONFIG.get_config()}, is_required=True, select_location=False, tooltip="baseline correction method to use, available options: 'arpls'", has_options=True, options=[
        {"None": NONE_DICT_CONFIG.get_config()},
        {ARPLS_TAG: DictConfig(parent_key=ARPLS_TAG, type_val=dict, current_state={
            LAMBDA_TAG: DictConfig(parent_key=LAMBDA_TAG, type_val=int, current_state=100000, tooltip="smoothness parameter (higher values give smoother baselines)").get_config(),
            STOP_RATIO: DictConfig(parent_key=STOP_RATIO, type_val=float, current_state=.000001, tooltip="convergence criterion").get_config(),
            MAX_ITER: DictConfig(parent_key=MAX_ITER, type_val=int, current_state=10, tooltip="maximum number of algorithm iterations").get_config()
        }).get_config()}
    ]).get_config()
}

dict_config_chemometrics = {
    ANALYSIS_TYPE: DictConfig(parent_key=ANALYSIS_TYPE, type_val=str, current_state=CHEMOMETRICS_ANALYSIS_TYPE, is_required=True, select_location=False, tooltip="analysis type", editable=False).get_config(),
    CHEMOMETRICS_DIRECTORY: DictConfig(parent_key=CHEMOMETRICS_DIRECTORY, type_val=str, current_state="", is_required=True, select_location=True, tooltip="contains full path of folder / directory with the files").get_config(),
    CHEMOMETRICS_SAMPLE_STANDARDS_FILE_LIST: DictConfig(parent_key=CHEMOMETRICS_SAMPLE_STANDARDS_FILE_LIST, type_val=List[str], current_state=[], is_required=True, select_location=False, tooltip="contains list of files to be used as sample standards").get_config(),
    CHEMOMETRICS_FILE_TYPE: DictConfig(parent_key=CHEMOMETRICS_FILE_TYPE, type_val=dict, current_state={"None": NONE_DICT_CONFIG.get_config()}, is_required=True, select_location=False, tooltip="file type of files, csv or txt", has_options=True, options=[
        {"None": NONE_DICT_CONFIG.get_config()},
        {"csv": DictConfig(parent_key="csv", type_val=dict, current_state={
            FREQUENCY_COLUMN: DictConfig(parent_key=FREQUENCY_COLUMN, type_val=str, current_state='frequency', tooltip="frequency column name").get_config(),
            INTENSITY_COLUMN: DictConfig(parent_key=INTENSITY_COLUMN, type_val=str, current_state='intensity', tooltip="intensity column name").get_config(),
            STARTING_ROW_NUMBER: DictConfig(parent_key=STARTING_ROW_NUMBER, type_val=int, current_state=0, tooltip="row number to start reading data").get_config(),
            ENDING_ROW_NUMBER: DictConfig(parent_key=ENDING_ROW_NUMBER, type_val=int, current_state=-1, tooltip="row number to end reading data").get_config()
        }).get_config()},
        {"txt": DictConfig(parent_key="txt", type_val=dict, current_state=dict()).get_config()}
    ]).get_config(),
    CONFIDENCE_INTERVAL: DictConfig(parent_key=CONFIDENCE_INTERVAL, type_val=float, current_state=0.95, is_required=True, select_location=False, tooltip="confidence interval for chemometrics").get_config(),
    CONSTRAINED_FITTING_CONFIGS: DictConfig(parent_key=CONSTRAINED_FITTING_CONFIGS, type_val=dict, current_state={
        CONSTRAINED_FIT: DictConfig(parent_key=CONSTRAINED_FIT, type_val=bool, current_state=True, tooltip="to perform non-negative linear least squares").get_config(),
        EPSILON: DictConfig(parent_key=EPSILON, type_val=float, current_state=10e-6, tooltip="epsilon").get_config(),
        MAX_ITER: DictConfig(parent_key=MAX_ITER, type_val=int, current_state=1000, tooltip="max iters for constraint").get_config()
    }, is_required=True, select_location=False, tooltip="configs for constrained fitting").get_config(),
    NORMALIZE_FIT: DictConfig(parent_key=NORMALIZE_FIT, type_val=bool, current_state=True, is_required=True, select_location=False, tooltip="normalize fit").get_config()
}

dict_config_phase = {
    ANALYSIS_TYPE: DictConfig(parent_key=ANALYSIS_TYPE, type_val=str, current_state=PHASE_ANALYSIS_TYPE, is_required=True, select_location=False, tooltip="analysis type", editable=False).get_config(),
    LIST_OF_HARMONICS: DictConfig(parent_key=LIST_OF_HARMONICS, type_val=List[int], current_state=[1], is_required=False, select_location=False, tooltip="list of harmonics").get_config(),
    SAVE_PHASE_DATA: DictConfig(parent_key=SAVE_PHASE_DATA, type_val=bool, current_state=True, is_required=True, select_location=False, tooltip="save phase data").get_config(),
    SAVE_FFT_DATA: DictConfig(parent_key=SAVE_FFT_DATA, type_val=bool, current_state=True, is_required=True, select_location=False, tooltip="save fft data").get_config(),
    SAVE_IFFT_DATA: DictConfig(parent_key=SAVE_IFFT_DATA, type_val=bool, current_state=True, is_required=True, select_location=False, tooltip="save ifft data").get_config()
}

dict_config_rate_data = {
    ANALYSIS_TYPE: DictConfig(parent_key=ANALYSIS_TYPE, type_val=str, current_state=RATE_DATA_ANALYSIS_TYPE, is_required=True, select_location=False, tooltip="analysis type", editable=False).get_config(),
    AVERAGE_RATE: DictConfig(parent_key=AVERAGE_RATE, type_val=bool, current_state=True, is_required=True, select_location=False, tooltip="average rate, setting this to false is very slow").get_config(),
    PLOT_DATA: DictConfig(parent_key=PLOT_DATA, type_val=bool, current_state=False, is_required=True, select_location=False, tooltip="plot data").get_config(),
    FITTING_FUNCTION: DictConfig(parent_key=FITTING_FUNCTION, type_val=str, current_state="exponential", is_required=True, select_location=False, tooltip="fitting function to use the only current option is 'exponential' more may be added in the future").get_config(),
    DESIRED_FREQUENCY_FOR_RATE_DATA: DictConfig(parent_key=DESIRED_FREQUENCY_FOR_RATE_DATA, type_val=int, current_state=100, is_required=True, select_location=False, tooltip="The desired frequency in the spectrum to extract rates from set to -1 for all frequencies").get_config()
}

if __name__ == "__main__":
    print("loaded")
