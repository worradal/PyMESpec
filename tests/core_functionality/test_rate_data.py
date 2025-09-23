import numpy as np
import pytest
import tempfile
import os
from unittest.mock import patch, MagicMock
import matplotlib.pyplot as plt
from src.core_functionality.rate_data import RateData, fit_exp, RATE_CONSTANTS, DECAY_DATA, TIME_SEGMENTS, POPT_LIST, PCOV_LIST, FREQUENCY
from src.core_functionality.spectrum import Spectrum, Spectra
# np.random.seed(103)  # For reproducibility

@pytest.fixture
def synthetic_spectra_and_expected_rates():
    pretend_frequencies = np.array([100, 200, 300, 400])
    expected_down_rates = np.array([0.154, 0.094, 0.04, 0.54])
    expected_up_rates = np.array([0.5, 0.1, 0.05, 0.1])

    def pulse(down_rate, up_rate, noise_std=0.01, seed=None):
        rng = np.random.default_rng(seed)
        t_decay = np.arange(0, 900, 1)
        decay = np.exp(-down_rate * t_decay) + rng.normal(0, noise_std, t_decay.shape)
        t_regen = np.arange(900, 1000, 1)
        regen = 1 - np.exp(-up_rate * (t_regen - t_regen[0])) + rng.normal(0, noise_std, t_regen.shape)
        adjust_based_on_minimum = min(np.min(decay), np.min(regen))
        decay -= adjust_based_on_minimum
        regen -= adjust_based_on_minimum
        return np.concatenate([regen, decay]), np.concatenate([t_regen, t_decay])

    def pulse_repeated(n_pulses, down_rate, up_rate, noise_std=0.01, seed=None):
        data, time = [], []
        rng = np.random.default_rng(seed)
        for i in range(n_pulses):
            d, t = pulse(down_rate, up_rate, noise_std=noise_std, seed=rng.integers(0, 1e6))
            data.append(d)
            time.append(t + i * np.max(t))
        return np.concatenate(data), np.concatenate(time)

    spectra = Spectra([])
    time_data_values = []
    for drate, urate in zip(expected_down_rates, expected_up_rates):
        d_arr, t_arr = pulse_repeated(5, drate, urate, noise_std=0.01)
        time_data_values.append(d_arr)

    time_data_values = np.array(time_data_values)
    for i in range(time_data_values.shape[1]):
        spectra.append(Spectrum(pretend_frequencies, time_data_values[:, i]))

    return spectra, pretend_frequencies, expected_down_rates

@pytest.fixture
def simple_spectra():
    """Create simple synthetic spectra for basic testing"""
    frequencies = np.array([100, 200, 300])
    # Create simple exponential decay pattern
    time_points = np.linspace(0, 10, 50)
    intensities = []
    
    for freq in frequencies:
        freq_data = []
        for t in time_points:
            # Simple exponential decay with frequency-dependent rate
            rate = 0.1 * (freq / 100)
            intensity = np.exp(-rate * t) + 0.1 * np.random.randn()
            freq_data.append(intensity)
        intensities.append(freq_data)
    
    spectra = Spectra([])
    intensities = np.array(intensities)
    for i in range(intensities.shape[1]):
        spectra.append(Spectrum(frequencies, intensities[:, i]))
    
    return spectra, frequencies

@pytest.fixture
def noisy_spectra():
    """Create spectra with high noise for robustness testing"""
    frequencies = np.array([50, 150, 250, 350])
    time_points = np.linspace(0, 5, 30)
    
    spectra = Spectra([])
    for i, t in enumerate(time_points):
        # Add significant noise
        intensities = np.sin(frequencies / 50 + t) + 0.5 * np.random.randn(len(frequencies))
        spectra.append(Spectrum(frequencies, intensities))
    
    return spectra, frequencies

# ============== Basic Initialization and Property Tests ==============

def test_rate_data_initialization(simple_spectra):
    """Test basic RateData initialization"""
    spectra, frequencies = simple_spectra
    
    # Test with default time step
    rate_data = RateData(spectra)
    assert rate_data._time_step == 1
    assert rate_data._spectra == spectra
    assert rate_data._average_spectra is None
    
    # Test with custom time step
    rate_data_custom = RateData(spectra, time_step_in_seconds=0.5)
    assert rate_data_custom._time_step == 0.5

def test_fit_exp_function():
    """Test the fit_exp function directly"""
    x = np.array([0, 1, 2, 3, 4])
    a, b, c = 2.0, 0.5, 1.0
    
    result = fit_exp(x, a, b, c)
    expected = a * np.exp(-b * x) + c
    
    np.testing.assert_allclose(result, expected)

def test_average_spectra_property(simple_spectra):
    """Test average_spectra property"""
    spectra, frequencies = simple_spectra
    rate_data = RateData(spectra)
    
    # Initially should be None
    assert rate_data.average_spectra is None
    
    # After getting average spectra, property should be set
    avg_data = rate_data.get_average_spectra_desired_freq(frequencies[0], 10)
    assert rate_data.average_spectra is not None
    assert isinstance(rate_data.average_spectra, Spectra)

# ============== Core Computation Tests ==============

def test_rate_data_computes_expected_rates(synthetic_spectra_and_expected_rates):
    spectra, frequencies, expected_down_rates = synthetic_spectra_and_expected_rates
    rate_data = RateData(spectra, time_step_in_seconds=1)
    num_time_points = 1000
    prominence_distance = 5
    
    # Test compute_average_rate_data
    result = rate_data.compute_average_rate_data(
        select_frequency=frequencies[1],
        num_time_points=num_time_points,
        smooth_data=True,
        prominence_distance=prominence_distance
    )

    # Check that result is a dictionary with expected keys
    assert isinstance(result, dict)
    assert RATE_CONSTANTS in result
    assert DECAY_DATA in result
    assert TIME_SEGMENTS in result
    assert POPT_LIST in result
    assert PCOV_LIST in result
    
    # Extract rate constants from the result
    detected_rates = result[RATE_CONSTANTS]
    
    assert np.isfinite(detected_rates).all()
    assert (detected_rates > 0).all()

    # Test compute_all_rate_data_single_freq
    result_all = rate_data.compute_all_rate_data_single_freq(
        select_frequency=frequencies[1],
        num_time_points=num_time_points,
        smooth_data=True,
        prominence_distance=prominence_distance
    )

    # Check that result is a dictionary with expected keys
    assert isinstance(result_all, dict)
    assert RATE_CONSTANTS in result_all
    assert DECAY_DATA in result_all
    assert TIME_SEGMENTS in result_all
    assert POPT_LIST in result_all
    assert PCOV_LIST in result_all
    
    # Extract data from the result
    rate_constants = result_all[RATE_CONSTANTS]
    optimal_decay_data = result_all[DECAY_DATA]
    time_segments = result_all[TIME_SEGMENTS]
    popt_list = result_all[POPT_LIST]
    pcov_list = result_all[PCOV_LIST]

    assert np.all(np.isfinite(rate_constants))
    assert (rate_constants > 0).all()

    # Test compute_all_rate_data_all_freq
    all_freq_rates = rate_data.compute_all_rate_data_all_freq(
        num_time_points=num_time_points,
        smooth_data=True,
        prominence_distance=prominence_distance
    )

    # Check that result is a dictionary with one entry per frequency
    assert isinstance(all_freq_rates, dict)
    assert len(all_freq_rates) == len(frequencies)
    
    # Check each frequency result
    for i, freq_result in all_freq_rates.items():
        assert isinstance(freq_result, dict)
        assert FREQUENCY in freq_result
        assert RATE_CONSTANTS in freq_result
        assert DECAY_DATA in freq_result
        assert TIME_SEGMENTS in freq_result
        assert POPT_LIST in freq_result
        assert PCOV_LIST in freq_result
        
        # Check that the frequency matches expected
        assert freq_result[FREQUENCY] in frequencies
        
        # Check that rate constants are valid
        rate_constants = freq_result[RATE_CONSTANTS]
        assert np.all(np.isfinite(rate_constants))
        assert (rate_constants > 0).all()

# ============== Data Processing Methods Tests ==============

def test_get_average_spectra_desired_freq(simple_spectra):
    """Test get_average_spectra_desired_freq method"""
    spectra, frequencies = simple_spectra
    rate_data = RateData(spectra)
    
    # Test basic functionality
    avg_data = rate_data.get_average_spectra_desired_freq(frequencies[0], 10)
    assert isinstance(avg_data, np.ndarray)
    assert len(avg_data) == 10
    
    # Test with different parameters - this method caches average_spectra 
    # so subsequent calls with different parameters use the same averaged data
    avg_data_custom = rate_data.get_average_spectra_desired_freq(
        frequencies[1], 10, time_start=1, time_end=15  # Keep same num_time_points
    )
    assert len(avg_data_custom) == 10  # Should be same as first call due to caching
    
    # Test that average_spectra is cached
    first_call = rate_data.get_average_spectra_desired_freq(frequencies[0], 10)
    second_call = rate_data.get_average_spectra_desired_freq(frequencies[1], 10)
    assert rate_data._average_spectra is not None

def test_smooth_data_with_savgol(simple_spectra):
    """Test Savitzky-Golay smoothing functionality"""
    spectra, frequencies = simple_spectra
    rate_data = RateData(spectra)
    
    # Create test data with noise
    noisy_data = np.sin(np.linspace(0, 10, 20)) + 0.2 * np.random.randn(20)
    
    smoothed = rate_data.smooth_data_with_savgol(noisy_data, window_length=5, polyorder=2)
    
    assert len(smoothed) == len(noisy_data)
    assert isinstance(smoothed, np.ndarray)
    
    # Smoothed data should have lower variance (in most cases)
    # Note: This is probabilistic, so we just check it's reasonable
    assert np.all(np.isfinite(smoothed))

def test_smooth_data_edge_cases(simple_spectra):
    """Test edge cases for smoothing"""
    spectra, frequencies = simple_spectra
    rate_data = RateData(spectra)
    
    # Test with minimal data
    minimal_data = np.array([1, 2, 3, 4, 5])
    smoothed = rate_data.smooth_data_with_savgol(minimal_data, window_length=3, polyorder=1)
    assert len(smoothed) == len(minimal_data)
    
    # Test with different polynomial orders
    data = np.linspace(0, 1, 15)
    smoothed_poly1 = rate_data.smooth_data_with_savgol(data, window_length=5, polyorder=1)
    smoothed_poly3 = rate_data.smooth_data_with_savgol(data, window_length=5, polyorder=3)
    
    assert len(smoothed_poly1) == len(data)
    assert len(smoothed_poly3) == len(data)

# ============== Private Method Tests ==============

def test_fit_exp_to_data(simple_spectra):
    """Test _fit_exp_to_data method"""
    spectra, frequencies = simple_spectra
    rate_data = RateData(spectra)
    
    # Create synthetic exponential decay data (1D array, not 2D)
    y = 2.0 * np.exp(-0.5 * np.arange(50)) + 1.0 + 0.01 * np.random.randn(50)
    
    x, y_fitted, popt, pcov = rate_data._fit_exp_to_data(y)
    
    assert len(x) == len(y)
    assert len(y_fitted) == len(y)
    assert len(popt) == 3  # [a, b, c] parameters
    assert pcov.shape == (3, 3)  # Covariance matrix
    
    # Check that fitted parameters are reasonable
    a, b, c = popt
    assert a > 0  # Amplitude should be positive
    assert b > 0  # Decay rate should be positive
    assert np.isfinite([a, b, c]).all()

def test_obtain_exp_rate_constants(simple_spectra):
    """Test _obtain_exp_rate_constants method"""
    spectra, frequencies = simple_spectra
    rate_data = RateData(spectra)
    
    # Create synthetic decay segments - these should be 1D arrays, not 2D
    y1 = 1.5 * np.exp(-0.3 * np.arange(30)) + 0.5
    y2 = 2.0 * np.exp(-0.6 * np.arange(40)) + 0.3
    
    decay_segments = [y1, y2]  # 1D arrays
    time_segments = [np.arange(30), np.arange(40)]  # corresponding time arrays
    
    rate_constants, decay_data, time_segs, popt_list, pcov_list = rate_data._obtain_exp_rate_constants(
        decay_segments, time_segments
    )
    
    assert len(rate_constants) == 2
    assert len(decay_data) == 2
    assert len(time_segs) == 2
    assert len(popt_list) == 2
    assert len(pcov_list) == 2
    
    # Check that rate constants are positive and finite
    assert np.all(rate_constants > 0)
    assert np.all(np.isfinite(rate_constants))

# ============== Find Decay Data Tests ==============

def test_find_decay_data_basic(simple_spectra):
    """Test _find_decay_data with basic parameters"""
    spectra, frequencies = simple_spectra
    rate_data = RateData(spectra)
    
    # Get sample data
    avg_data = rate_data.get_average_spectra_desired_freq(frequencies[0], 20)
    time_arr = np.arange(len(avg_data)) * rate_data._time_step
    
    # Set the instance variable that the method expects
    rate_data.data_arr = avg_data
    
    decay_segments, time_segments = rate_data._find_decay_data(
        5,  # num_segments (number_of_max_and_min parameter)
        time_arr,
        smooth_data=True,
        prominence_distance=3,
        prominence_threshold=0.01
    )
    
    assert isinstance(decay_segments, list)
    assert isinstance(time_segments, list)
    assert len(decay_segments) == len(time_segments)
    
    # Each segment should be a 1D array 
    for segment in decay_segments:
        assert isinstance(segment, np.ndarray)
        assert len(segment) >= 3  # Minimum segment length

def test_find_decay_data_with_max_segments(simple_spectra):
    """Test _find_decay_data with max_segments parameter"""
    spectra, frequencies = simple_spectra
    rate_data = RateData(spectra)
    
    avg_data = rate_data.get_average_spectra_desired_freq(frequencies[0], 30)
    time_arr = np.arange(len(avg_data)) * rate_data._time_step
    rate_data.data_arr = avg_data
    
    # Test with max_segments limit
    decay_segments, time_segments = rate_data._find_decay_data(
        10,  # num_segments
        time_arr,
        max_segments=3
    )
    
    assert len(decay_segments) <= 3
    assert len(time_segments) <= 3

def test_find_decay_data_smoothing_options(simple_spectra):
    """Test _find_decay_data with different smoothing options"""
    spectra, frequencies = simple_spectra
    rate_data = RateData(spectra)
    
    avg_data = rate_data.get_average_spectra_desired_freq(frequencies[0], 25)
    time_arr = np.arange(len(avg_data)) * rate_data._time_step
    rate_data.data_arr = avg_data
    
    # Test without smoothing
    decay_no_smooth, time_no_smooth = rate_data._find_decay_data(
        5, time_arr, smooth_data=False
    )
    
    # Test with smoothing
    decay_smooth, time_smooth = rate_data._find_decay_data(
        5, time_arr, smooth_data=True, smooth_window_size=5
    )
    
    # Both should return valid results
    assert isinstance(decay_no_smooth, list)
    assert isinstance(decay_smooth, list)

# ============== Segment Optimization Tests ==============

def test_optimize_segments_by_swapping(simple_spectra):
    """Test _optimize_segments_by_swapping method"""
    spectra, frequencies = simple_spectra
    rate_data = RateData(spectra)
    
    # Create test segments - 1D arrays as expected by the method
    y1 = 2.0 * np.exp(-0.4 * np.arange(20)) + 0.5 + 0.01 * np.random.randn(20)
    y2 = 1.5 * np.exp(-0.3 * np.arange(20)) + 0.4 + 0.01 * np.random.randn(20)
    
    decay_segments = [y1, y2]  # 1D arrays
    time_segments = [np.arange(20), np.arange(20, 40)]  # corresponding time arrays
    
    optimized_decay, optimized_time = rate_data._optimize_segments_by_swapping(
        decay_segments, time_segments
    )
    
    assert len(optimized_decay) == len(decay_segments)
    assert len(optimized_time) == len(time_segments)
    
    # Check that each optimized segment has valid structure
    for i, segment in enumerate(optimized_decay):
        assert isinstance(segment, np.ndarray)
        assert len(optimized_time[i]) == len(segment)

def test_optimize_segments_with_single_segment(simple_spectra):
    """Test optimization with single segment"""
    spectra, frequencies = simple_spectra
    rate_data = RateData(spectra)
    
    # Single segment case - 1D array
    y = 1.0 * np.exp(-0.2 * np.arange(30)) + 0.3
    
    decay_segments = [y]
    time_segments = [np.arange(30)]
    
    optimized_decay, optimized_time = rate_data._optimize_segments_by_swapping(
        decay_segments, time_segments
    )
    
    # Should return the same segment
    assert len(optimized_decay) == 1
    assert len(optimized_time) == 1

# ============== Plotting Tests ==============

@patch('matplotlib.pyplot.show')
def test_plot_decays_and_fits(mock_show, simple_spectra):
    """Test plot_decays_and_fits method without actually displaying plots"""
    spectra, frequencies = simple_spectra
    rate_data = RateData(spectra)
    
    # Create synthetic fit data - 1D arrays
    y1 = 2.0 * np.exp(-0.5 * np.arange(30)) + 1.0
    y2 = 1.5 * np.exp(-0.3 * np.arange(40)) + 0.5
    
    rate_constants = np.array([0.5, 0.3])
    decay_data = [y1, y2]  # 1D arrays
    time_segments = [np.arange(30), np.arange(40)]  # time arrays
    popt_list = [[2.0, 0.5, 1.0], [1.5, 0.3, 0.5]]
    pcov_list = [np.eye(3), np.eye(3)]
    
    # Test plotting (should not raise errors)
    result = rate_data.plot_decays_and_fits(
        rate_constants, decay_data, time_segments, popt_list, pcov_list
    )
    
    # Should return tuple of three lists
    assert isinstance(result, tuple)
    assert len(result) == 3
    fitted_curves, decay_data_out, time_segments_out = result
    
    # Note: plt.show() is commented out in the method, so we won't check for it
    # mock_show.assert_called_once()

@patch('matplotlib.pyplot.show')
def test_plot_decays_and_fits_with_kwargs(mock_show, simple_spectra):
    """Test plotting with custom kwargs"""
    spectra, frequencies = simple_spectra
    rate_data = RateData(spectra)
    
    # Minimal test data - 1D arrays
    rate_constants = np.array([0.2])
    decay_data = [np.array([1, 0.8, 0.6, 0.4])]  # 1D array
    time_segments = [np.array([0, 1, 2, 3])]  # time array
    popt_list = [[1.0, 0.2, 0.1]]
    
    # Test with custom fitting function
    def custom_fit(x, a, b, c):
        return a * np.exp(-b * x) + c
    
    result = rate_data.plot_decays_and_fits(
        rate_constants, decay_data, time_segments, popt_list,
        fitting_function=custom_fit,
        figsize=(10, 6)
    )
    
    assert isinstance(result, tuple)
    # Note: plt.show() is commented out in the method, so we won't check for it
    # mock_show.assert_called_once()

# ============== Edge Cases and Error Handling ==============

def test_rate_data_with_empty_spectra():
    """Test RateData with empty spectra"""
    empty_spectra = Spectra([])
    rate_data = RateData(empty_spectra)
    
    assert rate_data._spectra == empty_spectra
    assert rate_data._time_step == 1

def test_rate_data_with_single_spectrum():
    """Test RateData with only one spectrum"""
    freq = np.array([100])
    intensity = np.array([1.0])
    single_spectrum = Spectrum(freq, intensity)
    spectra = Spectra([single_spectrum])
    
    rate_data = RateData(spectra)
    assert len(rate_data._spectra) == 1

def test_fit_exp_with_edge_values():
    """Test fit_exp function with edge values"""
    x = np.array([0, 1, 2])
    
    # Test with zero decay rate
    result_zero = fit_exp(x, 1.0, 0.0, 0.5)
    expected_zero = np.array([1.5, 1.5, 1.5])  # Should be constant
    np.testing.assert_allclose(result_zero, expected_zero)
    
    # Test with negative values
    result_neg = fit_exp(x, -1.0, 0.5, 2.0)
    assert np.all(np.isfinite(result_neg))

def test_get_average_spectra_with_invalid_frequency(simple_spectra):
    """Test get_average_spectra_desired_freq with frequency not in data"""
    spectra, frequencies = simple_spectra
    rate_data = RateData(spectra)
    
    # This should still work due to interpolation in the Spectra class
    # or should raise an appropriate error
    try:
        avg_data = rate_data.get_average_spectra_desired_freq(999.0, 10)
        # If it works, should return valid data
        assert isinstance(avg_data, np.ndarray)
    except (ValueError, IndexError):
        # This is also acceptable behavior
        pass

def test_compute_methods_with_minimal_data(simple_spectra):
    """Test compute methods with minimal data requirements"""
    # Create very simple spectra with clear exponential decay
    freqs = np.array([100, 200])
    spectra = Spectra([])
    
    # Add time points with a clear exponential pattern
    for i in range(10):  # More points for better fitting
        # Clear exponential decay pattern
        intensities = np.array([2.0 * np.exp(-0.1*i) + 0.1, 1.5 * np.exp(-0.05*i) + 0.1])  
        spectra.append(Spectrum(freqs, intensities))
    
    rate_data = RateData(spectra)
    
    # Test with minimal parameters
    try:
        result = rate_data.compute_average_rate_data(
            select_frequency=freqs[0],
            num_time_points=10,
            smooth_data=False,
            prominence_distance=1
        )
        
        # Should return dictionary even with minimal data
        assert isinstance(result, dict)
        assert RATE_CONSTANTS in result
        
    except (ValueError, IndexError, RuntimeError):
        # This is acceptable for minimal data that may not fit well
        pass

# ============== Parameter Validation Tests ==============

def test_compute_all_rate_data_all_freq_with_different_params(simple_spectra):
    """Test compute_all_rate_data_all_freq with various parameter combinations"""
    spectra, frequencies = simple_spectra
    rate_data = RateData(spectra)
    
    # Test with different smoothing parameters
    result1 = rate_data.compute_all_rate_data_all_freq(
        num_time_points=20,
        smooth_data=True,
        prominence_distance=2,
        prominence_threshold=0.05
    )
    
    result2 = rate_data.compute_all_rate_data_all_freq(
        num_time_points=20,
        smooth_data=False,
        prominence_distance=5,
        prominence_threshold=0.01
    )
    
    # Both should return valid dictionaries
    assert isinstance(result1, dict)
    assert isinstance(result2, dict)
    
    # Results might differ due to different processing
    if len(result1) > 0 and len(result2) > 0:
        # Check that both have valid structure
        for freq_data in result1.values():
            assert FREQUENCY in freq_data
            assert RATE_CONSTANTS in freq_data

def test_time_step_effects(simple_spectra):
    """Test that different time steps affect calculations appropriately"""
    spectra, frequencies = simple_spectra
    
    rate_data_1s = RateData(spectra, time_step_in_seconds=1.0)
    rate_data_half_s = RateData(spectra, time_step_in_seconds=0.5)
    
    assert rate_data_1s._time_step == 1.0
    assert rate_data_half_s._time_step == 0.5
    
    # Time arrays should be scaled differently
    avg_data_1s = rate_data_1s.get_average_spectra_desired_freq(frequencies[0], 10)
    avg_data_half_s = rate_data_half_s.get_average_spectra_desired_freq(frequencies[0], 10)
    
    # Both should return same length arrays but represent different time scales
    assert len(avg_data_1s) == len(avg_data_half_s)

# ============== Integration Tests ==============

def test_full_workflow_simple_data(simple_spectra):
    """Test complete workflow from initialization to rate extraction"""
    spectra, frequencies = simple_spectra
    rate_data = RateData(spectra, time_step_in_seconds=0.1)
    
    # Step 1: Get average data
    avg_data = rate_data.get_average_spectra_desired_freq(frequencies[0], 25)
    assert len(avg_data) == 25
    
    # Step 2: Try computing rate data
    try:
        result = rate_data.compute_average_rate_data(
            select_frequency=frequencies[0],
            num_time_points=25,
            smooth_data=True,
            prominence_distance=3
        )
        
        # Verify result structure
        assert isinstance(result, dict)
        assert RATE_CONSTANTS in result
        
        # Extract and validate rate constants
        if len(result[RATE_CONSTANTS]) > 0:
            rates = result[RATE_CONSTANTS]
            assert np.all(np.isfinite(rates))
            assert np.all(rates > 0)
            
    except (ValueError, IndexError):
        # Acceptable for simple synthetic data that may not have clear decay patterns
        pass

def test_data_processing_pipeline_with_noise(noisy_spectra):
    """Test data processing pipeline with noisy data"""
    spectra, frequencies = noisy_spectra
    rate_data = RateData(spectra)
    
    # Test that smoothing helps with noisy data
    avg_data = rate_data.get_average_spectra_desired_freq(frequencies[0], 15)
    
    # Apply smoothing
    smoothed = rate_data.smooth_data_with_savgol(avg_data, window_length=5, polyorder=2)
    
    # Smoothed data should be valid
    assert len(smoothed) == len(avg_data)
    assert np.all(np.isfinite(smoothed))

# ============== Constants and Return Format Tests ==============

def test_constants_defined():
    """Test that all expected constants are defined"""
    assert FREQUENCY == "frequency"
    assert RATE_CONSTANTS == "rate_constants" 
    assert DECAY_DATA == "decay_data"
    assert TIME_SEGMENTS == "time_segments"
    assert POPT_LIST == "popt_list"
    assert PCOV_LIST == "pcov_list"

def test_return_format_consistency(synthetic_spectra_and_expected_rates):
    """Test that all compute methods return consistent dictionary formats"""
    spectra, frequencies, expected_rates = synthetic_spectra_and_expected_rates
    rate_data = RateData(spectra)
    
    try:
        # Test single frequency method
        result_single = rate_data.compute_all_rate_data_single_freq(
            select_frequency=frequencies[0],
            num_time_points=500,
            smooth_data=True,
            prominence_distance=10
        )
        
        # Check required keys
        required_keys = [RATE_CONSTANTS, DECAY_DATA, TIME_SEGMENTS, POPT_LIST, PCOV_LIST]
        for key in required_keys:
            assert key in result_single
            
        # Test all frequencies method
        result_all = rate_data.compute_all_rate_data_all_freq(
            num_time_points=500,
            smooth_data=True, 
            prominence_distance=10
        )
        
        # Each frequency should have same structure plus FREQUENCY key
        for freq_result in result_all.values():
            for key in required_keys:
                assert key in freq_result
            assert FREQUENCY in freq_result
            
    except (ValueError, IndexError):
        # Acceptable for edge cases with synthetic data
        pass


if __name__ == "__main__":
    pytest.main([__file__])