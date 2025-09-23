import numpy as np
import os
import pytest
import tempfile
import warnings
from tempfile import NamedTemporaryFile
from unittest.mock import patch
import pandas as pd

from src.core_functionality.phase import Phase
from src.core_functionality.spectrum import Spectrum, Spectra


@pytest.fixture
def dummy_spectra():
    """Create dummy spectra for testing"""
    freqs = np.linspace(0, 10, 100)
    spectra = [Spectrum(freqs, np.sin(freqs + i)) for i in range(10)]
    return Spectra(spectra)

@pytest.fixture
def phase_with_fft(dummy_spectra):
    """Create a Phase object with FFT already computed"""
    return Phase(dummy_spectra, num_time_points=5).fourier_transform_on_avg_data()

@pytest.fixture
def sinusoidal_spectra():
    """Create spectra with known sinusoidal pattern for testing phase analysis"""
    freqs = np.linspace(1000, 4000, 200)  # IR-like frequencies
    num_time_points = 8
    
    spectra_list = []
    for t in range(num_time_points):
        # Create a sinusoidal pattern with known phase and amplitude
        phase_shift = 2 * np.pi * t / num_time_points
        intensity = 2 + np.sin(freqs/1000 + phase_shift) + 0.5 * np.cos(2 * freqs/1000)
        spectra_list.append(Spectrum(freqs, intensity))
    
    return Spectra(spectra_list)

@pytest.fixture
def complex_spectra():
    """Create more complex spectra with multiple harmonics"""
    freqs = np.linspace(500, 2500, 150)
    num_time_points = 12
    
    spectra_list = []
    for t in range(num_time_points):
        # Multiple frequency components with different phases
        t_norm = t / num_time_points
        intensity = (
            3.0 +  # DC offset
            2.0 * np.sin(2 * np.pi * t_norm + freqs/1000) +  # Fundamental
            1.0 * np.cos(4 * np.pi * t_norm + freqs/2000) +  # Second harmonic
            0.5 * np.sin(6 * np.pi * t_norm)  # Third harmonic
        )
        spectra_list.append(Spectrum(freqs, intensity))
    
    return Spectra(spectra_list)

# ============== Basic Initialization Tests ==============

def test_phase_initialization(dummy_spectra):
    phase = Phase(dummy_spectra, num_time_points=5)
    assert phase._ft_all_components.shape[0] == len(phase._average_data)
    assert phase._ft_all_components.shape[1] == len(phase._average_data[0].frequencies)

def test_phase_initialization_with_start_end(dummy_spectra):
    """Test initialization with custom start and end parameters"""
    phase = Phase(dummy_spectra, num_time_points=3, start=1, end=7)
    
    # Should have 3 time points from the averaged data
    assert len(phase._average_data) == 3
    assert phase._ft_all_components.shape[0] == 3

def test_phase_initialization_properties(dummy_spectra):
    """Test that all internal arrays are properly initialized"""
    phase = Phase(dummy_spectra, num_time_points=4)
    
    expected_shape = (4, len(dummy_spectra[0].frequencies))
    
    assert phase._plot_phase_data.shape == expected_shape
    assert phase._ft_all_components.shape == expected_shape
    assert phase._ft_real_fft_component.shape == expected_shape
    assert phase._ft_imaginary_component.shape == expected_shape
    assert phase._phase_shift.shape == expected_shape
    assert phase._ifft_data.shape == expected_shape

def test_phase_property_accessors(dummy_spectra):
    """Test property accessors return correct data"""
    phase = Phase(dummy_spectra, num_time_points=5)
    
    # Before FFT, should be zeros
    assert np.allclose(phase.ft_all_components, 0)
    assert np.allclose(phase.ifft_data, 0)
    assert np.allclose(phase.phase_shift, 0)
    
    # After FFT, should have non-zero values
    phase.fourier_transform_on_avg_data()
    assert not np.allclose(phase.ft_all_components, 0)

# ============== Fourier Transform Tests ==============

def test_fourier_transform_output(dummy_spectra):
    phase = Phase(dummy_spectra, num_time_points=5).fourier_transform_on_avg_data()
    assert np.any(np.abs(phase.ft_all_components) > 0)

    expected_magnitude = np.hypot(
        phase._ft_real_fft_component, phase._ft_imaginary_component
    )
    actual_magnitude = np.abs(phase.ft_all_components)
    np.testing.assert_allclose(actual_magnitude, expected_magnitude, rtol=1e-5)

def test_fourier_transform_with_different_harmonics(dummy_spectra):
    """Test FFT with different harmonic settings"""
    phase1 = Phase(dummy_spectra, num_time_points=6).fourier_transform_on_avg_data(harmonic=1)
    phase2 = Phase(dummy_spectra, num_time_points=6).fourier_transform_on_avg_data(harmonic=2)
    
    # Results should be different for different harmonics
    assert not np.allclose(phase1._plot_phase_data, phase2._plot_phase_data)

def test_fourier_transform_components_consistency(dummy_spectra):
    """Test that real and imaginary components are consistent with complex FFT"""
    phase = Phase(dummy_spectra, num_time_points=5).fourier_transform_on_avg_data()
    
    # Real part should match
    np.testing.assert_allclose(
        phase._ft_real_fft_component, 
        np.real(phase._ft_all_components),
        rtol=1e-10
    )
    
    # Imaginary part should match
    np.testing.assert_allclose(
        phase._ft_imaginary_component, 
        np.imag(phase._ft_all_components),
        rtol=1e-10
    )

def test_fourier_transform_phase_calculation(sinusoidal_spectra):
    """Test phase calculation with known sinusoidal data"""
    phase = Phase(sinusoidal_spectra, num_time_points=8).fourier_transform_on_avg_data()
    
    # Phase data should be computed
    assert not np.allclose(phase._phase_shift, 0)
    
    # Phase values should be in [0, 2π] range
    assert np.all(phase._phase_shift >= 0)
    assert np.all(phase._phase_shift <= 2 * np.pi)

# ============== Weighting Tests ==============

def test_weight_application(dummy_spectra):
    phase = Phase(dummy_spectra, num_time_points=5).fourier_transform_on_avg_data()
    original = np.copy(phase.ft_all_components)
    phase.weight(list_of_harmonics=[1])

    assert np.allclose(phase.ft_all_components[:, 1:], 0, atol=1e-12)
    assert not np.allclose(phase.ft_all_components[:, 0], 0)

def test_weight_multiple_harmonics(dummy_spectra):
    """Test weighting with multiple harmonics"""
    phase = Phase(dummy_spectra, num_time_points=6).fourier_transform_on_avg_data()
    original = np.copy(phase.ft_all_components)
    
    # Keep harmonics 1, 3, and 5
    phase.weight(list_of_harmonics=[1, 3, 5])
    
    # Check that only specified harmonics are kept
    assert not np.allclose(phase.ft_all_components[:, 0], 0)  # Harmonic 1
    assert np.allclose(phase.ft_all_components[:, 1], 0, atol=1e-12)  # Harmonic 2
    assert not np.allclose(phase.ft_all_components[:, 2], 0)  # Harmonic 3
    assert np.allclose(phase.ft_all_components[:, 3], 0, atol=1e-12)  # Harmonic 4
    assert not np.allclose(phase.ft_all_components[:, 4], 0)  # Harmonic 5

def test_weight_all_harmonics_none(dummy_spectra):
    """Test that None keeps all harmonics"""
    phase = Phase(dummy_spectra, num_time_points=5).fourier_transform_on_avg_data()
    original = np.copy(phase.ft_all_components)
    
    phase.weight(list_of_harmonics=None)
    
    # All components should remain unchanged
    np.testing.assert_allclose(phase.ft_all_components, original, rtol=1e-10)

def test_weight_out_of_bounds(dummy_spectra):
    phase = Phase(dummy_spectra, num_time_points=5).fourier_transform_on_avg_data()
    with pytest.warns(UserWarning, match="out of bounds"):
        phase.weight(list_of_harmonics=[999])

def test_weight_negative_harmonic(dummy_spectra):
    """Test weighting with negative harmonic numbers"""
    phase = Phase(dummy_spectra, num_time_points=5).fourier_transform_on_avg_data()
    
    with pytest.warns(UserWarning):
        phase.weight(list_of_harmonics=[-1, 0])

def test_weight_affects_all_components(dummy_spectra):
    """Test that weighting affects all FFT component arrays"""
    phase = Phase(dummy_spectra, num_time_points=5).fourier_transform_on_avg_data()
    
    original_real = np.copy(phase._ft_real_fft_component)
    original_imag = np.copy(phase._ft_imaginary_component)
    original_complex = np.copy(phase._ft_all_components)
    
    phase.weight(list_of_harmonics=[1, 2])
    
    # All should be modified
    assert not np.allclose(phase._ft_real_fft_component, original_real)
    assert not np.allclose(phase._ft_imaginary_component, original_imag)
    assert not np.allclose(phase._ft_all_components, original_complex)

# ============== Save Methods Tests ==============

def test_save_fft_to_file(phase_with_fft):
    """Test saving FFT data to CSV file"""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        phase_with_fft.save_fft_to(tmp_path)
        assert os.path.exists(tmp_path)
        
        # Check file content
        df = pd.read_csv(tmp_path, index_col=0)
        # The CSV file includes both real and imaginary parts, plus extra metadata rows
        # Check that we have at least the expected number of time points
        assert df.shape[0] >= len(phase_with_fft._average_data)  # At least the number of time points
        assert df.shape[1] == len(phase_with_fft._general_frequency_values)  # Number of frequencies
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def test_save_ifft_to_file(phase_with_fft):
    """Test saving IFFT data to CSV file"""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        phase_with_fft.save_ifft_to(tmp_path)
        assert os.path.exists(tmp_path)
        
        # Check that ifft_data property is updated
        assert not np.allclose(phase_with_fft.ifft_data, 0)
        
        # Check file content
        df = pd.read_csv(tmp_path)
        expected_rows = len(phase_with_fft._average_data) + 1  # +1 for frequency row
        assert df.shape[0] == expected_rows
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except (PermissionError, OSError):
            pass

def test_save_ifft_real_component(phase_with_fft):
    """Test saving IFFT with real component only"""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        phase_with_fft.save_ifft_to(tmp_path, fft_real_component=True)
        assert os.path.exists(tmp_path)
        
        # Verify file was created and has content
        df = pd.read_csv(tmp_path)
        assert df.shape[0] > 0
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except (PermissionError, OSError):
            pass

def test_ifft_reconstruction(dummy_spectra):
    phase = Phase(dummy_spectra, num_time_points=5).fourier_transform_on_avg_data()
    tmp = NamedTemporaryFile(suffix=".csv", delete=False)
    tmp.close()
    try:
        phase.save_ifft_to(tmp.name)
        assert os.path.exists(tmp.name)
        assert tmp.name.endswith(".csv")
    finally:
        os.remove(tmp.name)

def test_save_phase_to_file(phase_with_fft):
    """Test saving phase data to CSV file"""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        phase_with_fft.save_phase_to(tmp_path)
        assert os.path.exists(tmp_path)
        
        # Check file content
        df = pd.read_csv(tmp_path, index_col=0)
        assert df.shape[0] == len(phase_with_fft._average_data)  # Number of harmonics
        assert df.shape[1] == len(phase_with_fft._general_frequency_values)  # Number of frequencies
        
        # Check that index contains harmonic labels
        assert all("Harmonic" in str(idx) for idx in df.index)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except (PermissionError, OSError):
            pass

def test_save_phase_to_file_legacy(dummy_spectra):
    """Test legacy save_phase_to method"""
    phase = Phase(dummy_spectra, num_time_points=5).fourier_transform_on_avg_data()
    tmp = NamedTemporaryFile(suffix=".csv", delete=False)
    tmp.close()
    try:
        phase.save_phase_to(tmp.name)
        assert os.path.exists(tmp.name)
        assert tmp.name.endswith(".csv")
    finally:
        os.remove(tmp.name)

def test_save_file_permission_error(phase_with_fft):
    """Test handling of permission errors when saving files"""
    # Mock a permission error
    with patch('pandas.DataFrame.to_csv', side_effect=PermissionError("Mock permission error")):
        with pytest.raises(IOError, match="Cannot write to"):
            phase_with_fft.save_ifft_to("test.csv")
        
        with pytest.raises(IOError, match="Cannot write to"):
            phase_with_fft.save_phase_to("test.csv")

# ============== Phase Analysis Tests ==============

def test_fft_harmonic_amp_phase_sin_PSD(sinusoidal_spectra):
    """Test FFT harmonic amplitude and phase calculation"""
    phase = Phase(sinusoidal_spectra, num_time_points=8).fourier_transform_on_avg_data()
    
    # Test for a specific frequency
    test_freq = sinusoidal_spectra[0].frequencies[50]
    amp, phi = phase.fft_harmonic_amp_phase_sin_PSD(test_freq, harmonic=1)
    
    # Amplitude should be positive
    assert amp >= 0
    
    # Phase should be in [0, 2π] range
    assert 0 <= phi <= 2 * np.pi

def test_fft_harmonic_amp_phase_different_harmonics(complex_spectra):
    """Test amplitude and phase calculation for different harmonics"""
    phase = Phase(complex_spectra, num_time_points=12).fourier_transform_on_avg_data()
    
    test_freq = complex_spectra[0].frequencies[75]
    
    # Test multiple harmonics
    for harmonic in [0, 1, 2, 3]:
        amp, phi = phase.fft_harmonic_amp_phase_sin_PSD(test_freq, harmonic)
        
        assert amp >= 0
        assert 0 <= phi <= 2 * np.pi

def test_fft_harmonic_dc_component(complex_spectra):
    """Test DC component (harmonic 0) handling"""
    phase = Phase(complex_spectra, num_time_points=12).fourier_transform_on_avg_data()
    
    test_freq = complex_spectra[0].frequencies[60]
    amp_dc, phi_dc = phase.fft_harmonic_amp_phase_sin_PSD(test_freq, harmonic=0)
    
    # DC component should have reasonable amplitude
    assert amp_dc >= 0

# ============== Plotting Tests (without actually displaying) ==============

@patch('matplotlib.pyplot.show')
def test_plot_sinusoidal_PD_spectra(mock_show, phase_with_fft):
    """Test plotting sinusoidal PD spectra without actually showing plot"""
    # Should not raise any errors
    phase_with_fft.plot_sinusodial_PD_spectra()
    mock_show.assert_called_once()

@patch('matplotlib.pyplot.show')
def test_plot_sinusoidal_PD_spectra_with_sampling(mock_show, phase_with_fft):
    """Test plotting with sample factor"""
    phase_with_fft.plot_sinusodial_PD_spectra(sample_factor=2)
    mock_show.assert_called_once()

@patch('matplotlib.pyplot.show')
def test_plot_sinusoidal_PD_spectra_custom_params(mock_show, phase_with_fft):
    """Test plotting with custom matplotlib parameters"""
    custom_params = {
        "font.size": 12,
        "lines.linewidth": 2.0
    }
    phase_with_fft.plot_sinusodial_PD_spectra(matplotlib_param_dict=custom_params)
    mock_show.assert_called_once()

def test_plot_phase_as_sine(phase_with_fft):
    """Test plotting phase as sine wave"""
    test_freq = phase_with_fft._general_frequency_values[25]
    
    # Should not raise any errors
    with patch('matplotlib.pyplot.show'):
        phase_with_fft.plot_phase_as_sine(test_freq, harmonic=1)

def test_plot_phase_as_sine_custom_params(phase_with_fft):
    """Test plotting phase as sine with custom parameters"""
    test_freq = phase_with_fft._general_frequency_values[30]
    custom_params = {"font.size": 14}
    
    with patch('matplotlib.pyplot.show'):
        phase_with_fft.plot_phase_as_sine(test_freq, harmonic=2, matplotlib_param_dict=custom_params)

# ============== Edge Cases and Error Handling ==============

def test_phase_with_minimal_data():
    """Test Phase with minimal data points"""
    freqs = np.array([1, 2, 3])
    spectra = [Spectrum(freqs, np.array([1, 2, 1]))]
    phase_data = Spectra(spectra)
    
    phase = Phase(phase_data, num_time_points=1)
    
    # Should not crash - use harmonic 0 for single data point
    phase.fourier_transform_on_avg_data(harmonic=0)
    assert phase.ft_all_components.shape == (1, 3)

def test_phase_frequency_search(dummy_spectra):
    """Test frequency searching functionality"""
    phase = Phase(dummy_spectra, num_time_points=5).fourier_transform_on_avg_data()
    
    # Test with exact frequency
    exact_freq = phase._general_frequency_values[10]
    amp, phi = phase.fft_harmonic_amp_phase_sin_PSD(exact_freq, harmonic=1)
    
    assert isinstance(amp, (int, float, np.number))
    assert isinstance(phi, (int, float, np.number))

def test_phase_with_zero_data():
    """Test Phase with zero intensity data"""
    freqs = np.linspace(0, 10, 20)
    spectra = [Spectrum(freqs, np.zeros_like(freqs)) for _ in range(3)]
    phase_data = Spectra(spectra)
    
    phase = Phase(phase_data, num_time_points=3).fourier_transform_on_avg_data()
    
    # Should handle zero data gracefully
    test_freq = freqs[5]
    amp, phi = phase.fft_harmonic_amp_phase_sin_PSD(test_freq, harmonic=1)
    
    assert amp >= 0  # Amplitude should be non-negative even for zero data

def test_phase_nyquist_frequency_handling(dummy_spectra):
    """Test handling of Nyquist frequency for even-length data"""
    # Create data with even number of time points
    phase = Phase(dummy_spectra, num_time_points=6).fourier_transform_on_avg_data()
    
    # Test Nyquist frequency (should be handled specially)
    test_freq = phase._general_frequency_values[10]
    nyquist_harmonic = len(phase._average_data) // 2
    
    amp, phi = phase.fft_harmonic_amp_phase_sin_PSD(test_freq, nyquist_harmonic)
    assert amp >= 0

# ============== Integration Tests ==============

def test_full_workflow_pipeline(complex_spectra):
    """Test complete workflow from initialization to analysis"""
    # Initialize
    phase = Phase(complex_spectra, num_time_points=12)
    
    # Perform FFT
    phase.fourier_transform_on_avg_data(harmonic=1)
    
    # Apply weighting
    phase.weight(list_of_harmonics=[1, 2, 3])
    
    # Test analysis
    test_freq = complex_spectra[0].frequencies[50]
    amp, phi = phase.fft_harmonic_amp_phase_sin_PSD(test_freq, harmonic=1)
    
    # Save data
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        phase.save_ifft_to(tmp_path)
        assert os.path.exists(tmp_path)
    finally:
        # Use try-except to handle Windows permission issues
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except (PermissionError, OSError):
            # On Windows, sometimes temp files are locked briefly
            pass
    
    # Verify final state
    assert not np.allclose(phase.ft_all_components, 0)
    assert not np.allclose(phase.ifft_data, 0)
    assert amp >= 0
    assert 0 <= phi <= 2 * np.pi

def test_chained_operations(dummy_spectra):
    """Test that methods can be chained together"""
    result = (Phase(dummy_spectra, num_time_points=5)
              .fourier_transform_on_avg_data()
              .weight(list_of_harmonics=[1, 2]))
    
    assert isinstance(result, Phase)
    assert not np.allclose(result.ft_all_components, 0)

def test_property_consistency_after_operations(dummy_spectra):
    """Test that properties remain consistent after various operations"""
    phase = Phase(dummy_spectra, num_time_points=5)
    
    # Initial state
    initial_shape = phase.ft_all_components.shape
    
    # After FFT
    phase.fourier_transform_on_avg_data()
    assert phase.ft_all_components.shape == initial_shape
    assert phase.ifft_data.shape == initial_shape
    assert phase.phase_shift.shape == initial_shape
    
    # After weighting
    phase.weight(list_of_harmonics=[1, 3])
    assert phase.ft_all_components.shape == initial_shape

def test_real_world_like_scenario():
    """Test with realistic IR spectroscopy-like data"""
    # Simulate IR spectrum with time-resolved measurements
    freqs = np.linspace(1000, 4000, 300)  # IR wavenumber range
    num_time_points = 16
    
    spectra_list = []
    for t in range(num_time_points):
        # Simulate time-dependent IR absorption with phase information
        time_phase = 2 * np.pi * t / num_time_points
        
        # Multiple IR bands with different phase behavior
        intensity = (
            1.0 +  # Baseline
            0.5 * np.exp(-(freqs - 1650)**2 / (2 * 50**2)) * (1 + 0.3 * np.sin(time_phase)) +  # C=O band
            0.3 * np.exp(-(freqs - 3000)**2 / (2 * 80**2)) * (1 + 0.2 * np.cos(time_phase + np.pi/4)) +  # O-H band
            0.2 * np.exp(-(freqs - 1450)**2 / (2 * 30**2)) * (1 + 0.1 * np.sin(2 * time_phase))  # C-H band
        )
        
        spectra_list.append(Spectrum(freqs, intensity))
    
    ir_spectra = Spectra(spectra_list)
    
    # Analyze phase information
    phase = Phase(ir_spectra, num_time_points=16).fourier_transform_on_avg_data()
    
    # Test analysis at characteristic IR frequencies
    co_freq = 1650  # C=O stretch
    amp_co, phi_co = phase.fft_harmonic_amp_phase_sin_PSD(co_freq, harmonic=1)
    
    assert amp_co > 0
    assert 0 <= phi_co <= 2 * np.pi
    
    # Apply selective weighting to isolate fundamental frequency
    phase.weight(list_of_harmonics=[1])
    
    # Verify that higher harmonics are suppressed
    assert np.allclose(phase.ft_all_components[2:, :], 0, atol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__])
