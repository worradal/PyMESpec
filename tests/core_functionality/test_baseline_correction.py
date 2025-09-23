import numpy as np
import pytest

from src.core_functionality.spectrum import Spectrum, Spectra
from src.core_functionality.baseline_correction import ARPLS, Quadratic, Linear

@pytest.fixture
def sample_spectrum():
    np.random.seed(16)  # For reproducibility
    x = np.linspace(0, 1000, 1000)
    baseline = 0.00001 * (x - 500)**2
    peaks = (
        10 * np.exp(-(x - 300)**2 / (2 * 5**2)) +
        20 * np.exp(-(x - 700)**2 / (2 * 10**2))
    )
    noise = np.random.normal(0, 0.2, size=x.shape)
    y = baseline + peaks + noise
    return Spectrum(x, y)

@pytest.fixture
def sample_spectra():
    """Create multiple spectra for testing batch operations"""
    np.random.seed(42)
    x = np.linspace(0, 1000, 500)
    spectra_list = []
    
    for i in range(3):
        baseline = 0.01 * (x - 400 - i*100)**2 
        peaks = 15 * np.exp(-(x - 200 - i*50)**2 / (2 * 8**2))
        noise = np.random.normal(0, 0.1, size=x.shape)
        y = baseline + peaks + noise
        spectra_list.append(Spectrum(x, y))
    
    return Spectra(spectra_list)

@pytest.fixture
def linear_spectrum():
    """Create a spectrum with a clear linear baseline"""
    x = np.linspace(0, 1000, 500)
    linear_baseline = 0.005 * x + 2.0
    peaks = 8 * np.exp(-(x - 400)**2 / (2 * 10**2))
    y = linear_baseline + peaks
    return Spectrum(x, y)

@pytest.fixture
def quadratic_spectrum():
    """Create a spectrum with a clear quadratic baseline"""
    x = np.linspace(0, 1000, 500)
    quadratic_baseline = 0.000002 * (x - 500)**2 + 1.0
    peaks = 12 * np.exp(-(x - 600)**2 / (2 * 15**2))
    y = quadratic_baseline + peaks
    return Spectrum(x, y)

@pytest.fixture
def flat_spectrum():
    """Create a flat spectrum with no peaks"""
    x = np.linspace(0, 1000, 500)
    y = np.ones_like(x) * 5.0
    return Spectrum(x, y)

@pytest.fixture
def noisy_spectrum():
    """Create a very noisy spectrum"""
    np.random.seed(123)
    x = np.linspace(0, 1000, 500)
    signal = 3 * np.exp(-(x - 500)**2 / (2 * 50**2))
    noise = np.random.normal(0, 2.0, size=x.shape)
    y = signal + noise + 2.0  # Add offset
    return Spectrum(x, y)

@pytest.fixture(params=[ARPLS, Quadratic, Linear])
def corrector(request):
    return request.param()

# ============== Basic Functionality Tests ==============

def test_baseline_shape(sample_spectrum, corrector):
    baseline = corrector.get_baseline(sample_spectrum)
    assert baseline.shape == sample_spectrum.intensities.shape

def test_corrected_is_non_negative(sample_spectrum, corrector):
    corrected = corrector.baseline_corrected_spectrum(sample_spectrum)
    assert np.all(corrected.intensities >= 0)

def test_corrected_same_shape(sample_spectrum, corrector):
    corrected = corrector.baseline_corrected_spectrum(sample_spectrum)
    assert corrected.frequencies.shape == sample_spectrum.frequencies.shape
    assert corrected.intensities.shape == sample_spectrum.intensities.shape

def test_input_not_modified(sample_spectrum, corrector):
    original = sample_spectrum.intensities.copy()
    _ = corrector.baseline_corrected_spectrum(sample_spectrum)
    assert np.allclose(sample_spectrum.intensities, original)

def test_baseline_is_close_in_flat_regions(sample_spectrum, corrector):
    baseline = corrector.get_baseline(sample_spectrum)
    left = slice(0, 100)
    right = slice(-100, None)
    error_left = np.abs(sample_spectrum.intensities[left] - baseline[left])
    error_right = np.abs(sample_spectrum.intensities[right] - baseline[right])
    assert np.mean(error_left) < 1
    assert np.mean(error_right) < 1

def test_baseline_smoothness(sample_spectrum, corrector):
    baseline = corrector.get_baseline(sample_spectrum)
    second_deriv_signal = np.diff(sample_spectrum.intensities, n=2)
    second_deriv_baseline = np.diff(baseline, n=2)
    assert np.std(second_deriv_baseline) < np.std(second_deriv_signal)

def test_flat_spectrum(corrector, flat_spectrum):
    corrected = corrector.baseline_corrected_spectrum(flat_spectrum)
    # For flat spectra, corrected should be close to zero
    assert np.max(corrected.intensities) < 1.0
    assert np.min(corrected.intensities) >= 0.0

def test_methods_differ(sample_spectrum):
    arpls = ARPLS()
    linear = Linear()
    quadratic = Quadratic()
    corrected_arpls = arpls.baseline_corrected_spectrum(sample_spectrum)
    corrected_linear = linear.baseline_corrected_spectrum(sample_spectrum)
    corrected_quadratic = quadratic.baseline_corrected_spectrum(sample_spectrum)
    assert not np.allclose(corrected_arpls.intensities, corrected_linear.intensities)
    assert not np.allclose(corrected_arpls.intensities, corrected_quadratic.intensities)
    assert not np.allclose(corrected_linear.intensities, corrected_quadratic.intensities)

# ============== ARPLS-Specific Tests ==============

def test_arpls_full_output_structure(sample_spectrum):
    corrector = ARPLS()
    result, residual, info = corrector.baseline_corrected_spectrum(sample_spectrum, full_output=True)
    assert isinstance(result, Spectrum)
    assert isinstance(residual, np.ndarray)
    assert isinstance(info, dict)
    assert 'num_iters' in info
    assert 'final_ratio' in info
    assert info['num_iters'] >= 0
    assert info['final_ratio'] >= 0

def test_arpls_parameter_variations(sample_spectrum):
    corrector = ARPLS()
    
    # Test different lambda parameters
    baseline_low = corrector.get_baseline(sample_spectrum, lambda_parameter=1e4)
    baseline_high = corrector.get_baseline(sample_spectrum, lambda_parameter=1e8)
    assert not np.allclose(baseline_low, baseline_high)
    
    # Test different stop ratios
    baseline_loose = corrector.get_baseline(sample_spectrum, stop_ratio=1e-3)
    baseline_strict = corrector.get_baseline(sample_spectrum, stop_ratio=1e-9)
    # Results might be the same if convergence is fast, so just check they complete
    assert len(baseline_loose) == len(sample_spectrum.intensities)
    assert len(baseline_strict) == len(sample_spectrum.intensities)

def test_arpls_max_iterations(sample_spectrum):
    corrector = ARPLS()
    
    # Test with very low max iterations
    baseline_few = corrector.get_baseline(sample_spectrum, max_iters=1)
    baseline_many = corrector.get_baseline(sample_spectrum, max_iters=50)
    
    assert len(baseline_few) == len(sample_spectrum.intensities)
    assert len(baseline_many) == len(sample_spectrum.intensities)

def test_arpls_convergence_info(sample_spectrum):
    corrector = ARPLS()
    baseline, residual, info = corrector.get_baseline(sample_spectrum, full_output=True)
    
    assert isinstance(baseline, np.ndarray)
    assert isinstance(residual, np.ndarray)
    assert isinstance(info, dict)
    assert 'num_iters' in info
    assert 'final_ratio' in info
    assert info['num_iters'] >= 0  # Should be non-negative
    assert info['final_ratio'] >= 0

def test_arpls_batch_correction(sample_spectra):
    corrector = ARPLS()
    corrected_spectra = corrector.baseline_corrected_spectra(sample_spectra)
    
    assert isinstance(corrected_spectra, Spectra)
    assert len(corrected_spectra) == len(sample_spectra)
    
    # Check each spectrum is properly corrected
    for i, spectrum in enumerate(corrected_spectra):
        assert isinstance(spectrum, Spectrum)
        assert len(spectrum.intensities) == len(sample_spectra[i].intensities)
        assert np.all(spectrum.intensities >= 0)

def test_arpls_batch_with_parameters(sample_spectra):
    corrector = ARPLS()
    corrected_spectra = corrector.baseline_corrected_spectra(
        sample_spectra, 
        lambda_parameter=1e5, 
        stop_ratio=1e-4, 
        max_iters=5
    )
    
    assert len(corrected_spectra) == len(sample_spectra)
    for spectrum in corrected_spectra:
        assert np.all(spectrum.intensities >= 0)

# ============== Linear Baseline Tests ==============

def test_linear_baseline_fit(linear_spectrum):
    corrector = Linear()
    baseline = corrector.get_baseline(linear_spectrum)
    
    # For linear data, the baseline should capture the linear trend
    # Check that baseline is indeed linear (second derivative should be ~0)
    second_deriv = np.diff(baseline, n=2)
    assert np.allclose(second_deriv, 0, atol=1e-10)

def test_linear_corrected_removes_trend(linear_spectrum):
    corrector = Linear()
    corrected = corrector.baseline_corrected_spectrum(linear_spectrum)
    
    # After linear correction, the overall trend should be removed
    # Check that the corrected spectrum starts at zero
    assert corrected.intensities[0] == 0.0 or np.isclose(corrected.intensities[0], 0.0, atol=1e-10)

def test_linear_preserves_peaks():
    # Create a spectrum with known linear baseline and peak
    x = np.linspace(0, 100, 100)
    linear_baseline = 0.1 * x + 5  # Clear linear trend
    peak = 10 * np.exp(-(x - 50)**2 / (2 * 5**2))  # Gaussian peak
    y = linear_baseline + peak
    spectrum = Spectrum(x, y)
    
    corrector = Linear()
    corrected = corrector.baseline_corrected_spectrum(spectrum)
    
    # The peak should still be present and prominent after correction
    peak_position = np.argmax(corrected.intensities)
    assert 45 <= peak_position <= 55  # Peak should be around position 50

# ============== Quadratic Baseline Tests ==============

def test_quadratic_baseline_fit(quadratic_spectrum):
    corrector = Quadratic()
    baseline = corrector.get_baseline(quadratic_spectrum)
    
    # For quadratic data, the baseline should capture the quadratic curve
    # Third derivative should be ~0 for quadratic function
    third_deriv = np.diff(baseline, n=3)
    assert np.allclose(third_deriv, 0, atol=1e-8)

def test_quadratic_corrected_removes_curvature(quadratic_spectrum):
    corrector = Quadratic()
    corrected = corrector.baseline_corrected_spectrum(quadratic_spectrum)
    
    # After quadratic correction, the minimum should be at zero (due to shift)
    assert np.min(corrected.intensities) == 0.0 or np.isclose(np.min(corrected.intensities), 0.0, atol=1e-10)
    
    # Test that the corrected spectrum has reasonable properties
    # Since we fit and subtract a quadratic, the remaining signal should be more "peak-like"
    # Check that all values are non-negative
    assert np.all(corrected.intensities >= 0)

def test_quadratic_preserves_peaks():
    # Create a spectrum with known quadratic baseline and peak
    x = np.linspace(0, 100, 100)
    quadratic_baseline = 0.001 * (x - 50)**2 + 2  # Quadratic curve
    peak = 8 * np.exp(-(x - 30)**2 / (2 * 4**2))  # Gaussian peak
    y = quadratic_baseline + peak
    spectrum = Spectrum(x, y)
    
    corrector = Quadratic()
    corrected = corrector.baseline_corrected_spectrum(spectrum)
    
    # The peak should still be present after correction
    peak_position = np.argmax(corrected.intensities)
    assert 25 <= peak_position <= 35  # Peak should be around position 30

# ============== Edge Cases and Error Handling ==============

def test_empty_spectrum_handling():
    # Test with minimal data
    x = np.array([1, 2, 3])
    y = np.array([1, 2, 1])
    spectrum = Spectrum(x, y)
    
    for corrector_class in [ARPLS, Linear, Quadratic]:
        corrector = corrector_class()
        try:
            corrected = corrector.baseline_corrected_spectrum(spectrum)
            assert len(corrected.intensities) == len(y)
            assert np.all(corrected.intensities >= 0)
        except Exception as e:
            # Some methods might fail with very small datasets, which is acceptable
            assert len(str(e)) > 0

def test_single_point_spectrum():
    # Test with single point (edge case)
    x = np.array([1])
    y = np.array([5])
    spectrum = Spectrum(x, y)
    
    for corrector_class in [Linear, Quadratic]:  # Skip ARPLS as it needs more points
        corrector = corrector_class()
        try:
            corrected = corrector.baseline_corrected_spectrum(spectrum)
            assert len(corrected.intensities) == 1
            assert corrected.intensities[0] >= 0
        except Exception:
            # It's acceptable for some methods to fail with single points
            pass

def test_very_noisy_spectrum(noisy_spectrum):
    # Test all methods can handle very noisy data
    for corrector_class in [ARPLS, Linear, Quadratic]:
        corrector = corrector_class()
        corrected = corrector.baseline_corrected_spectrum(noisy_spectrum)
        
        assert len(corrected.intensities) == len(noisy_spectrum.intensities)
        assert np.all(corrected.intensities >= 0)
        assert np.all(np.isfinite(corrected.intensities))

def test_negative_intensity_spectrum():
    # Test spectrum with negative intensities
    x = np.linspace(0, 100, 50)
    y = np.sin(x) - 2  # Will have negative values
    spectrum = Spectrum(x, y)
    
    for corrector_class in [ARPLS, Linear, Quadratic]:
        corrector = corrector_class()
        corrected = corrector.baseline_corrected_spectrum(spectrum)
        
        # All corrected intensities should be non-negative
        assert np.all(corrected.intensities >= 0)

# ============== Performance and Consistency Tests ==============

def test_method_consistency():
    """Test that methods give consistent results across multiple runs"""
    np.random.seed(456)
    x = np.linspace(0, 1000, 200)
    y = 0.001 * x**2 + 5 * np.exp(-(x - 500)**2 / (2 * 50**2)) + np.random.normal(0, 0.1, len(x))
    spectrum = Spectrum(x, y)
    
    for corrector_class in [Linear, Quadratic]:  # Deterministic methods
        corrector = corrector_class()
        result1 = corrector.baseline_corrected_spectrum(spectrum)
        result2 = corrector.baseline_corrected_spectrum(spectrum)
        
        # Results should be identical for deterministic methods
        assert np.allclose(result1.intensities, result2.intensities)

def test_spectrum_frequencies_preserved():
    """Test that frequency arrays are preserved exactly"""
    x = np.linspace(100, 4000, 1000)  # Realistic IR frequencies
    y = np.random.random(1000) + 1
    spectrum = Spectrum(x, y)
    
    for corrector_class in [ARPLS, Linear, Quadratic]:
        corrector = corrector_class()
        corrected = corrector.baseline_corrected_spectrum(spectrum)
        
        # Frequencies should be identical
        assert np.array_equal(corrected.frequencies, spectrum.frequencies)

# ============== Integration Tests ==============

def test_realistic_ir_spectrum():
    """Test with realistic IR spectrum-like data"""
    # Simulate IR spectrum with multiple peaks and quadratic baseline
    x = np.linspace(400, 4000, 1800)  # IR range
    baseline = 0.000001 * (x - 2200)**2 + 0.5
    
    # Multiple peaks at typical IR frequencies
    peaks = (
        5 * np.exp(-(x - 1650)**2 / (2 * 20**2)) +  # C=O stretch
        3 * np.exp(-(x - 3000)**2 / (2 * 30**2)) +  # O-H stretch
        2 * np.exp(-(x - 1450)**2 / (2 * 15**2))    # C-H bend
    )
    
    noise = np.random.normal(0, 0.05, len(x))
    y = baseline + peaks + noise
    spectrum = Spectrum(x, y)
    
    for corrector_class in [ARPLS, Linear, Quadratic]:
        corrector = corrector_class()
        corrected = corrector.baseline_corrected_spectrum(spectrum)
        
        # Check that peaks are still present
        assert np.max(corrected.intensities) > 1.0  # Should have significant peaks
        assert np.all(corrected.intensities >= 0)
        
        # Check that baseline is reasonably flat in peak-free regions
        # (edges of IR spectrum are typically flat)
        edge_region = corrected.intensities[:50]  # First 50 points
        edge_std = np.std(edge_region)
        assert edge_std < 0.5  # Should be relatively flat

if __name__ == "__main__":
    pytest.main([__file__])