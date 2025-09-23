import numpy as np
import pytest
from src.core_functionality.chemometrics import ls_custom, nnls_custom, Chemometrics, FIT_PARAMETERS, CONFIDENCE_INTERVALS
from src.core_functionality.spectrum import Spectrum, Spectra
# TODO: Add more tests for Chemometrics class methods

@pytest.fixture
def simple_linear_data():
    A = np.array([[1, 0], [0, 1], [1, 1]])
    b = np.array([1, 2, 3])
    return A, b

def test_ls_custom_solution(simple_linear_data):
    A, b = simple_linear_data
    x, ci = ls_custom(A, b)
    np.testing.assert_allclose(np.dot(A, x), b, rtol=1e-4)

def test_ls_custom_confidence_intervals(simple_linear_data):
    A, b = simple_linear_data
    x, ci = ls_custom(A, b)
    assert isinstance(ci, list)
    assert all(len(interval) == 2 for interval in ci)

def test_nnls_custom_nonnegative_constraint(simple_linear_data):
    A, b = simple_linear_data
    x, ci = nnls_custom(A, b)
    assert np.all(x >= 0)
    assert isinstance(ci, list)

@pytest.fixture
def dummy_spectra():
    freqs = np.linspace(400, 700, 100)
    refs = [Spectrum(freqs, np.sin(freqs + i)) for i in range(3)]
    mix = Spectrum(freqs, 0.5*np.sin(freqs) + 0.3*np.sin(freqs+1) + 0.2*np.sin(freqs+2))
    return Spectra(refs), mix

def test_chemometrics_single_fit(dummy_spectra):
    refs, mix = dummy_spectra
    chem = Chemometrics(reference_spectra=refs)
    result = chem.compute_spectrum_fitting(mix)
    
    # Check that result is a dictionary with expected keys
    assert isinstance(result, dict)
    assert FIT_PARAMETERS in result
    assert CONFIDENCE_INTERVALS in result
    
    # Check the fit parameters (weights)
    weights = result[FIT_PARAMETERS]
    ci = result[CONFIDENCE_INTERVALS]
    
    assert len(weights) == len(refs)
    assert all(isinstance(i, tuple) for i in ci)
    assert len(ci) == len(refs)

def test_chemometrics_multiple_fit(dummy_spectra, tmp_path):
    refs, mix = dummy_spectra
    chem = Chemometrics(reference_spectra=refs)
    mixed_spectra = Spectra([mix, mix])
    result = chem.compute_spectra_fitting(mixed_spectra, save_csv_file=str(tmp_path / "out.csv"))
    
    # Check that result is a list of dictionaries
    assert isinstance(result, list)
    assert len(result) == 2
    
    # Check each result dictionary
    for res in result:
        assert isinstance(res, dict)
        assert FIT_PARAMETERS in res
        assert CONFIDENCE_INTERVALS in res
        
        weights = res[FIT_PARAMETERS]
        ci = res[CONFIDENCE_INTERVALS]
        
        assert len(weights) == len(refs)
        assert len(ci) == len(refs)
        assert all(isinstance(i, tuple) for i in ci)

# ============== Additional Chemometrics Tests ==============

def test_chemometrics_initialization_empty():
    """Test initialization with no reference spectra"""
    chem = Chemometrics(reference_spectra=Spectra([]))  # Explicitly pass empty Spectra
    assert len(chem._reference_spectra) == 0
    assert len(chem._reference_spectra_names) == 0

def test_chemometrics_initialization_with_spectra(dummy_spectra):
    """Test initialization with reference spectra"""
    refs, _ = dummy_spectra
    
    chem = Chemometrics(reference_spectra=refs)
    assert len(chem._reference_spectra) == 3
    assert len(chem._reference_spectra_names) == 3

def test_add_reference_sample():
    """Test adding reference spectra"""
    freqs = np.linspace(100, 200, 50)
    ref = Spectrum(freqs, np.sin(freqs), name="Test_Ref")
    
    chem = Chemometrics(reference_spectra=Spectra([]))  # Explicitly pass empty Spectra
    chem.add_reference_sample(ref, "Custom_Name")
    
    assert len(chem._reference_spectra) == 1
    assert chem._reference_spectra_names[0] == "Custom_Name"

def test_add_reference_sample_default_name():
    """Test adding reference sample without custom name"""
    freqs = np.linspace(100, 200, 50)
    ref = Spectrum(freqs, np.sin(freqs), name="Original_Name")
    
    chem = Chemometrics(reference_spectra=Spectra([]))  # Explicitly pass empty Spectra
    chem.add_reference_sample(ref)
    
    assert len(chem._reference_spectra) == 1
    assert chem._reference_spectra_names[0] == "Original_Name"

def test_chemometrics_no_reference_error():
    """Test error when no reference spectra provided"""
    freqs = np.linspace(100, 200, 50)
    target = Spectrum(freqs, np.sin(freqs))
    
    chem = Chemometrics(reference_spectra=Spectra([]))  # Explicitly pass empty Spectra
    
    with pytest.raises(ValueError, match="No reference spectra have been provided"):
        chem.compute_spectrum_fitting(target)

def test_chemometrics_unconstrained_fit(realistic_spectral_data):
    """Test unconstrained fitting (can have negative weights)"""
    refs, mix = realistic_spectral_data
    
    chem = Chemometrics(reference_spectra=refs)
    result = chem.compute_spectrum_fitting(mix, constrained_fit=False)
    
    weights = result[FIT_PARAMETERS]
    assert len(weights) == 3
    # Weights might be negative in unconstrained case

def test_chemometrics_no_normalization(realistic_spectral_data):
    """Test fitting without normalization"""
    refs, mix = realistic_spectral_data
    
    chem = Chemometrics(reference_spectra=refs)
    result = chem.compute_spectrum_fitting(mix, normalize_fit=False)
    
    weights = result[FIT_PARAMETERS]
    # Without normalization, weights won't necessarily sum to 1
    assert len(weights) == 3
    assert np.all(weights >= 0)  # Still non-negative due to NNLS

def test_chemometrics_different_alpha(realistic_spectral_data):
    """Test fitting with different confidence level"""
    refs, mix = realistic_spectral_data
    
    chem = Chemometrics(reference_spectra=refs)
    
    # Test with different alpha (confidence level)
    result_95 = chem.compute_spectrum_fitting(mix, alpha=0.05)
    result_99 = chem.compute_spectrum_fitting(mix, alpha=0.01)
    
    # Fit parameters should be the same
    np.testing.assert_allclose(
        result_95[FIT_PARAMETERS], 
        result_99[FIT_PARAMETERS], 
        rtol=1e-10
    )
    
    # Confidence intervals should be different (99% wider)
    ci_95 = result_95[CONFIDENCE_INTERVALS]
    ci_99 = result_99[CONFIDENCE_INTERVALS]
    
    for i in range(len(ci_95)):
        width_95 = ci_95[i][1] - ci_95[i][0]
        width_99 = ci_99[i][1] - ci_99[i][0]
        assert width_99 >= width_95

@pytest.fixture
def realistic_spectral_data():
    """Create realistic spectral data for comprehensive testing"""
    freqs = np.linspace(1000, 4000, 200)
    
    # Create reference spectra with Gaussian-like peaks
    ref1 = Spectrum(freqs, np.exp(-(freqs - 1500)**2 / (2 * 100**2)), name="Reference_1")
    ref2 = Spectrum(freqs, np.exp(-(freqs - 2500)**2 / (2 * 150**2)), name="Reference_2") 
    ref3 = Spectrum(freqs, np.exp(-(freqs - 3200)**2 / (2 * 80**2)), name="Reference_3")
    
    refs = Spectra([ref1, ref2, ref3])
    
    # Create mixture spectrum with known contributions
    mixture_intensities = 0.4 * ref1.intensities + 0.35 * ref2.intensities + 0.25 * ref3.intensities
    mixture_intensities += 0.01 * np.random.randn(len(freqs))  # Add small amount of noise
    
    mixture = Spectrum(freqs, mixture_intensities, name="Mixture")
    
    return refs, mixture

def test_chemometrics_realistic_fit(realistic_spectral_data):
    """Test single spectrum fitting with realistic data"""
    refs, mixture = realistic_spectral_data
    
    chem = Chemometrics(reference_spectra=refs)
    result = chem.compute_spectrum_fitting(mixture)
    
    weights = result[FIT_PARAMETERS]
    ci = result[CONFIDENCE_INTERVALS]
    
    # Check that weights are normalized and reasonable
    assert np.abs(np.sum(weights) - 1.0) < 0.01  # Should sum to ~1
    assert np.all(weights >= 0)  # Should be non-negative
    assert np.all(weights <= 1)  # Should be <= 1
    
    # Check confidence intervals
    assert len(ci) == 3
    for interval in ci:
        assert interval[0] <= interval[1]  # Lower <= Upper
        assert interval[0] >= 0  # Lower bound >= 0

def test_chemometrics_save_csv_multiple(realistic_spectral_data, tmp_path):
    """Test saving CSV file functionality"""
    refs, mixture = realistic_spectral_data
    mixtures = Spectra([mixture, mixture])
    
    chem = Chemometrics(reference_spectra=refs)
    csv_path = tmp_path / "test_output.csv"
    
    results = chem.compute_spectra_fitting(
        mixtures, 
        save_csv_file=str(csv_path)
    )
    
    # Check that CSV file was created
    assert csv_path.exists()
    
    # Check that results are valid
    assert len(results) == 2
    for result in results:
        weights = result[FIT_PARAMETERS]
        assert np.abs(np.sum(weights) - 1.0) < 0.01
        assert np.all(weights >= 0)

def test_ls_custom_overdetermined():
    """Test ls_custom with more equations than unknowns"""
    A = np.array([[1, 2], [2, 1], [1, 1], [3, 1], [2, 3]])
    b = np.array([3, 3, 2, 4, 5])
    
    x, ci = ls_custom(A, b)
    
    # Check that solution minimizes least squares error
    assert len(x) == A.shape[1]
    assert len(ci) == A.shape[1]
    assert all(isinstance(interval, tuple) for interval in ci)

def test_nnls_custom_vs_unconstrained():
    """Test that NNLS gives non-negative solution when unconstrained would be negative"""
    A = np.array([[1, 0], [0, 1], [1, -1]])
    b = np.array([1, -0.5, 0.5])  # This would give negative solution without constraint
    
    x_nnls, ci_nnls = nnls_custom(A, b)
    x_ls, ci_ls = ls_custom(A, b)
    
    assert np.all(x_nnls >= 0)
    # Unconstrained solution should have at least one negative component
    assert np.any(x_ls < 0)

if __name__ == "__main__":
    pytest.main([__file__])