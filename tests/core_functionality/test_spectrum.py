import numpy as np
import pytest
from src.core_functionality.spectrum import Spectrum 
from scipy.signal import savgol_filter


@pytest.fixture
def basic_spectrum():
    freqs = np.array([100, 200, 300, 400, 500])
    intens = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    return Spectrum(freqs, intens)


def test_init_valid():
    s = Spectrum(np.array([1, 2]), np.array([3, 4]))
    assert isinstance(s, Spectrum)


def test_init_shape_mismatch():
    with pytest.raises(ValueError):
        Spectrum(np.array([1, 2]), np.array([3]))


def test_repr(basic_spectrum):
    r = repr(basic_spectrum)
    assert "Spectrum(frequency=" in r
    assert "intensity=" in r


def test_eq_operator(basic_spectrum):
    s2 = Spectrum(basic_spectrum.frequencies.copy(), basic_spectrum.intensities.copy())
    assert basic_spectrum == s2


def test_copy(basic_spectrum):
    s_copy = basic_spectrum.copy()
    assert s_copy == basic_spectrum
    assert s_copy is not basic_spectrum


def test_interpolation(basic_spectrum):
    new_freqs = np.array([150, 250, 350])
    interp = basic_spectrum.interpolate(new_freqs)
    expected = np.interp(new_freqs, basic_spectrum.frequencies, basic_spectrum.intensities)
    assert np.allclose(interp.intensities, expected)
    assert np.allclose(interp.frequencies, new_freqs)


def test_find_nearest_frequency(basic_spectrum):
    assert basic_spectrum.find_nearest_frequency(230) == 200
    assert basic_spectrum.find_nearest_frequency(499.9) == 500


def test_isolate_section(basic_spectrum):
    section = basic_spectrum.isolate_spectrum_sections([(150, 350)])
    assert np.array_equal(section.frequencies, np.array([200, 300]))
    assert np.array_equal(section.intensities, np.array([2.0, 3.0]))


def test_add_data_to_spectrum(basic_spectrum):
    new_freqs = np.array([50, 600])
    new_intens = np.array([0.5, 6.0])
    basic_spectrum.add_data_to_spectrum(new_freqs, new_intens)
    assert np.array_equal(basic_spectrum.frequencies, np.array([50, 100, 200, 300, 400, 500, 600]))
    assert np.array_equal(basic_spectrum.intensities, np.array([0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))


def test_fft_smooth(basic_spectrum):
    smoothed = basic_spectrum.fft_smooth(0.4)
    assert isinstance(smoothed, Spectrum)
    assert len(smoothed.frequencies) == len(basic_spectrum.frequencies)


def test_fft_smooth_invalid_ratio(basic_spectrum):
    with pytest.raises(ValueError):
        basic_spectrum.fft_smooth(0.0)


def test_savitzky_golay_smooth(basic_spectrum):
    smoothed = basic_spectrum.savitzky_golay_smooth(window_size=5, order=2)
    expected = savgol_filter(basic_spectrum.intensities, 5, 2)
    assert np.allclose(smoothed.intensities, expected)
    assert np.array_equal(smoothed.frequencies, basic_spectrum.frequencies)

if __name__ == "__main__":
    pytest.main([__file__])