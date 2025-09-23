import numpy as np
import pytest

from src.core_functionality.spectrum import Spectra, Spectrum

# Tests
def test_init_and_len():
    s = Spectrum(np.array([1,2,3]), np.array([4,5,6]))
    spectra = Spectra([s, s.copy()])
    assert len(spectra) == 2

def test_append_valid_and_repr():
    s = Spectrum(np.array([1,2]), np.array([4,5]))
    spectra = Spectra([s])
    spectra.append(s.copy())
    assert len(spectra) == 2
    assert "Spectra" in repr(spectra)

def test_append_invalid_type():
    s = Spectrum(np.array([1,2]), np.array([3,4]))
    spectra = Spectra([s])
    with pytest.raises(TypeError):
        spectra.append("not a spectrum")

def test_append_shape_mismatch():
    s1 = Spectrum(np.array([1,2]), np.array([3,4]))
    s2 = Spectrum(np.array([1,2,3]), np.array([3,4,5]))
    spectra = Spectra([s1])
    with pytest.raises(ValueError):
        spectra.append(s2)

def test_average_spectra():
    s1 = Spectrum(np.array([1,2,3]), np.array([1,2,3]))
    s2 = Spectrum(np.array([1,2,3]), np.array([3,2,1]))
    spectra = Spectra([s1, s2])
    avg = spectra.average_spectra()
    np.testing.assert_array_almost_equal(avg.intensities, [2,2,2])

def test_interpolation():
    s = Spectrum(np.array([1,2,3]), np.array([4,5,6]))
    new_freqs = np.array([1.5, 2.5])
    spectra = Spectra([s])
    new_spectra = spectra.interpolate(new_freqs)
    assert isinstance(new_spectra, Spectra)
    np.testing.assert_array_almost_equal(new_spectra[0].frequencies, new_freqs)

def test_slice_specific_frequency_exact():
    freqs = np.array([10, 20, 30])
    ints = np.array([100, 200, 300])
    spectra = Spectra([Spectrum(freqs, ints) for _ in range(3)])
    result = spectra.slice_specific_frequency(20)
    np.testing.assert_array_equal(result, [200, 200, 200])

def test_average_over_data_repeats():
    s = Spectrum(np.array([1,2,3]), np.array([1,2,3]))
    spectra = Spectra([s.copy() for _ in range(6)])
    avg = spectra.average_over_data_repeats(num_time_points=3)
    assert isinstance(avg, Spectra)
    assert len(avg) == 3

def test_fft_smooth():
    f = np.linspace(0, 1, 100)
    i = np.sin(2 * np.pi * f)
    spectra = Spectra([Spectrum(f, i)])
    smoothed = spectra.fft_smooth()
    assert isinstance(smoothed[0], Spectrum)



if __name__ == "__main__":
    pytest.main([__file__])