import pytest
import numpy as np

from src.core_functionality.data_processing import DataProcessingTXT
from src.core_functionality.spectrum import Spectrum, Spectra
from src.core_functionality.exceptions import DataError

@pytest.fixture
def sample_txt_dir(tmp_path):
    frequencies = np.linspace(400, 1600, 5)
    intensities_list = [
        np.array([1, 2, 3, 4, 5]),
        np.array([2, 3, 4, 5, 6]),
        np.array([3, 4, 5, 6, 7]),
    ]
    for i, intensities in enumerate(intensities_list):
        data = "\n".join(f"{f},{j}" for f, j in zip(frequencies, intensities))
        (tmp_path / f"spectrum_{i}.txt").write_text(data)
    return tmp_path

def test_load_all_txt_files(sample_txt_dir):
    dp = DataProcessingTXT(in_dir=str(sample_txt_dir))
    spectra = dp.get_spectra()

    assert isinstance(spectra, Spectra)
    assert len(spectra) == 3
    assert np.allclose(spectra[0].frequencies, np.linspace(400, 1600, 5))

def test_specific_file_loading(sample_txt_dir):
    dp = DataProcessingTXT(
        in_dir=str(sample_txt_dir),
        specific_files=["spectrum_1.txt"]
    )
    spectra = dp.get_spectra()
    assert len(spectra) == 1
    assert np.allclose(spectra[0].intensities, [2, 3, 4, 5, 6])

def test_missing_file_raises_error(sample_txt_dir):
    with pytest.raises(FileNotFoundError):
        DataProcessingTXT(
            in_dir=str(sample_txt_dir),
            specific_files=["nonexistent.txt"]
        )

def test_shape_mismatch_raises_data_error(tmp_path):
    # Create valid file
    (tmp_path / "valid.txt").write_text("400,1\n800,2\n1200,3\n1600,4\n2000,5")
    # Create mismatched file
    (tmp_path / "bad.txt").write_text("400,1\n800,2")

    with pytest.raises(DataError):
        DataProcessingTXT(in_dir=str(tmp_path))

@pytest.fixture
def txt_file_with_header(tmp_path):
    file = tmp_path / "spectrum_with_header.txt"
    file.write_text("freq, intensity\n100.0, 1.0\n200.0, 2.0\n300.0, 3.0\n")
    return tmp_path

def test_file_with_header_is_loaded_correctly(txt_file_with_header):
    dp = DataProcessingTXT(in_dir=str(txt_file_with_header))
    spectra = dp.get_spectra()
    
    assert len(spectra) == 1
    spec = spectra[0]
    np.testing.assert_array_equal(spec.frequencies, np.array([100.0, 200.0, 300.0]))
    np.testing.assert_array_equal(spec.intensities, np.array([1.0, 2.0, 3.0]))

def test_files_are_naturally_sorted(tmp_path):
    # Create files in deliberately unsorted order
    filenames = ["spectrum_10.txt", "spectrum_2.txt", "spectrum_1.txt"]
    frequencies = [100.0, 200.0, 300.0]
    intensities = [1.0, 2.0, 3.0]
    content = "\n".join(f"{f},{i}" for f, i in zip(frequencies, intensities))

    for fname in filenames:
        (tmp_path / fname).write_text(content)

    dp = DataProcessingTXT(in_dir=str(tmp_path))
    sorted_filenames = [f.name for f in dp.all_files]

    expected_order = ["spectrum_1.txt", "spectrum_2.txt", "spectrum_10.txt"]
    assert sorted_filenames == expected_order

def test_files_are_naturally_sorted_different_file_names(tmp_path):
    filenames = ["spectrum-10.txt", "spectrum-2.txt", "spectrum-1.txt", "spectrum-20.txt", "spectrum-3.txt"]
    frequencies = [100.0, 200.0, 300.0]
    intensities = [1.0, 2.0, 3.0]
    content = "\n".join(f"{f},{i}" for f, i in zip(frequencies, intensities))

    for fname in filenames:
        (tmp_path / fname).write_text(content)

    dp = DataProcessingTXT(in_dir=str(tmp_path))
    sorted_filenames = [f.name for f in dp.all_files]

    expected_order = ["spectrum-1.txt", "spectrum-2.txt", "spectrum-3.txt", "spectrum-10.txt", "spectrum-20.txt"]
    assert sorted_filenames == expected_order


if __name__ == "__main__":
    pytest.main([__file__])
    # Run the tests if this script is executed directly