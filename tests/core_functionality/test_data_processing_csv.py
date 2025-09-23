import pytest
from pathlib import Path
import numpy as np

from src.core_functionality.data_processing import DataProcessingCSV
from src.core_functionality.exceptions import DataError


def create_csv_file(path: Path, content: str):
    path.write_text(content)

def test_csv_loading_with_headers(tmp_path):
    content = "freq,intensity\n100,1\n200,2\n300,3"
    create_csv_file(tmp_path / "spec1.csv", content)
    create_csv_file(tmp_path / "spec2.csv", content)


    dp = DataProcessingCSV(
        in_dir=str(tmp_path),
        csv_frequency_column="freq",
        csv_intensity_column="intensity"
    )

    assert len(dp.spectra) == 2
    np.testing.assert_array_equal(dp.spectra[0].frequencies, [100, 200, 300])
    np.testing.assert_array_equal(dp.spectra[0].intensities, [1, 2, 3])

def test_csv_loading_without_headers(tmp_path):
    content = "100,1\n200,2\n300,3"
    create_csv_file(tmp_path / "data1.csv", content)

    dp = DataProcessingCSV(in_dir=str(tmp_path))

    assert len(dp.spectra) == 1
    np.testing.assert_array_equal(dp.spectra[0].frequencies, [100, 200, 300])
    np.testing.assert_array_equal(dp.spectra[0].intensities, [1, 2, 3])

def test_csv_specific_file_loading(tmp_path):
    content = "freq,intensity\n100,1\n200,2\n300,3"
    create_csv_file(tmp_path / "target.csv", content)
    create_csv_file(tmp_path / "ignored.csv", content)


    dp = DataProcessingCSV(
        in_dir=str(tmp_path),
        specific_files=["target.csv"],
        csv_frequency_column="freq",
        csv_intensity_column="intensity"
    )

    assert len(dp.spectra) == 1
    assert dp.all_files[0].name == "target.csv"

def test_csv_invalid_column_raises(tmp_path):
    content = "a,b\n1,2\n3,4"
    create_csv_file(tmp_path / "bad.csv", content)


    with pytest.raises(DataError):
        DataProcessingCSV(
            in_dir=str(tmp_path),
            csv_frequency_column="nonexistent",
            csv_intensity_column="b"
        )

def test_csv_shape_mismatch_raises(tmp_path):
    create_csv_file(tmp_path / "good.csv", "freq,intensity\n1,2\n3,4")
    create_csv_file(tmp_path / "bad.csv", "freq,intensity\n1,2\n3,4\n5,6")


    with pytest.raises(DataError):
        DataProcessingCSV(
            in_dir=str(tmp_path),
            csv_frequency_column="freq",
            csv_intensity_column="intensity"
        )

if __name__ == "__main__":
    pytest.main([__file__])