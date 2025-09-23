#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
File created to decouple data analysis from reading data from files.
'''

# Written by
# Alfred Worrad <worrada@udel.edu>,

__author__ = "Afred Worrad"
__version__ = "1.2.1"
__maintainer__ = "Alfred Worrad"
__email__ = "worrada@udel.edu"
__status__ = "Development"
__project__ = "MES wavey"
__created__ = "December 05, 2023"
__updated__ = "June 23, 2025"

# built-in modules
from typing import (List, Optional)
from pathlib import Path
import csv
import re

# third-party modules
import numpy as np
import pandas as pd

# project modules
from src.core_functionality.spectrum import Spectrum, Spectra
from src.core_functionality.exceptions import DataError

# Avoid this class if possible, was used for very specific legacy data.
class DataProcessingTXT:
    """
    Load a series of delimited text spectra from a directory (or a specific list of files),
    ensure they all share the same shape, and collect them into a Spectra container.
    Avoid this class if possible, was used for very specific legacy data. No guarantees this will work.
    """

    def __init__(
        self,
        in_dir: str,
        specific_files: Optional[List[str]] = None,
    ):
        """ Initialize the DataProcessingTXT instance.

        Args:
            in_dir (str): Directory containing the .txt files.
            specific_files (Optional[List[str]], optional): List of specific file names to load.
                If None, all .txt files in the directory will be loaded. Defaults to None.

        Raises:
            DataError: If no .txt files are found in the specified directory.
        """
        self.in_dir = Path(in_dir)
        self._all_files = self._discover_files(specific_files)
        if not self._all_files:
            raise DataError(f"No .txt files found in directory {self.in_dir}")
        self._spectra = self._load_all()
    
    @staticmethod
    def _natural_sort_key(name: str) -> List:
        """Generate a key for natural sorting of strings with numbers."""
        return [int(part) if part.isdigit() else part.lower()
                for part in re.split(r'(\d+)', name)]

    def _discover_files(self, specific_files: Optional[List[str]]) -> List[Path]:
        """ Discover .txt files in the specified directory or from a specific list.

        Args:
            specific_files (Optional[List[str]]): List of specific file names to load.

        Raises:
            FileNotFoundError: If any specified files are missing in the directory.

        Returns:
            List[Path]: List of Path objects for the discovered .txt files.
        """
        if specific_files:
            paths = [self.in_dir / fname for fname in specific_files]
            missing = [p.name for p in paths if not p.exists()]
            if missing:
                raise FileNotFoundError(f"Missing files in {self.in_dir}: {missing}")
            return paths

        # Discover all .txt files in the directory (case-insensitive)
        files = [f for f in self.in_dir.iterdir() if f.is_file() and f.suffix.lower() == ".txt"]
        return sorted(files, key=lambda p: self._natural_sort_key(p.name))

    def _load_all(self) -> Spectra:
        """ Load all spectra from the discovered .txt files.

        Raises:
            DataError: If any loaded spectrum does not match the shape of the first spectrum.

        Returns:
            Spectra: A Spectra object containing all loaded spectra.
        """
        spectra = Spectra([])
        for idx, fpath in enumerate(self._all_files):
            spec = self._load_data_txt(fpath)
            if idx > 0 and not spectra[0].compare_spectra_shape(spec):
                raise DataError(
                    f"Shape mismatch in {fpath.name}: dimensions differ from first spectrum"
                )
            spectra.append(spec)
        return spectra

    def _load_data_txt(self, fpath: Path) -> Spectrum:
        """ Load a single spectrum from a delimited text file.

        Args:
            fpath (Path): Path object pointing to the .txt file.

        Returns:
            Spectrum: A Spectrum object containing the frequency and intensity data extracted from the file.
        """
        lines = fpath.read_text().splitlines()
        sample = "\n".join(lines[:min(5, len(lines))])
        try:
            delim = csv.Sniffer().sniff(sample).delimiter
        except csv.Error:
            delim = ','  # Default to comma if sniffing fails

        x_vals: List[float] = []
        y_vals: List[float] = []
        for line in lines:
            parts = [tok.strip() for tok in line.split(delim) if tok.strip()]
            try:
                x, y = float(parts[0]), float(parts[-1])
            except (ValueError, IndexError):
                # skip headers or malformed rows
                continue
            x_vals.append(x)
            y_vals.append(y)

        return Spectrum(np.array(x_vals), np.array(y_vals))

    @property
    def all_files(self) -> List[Path]:
        """List of all files (as Path objects) that were loaded."""
        return self._all_files
    
    @property
    def spectra(self) -> Spectra:
        """The loaded Spectra object in the same order as all_files."""
        return self._spectra

    def get_spectra(self) -> Spectra:
        """Return the loaded Spectra in the order of self.all_files."""
        return self._spectra


class DataProcessingCSV:
    """
    Load a series of CSV spectra from a directory (or a specific list of files),
    extract frequency/intensity columns over an optional row range,
    ensure consistent shape across all spectra, and collect them into a Spectra.
    """

    def __init__(
        self,
        in_dir: str,
        specific_files: Optional[List[str]] = None,
        csv_frequency_column: Optional[str] = None,
        csv_intensity_column: Optional[str] = None,
        csv_row_num_start: int = 0,
        csv_row_num_end: Optional[int] = None,
    ):
        """ Initialize the DataProcessingCSV instance.

        Args:
            in_dir (str): Directory containing the CSV files.
            specific_files (Optional[List[str]], optional): List of specific file names to load.
                If None, all CSV files in the directory will be loaded. Defaults to None.
            csv_frequency_column (Optional[str], optional): Column name for frequency data. Defaults to None.
            csv_intensity_column (Optional[str], optional): Column name for intensity data. Defaults to None.
            csv_row_num_start (int, optional): Starting row number for data extraction. Defaults to 0.
            csv_row_num_end (Optional[int], optional): Ending row number for data extraction. Defaults to None.

        Raises:
            DataError: If no CSV files are found in the specified directory.
        """
        self.in_dir = Path(in_dir)
        self.csv_frequency_column = csv_frequency_column
        self.csv_intensity_column = csv_intensity_column
        self.csv_row_num_start = csv_row_num_start
        self.csv_row_num_end = csv_row_num_end

        self._all_files = self._discover_files(specific_files)
        if not self._all_files:
            raise DataError(f"No CSV files found in directory {self.in_dir}")
        self._spectra = self._load_all()

    @staticmethod
    def _natural_sort_key(name: str):
        return [int(part) if part.isdigit() else part.lower() for part in re.split(r'(\d+)', name)]

    def _discover_files(self, specific_files: Optional[List[str]]) -> List[Path]:
        """ Discover CSV files in the specified directory or from a specific list.

        Args:
            specific_files (Optional[List[str]]): List of specific file names to load.

        Returns:
            List[Path]: List of Path objects for the discovered CSV files.
        """
        if specific_files:
            return [self.in_dir / fname for fname in specific_files]
        files = list(self.in_dir.glob("*.csv"))
        return sorted(files, key=lambda p: self._natural_sort_key(p.name))

    def _load_all(self) -> Spectra:
        spectra = Spectra([])
        for idx, fpath in enumerate(self._all_files):
            spec = self._load_data_csv(
                fpath,
                self.csv_frequency_column,
                self.csv_intensity_column,
                self.csv_row_num_start,
                self.csv_row_num_end,
            )
            if idx > 0 and not spectra[0].compare_spectra_shape(spec):
                raise DataError(
                    f"Shape mismatch in {fpath.name}: dimensions differ from the first spectrum"
                )
            spectra.append(spec)
        return spectra

    def _load_data_csv(
        self,
        fpath: Path,
        frequency_column: Optional[str],
        intensity_column: Optional[str],
        row_num_start: int,
        row_num_end: Optional[int],
    ) -> Spectrum:
        """ Load a single spectrum from a CSV file.

        Args:
            fpath (Path): Path object pointing to the CSV file.
            frequency_column (str): Name of the column containing frequency data.
            intensity_column (str): Name of the column containing intensity data.
            row_num_start (int): Index of the first row to include in the spectrum.
            row_num_end (Optional[int]): Index of the last row to include in the spectrum.

        Raises:
            DataError: If the CSV file cannot be read or does not contain the required columns.
            DataError: If the specified columns are missing or if the file does not contain enough rows.
            DataError: If the specified row range is invalid.

        Returns:
            Spectrum: A Spectrum object containing the frequency and intensity data extracted from the file.
        """
        try:
            if frequency_column is None or intensity_column is None:
                df = pd.read_csv(fpath, header=None)
            else:
                df = pd.read_csv(fpath)
        except Exception as e:
            raise DataError(f"Failed to read {fpath.name}: {e}")

        if frequency_column is None or intensity_column is None:
            # Assume first two columns if headers are not provided
            if df.shape[1] < 2:
                raise DataError(f"{fpath.name} must contain at least two columns.")
            df.columns = [f"col{i}" for i in range(df.shape[1])]  # temporary names
            frequency_column = "col0"
            intensity_column = "col1"


        # Verify required columns exist
        missing = [c for c in (frequency_column, intensity_column) if c not in df.columns]
        if missing:
            raise DataError(
                f"{fpath.name} is missing columns: {', '.join(missing)}. "
                f"Available columns: {', '.join(df.columns)}"
            )

        # Slice rows
        if row_num_end is not None and row_num_end > row_num_start:
            df = df.iloc[row_num_start:row_num_end]
        else:
            df = df.iloc[row_num_start:]

        # Convert to numeric, drop invalid rows
        x = pd.to_numeric(df[frequency_column], errors="coerce")
        y = pd.to_numeric(df[intensity_column], errors="coerce")
        mask = x.notna() & y.notna()

        return Spectrum(x[mask].to_numpy(), y[mask].to_numpy(), name=fpath.stem)

    @property
    def all_files(self) -> List[Path]:
        """Paths of all CSV files that were processed."""
        return self._all_files

    @property
    def spectra(self) -> Spectra:
        """The loaded Spectra in the same order as all_files."""
        return self._spectra
    
    # This is for legacy implementation compatibility
    def get_spectra(self) -> Spectra:
        return self._spectra


if __name__ == '__main__':
    pass
