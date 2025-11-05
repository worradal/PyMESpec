#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Spectrum and Spectra data structures.
'''

# Written by
# Alfred Worrad <worrada@udel.edu>,

__author__ = "Afred Worrad"
__version__ = "1.2.1"
__maintainer__ = "Alfred Worrad"
__email__ = "worrada@udel.edu"
__status__ = "Development"
__project__ = "MES wavey"
__created__ = "December 06, 2023"
__updated__ = "June 23, 2025"

# built-in modules
from typing import ( List, Tuple,
                    Union)
import os
from copy import deepcopy
import warnings

# third-party modules
import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt


class Spectrum():

    def __init__(self, frequencies: np.ndarray, intensities: np.ndarray, name: str = ''):
        """        Initializes a Spectrum object with frequencies and intensities.

        Args:
            frequencies (np.ndarray): Array of frequencies.
            intensities (np.ndarray): Array of intensities corresponding to the frequencies.

        Raises:
            ValueError: If frequencies and intensities do not have the same shape.
        """
        if frequencies.shape != intensities.shape:
            raise ValueError("Frequencies and intensities must have the same shape")
        self._frequencies = frequencies
        self._intensities = intensities
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def frequencies(self) -> np.ndarray:
        return self._frequencies

    @frequencies.setter
    def frequencies(self, new_frequencies: np.ndarray) -> None:
        if new_frequencies.shape != self._intensities.shape:
            raise ValueError("New frequencies must have the same shape as intensities")
        self._frequencies = new_frequencies

    @property
    def intensities(self) -> np.ndarray:
        return self._intensities

    @intensities.setter
    def intensities(self, new_intensities: np.ndarray) -> None:
        if new_intensities.shape != self._frequencies.shape:
            raise ValueError("New intensities must have the same shape as frequencies")
        self._intensities = new_intensities

    def set_data(self, new_frequencies: np.ndarray, new_intensities: np.ndarray) -> None:
        """ Sets the frequencies and intensities of the spectrum.

        Args:
            new_frequencies (np.ndarray): Array of new frequencies to set for the spectrum.
            new_intensities (np.ndarray): Array of new intensities corresponding to the new frequencies.

        Raises:
            ValueError: If the shapes of new_frequencies and new_intensities do not match.
        """
        if new_frequencies.shape != new_intensities.shape:
            raise ValueError("Frequencies and intensities must have the same shape")
        self._frequencies = new_frequencies
        self._intensities = new_intensities

    def __repr__(self) -> str:
        return f'Spectrum(frequency={self.frequencies}, intensity={self.intensities})'
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Spectrum):
            return NotImplemented
        return self.compare_frequencies(other) and self.compare_intensities(other)

    def compare_frequencies(self, other_spectrum: 'Spectrum') -> bool:
        """ Compares the frequencies of the current spectrum with another spectrum.

        Args:
            other_spectrum (Spectrum): The other spectrum to compare with.

        Returns:
            bool: True if both spectra have the same frequencies, False otherwise.
        """
        return (
            len(self.frequencies) == len(other_spectrum.frequencies) and
            np.allclose(self.frequencies, other_spectrum.frequencies)
        )
    
    def compare_intensities(self, other_spectrum: 'Spectrum') -> bool:
        """ Compares the intensities of the current spectrum with another spectrum.

        Args:
            other_spectrum (Spectrum): The other spectrum to compare with.

        Returns:
            bool: True if both spectra have the same intensities, False otherwise.
        """
        return (
            len(self.intensities) == len(other_spectrum.intensities) and
            np.allclose(self.intensities, other_spectrum.intensities)
        )

    def copy(self) -> 'Spectrum':
        return Spectrum(self.frequencies.copy(), self.intensities.copy(), name=self.name)

    def interpolate(self, new_frequencies: np.ndarray) -> 'Spectrum':
        """ Interpolates the spectrum to a new frequency array.

        Args:
            new_frequencies (np.ndarray): Array of new frequencies to interpolate the spectrum to.

        Returns:
            Spectrum: A new Spectrum object with the interpolated intensities corresponding to the new frequencies.
        """
        new_intensities = np.interp(
            new_frequencies, self.frequencies, self.intensities)
        return Spectrum(new_frequencies, new_intensities, name=self.name)

    def compare_spectra_shape(self, other_spectrum: 'Spectrum') -> bool:
        """ Compares the shape of the current spectrum with another spectrum.

        Args:
            other_spectrum (Spectrum): The other spectrum to compare with.

        Returns:
            bool: True if both spectra have the same number of frequencies and intensities, False otherwise.
        """
        return len(self.frequencies) == len(other_spectrum.frequencies) and len(self.intensities) == len(other_spectrum.intensities)

    def find_nearest_frequency(self, frequency: float) -> float:
        """ Finds the nearest frequency in the spectrum to a given frequency.

        Args:
            frequency (float): The frequency to find the nearest match for.

        Returns:
            float: The nearest frequency in the spectrum to the given frequency.
        """
        return self.frequencies[np.abs(self.frequencies - frequency).argmin()]

    def isolate_spectrum_sections(self, frequency_sections: List[Tuple[float, float]]) -> 'Spectrum':
        """ Isolates sections of the spectrum based on provided frequency ranges.

        Args:
            frequency_sections (List[Tuple[float, float]]): List of tuples where each tuple contains a start and end frequency defining a section to isolate.

        Returns:
            Spectrum: A new Spectrum object containing only the frequencies and intensities that fall within the specified sections.
        """
        mask = np.zeros_like(self.frequencies, dtype=bool)
        for start, end in frequency_sections:
            mask |= (self.frequencies >= start) & (self.frequencies <= end)
        return Spectrum(self.frequencies[mask], self.intensities[mask], name=self.name)

    def savitzky_golay_smooth(self, window_size: int = 7, order: int = 3, deriv: int = 0, rate: int = 1) -> 'Spectrum':
        """ Smooths the spectrum using the Savitzky-Golay filter.

        Args:
            window_size (int, optional): Size of the filter window. Must be a positive odd integer. Defaults to 7.
            order (int, optional): Order of the polynomial used to fit the samples. Must be less than window_size. Defaults to 3.
            deriv (int, optional): Order of the derivative to compute. Defaults to 0 (no derivative).
            rate (int, optional): Rate of the derivative. Defaults to 1.

        Returns:
            Spectrum: A new Spectrum object with the smoothed intensities.
        """
        return Spectrum(self.frequencies, savgol_filter(self.intensities, window_size, order, deriv, rate), name=self.name)

    def fft_smooth(self, ratio_of_spectra_size: float = 0.1) -> 'Spectrum':
        """ Smooths the spectrum using the Fast Fourier Transform (FFT) method.

        Args:
            ratio_of_spectra_size (float, optional): Ratio of the size of the spectrum to keep in the frequency domain. Must be greater than 0. Defaults to 0.1.

        Raises:
            ValueError: If ratio_of_spectra_size is not greater than 0.

        Returns:
            Spectrum: A new Spectrum object with the smoothed intensities after applying the FFT filter.
        """
        if not (0 < ratio_of_spectra_size):
            raise ValueError("ratio_of_spectra_size must be be greater than 0")
        fft_intensities = np.fft.fft(self.intensities)
        freq_to_keep = int(ratio_of_spectra_size * len(fft_intensities))
        fft_intensities[freq_to_keep:-freq_to_keep] = 0
        return Spectrum(self.frequencies, np.fft.ifft(fft_intensities).real, name=self.name)

    # check this function
    def add_data_to_spectrum(self, new_frequencies: np.ndarray, new_intensities: np.ndarray) -> None:
        """ Adds new frequencies and intensities to the existing spectrum.

        Args:
            new_frequencies (np.ndarray): Array of new frequencies to add to the spectrum.
            new_intensities (np.ndarray): Array of new intensities corresponding to the new frequencies.

        Raises:
            ValueError: If the lengths of new_frequencies and new_intensities do not match.
        """
        if len(new_frequencies) != len(new_intensities):
            raise ValueError("New frequencies and intensities must have the same length")
        frequencies = np.concatenate((self.frequencies, new_frequencies))
        intensities = np.concatenate((self.intensities, new_intensities))
        # Sort the spectrum by frequency
        sorted_indices = np.argsort(frequencies)
        self.set_data(
            frequencies[sorted_indices],
            intensities[sorted_indices]
        )

    def plot_spectrum(self, title: str = '', xlabel: str = 'Frequency (cm-1)', ylabel: str = 'Intensity (a.u.)', show_plot: bool = False, **kwargs) -> None:
        """ Simple plot of the spectrum using matplotlib.

        Args:
            title (str, optional): Title of the plot. Defaults to ''.
            xlabel (str, optional): X-axis label. Defaults to 'Frequency (cm-1)'.
            ylabel (str, optional): Y-axis label. Defaults to 'Intensity (a.u.)'.
            show_plot (bool, optional): If True, displays the plot immediately. Defaults to False.
            **kwargs: Additional keyword arguments to pass to the matplotlib plot function.
        """
        plt.plot(self.frequencies, self.intensities, **kwargs)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if show_plot:
            plt.show()


class Spectra():

    def __init__(self, spectra: List[Spectrum]):
        """ Initializes a Spectra object with a list of Spectrum objects.

        Args:
            spectra (List[Spectrum]): List of Spectrum objects to initialize the Spectra object.

        Raises:
            TypeError: If any element in the spectra list is not an instance of Spectrum.
            ValueError: If the spectra do not have the same shape
            ValueError: If the spectra do not have the same frequencies.
        """
        if not all(isinstance(s, Spectrum) for s in spectra):
            raise TypeError("All elements must be Spectrum instances")
        
        if len(spectra) > 1:
            first_spectrum = spectra[0]
            frequencies = first_spectrum.frequencies
            for spectrum in spectra[1:]:
                if not first_spectrum.compare_spectra_shape(spectrum):
                    raise ValueError("All spectra must have the same shape")
            if not first_spectrum.compare_frequencies(spectrum):
                raise ValueError("All spectra must have the same frequencies")
        

        self._spectra: List[Spectrum] = spectra

    @property
    def spectra(self) -> List[Spectrum]:
        return self._spectra  

    @property
    def frequencies(self) -> np.ndarray:
        """
        Return the frequencies of the spectra.
        Assumes all spectra have the same frequencies.
        """
        if not self._spectra:
            return np.array([])
        return self._spectra[0].frequencies  
    
    @property
    def data(self) -> np.ndarray:
        """
        Return all intensities as an (m × n) array:
        m = number of spectra
        n = number of frequency bins
        """
        return np.vstack([s.intensities for s in self._spectra])

    def __repr__(self) -> str:
        return f'Spectra(spectra={self._spectra})'

    def __getitem__(self, key: Union[int, slice]) -> Union[Spectrum, 'Spectra']:
        if isinstance(key, slice):
            return Spectra(self._spectra[key])
        elif isinstance(key, int):
            return self._spectra[key]
        else:
            raise TypeError("Unsupported index type")

    def __len__(self) -> int:
        return len(self._spectra)
    
    def __iter__(self):
        return iter(self._spectra)
    
    def slice_spectra_at_freq(self,freq):
        return np.array([s.intensities[np.where(s.frequencies==freq)[0][0]] for s in self._spectra])

    def copy(self) -> 'Spectra':
        return Spectra([spectrum.copy() for spectrum in self._spectra])

    def append(self, new_spectrum: Spectrum) -> None:
        """ Appends a new Spectrum object to the Spectra object.

        Args:
            new_spectrum (Spectrum): The Spectrum object to append to the Spectra object.

        Raises:
            TypeError: If new_spectrum is not an instance of Spectrum.
            ValueError: If the new spectrum does not have the same shape as the existing spectra.
            ValueError: If the new spectrum does not have the same frequencies as existing spectra.
        """
        if not isinstance(new_spectrum, Spectrum):
            raise TypeError("new_spectrum must be an instance of Spectrum")
        if len(self._spectra) > 0:
            if not self._spectra[0].compare_spectra_shape(new_spectrum):
                raise ValueError("New spectrum must have the same shape as existing spectra")
            if not self._spectra[0].compare_frequencies(new_spectrum):
                raise ValueError("New spectrum must have the same frequencies as existing spectra")
        self._spectra.append(new_spectrum)
    
    def save_to_csv(
            self, 
            folder_path: str, 
            file_prefix: str, 
            frequency_column_name: str = "frequencies", 
            intensity_column_names: str = "intensities") -> None:
        """ Saves each Spectrum object in the Spectra object to a separate CSV file.

        Args:
            folder_path (str): The folder path where the CSV files will be saved.
            file_prefix (str): The prefix for the CSV file names. Each file will be named as '{file_prefix}_{index}.csv'.
            frequency_column_name (str, optional): The column name for frequencies. Defaults to "frequencies".
            intensity_column_names (str, optional): The column name for intensities. Defaults to "intensities".
        """
        
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        for i, spectrum in enumerate(self._spectra):
            data = np.column_stack((spectrum.frequencies, spectrum.intensities))
            header = f"{frequency_column_name},{intensity_column_names}"
            file_path = os.path.join(folder_path, f"{file_prefix}_{i}.csv")
            np.savetxt(file_path, data, delimiter=",", header=header, comments='')

    def average_spectra(self) -> Spectrum:
        """ Averages the intensities of all Spectrum objects in the Spectra object.

        Raises:
            ValueError: If the Spectra object is empty.

        Returns:
            Spectrum: A new Spectrum object containing the average intensities of the spectra, with the same frequencies as the first spectrum.
        """
        if not self._spectra:
            raise ValueError("Cannot average an empty Spectra list")

        for spectrum in self._spectra:
            assert self._spectra[0].compare_spectra_shape(
                spectrum), 'Spectra must have the same shape'
        average_intensities: np.ndarray = np.average(
            [x.intensities for x in self._spectra], axis=0)
        return Spectrum(self._spectra[0].frequencies, average_intensities)

    def interpolate(self, new_frequencies: np.ndarray) -> 'Spectra':
        """ Interpolates all spectra in the Spectra object to a new set of frequencies.

        Args:
            new_frequencies (np.ndarray): Array of new frequencies to interpolate the spectra to.

        Returns:
            Spectra: A new Spectra object containing Spectrum objects with the interpolated intensities corresponding to the new frequencies.
        """
        new_spectra: Spectra = Spectra([])
        for spectrum in self._spectra:
            new_spectra.append(spectrum.interpolate(new_frequencies))
        return new_spectra

    def average_over_data_repeats(self, num_time_points: int, start: int = 0, end: int = -1) -> 'Spectra':
        """ Averages the spectra over a specified number of time points, effectively grouping the spectra into cycles.

        Args:
            num_time_points (int): The number of time points to average over. This should be a divisor of the total number of spectra.
            start (int, optional): Beginning index for slicing the spectra. Defaults to 0.
            end (int, optional): Ending index for slicing the spectra. Defaults to -1, which means to the end of the spectra.

        Returns:
            Spectra: A new Spectra object containing averaged Spectrum objects, each representing the average of the spectra at each time point.
        """
        if end == -1:
            spectra_subset = self[start:]
        else:
            spectra_subset = self[start:end]
        offset = len(spectra_subset) % num_time_points
        if offset != 0:

            warnings.warn(
                f"The slicing you provided {num_time_points} is not evenly divisible by"
                f" the number of time points {len(spectra_subset)}. Rounding in the back end."
            )
            spectra_subset = spectra_subset[:-offset]
        # number_of_cycles: int = len(spectra_subset) // num_time_points
        averaged_spectra: Spectra = Spectra([])
        for i in range(num_time_points):
            averaged_spectra.append(
                spectra_subset[i::num_time_points].average_spectra()
            )
        return averaged_spectra

    def slice_frequency_at_index(self, frequency_index: int, start: int = 0, end: int = -1) -> np.ndarray:
        """ Slices the spectra at a specific frequency index and returns the intensity at that frequency.

        Args:
            frequency_index (int): The index of the frequency to slice at.
            start (int, optional): Beginning index for slicing. Defaults to 0.
            end (int, optional): Ending index for slicing. Defaults to -1, which means to the end of the spectra.

        Returns:
            np.ndarray: An array of intensities at the specified frequency for each time point collected.
        """
        intensity_at_specific_frequency: List[float] = []
        if end == -1:
            desired_spectra: Spectra = self[start:]
        else:
            desired_spectra: Spectra = self[start:end]
        for spectrum in desired_spectra:
            intensity_at_specific_frequency.append(
                spectrum.intensities[frequency_index])
        return np.array(intensity_at_specific_frequency)

    def slice_specific_frequency(
        self,
        specific_frequency: float,
        start: int = 0,
        end: int = -1,
        round_freq_if_necessary: bool = True
    ) -> np.ndarray:
        """ Slices the spectra at a specific frequency and returns the intensity at that frequency.

        Args:
            specific_frequency (float): The frequency at which to slice the spectra.
            start (int, optional): Beginning index for slicing. Defaults to 0.
            end (int, optional): Ending index for slicing. Defaults to -1, which means to the end of the spectra.
            round_freq_if_necessary (bool, optional): If True, rounds the frequency to the nearest available frequency in the spectra. If False, raises an error if the frequency is not found. Defaults to True.

        Raises:
            ValueError: If specific_frequency is not found in the spectra and round_freq_if_necessary is False.

        Returns:
            np.ndarray: An array of intensities at the specified frequency for each time point collected.
        """

        if specific_frequency not in self._spectra[0].frequencies:
            if not round_freq_if_necessary:
                raise ValueError(f"The frequency {specific_frequency} is not in the data.")
            else:
                old_freq = deepcopy(specific_frequency)
                specific_frequency = self._spectra[0].find_nearest_frequency(
                    specific_frequency)
                warnings.warn(
                    f"The frequency {old_freq} is not in the data. The nearest frequency {specific_frequency} was used instead."
                )

        frequency_index: int = np.where(
            np.array(self._spectra[0].frequencies) == specific_frequency)[0][0]
        intensity_at_specific_frequency: np.ndarray = self.slice_frequency_at_index(
            frequency_index, start=start, end=end)
        return intensity_at_specific_frequency

    def isolate_spectra_sections(self, frequency_sections: List[Tuple[float, float]]) -> 'Spectra':
        """ Isolates sections of the spectra based on provided frequency ranges.

        Args:
            frequency_sections (List[Tuple[float, float]]): List of tuples where each tuple contains a start and end frequency defining a section to isolate.

        Returns:
            Spectra: A new Spectra object containing Spectrum objects with only the frequencies and intensities that fall within the specified sections.
        """
        new_spectra: Spectra = Spectra([])
        for spectrum in self._spectra:
            new_spectra.append(
                spectrum.isolate_spectrum_sections(frequency_sections))
        return new_spectra

    def savitzky_golay_smooth(self, window_size: int = 7, order: int = 3, deriv: int = 0, rate: int = 1) -> 'Spectra':
        """ Smooths all spectra in the Spectra object using the Savitzky-Golay filter.

        Args:
            window_size (int, optional): Size of the filter window. Must be a positive odd integer. Defaults to 7.
            order (int, optional): Order of the polynomial used to fit the samples. Must be less than window_size. Defaults to 3.
            deriv (int, optional): Order of the derivative to compute. Defaults to 0 (no derivative).
            rate (int, optional): Rate of the derivative. Defaults to 1.

        Returns:
            Spectra: A new Spectra object containing Spectrum objects with the smoothed intensities.
        """
        new_spectra: Spectra = Spectra([])
        for spectrum in self._spectra:
            new_spectra.append(spectrum.savitzky_golay_smooth(
                window_size, order, deriv, rate))
        return new_spectra

    def fft_smooth(self, ratio_of_spectra_size: float = 0.1) -> 'Spectra':
        """ Smooths all spectra in the Spectra object using the Fast Fourier Transform (FFT) method.

        Args:
            ratio_of_spectra_size (float, optional): Ratio of the size of the spectrum to keep in the frequency domain. Must be greater than 0. Defaults to 0.1.

        Returns:
            Spectra: A new Spectra object containing Spectrum objects with the smoothed intensities after applying the FFT filter.
        """
        new_spectra: Spectra = Spectra([])
        for spectrum in self._spectra:
            new_spectra.append(spectrum.fft_smooth(ratio_of_spectra_size))
        return new_spectra

    def plot_spectra(self, title: str = '', xlabel: str = 'Frequency (cm-1)', ylabel: str = 'Intensity (a.u.)', **kwargs) -> None:
        """ Simplistic plot of all spectra in the Spectra object using matplotlib.

        Args:
            title (str, optional): Title of the plot. Defaults to ''.
            xlabel (str, optional): X-axis label. Defaults to 'Frequency (cm-1)'.
            ylabel (str, optional): Y-axis label. Defaults to 'Intensity (a.u.)'.
            **kwargs: Additional keyword arguments to pass to the matplotlib plot function.
        """
        
        for spectrum in self._spectra:
            plt.plot(spectrum.frequencies, spectrum.intensities, **kwargs)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()


if __name__ == '__main__':
    pass
