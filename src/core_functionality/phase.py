#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
This file provides functionality for performing Fourier Transform on spectral data,
calculating phase shifts, and applying weights to the Fourier transformed data.
'''

# Written by
# Alfred Worrad <worrada@udel.edu>,

__author__ = "Afred Worrad"
__version__ = "0.1.3"
__maintainer__ = "Alfred Worrad"
__email__ = "worrada@udel.edu"
__status__ = "Development"
__project__ = "MES wavey"
__created__ = "December 06, 2023"
__updated__ = "June 26, 2025"

# built-in modules
from typing import List, Optional
from itertools import count
import warnings

# third-party modules
import numpy as np
from scipy.fft import fft, ifft
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# project modules
from src.core_functionality.spectrum import Spectra


class Phase():


    def __init__(self, data: Spectra, num_time_points: int, start: int = 0, end: int = -1):
        
        self._average_data = data.average_over_data_repeats(num_time_points, start, end)
        self._general_frequency_values = self._average_data[0].frequencies

        num_spectra = len(self._average_data)
        num_freqs = len(self._general_frequency_values)
        shape = (num_spectra, num_freqs)

        self._plot_phase_data = np.zeros(shape)
        self._ft_all_components = np.zeros(shape, dtype=complex)
        self._ft_real_fft_component = np.zeros(shape)
        self._ft_imaginary_component = np.zeros(shape) 
        self._phase_shift = np.zeros(shape) # shape of (# harmonics, # real frequencies)
        self._ifft_data = np.zeros(shape)

    @property
    def ft_all_components(self) -> np.ndarray:
        return self._ft_all_components

    @property
    def ifft_data(self) -> np.ndarray:
        return self._ifft_data

    @property
    def phase_shift(self) -> np.ndarray:
        return self._phase_shift
    
    # def phase_test_direct_demod(self, desired_frequency:int, harmonic_index:int = 1):
    #     T = self._average_data.shape[0]
    #     E = 2 / T * np.trapezoid(self._average_data * sin())

    def fourier_transform_on_avg_data(self, harmonic=1):
        
        frequencies = self._average_data[0].frequencies
        for freq, i in zip(frequencies, count()):
            time_slice = self._average_data.slice_frequency_at_index(i)
            fft_signal = fft(time_slice)
            filtered_signal = np.zeros_like(fft_signal)
            phases_data = np.zeros_like(fft_signal)
            magnitude_p = np.abs(fft_signal[harmonic])
            magnitude_n = np.abs(fft_signal[-harmonic])

            # if phase_angle < 0:
            true_phase = (-np.angle(fft_signal[harmonic]) + np.pi/2) % (2 * np.pi)
            phi_psd_deg = np.rad2deg(true_phase)
            filtered_signal[harmonic] = magnitude_p * np.exp(1j * true_phase)
            filtered_signal[-harmonic] = magnitude_n * np.exp(1j * -true_phase)
            inv_fft = -ifft(filtered_signal) 
            phases_data[harmonic] = (np.rad2deg(np.angle(fft_signal[harmonic]) )+np.pi/2) %(2*np.pi)
            # if np.rad2deg(np.angle(fft_signal[harmonic]))+90 < 0:
            #     print(f"FFT phase angle at harmonic {harmonic}: {(np.rad2deg(np.angle(fft_signal[harmonic]) )+90) %360} degrees")
            # else:
            #     print(f"FFT phase angle at harmonic {harmonic}: {(np.rad2deg(np.angle(fft_signal[harmonic]) )+90) %360} degrees")
            self._plot_phase_data[:, i] = -ifft(filtered_signal).real
            self._ft_all_components[:, i] = fft_signal
            self._ft_real_fft_component[:, i] = np.real(fft_signal)
            self._ft_imaginary_component[:, i] = np.imag(fft_signal)
            self._phase_shift[:, i] = phases_data
        return self

    def fourier_transform_on_avg_data_old(self):
        """ Performs Fourier Transform on the average data by isolating each individual
            spectral frequency as a timeseries and applying FFT. Time is the time the spectrum
            was collected so after the FFT, the fourier transform frequency space is frequency 
            of data collection.

        Returns:
            _type_: self with the Fourier transformed data stored in the attributes.
        """
        frequencies = self._average_data[0].frequencies
        for freq, i in zip(frequencies, count()):
            time_slice = self._average_data.slice_frequency_at_index(i)
            fft_slice = fft(time_slice)
            
            self._ft_all_components[:, i] = fft_slice
            self._ft_real_fft_component[:, i] = np.real(fft_slice)
            self._ft_imaginary_component[:, i] = np.imag(fft_slice)
            self._phase_shift[:, i] = np.angle(fft_slice)

        return self

    # Made a breaking change with the weight files by replacing it with a list entry
    # This is useful for the GUI but not really for more flexible code options.
    def weight(self, list_of_harmonics: Optional[List[int]] = None):
        """ Applies weights to the Fourier transformed data based on the specified harmonics.

        Args:
            list_of_harmonics (Optional[List[int]], optional): A list of harmonics to keep with weight 1. If None, all harmonics are kept with weight 1.

        Returns:
            _type_: self with the Fourier transformed data weighted according to the specified harmonics provided in the list_of_harmonics.
        """

        num_spectra, num_harmonics = self._ft_all_components.shape

        # Default: keep all harmonics with weight 1
        if list_of_harmonics is None:
            weights = np.ones(num_harmonics)
        else:
            weights = np.zeros(num_harmonics)
            for h in list_of_harmonics:
                if 1 <= h <= num_harmonics:
                    weights[h - 1] = 1  # Convert to 0-based index
                else:
                    warnings.warn(f"Harmonic {h} is out of bounds (max {num_harmonics}). Ignored.")

        # Apply weights across all spectra
        self._ft_real_fft_component *= weights
        self._ft_imaginary_component *= weights
        self._ft_all_components *= weights

        return self

    def save_fft_to(self, fpath: str):
        """ Saves the Fourier transformed data to a CSV file.

        Args:
            fpath (str): The file path where the Fourier transformed data will be saved.
        """
        freq_data = np.array([self._general_frequency_values])
        data_for_df = np.concatenate(
            (freq_data, self._ft_real_fft_component), axis=0)
        df = pd.DataFrame(data_for_df)
        print('Writing to ', fpath)
        df.to_csv(fpath)
    
    def save_ifft_to(self, fpath: str, fft_real_component: bool = False):
        """ Saves the Inverse Fourier Transform data to a CSV file.

        Args:
            fpath (str): The file path where the Inverse Fourier Transform data will be saved.
            fft_real_component (bool, optional): If True, saves only the real component of the FFT. Defaults to False.

        Raises:
            IOError: If there is an issue writing to the file, such as it being open in another program.
        """
        freq_data = np.array([self._general_frequency_values])
        ifft_data: List[np.ndarray] = []

        if fft_real_component:
            for i in range(self._ft_real_fft_component.shape[1]):
                ifft_data.append(np.real(ifft(self._ft_real_fft_component[:, i])))
        else:
            for i in range(self._ft_all_components.shape[1]):
                ifft_data.append(np.real(ifft(self._ft_all_components[:, i])))

        ifft_data = np.array(ifft_data).T
        self._ifft_data = ifft_data

        data_for_df = np.concatenate((freq_data, ifft_data), axis=0)
        df = pd.DataFrame(data_for_df)

        try:
            print('Writing to', fpath)
            df.to_csv(fpath, index=False)
        except PermissionError:
            raise IOError(f"Cannot write to '{fpath}'. It may be open in another program (e.g. Excel). Please close it and try again.")


    def save_phase_to(self, fpath: str):
        """ Saves the phase shift data to a CSV file.   

        Args:
            fpath (str): The file path where the phase shift data will be saved.

        Raises:
            IOError: If there is an issue writing to the file, such as it being open in another program.
        """
        freq = self._general_frequency_values
        phase = self._phase_shift 

        df = pd.DataFrame(phase, columns=freq)
        df.index = [f"Harmonic {i+1} (rad)" for i in range(len(phase))]
        df.index.name = "Component"

        try:
            print(f"Writing phase shift data to {fpath}")
            df.to_csv(fpath)
        except PermissionError:
            raise IOError(f"Cannot write to '{fpath}'. It may be open in another program (e.g. Excel). Please close it and try again.")

    def fft_harmonic_amp_phase_sin_PSD(self, frequency: float, harmonic: int):
        """
        This function is designed to get all the relevant information from a single FFT coefficient
        for a specific harmonic.
        """
        
        k = harmonic  
        index_of_the_frequency = self._general_frequency_values.searchsorted(frequency)
        Xk = self._ft_all_components[k, index_of_the_frequency]
        N = self._ft_all_components.shape[0]


        if k == 0 or (N % 2 == 0 and k == N // 2):
            amp = np.abs(Xk) / N
        else:
            amp = 2.0 * np.abs(Xk) / N

        # adjustment to match sin convention not FFT
        phi_sin = np.arctan2(-Xk.imag, Xk.real)

        phi_sin = (phi_sin) % (2*np.pi)

        return amp, phi_sin

    def plot_sinusodial_PD_spectra(self, sample_factor: int = 1, matplotlib_param_dict: Optional[dict] = None):
        """ Plots the phase domain spectra data

        Args:
            sample_factor (int, optional): Factor to reduce the number of spectra plotted for clarity. Defaults to 1 which plots all.
            matplotlib_param_dict (Optional[dict], optional): Custom matplotlib parameters. Defaults to None.
        """

        # This should probably be adjusted, currently things look improperly sized
        if matplotlib_param_dict:
            mpl.rcParams.update(matplotlib_param_dict)
        else:
            mpl.rcParams.update({
                "font.family": "sans-serif",
                "font.sans-serif": ["Arial"],
                "axes.linewidth": 1.2,
                "axes.labelsize": 28,
                "xtick.labelsize": 20,
                "ytick.labelsize": 20,
                "legend.fontsize": 26,
                "lines.linewidth": 1.0,
            })


        fig, ax = plt.subplots(figsize=(8, 6))
        num_lines = self._plot_phase_data.shape[0] // sample_factor +1
        colors = plt.cm.viridis(np.linspace(0, 1, num_lines))

        for idx, i in enumerate(range(0, self._plot_phase_data.shape[0], sample_factor)):
            ax.plot(
                self._general_frequency_values,
                -self._plot_phase_data[i, :],
                color=colors[idx],
                alpha=0.85,
            )


        ax.set_xlabel("Wavenumber (nm)")
        ax.set_ylabel("Absorbance (a.u.)")
        sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=0, vmax=self._plot_phase_data.shape[0]))
        cbar = plt.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label("Time Index", rotation=270, labelpad=30)

        plt.tight_layout()
        plt.show()

    def plot_phase_as_sine(self, frequency: float, harmonic: int = 1, matplotlib_param_dict: Optional[dict] = None):
        """ Plots the phase shift as a sine wave for a specific frequency and harmonic.

        Args:
            frequency (float): The frequency at which to plot the phase shift.
            harmonic (int, optional): The harmonic to consider. Defaults to 1.
            matplotlib_param_dict (Optional[dict], optional): Custom matplotlib parameters. Defaults to None.
        """
        
        if matplotlib_param_dict:
            mpl.rcParams.update(matplotlib_param_dict)
        else:
            mpl.rcParams.update({
                "font.family": "sans-serif",
                "font.sans-serif": ["Arial"],
                "axes.linewidth": 1.2,
                "axes.labelsize": 18,
                "xtick.labelsize": 20,
                "ytick.labelsize": 20,
                "legend.fontsize": 18,
                "lines.linewidth": 1.0,
            })
        
        amp_fft, phi_fft = self.fft_harmonic_amp_phase_sin_PSD(frequency, harmonic)
        psi = np.linspace(0, 2*np.pi, len(self._general_frequency_values))
        plt.plot(psi, amp_fft * np.sin(psi + phi_fft), "--",
            label=f"FFT φ={np.degrees(phi_fft):.1f}°")
        plt.xlabel("Phase (radians)")
        plt.ylabel("Amplitude-scaled sine (a.u.)")
        plt.xlim(0, 2*np.pi)
        plt.xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi],
                ["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
        plt.legend(); plt.tight_layout()

        
if __name__ == "__main__":
    pass
