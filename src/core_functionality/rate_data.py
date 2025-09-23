#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
File for extracting rate constant data from the transient spectra.
This file contains the work related to extracting rate constant data
from each pulse in the transient spectra.
'''

# Written by
# Alfred Worrad <worrada@udel.edu>,

__author__ = "Afred Worrad"
__version__ = "1.2.1"
__maintainer__ = "Alfred Worrad"
__email__ = "worrada@udel.edu"
__status__ = "Development"
__project__ = "MES wavey"
__created__ = "December 07, 2023"
__updated__ = "July 07, 2025"

# built-in modules
from typing import List, Optional, Tuple, Callable, Dict, Any
from itertools import count

# third-party modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.signal import savgol_filter


# project modules
from src.core_functionality.spectrum import Spectrum, Spectra

# types of fitting functions
FREQUENCY = "frequency"
RATE_CONSTANTS = "rate_constants"
DECAY_DATA = "decay_data"
TIME_SEGMENTS = "time_segments"
POPT_LIST = "popt_list"
PCOV_LIST = "pcov_list"

def fit_exp(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """
    Fit the data to an exponential function
    """
    return a * np.exp(-b * x) + c

# Note to self, after deconvolutions, we can always go back and do this fitting.
class RateData:
    def __init__(self, spectra: Spectra, time_step_in_seconds: float = 1) -> None:
        """ Initialize the RateData object.

        Args:
            spectra (Spectra): The spectra object containing transient spectra data.
            time_step_in_seconds (float, optional): Time step in seconds for the spectra data. Defaults to 1.
        """
        self._spectra = spectra
        self._time_step = time_step_in_seconds
        self._average_spectra = None

    @property
    def average_spectra(self) -> Spectra:
        """Return the averaged transient spectra."""
        return self._average_spectra

    def get_average_spectra_desired_freq(
        self,
        select_frequency: float,
        num_time_points: int,
        time_start: float = 0,
        time_end: float = -1,
    ) -> np.ndarray:
        """ Get the average spectra at a specific frequency.

        Args:
            select_frequency (float): The frequency at which to get the average spectra.
            num_time_points (int): Number of time points to consider for the average spectra.
            time_start (float, optional): The start time for the average spectra. Defaults to 0.
            time_end (float, optional): The end time for the average spectra. Defaults to -1, which means it will use the length of the spectra.

        Returns:
            np.ndarray: Average spectra at the specified frequency.
        """
        if self._average_spectra is None:
            self._average_spectra = self._spectra.average_over_data_repeats(
                num_time_points, time_start, time_end
            )
        return self._average_spectra.slice_specific_frequency(select_frequency)

    def _compute_rate_single_freq(
        self,
        data_arr: np.ndarray,
        time_arr: np.ndarray,
        num_segments: int,
        fitting_function: Callable = fit_exp,
        adaptive_fitting: bool = False,
        adaptive_swapping_fraction: float = 0.75,
        smooth_data: bool = False,
        prominence_distance: int = 10,
        prominence_threshold: float = 0.01,
        smooth_window_size: int = 5,
        smooth_polyorder: int = 3,
        max_segments: Optional[int] = None
    ) -> Dict[str, Any]:
        """ Compute rate constants for a single frequency.

        Args:
            data_arr (np.ndarray): Array of data values for a specific frequency.
            time_arr (np.ndarray): Array of time values corresponding to the data.
            num_segments (int): Number of segments to divide the data into for rate calculation.
            fitting_function (Callable, optional): Function to fit the data. Defaults to fit_exp.
            adaptive_fitting (bool, optional): If True, uses adaptive fitting for each segment. Defaults to False.
            adaptive_swapping_fraction (float, optional): The fraction of segments to swap during adaptive fitting. Defaults to 0.75.
            smooth_data (bool, optional): If True, applies Savitzky-Golay smoothing to the data. Defaults to True.
            prominence_distance (int, optional): The distance parameter for prominence calculation. Defaults to 10.
            prominence_threshold (float, optional): The threshold for prominence calculation. Defaults to 0.01.
            smooth_window_size (int, optional): The window size for Savitzky-Golay smoothing. Defaults to 5.
            smooth_polyorder (int, optional): The polynomial order for Savitzky-Golay smoothing. Defaults to 3.
            max_segments: Optional[int] : The maximum number of segments to divide the signal into. Defaults to None.

        Returns:
            Dict[str, Any]: A dictionary containing the rate constants, optimal decay data, time segments, popt list, and pcov list from fitting.
        """
        self.data_arr = data_arr

        decay_data, time_segments = self._find_decay_data(
            num_segments, time_arr, smooth_data=smooth_data, prominence_distance=prominence_distance, prominence_threshold=prominence_threshold,
            smooth_window_size=smooth_window_size, smooth_polyorder=smooth_polyorder, max_segments=max_segments
        )

        rate_constants, optimal_decay_data, time_segments, popt_list, pcov_list = self._obtain_exp_rate_constants(
            decay_data,
            time_segments=time_segments,
            fitting_function=fitting_function,
            adaptive_fitting=adaptive_fitting,
            adaptive_swapping_fraction=adaptive_swapping_fraction,
            max_segments=max_segments
        )
        # adjust exp based on the timestep assuming exp
        popt_list = [np.array([p[0], p[1] / self._time_step, p[2]]) for p in popt_list]
        output_dict = {
            RATE_CONSTANTS: rate_constants / self._time_step,  # convert to per second
            DECAY_DATA: optimal_decay_data,
            TIME_SEGMENTS: time_segments,
            POPT_LIST: popt_list,
            PCOV_LIST: pcov_list
        }

        return output_dict
    # I need to reformat this because the thing being passed is now a dictionary and not all the individual elements are being passed
    def plot_decays_and_fits(
        self,
        rate_constants: np.ndarray,
        decay_data: List[np.ndarray],
        time_segments: List[np.ndarray],
        popt_list: List[np.ndarray],
        pcov_list:Optional[List[np.ndarray]] = None,
        fitting_function: Callable = fit_exp,
        **kwargs
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Plot decay data segments with overlaid exponential fits for visual validation.

        Args:
            rate_constants (np.ndarray): Array of extracted rate constants for each segment.
            decay_data (List[np.ndarray]): List of decay data arrays.
            timesegments (List[np.ndarray]): List of corresponding time arrays for each segment.
            popt_list (List[np.ndarray]): List of fitted parameters [a, k, c] for each segment.
            fitting_function (Callable, optional): Fitting function to overlay. Defaults to fit_exp.
            **kwargs

        Returns:
            Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]: A tuple containing the fitted curves, decay data, and time segments.
        """
        def format_k(val):
            s = f"{val:.2e}"
            base, exp = s.split("e")
            exp = exp.replace("+0", "+").replace("-0", "-")
            return f"{base}e{exp}"
        
        num_segments = len(decay_data)
       
        


        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial"],
            "axes.labelsize": 28,
            "axes.titlesize": 28,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 18,
            "lines.linewidth": 3,
            "lines.markersize": 6,
        })

        fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
        colors = plt.cm.viridis(np.linspace(0, 1, num_segments))
        legend_handles_colored = []
        curve_fits = []

        for i in range(num_segments):
            data_seg = decay_data[i]
            time_seg = time_segments[i]
            popt = popt_list[i]

            color = colors[i]
            fit_curve = fitting_function(time_seg - time_seg[0], *popt)

            ax.plot(time_seg, data_seg, 'o', color=color, alpha=0.7)
            ax.plot(time_seg, fit_curve, '-', color=color)
            curve_fits.append(fit_curve)

            legend_handles_colored.append(Line2D(
                [0], [0],
                color=color,
                marker='o',
                linestyle='-',
                linewidth=3,
                markersize=6,
                label=f"Seg {i+1} (k={format_k(popt[1])})"
            ))

        # === Legend 1: Data vs Fit ===
        legend_handles_datafit = [
            Line2D([0], [0], linestyle='none', marker='o', color='black', label="Data", markersize=6),
            Line2D([0], [0], linestyle='-', color='black', label="Fit", linewidth=3),
        ]

        first_legend = ax.legend(
            handles=legend_handles_datafit,
            loc='upper left',
            bbox_to_anchor=(1.03, 1.0),
            bbox_transform=ax.transAxes,
            frameon=False
        )
        ax.add_artist(first_legend)

        # === Legend 2: Segments ===
        ax.legend(
            handles=legend_handles_colored,
            loc='center left',
            bbox_to_anchor=(1.03, 0.4),
            bbox_transform=ax.transAxes,
            frameon=False
        )

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Signal (a.u.)")
        plt.tight_layout()
        # plt.show()
        return curve_fits, decay_data, time_segments

    def compute_average_rate_data(
        self,
        select_frequency: float,
        num_time_points: int,
        time_start: float = 0,
        time_end: float = -1,
        fitting_function: Callable = fit_exp,
        adaptive_fitting: bool = False,
        adaptive_swapping_fraction: float = 0.75,
        smooth_data: bool = False,
        prominence_distance: int = 10,
        prominence_threshold: float = 0.01,
        smooth_window_size: int = 5,
        smooth_polyorder: int = 3,
        max_segments: Optional[int] = None
    ) -> Dict[str, Any]:
        """ Compute rate constants for a specific frequency in the averaged spectra.

        Args:
            select_frequency (float): The frequency at which to compute the rate constants.
            num_time_points (int): Number of time points to consider for the rate calculation.
            time_start (float, optional): The start time for the rate calculation. Defaults to 0.
            time_end (float, optional): The end time for the rate calculation. Defaults to -1, which means it will use the length of the spectra.
            fitting_function (Callable, optional): Function to fit the data. Defaults to fit_exp.
            adaptive_fitting (bool, optional): If True, uses adaptive fitting for each segment. Defaults to False.
            adaptive_swapping_fraction (float, optional): The fraction of segments to swap during adaptive fitting. Defaults to 0.75.
            smooth_data (bool, optional): If True, applies Savitzky-Golay smoothing to the data. Defaults to False.
            prominence_distance (int, optional): The distance parameter for prominence calculation. Defaults to 10.
            smooth_window_size (int, optional): The window size for Savitzky-Golay smoothing. Defaults to 5.
            smooth_polyorder (int, optional): The polynomial order for Savitzky-Golay smoothing. Defaults to 3.
            max_segments (Optional[int], optional): The maximum number of segments to divide the signal into. Defaults to None.

        Returns:
            Dict[str, Any]: A dictionary containing the rate constants, optimal decay data, time segments, popt list, and pcov list from fitting.
        """
        data = self._spectra.average_over_data_repeats(
            num_time_points, time_start, time_end
        )
        self._average_spectra = data

        if time_end == -1:
            time_end = num_time_points

        time_arr = np.arange(time_start, time_end, 1) * self._time_step
        data_arr = data.slice_specific_frequency(select_frequency)

        return self._compute_rate_single_freq(
            data_arr, time_arr, num_segments=1, fitting_function=fitting_function,
            adaptive_fitting=adaptive_fitting, adaptive_swapping_fraction=adaptive_swapping_fraction,smooth_data=smooth_data, 
            prominence_distance=prominence_distance, prominence_threshold=prominence_threshold,
            smooth_window_size=smooth_window_size, smooth_polyorder=smooth_polyorder, max_segments=max_segments
        )

    def compute_all_rate_data_single_freq(
        self,
        select_frequency: float,
        num_time_points: int,
        time_start: float = 0,
        time_end: float = -1,
        fitting_function: Callable = fit_exp,
        adaptive_fitting: bool = False,
        adaptive_swapping_fraction: float = 0.75,
        smooth_data: bool = False,
        prominence_distance: int = 10,
        prominence_threshold: float = 0.0,
        smooth_window_size: int = 5,
        smooth_polyorder: int = 3,
        max_segments: Optional[int] = None
    ) -> Dict[str, Any]:
        """ Compute rate constants for a specific frequency in the spectra.

        Args:
            select_frequency (float): The frequency at which to compute the rate constants.
            num_time_points (int): Number of time points to consider for the rate calculation.
            time_start (float, optional): The start time for the rate calculation. Defaults to 0.
            time_end (float, optional): The end time for the rate calculation. Defaults to -1, which means it will use the length of the spectra.
            fitting_function (Callable, optional): Function to fit the data. Defaults to fit_exp.
            adaptive_fitting (bool, optional): If True, uses adaptive fitting for each segment. Defaults to False.
            adaptive_swapping_fraction (float, optional): The fraction of segments to swap during adaptive fitting. Defaults to 0.75.
            smooth_data (bool, optional): If True, applies Savitzky-Golay smoothing to the data. Defaults to False.
            prominence_distance (int, optional): The distance parameter for prominence calculation. Defaults to 10.
            prominence_threshold (float, optional): The threshold for prominence calculation. Defaults to 0.0.
            smooth_window_size (int, optional): The window size for Savitzky-Golay smoothing. Defaults to 5.
            smooth_polyorder (int, optional): The polynomial order for Savitzky-Golay smoothing. Defaults to 3.
            max_segments (Optional[int], optional): The maximum number of segments to divide the signal into. Defaults to None.

        Returns:
            Dict[str, Any]: A dictionary containing the rate constants, optimal decay data, time segments, popt list, and pcov list from fitting.
        """
        if time_end == -1:
            time_end = len(self._spectra)

        time_arr = np.arange(time_start, time_end, 1) * self._time_step
        data_arr = self._spectra.slice_specific_frequency(
            select_frequency, time_start, time_end
        )

        num_segments = len(data_arr) // num_time_points

        return self._compute_rate_single_freq(
            data_arr, time_arr, num_segments=num_segments, fitting_function=fitting_function,
            adaptive_fitting=adaptive_fitting, adaptive_swapping_fraction=adaptive_swapping_fraction, smooth_data=smooth_data, 
            prominence_distance=prominence_distance, prominence_threshold=prominence_threshold,
            smooth_window_size=smooth_window_size, smooth_polyorder=smooth_polyorder, max_segments=max_segments
        )

    def compute_all_rate_data_all_freq(
        self,
        num_time_points: int,
        time_start: float = 0,
        time_end: float = -1,
        fitting_function: Callable = fit_exp,
        adaptive_fitting: bool = False,
        adaptive_swapping_fraction: float = 0.75,
        smooth_data: bool = False,
        prominence_distance: int = 10,
        prominence_threshold: float = 0.01,
        smooth_window_size: int = 5,
        smooth_polyorder: int = 3,
        max_segments: Optional[int] = None
    ) -> Dict[int, Dict[str, Any]]:
        """ Compute rate constants for all frequencies in the spectra.

        Args:
            num_time_points (int): Number of time points to consider for each frequency.
            time_start (float, optional): The start time for the rate calculation. Defaults to 0.
            time_end (float, optional): The end time for the rate calculation. Defaults to -1, which means it will use the length of the spectra.
            fitting_function (Callable, optional): Function to fit the data. Defaults to fit_exp.
            adaptive_fitting (bool, optional): If True, uses adaptive fitting for each segment. Defaults to False.
            adaptive_swapping_fraction (float, optional): The fraction of segments to swap during adaptive fitting. Defaults to 0.75.
            smooth_data (bool, optional): If True, applies Savitzky-Golay smoothing to the data. Defaults to False.
            prominence_distance (int, optional): The distance parameter for prominence calculation. Defaults to 10.
            prominence_threshold (float, optional): The threshold for prominence calculation. Defaults to 0.01.
            smooth_window_size (int, optional): The window size for Savitzky-Golay smoothing. Defaults to 5.
            smooth_polyorder (int, optional): The polynomial order for Savitzky-Golay smoothing. Defaults to 3.
            max_segments (Optional[int], optional): The maximum number of segments to divide the signal into. Defaults to None.

        Returns:
            Dict[int, Dict[str, Any]]: A dictionary containing the rate constants, optimal decay data, time segments, popt list, and pcov list from fitting for each frequency.

        """
        rates_per_frequency_dict = dict()
        for freq, i in zip(self._spectra[0].frequencies, count()):
            output_dict_at_curr_freq = self.compute_all_rate_data_single_freq(
                select_frequency=freq,
                num_time_points=num_time_points,
                time_start=time_start,
                time_end=time_end,
                fitting_function=fitting_function,
                adaptive_fitting=adaptive_fitting,
                adaptive_swapping_fraction=adaptive_swapping_fraction,
                smooth_data=smooth_data,
                prominence_distance=prominence_distance,
                prominence_threshold=prominence_threshold,
                smooth_window_size=smooth_window_size,
                smooth_polyorder=smooth_polyorder,
                max_segments=max_segments
            )
            rates_per_frequency_dict[i] = {
                FREQUENCY: freq,
                **output_dict_at_curr_freq
            }
        return rates_per_frequency_dict

    def _fit_exp_to_data(
        self,
        data: np.ndarray,
        fitting_function: Callable = fit_exp,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Fit data to exponential function."""
        x = np.arange(len(data))
        popt, pcov = curve_fit(
            fitting_function,
            x,
            data,
            maxfev=5000,
            # bounds=([0, 0, -np.inf], [np.inf, np.inf, np.inf])  # enforce a >= 0, b >= 0
        )
        fit = fitting_function(x, *popt)
        return x, fit, popt, pcov

    def _optimize_segments_by_swapping(
        self,
        decay_data: List[np.ndarray],
        time_segments: List[np.ndarray],
        max_swap_fraction: float = 0.5,
        fitting_function: Callable = fit_exp,
        fit_eval_function: Callable = None,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """ Optimize decay segments by swapping points between adjacent segments to improve fit quality.
        
        This method checks both directions for each segment boundary:
        1. Moving points from current segment to previous segment
        2. Moving points from previous segment to current segment
        The direction and number of points that gives the best overall fit is selected.

        Args:
            decay_data (List[np.ndarray]): List of decay data segments to optimize.
            time_segments (List[np.ndarray]): List of time segments corresponding to the decay data.
            max_swap_fraction (float, optional): Maximum fraction of points to swap between segments. Defaults to 0.5.
            fitting_function (Callable, optional): Function to fit the data. Defaults to fit_exp.
            fit_eval_function (Callable, optional): Function to evaluate the fit quality. Defaults to None.

        Returns:
            Tuple[List[np.ndarray], List[np.ndarray]]: Updated decay data and time segments
        """
        

        if fit_eval_function is None:
            def fit_eval_function(x, y, popt):
                return np.sum((fitting_function(x, *popt) - y) ** 2)

        n_segments = len(decay_data)
        optimized_data = [arr.copy() for arr in decay_data]
        optimized_time_segments = [arr.copy() for arr in time_segments]

        for i in range(1, n_segments):
            prev_seg = optimized_data[i - 1]
            curr_seg = optimized_data[i]
            prev_time = optimized_time_segments[i - 1]
            curr_time = optimized_time_segments[i]

            # Calculate maximum points we can shift in each direction
            max_shift_from_curr = int(len(curr_seg) * max_swap_fraction)
            max_shift_from_prev = int(len(prev_seg) * max_swap_fraction)
            
            best_score = None
            best_shift = 0
            best_direction = None

            # Try shifting points from current segment to previous segment (positive shifts)
            for shift_count in range(0, max_shift_from_curr + 1):
                if shift_count >= len(curr_seg):
                    break

                new_prev = np.concatenate([prev_seg, curr_seg[:shift_count]])
                new_curr = curr_seg[shift_count:]

                new_prev_time = np.concatenate([prev_time, curr_time[:shift_count]])
                new_curr_time = curr_time[shift_count:]

                if len(new_curr) < 5 or len(new_prev) < 5:
                    continue  # avoid overly small segments

                x_prev = np.arange(len(new_prev))
                x_curr = np.arange(len(new_curr))

                try:
                    popt_prev, _ = curve_fit(fitting_function, x_prev, new_prev, maxfev=5000)
                    popt_curr, _ = curve_fit(fitting_function, x_curr, new_curr, maxfev=5000)

                    score = fit_eval_function(x_prev, new_prev, popt_prev) + fit_eval_function(x_curr, new_curr, popt_curr)

                    if best_score is None or score < best_score:
                        best_score = score
                        best_shift = shift_count
                        best_direction = 'curr_to_prev'

                except Exception:
                    continue

            # Try shifting points from previous segment to current segment (negative shifts)
            for shift_count in range(1, max_shift_from_prev + 1):
                if shift_count >= len(prev_seg):
                    break

                new_prev = prev_seg[:-shift_count]
                new_curr = np.concatenate([prev_seg[-shift_count:], curr_seg])

                new_prev_time = prev_time[:-shift_count]
                new_curr_time = np.concatenate([prev_time[-shift_count:], curr_time])

                if len(new_curr) < 5 or len(new_prev) < 5:
                    continue  # avoid overly small segments

                x_prev = np.arange(len(new_prev))
                x_curr = np.arange(len(new_curr))

                try:
                    popt_prev, _ = curve_fit(fitting_function, x_prev, new_prev, maxfev=5000)
                    popt_curr, _ = curve_fit(fitting_function, x_curr, new_curr, maxfev=5000)

                    score = fit_eval_function(x_prev, new_prev, popt_prev) + fit_eval_function(x_curr, new_curr, popt_curr)

                    if best_score is None or score < best_score:
                        best_score = score
                        best_shift = shift_count
                        best_direction = 'prev_to_curr'

                except Exception:
                    continue

            # Apply the best shift in the best direction
            if best_shift > 0 and best_direction is not None:
                if best_direction == 'curr_to_prev':
                    # Move points from current to previous
                    optimized_data[i - 1] = np.concatenate([prev_seg, curr_seg[:best_shift]])
                    optimized_data[i] = curr_seg[best_shift:]
                    optimized_time_segments[i - 1] = np.concatenate([prev_time, curr_time[:best_shift]])
                    optimized_time_segments[i] = curr_time[best_shift:]
                elif best_direction == 'prev_to_curr':
                    # Move points from previous to current
                    optimized_data[i - 1] = prev_seg[:-best_shift]
                    optimized_data[i] = np.concatenate([prev_seg[-best_shift:], curr_seg])
                    optimized_time_segments[i - 1] = prev_time[:-best_shift]
                    optimized_time_segments[i] = np.concatenate([prev_time[-best_shift:], curr_time])

        return optimized_data, optimized_time_segments


    def _obtain_exp_rate_constants(
        self,
        decay_data: List[np.ndarray],
        time_segments: List[np.ndarray],
        fitting_function: Callable = fit_exp,
        adaptive_fitting: bool = False,
        adaptive_swapping_fraction: float = 0.75,
        max_segments: Optional[int] = None
    ) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
        """ Extract rate constants from decay data using exponential fitting.

        Args:
            decay_data (List[np.ndarray]): List of decay data segments to fit.
            fitting_function (Callable, optional): Function to fit the data. Defaults to fit_exp.

        Returns:
            Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]: Rate constants, optimal parameters, and covariance matrices.
        """
        if adaptive_fitting:
            optimal_decay_data, optimized_time_segments = self._optimize_segments_by_swapping(
                decay_data,
                time_segments=time_segments,
                max_swap_fraction=adaptive_swapping_fraction,
                fitting_function=fitting_function,
            )
        else:
            optimal_decay_data = decay_data
            optimized_time_segments = time_segments
        rate_constants = []
        popt_list = []
        pcov_list = []
        for data in optimal_decay_data:
            x, y, popt, pcov = self._fit_exp_to_data(data, fitting_function=fitting_function)
            rate_constants.append(popt[1])
            popt_list.append(popt)
            pcov_list.append(pcov)
        return np.array(rate_constants), optimal_decay_data, optimized_time_segments, popt_list, pcov_list

    def smooth_data_with_savgol(self, data, window_length:int, polyorder:int=3):
        # This function probably doesn't need to be here. The user can just call savgol_filter directly
        smoothed_data = savgol_filter(data, window_length=window_length, polyorder=polyorder)
        return smoothed_data

    def _find_decay_data(
        self,
        number_of_max_and_min: int,
        time_arr: np.ndarray,
        smooth_data: bool = True,
        prominence_distance: int = 10,
        prominence_threshold: float = 0.01,
        smooth_window_size: int = 5,
        smooth_polyorder: int = 3,
        max_segments: Optional[int] = None,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """ Identify decay segments in the data based on local maxima and minima.

        Args:
            number_of_max_and_min (int): The maximum number of maxima and minima to identify.
            time_arr (np.ndarray): Array of time values corresponding to the data.
            smooth_data (bool, optional): If True, applies Savitzky-Golay smoothing to the data. Defaults to True.
            prominence_distance (int, optional): Minimum horizontal distance (in samples) between neighboring peaks. Defaults to 10.
            prominence_threshold (float, optional): Minimum prominence required to consider a peak. Defaults to 0.01.
            smooth_window_size (int, optional): Window size for Savitzky-Golay smoothing. Defaults to 5.
            smooth_polyorder (int, optional): Polynomial order for Savitzky-Golay smoothing. Defaults to 3.
            max_segments (Optional[int], optional): Maximum number of segments to identify. Defaults to None.

        Raises:
            ValueError: If the length of the data array and time array do not match.
            ValueError: If number_of_max_and_min is not positive.

        Returns:
            Tuple[List[np.ndarray], List[np.ndarray]]: Identified decay segments and their corresponding time segments.
        """
        # Internal constraint: minimum points per segment for reliable curve fitting
        min_segment_length = 3
        
        if len(self.data_arr) != len(time_arr):
            raise ValueError("Length of data array and time array must match.")
        if number_of_max_and_min <= 0:
            raise ValueError("number_of_max_and_min must be positive.")

        data = np.asarray(self.data_arr, dtype=float)
        t    = np.asarray(time_arr, dtype=float)
        N    = data.size

        # Early check: if data is too small to create even one valid segment
        if N < min_segment_length:
            return [data], [t]

        # Calculate theoretical maximum segments based on minimum length constraint
        theoretical_max_segments = N // min_segment_length
        
        # If max_segments is specified, use the minimum of the two constraints
        if max_segments is not None:
            effective_max_segments = min(max_segments, theoretical_max_segments)
        else:
            effective_max_segments = theoretical_max_segments

        # --- Optional Savitzky–Golay smoothing
        if smooth_data:
            w = min(max(smooth_window_size, 5), N - 1)
            if w % 2 == 0:
                w -= 1
            w = max(w, 3)
            data_s = self.smooth_data_with_savgol(
                data, window_length=w, polyorder=min(smooth_polyorder, w - 2)
            )
        else:
            data_s = data

        # --- helper: wrapped/circular prominence for a given "target" signal
        def _wrapped_prom(target: np.ndarray, idx: int, win: int) -> float:
            L = target.size
            left  = np.arange(idx - win, idx) % L
            right = np.arange(idx + 1, idx + 1 + win) % L
            baseline = max(np.max(target[left]), np.max(target[right]))
            return target[idx] - baseline  # positive for peaks in `target`

        # --- find up to n wrapped peaks of chosen type
        def _find_wrapped_peaks(signal: np.ndarray, n: int, find_min: bool) -> Tuple[np.ndarray, np.ndarray]:
            target = -signal if find_min else signal
            distance = max(10, len(signal) // max(1, (n * 2)))
            cand_idx, _ = find_peaks(target, distance=distance, prominence=prominence_threshold)

            if cand_idx.size == 0:
                return np.array([], dtype=int), np.array([], dtype=float)

            win = min(max(1, prominence_distance), len(signal) - 1)
            prominences = np.array([_wrapped_prom(target, i, win) for i in cand_idx])

            keep = prominences > prominence_threshold
            idx_keep  = cand_idx[keep]
            prom_keep = prominences[keep]
            if idx_keep.size == 0:
                return np.array([], dtype=int), np.array([], dtype=float)

            take = min(n, idx_keep.size)
            order = np.argsort(prom_keep)  # ascending
            top_idx  = idx_keep[order[-take:]]
            top_prom = prom_keep[order[-take:]]
            sorter   = np.argsort(top_idx)
            return top_idx[sorter], top_prom[sorter]

        # --- maxima & minima
        max_idx, max_prom = _find_wrapped_peaks(data_s, number_of_max_and_min, find_min=False)
        min_idx, min_prom = _find_wrapped_peaks(data_s, number_of_max_and_min, find_min=True)

        # Combine boundaries and carry a "strength" (prominence) for each internal boundary
        internal_idx = np.concatenate([max_idx, min_idx]).astype(int)
        internal_prom = np.concatenate([max_prom, min_prom]).astype(float)

        # If duplicates (peak found as both due to noise), keep the larger strength
        if internal_idx.size:
            uniq, inv = np.unique(internal_idx, return_inverse=True)
            agg_prom = np.zeros_like(uniq, dtype=float)
            for k, u in enumerate(uniq):
                agg_prom[k] = np.max(internal_prom[inv == k])
            internal_idx = uniq
            internal_prom = agg_prom

        # Build boundary list [0, ...internal..., N]
        boundaries = np.unique(np.concatenate(([0], internal_idx, [N])))

        # --- Iteratively remove boundaries to satisfy both max_segments and min_segment_length constraints
        if internal_idx.size:
            # Map boundary index -> prominence (default small if missing)
            prom_map = {int(i): float(p) for i, p in zip(internal_idx, internal_prom)}
            
            # First check: do we have too many segments for max_segments constraint?
            while (len(boundaries) - 1) > effective_max_segments and len(boundaries) > 2:
                internals = boundaries[1:-1]
                if internals.size == 0:
                    break
                strengths = np.array([prom_map.get(int(b), 0.0) for b in internals])
                remove_at = int(internals[np.argmin(strengths)])
                boundaries = boundaries[boundaries != remove_at]
            
            # Second check: do any segments violate min_segment_length?
            # Keep removing weakest boundaries until all segments meet minimum length
            improved = True
            while improved and len(boundaries) > 2:
                improved = False
                segment_lengths = [boundaries[i+1] - boundaries[i] for i in range(len(boundaries)-1)]
                min_length = min(segment_lengths)
                
                if min_length >= min_segment_length:
                    break  # All segments are long enough
                
                # Find the shortest segment(s) and remove the boundary with weakest prominence
                # that's adjacent to a short segment
                candidates_to_remove = []
                for i, length in enumerate(segment_lengths):
                    if length < min_segment_length:
                        # Consider removing left boundary (if not the first) or right boundary (if not the last)
                        if i > 0:  # left boundary exists
                            candidates_to_remove.append(boundaries[i])
                        if i < len(segment_lengths) - 1:  # right boundary exists
                            candidates_to_remove.append(boundaries[i+1])
                
                if candidates_to_remove:
                    # Remove the candidate with the weakest prominence
                    candidate_strengths = [prom_map.get(int(b), 0.0) for b in candidates_to_remove]
                    weakest_idx = np.argmin(candidate_strengths)
                    remove_at = candidates_to_remove[weakest_idx]
                    
                    if remove_at in boundaries[1:-1]:  # Don't remove endpoints
                        boundaries = boundaries[boundaries != remove_at]
                        improved = True

        # --- Build contiguous, non-overlapping segments covering all samples
        decay_segments: List[np.ndarray] = []
        time_segments:  List[np.ndarray] = []
        for a, b in zip(boundaries[:-1], boundaries[1:]):
            a = int(a); b = int(b)
            if b <= a:
                continue
            segment_data = data[a:b]
            segment_time = t[a:b]
            
            # Final safety check - ensure no segment is too short
            if len(segment_data) >= min_segment_length:
                decay_segments.append(segment_data)
                time_segments.append(segment_time)

        # If we ended up with no valid segments (shouldn't happen with proper logic above),
        # return the entire data as one segment
        if not decay_segments:
            decay_segments = [data]
            time_segments = [t]

        # coverage sanity check
        total_len = sum(seg.size for seg in decay_segments)
        if total_len != N:
            # This might happen if we had to drop very short segments
            # In that case, we should merge the remaining data into the last segment
            if decay_segments:
                missing_points = N - total_len
                if missing_points > 0:
                    # Find which points are missing and add them to the last segment
                    covered_points = set()
                    current_pos = 0
                    for seg in decay_segments[:-1]:
                        covered_points.update(range(current_pos, current_pos + len(seg)))
                        current_pos += len(seg)
                    
                    # Add remaining points to last segment
                    last_seg_start = current_pos
                    decay_segments[-1] = data[last_seg_start:]
                    time_segments[-1] = t[last_seg_start:]

        return decay_segments, time_segments



if __name__ == '__main__':
    pass