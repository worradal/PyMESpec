#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Handles all of the baseline correction schemes.
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
__updated__ = "June 24, 2025"

# built-in modules
from abc import ABC, abstractmethod

# third-party modules
import numpy as np
from numpy.linalg import norm
from scipy import sparse
from scipy.sparse import linalg

# project modules
from src.core_functionality.spectrum import Spectrum, Spectra

class BaselineCorrection(ABC):
    """Abstract base class for baseline correction methods"""

    def __init__(self) -> None:
        pass

    @abstractmethod
    def get_baseline(self, Spectrum: Spectrum) -> np.ndarray:
        """Returns the baseline of the spectrum"""
        pass

    @abstractmethod
    def baseline_corrected_spectrum(self, Spectrum: Spectrum) -> Spectrum:
        """Returns the baseline corrected spectrum"""
        pass


class ARPLS(BaselineCorrection):
    """Implements the Asymmetrically reweighted penalized least square method from [1]

    References
    [1]: Baek, S.-J., Park, A., Ahn, Y.-J. & Choo, J.
        Baseline correction using asymmetrically reweighted penalized least squares smoothing. 
        Analyst 140, 250–257 (2014).

    """

    def __init__(self):
        pass

    def get_baseline(self, spectrum: Spectrum, lambda_parameter: float = 10e6, stop_ratio: float = 1e-6, max_iters: int = 10, full_output=False, verbose=False) -> np.ndarray:
        """ Calculates the baseline of a spectrum using the ARPLS method.

        Args:
            spectrum (Spectrum): The spectrum to baseline correct.
            lambda_parameter (float, optional): Smoothness parameter for the ARPLS method. Defaults to 10e6.
            stop_ratio (float, optional): The ratio of change in weights to stop the iteration. Defaults to 1e-6.
            max_iters (int, optional): Maximum number of iterations to perform. Defaults to 10.
            full_output (bool, optional): If True, returns additional information about the correction process. Defaults to False.

        Returns:
            np.ndarray: The baseline of the spectrum.
        """
        intensities: np.ndarray = spectrum.intensities
        L = len(intensities)
        diag = np.ones(L - 2)
        D = sparse.spdiags([diag, -2*diag, diag], [0, -1, -2], L, L - 2)
        # The transposes are flipped w.r.t the Algorithm on pg. 252
        H = lambda_parameter * D.dot(D.T)
        w = np.ones(L)
        W = sparse.spdiags(w, 0, L, L)

        current_ratio = 1
        num_iters = 0
        while current_ratio > stop_ratio:
            z = linalg.spsolve(W + H, W * intensities)
            d = intensities - z
            dn = d[d < 0]
            if len(dn) == 0 or np.allclose(d, 0):
                break 
            m = np.mean(dn)
            s = np.std(dn)
            w_new = 1 / (1 + np.exp(2 * (d - (2*s - m))/s))
            current_ratio = norm(w_new - w) / norm(w)
            w = w_new
            W.setdiag(w)

            num_iters += 1
            if num_iters > max_iters:
                if verbose:
                    print('Maximum number of iterations exceeded')
                break

        if full_output:
            info = {'num_iters': num_iters, 'final_ratio': current_ratio}
            return z, d, info
        else:
            return z

    def baseline_corrected_spectrum(
        self,
        spectrum: Spectrum,
        lambda_parameter: float = 10e6,
        stop_ratio: float = 1e-6,
        max_iters: int = 10,
        full_output=False,
        # default_to_min_zero: bool = True
    ) -> Spectrum:
        """ Returns a baseline corrected spectrum using the ARPLS method.

        Args:
            spectrum (Spectrum): The spectrum to baseline correct.
            lambda_parameter (float, optional): Smoothness parameter for the ARPLS method. Defaults to 10e6.
            stop_ratio (float, optional): The ratio of change in weights to stop the iteration. Defaults to 1e-6.
            max_iters (int, optional): Maximum number of iterations to perform. Defaults to 10.
            full_output (bool, optional): If True, returns additional information about the correction process. Defaults to False.

        Returns:
            Spectrum: A new Spectrum object with the baseline corrected intensities.
        """
        if full_output:
            baseline, d, info = self.get_baseline(
                spectrum, lambda_parameter, stop_ratio, max_iters, full_output)
            return Spectrum(spectrum.frequencies, spectrum.intensities - baseline), d, info
        else:
            baseline = self.get_baseline(
                spectrum, lambda_parameter, stop_ratio, max_iters, full_output)
        new_intensities = spectrum.intensities - baseline
        new_intensities = new_intensities - np.min(new_intensities)
        return Spectrum(spectrum.frequencies, new_intensities)

    def baseline_corrected_spectra(
        self,
        spectra: Spectra,
        lambda_parameter: float = 10e6,
        stop_ratio: float = 1e-6,
        max_iters: int = 10,
        full_output: bool = False
    ) -> Spectra:
        """ Returns a collection of baseline corrected spectra using the ARPLS method.

        Args:
            spectra (Spectra): The spectra to baseline correct.
            lambda_parameter (float, optional): Smoothness parameter for the ARPLS method. Defaults to 10e6.
            stop_ratio (float, optional): The ratio of change in weights to stop the iteration. Defaults to 1e-6.
            max_iters (int, optional): Maximum number of iterations to perform. Defaults to 10.
            full_output (bool, optional): If True, returns additional information about the correction process. Defaults to False.

        Returns:
            Spectra: A new Spectra object containing baseline corrected spectra.
        """
        baseline_corrected_spectra = Spectra([])
        for spectrum in spectra:
            baseline_corrected_spectra.append(self.baseline_corrected_spectrum(
                spectrum, lambda_parameter, stop_ratio, max_iters, full_output))
        return baseline_corrected_spectra


class Quadratic(BaselineCorrection):
    """Fits a quadratic baseline and shifts result to start at zero."""

    def __init__(self):
        pass

    def get_baseline(self, spectrum: Spectrum) -> np.ndarray:
        """ Returns a quadratic baseline for the given spectrum.

        Args:
            spectrum (Spectrum): The spectrum to baseline correct.

        Returns:
            np.ndarray: The baseline of the spectrum, fitted with a quadratic polynomial.
        """
        x = spectrum.frequencies
        y = spectrum.intensities
        coeffs = np.polyfit(x, y, deg=2)
        return np.polyval(coeffs, x)

    def baseline_corrected_spectrum(self, spectrum: Spectrum) -> Spectrum:
        """ Returns a baseline corrected spectrum by subtracting the fitted quadratic baseline.

        Args:
            spectrum (Spectrum): The spectrum to baseline correct.

        Returns:
            Spectrum: A new Spectrum object with the baseline corrected intensities.
        """
        baseline = self.get_baseline(spectrum)
        corrected = spectrum.intensities - baseline

        # Shift the minimum to zero (preserve relative shape)
        corrected -= corrected.min()

        return Spectrum(spectrum.frequencies, corrected)

class Linear(BaselineCorrection):

    def __init__(self):
        pass

    def get_baseline(self, spectrum: Spectrum) -> np.ndarray:
        """ Returns a linear baseline for the given spectrum.

        Args:
            spectrum (Spectrum): The spectrum to baseline correct.

        Returns:
            np.ndarray: The baseline of the spectrum, fitted with a linear polynomial.
        """
        x = spectrum.frequencies
        y = spectrum.intensities

        coeffs = np.polyfit(x, y, deg=1)
        baseline = np.polyval(coeffs, x)
        return baseline

    def baseline_corrected_spectrum(self, spectrum: Spectrum) -> Spectrum:
        """ Returns a baseline corrected spectrum by subtracting the fitted linear baseline.

        Args:
            spectrum (Spectrum): The spectrum to baseline correct.

        Returns:
            Spectrum: A new Spectrum object with the baseline corrected intensities.
        """
        baseline = self.get_baseline(spectrum)
        corrected = spectrum.intensities - baseline

        # Shift so the lowest point is 0
        corrected -= corrected.min()

        return Spectrum(spectrum.frequencies, corrected)


if __name__ == '__main__':
    pass