#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
This file handles the chemometrics analysis of the data.
'''

# Written by
# Alfred Worrad <worrada@udel.edu>,

__author__ = "Afred Worrad"
__version__ = "1.2.1"
__maintainer__ = "Alfred Worrad"
__email__ = "worrada@udel.edu"
__status__ = "Development"
__project__ = "MES wavey"
__created__ = "December 04, 2023"
__updated__ = "June 26, 2025"

# built-in modules
from typing import List, Optional, Tuple, Dict, Any
from itertools import count
import sys
import os


# third-party modules
import numpy as np
from scipy.optimize import nnls
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd

# project modules
from src.core_functionality.spectrum import Spectrum, Spectra

# Dictionary keywords
FIT_PARAMETERS = "fit parameters"
CONFIDENCE_INTERVALS = "confidence intervals"


def ls_custom(A: np.ndarray, b: np.ndarray, alpha: float = 0.05) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
    """
    Non-negative least squares solver.
    Solves argmin_x ||b - Ax||_2 subject to x >= 0.
    Args:
        A: Coefficient matrix.
        b: Target vector.
    Returns:
        x: Solution vector.
        confidence_intervals: List of confidence interval tuples for each parameter.
    Raises:
        ValueError: If the problem is infeasible.
    """

    # Not using scipy because we want to compute the confidence intervals.

    A_transpose = np.transpose(A)
    Q = np.linalg.inv(np.matmul(A_transpose, A))
    theta_hat = np.matmul(np.matmul(Q, A_transpose), b)

    confidence_intervals = []
    residual = b - np.matmul(A, theta_hat)
    denominator = len(b) - len(theta_hat)
    for i in range(len(theta_hat)):
        inverse_matrix_diagonal_element = Q[i, i]
        mse = np.sum(residual ** 2) / denominator if denominator != 0 else 0.0
        standard_error = np.sqrt(mse * inverse_matrix_diagonal_element)

        if standard_error > 0:
            ci = (
                theta_hat[i] - norm.ppf(1 - alpha / 2) * standard_error,
                theta_hat[i] + norm.ppf(1 - alpha / 2) * standard_error,
            )
        else:
            ci = (theta_hat[i], theta_hat[i])  # zero-width interval

        confidence_intervals.append(ci)

    return theta_hat, confidence_intervals

def _fnnls(
    A: np.ndarray,
    b: np.ndarray,
    eps: float = 1e-12,
    max_iter: int = 50000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fast Non-Negativity-Constrained Least Squares (FNNLS)
    Bro & de Jong (1997): uses normal equations A^T A and A^T b.

    Returns:
        x: solution vector (n,)
        P_mask: boolean mask of passive set (active coefficients > 0 at optimum)
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    m, n = A.shape

    AtA = A.T @ A
    Atb = A.T @ b

    x = np.zeros(n, dtype=float)
    P = np.zeros(n, dtype=bool)  # passive set mask
    R = ~P

    w = Atb - AtA @ x
    iters = 0

    # Outer loop: add variables with positive Lagrange multipliers
    while True:
        R = ~P
        if not np.any(R):
            break
        # choose j with largest positive w in the active (R) set
        w_R = np.where(R, w, -np.inf)
        j = int(np.argmax(w_R))
        if w_R[j] <= eps:
            break  # KKT satisfied

        P[j] = True
        R[j] = False

        # Solve normal equations on passive set
        while True:
            P_idx = np.where(P)[0]
            AtA_PP = AtA[np.ix_(P_idx, P_idx)]
            Atb_P  = Atb[P_idx]
            # s_P = (AtA_PP)^{-1} Atb_P ; s_R = 0
            try:
                s_P = np.linalg.solve(AtA_PP, Atb_P)
            except np.linalg.LinAlgError:
                s_P = np.linalg.pinv(AtA_PP) @ Atb_P

            s = np.zeros(n, dtype=float)
            s[P_idx] = s_P

            # If all s_P > 0 we're good; else step toward s and drop those that hit 0
            if np.all(s_P > eps):
                x = s
                break

            # Compute step size alpha to keep nonnegativity
            neg_in_P = P_idx[s_P <= eps]
            if neg_in_P.size == 0:
                x = s
                break
            alphas = []
            for k in neg_in_P:
                denom = x[k] - s[k]
                if denom > 0:
                    alphas.append(x[k] / denom)
            alpha = min(alphas) if alphas else 1.0
            x = x + alpha * (s - x)
            # Drop any variables that hit (or fall below) 0
            drop = (x <= eps) & P
            if np.any(drop):
                P[drop] = False

            iters += 1
            if iters >= max_iter:
                raise ValueError("FNNLS: maximum iterations exceeded during inner loop")

        # Update gradient
        w = Atb - AtA @ x
        iters += 1
        if iters >= max_iter:
            raise ValueError("FNNLS: maximum iterations exceeded")

    # Clean tiny negatives
    x[x < 0] = 0.0
    return x, P

def nnls_custom(
    A: np.ndarray,
    b: np.ndarray,
    alpha: float = 0.05,
    epsilon=1e-4,
    max_iter=10000,
    *,
    ci_mode: str = "fixed",                 # "fixed" or "bootstrap" or "none"
    spectra_standard_deviations: Optional[np.ndarray] = None,   # one SD per reference column (len = n)
    A_noise: str = "additive",               # "multiplicative" or "additive"
    n_boot: int = 500,
    random_state: Optional[int] = None,
    equalize_ci: bool = True,               # NEW: make all CI half-widths identical
) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
    """
    NNLS via FNNLS with optional CIs.
    If equalize_ci=True, all coefficients get the SAME CI half-width
    (pooled from component-wise widths), then clipped to [0,1].
    """
    A = np.asarray(A, float)
    b = np.asarray(b, float).reshape(-1)
    m, n = A.shape
    assert b.shape[0] == m

    # ---- FNNLS core (Bro & de Jong). Requires _fnnls(A,b) -> (x_hat, P_mask) defined elsewhere.
    x_hat, P_mask = _fnnls(A, b)

    if ci_mode.lower() == "none":
        return x_hat, [(x_hat[i], x_hat[i]) for i in range(n)]

    # ---------- fixed CIs (A fixed, Gaussian noise in b) ----------
    if ci_mode.lower() == "fixed":
        r = b - A @ x_hat
        P_idx = np.where(P_mask)[0]
        dof = int(m - np.linalg.matrix_rank(A[:, P_idx])) if P_idx.size else int(m - np.linalg.matrix_rank(A))
        dof = max(1, dof)
        sigma2 = float(r @ r) / dof

        ATA = A.T @ A
        try:
            ATA_inv = np.linalg.inv(ATA)
        except np.linalg.LinAlgError:
            ATA_inv = np.linalg.pinv(ATA)

        se = np.sqrt(np.clip(np.diag(ATA_inv), 0.0, np.inf) * sigma2)

        # refine SEs on active set
        if P_idx.size:
            AtA_PP = ATA[np.ix_(P_idx, P_idx)]
            try:
                inv_PP = np.linalg.inv(AtA_PP)
            except np.linalg.LinAlgError:
                inv_PP = np.linalg.pinv(AtA_PP)
            se_refined = np.sqrt(np.clip(np.diag(inv_PP), 0.0, np.inf) * sigma2)
            se[P_idx] = se_refined

        z = norm.ppf(1 - alpha / 2.0)

        if equalize_ci:
            # pooled half-width from RMS SE across components
            delta = float(z * np.sqrt(np.mean(se**2)))
            ci = [(max(0.0, x_hat[i] - delta), min(1.0, x_hat[i] + delta)) for i in range(n)]
        else:
            ci = []
            for i in range(n):
                delta = z * se[i]
                lo = max(0.0, x_hat[i] - delta)
                hi = min(1.0, x_hat[i] + delta)
                ci.append((lo, hi))
        return x_hat, ci

    # ---------- bootstrap CIs (uncertainty in A columns only, b fixed) ----------
    if ci_mode.lower() == "bootstrap":
        if spectra_standard_deviations is None:
            raise ValueError("Provide spectra_standard_deviations (length n) for bootstrap CIs.")
        sd = np.asarray(spectra_standard_deviations, float).reshape(-1)
        assert sd.size == n, "spectra_standard_deviations must have length equal to number of columns in A."

        rng = np.random.default_rng(random_state)
        samples = np.empty((n_boot, n), float)

        for k in range(n_boot):
            if A_noise == "multiplicative":
                scale = 1.0 + rng.normal(0.0, sd, size=(1, n))
                A_k = A * np.repeat(scale, m, axis=0)
            elif A_noise == "additive":
                noise = rng.normal(0.0, sd, size=(1, n))
                A_k = A + np.repeat(noise, m, axis=0)
            else:
                raise ValueError("A_noise must be 'multiplicative' or 'additive'.")
            try:
                x_k, _ = _fnnls(A_k, b)
            except Exception:
                x_k = x_hat
            samples[k] = x_k

        lo_q, hi_q = 100 * (alpha / 2.0), 100 * (1.0 - alpha / 2.0)
        if equalize_ci:
            # compute per-component half-widths, then pool to one scalar (RMS) and apply to all
            hi = np.percentile(samples, hi_q, axis=0)
            lo = np.percentile(samples, lo_q, axis=0)
            half_widths = 0.5 * (hi - lo)
            delta = float(np.sqrt(np.mean(half_widths**2)))
            ci = [(max(0.0, x_hat[i] - delta), min(1.0, x_hat[i] + delta)) for i in range(n)]
        else:
            ci = []
            for i in range(n):
                lo = float(np.percentile(samples[:, i], lo_q))
                hi = float(np.percentile(samples[:, i], hi_q))
                ci.append((max(0.0, lo), min(1.0, hi)))
        return x_hat, ci

    raise ValueError("ci_mode must be 'fixed', 'bootstrap', or 'none'")



class Chemometrics():

    def __init__(self, reference_spectra: Spectra = Spectra([]), reference_spectra_names: Optional[List[str]] = None):
        """ Initialize the Chemometrics object with reference spectra.

        Args:
            reference_spectra (Spectra, optional): A collection of reference spectra to be used for deconvolution. Defaults to an empty Spectra object: Spectra([]).
            reference_spectra_names (Optional[List[str]], optional): Names for the reference spectra. If None, names are taken from the reference spectra themselves or assigned default names. Defaults to None.
        """

        self._reference_spectra = reference_spectra
        self._manipulated_reference_spectra = Spectra([])
        if reference_spectra_names is None:
            self._reference_spectra_names = [reference_spectrum.name for reference_spectrum in self._reference_spectra]

    def add_reference_sample(self, reference_spectrum: Spectrum, reference_spectrum_name: Optional[str] = None) -> None:
        """ Add a reference spectrum to the Chemometrics object.

        Args:
            reference_spectrum (Spectrum): The reference spectrum to be added.
            reference_spectrum_name (Optional[str], optional): Name for the reference spectrum. If None, a default name will be assigned based on the number of reference spectra already present. Defaults to None.

        Raises:
            TypeError: If reference_spectrum is not an instance of Spectrum.
        """
        if not isinstance(reference_spectrum, Spectrum):
            raise TypeError("reference_spectrum must be an instance of Spectrum")
        
        self._reference_spectra.append(reference_spectrum)
        self._reference_spectra_names.append(reference_spectrum_name or reference_spectrum.name)

    def _adjust_reference_spectra(self, spectrum_to_match_frequencies: Spectrum) -> None:
        """ Adjust the reference spectra to match the frequencies of the provided spectrum.

        Args:
            spectrum_to_match_frequencies (Spectrum): The spectrum whose frequencies will be used to adjust the reference spectra.  
        """

        desired_frequency = spectrum_to_match_frequencies.frequencies
        self._manipulated_reference_spectra = []
        for reference_spectrum in self._reference_spectra:
            if not np.array_equal(reference_spectrum.frequencies, desired_frequency):
                altered_ref_spectrum = reference_spectrum.interpolate(
                    desired_frequency)
                self._manipulated_reference_spectra.append(
                    altered_ref_spectrum)
            else:
                self._manipulated_reference_spectra.append(reference_spectrum)

    def _deconvolute(
        self,
        spectrum_to_deconvolute: Spectrum,
        alpha: float = 0.05,
        constrained_fit: bool = True,
        epsilon: float = 10e-5,
        max_iter: int = 10000,
        normalize_fit: bool = True,
        ci_mode: str = "fixed",
        spectra_standard_deviations: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """ Perform the deconvolution of a single spectrum using the reference spectra.

        Args:
            spectrum_to_deconvolute (Spectrum): The spectrum to be deconvoluted.
            alpha (float, optional): Confidence level for the confidence intervals. Defaults to 0.05.
            constrained_fit (bool, optional): Whether to use constrained fitting (non-negative least squares). Defaults to True.
            epsilon (float, optional): Tolerance for convergence in the constrained fit. Defaults to 10e-5.
            max_iter (int, optional): Maximum number of iterations for the constrained fit. Defaults to 10000.
            normalize_fit (bool, optional): Whether to normalize the fit parameters. Defaults to True.
            ci_mode (str, optional): Method for computing confidence intervals ("fixed" or "bootstrap") . Defaults to "fixed".
            spectra_standard_deviations (Optional[np.ndarray], optional): Standard deviations for each reference spectrum, required if ci_mode is "bootstrap". Defaults to None.

        Returns:
            Dict[str, Any]: A dictionary containing the fit parameters (fitted fractions if normalized) and confidence intervals.
                - "fit parameters": np.ndarray of fit parameters.
                - "confidence intervals": List of tuples representing confidence intervals for each parameter.

        Raises:
            ValueError: If no reference spectra have been provided.
        """

        if not self._manipulated_reference_spectra:
            raise ValueError("No reference spectra have been provided.")

        intensity_matrix = np.vstack([x.intensities for x in self._manipulated_reference_spectra]).T
        
        if not constrained_fit:
            fit_params, confidence_intervals = ls_custom(
                intensity_matrix, spectrum_to_deconvolute.intensities)
        else:
            fit_params, confidence_intervals = nnls_custom(
                intensity_matrix, spectrum_to_deconvolute.intensities, alpha, epsilon, max_iter, ci_mode=ci_mode, spectra_standard_deviations=spectra_standard_deviations)
            # I include this note so I don't need to verify again.
            # This matches the implementation from scipy
            # fit_params_original, residuals = nnls(intensity_matrix, spectrum_to_deconvolute.intensities)
            # print("scipy computed fit params: ", fit_params_original)
            # print("custom computed fit params: ", fit_params)
        
        if normalize_fit:
            renormalization_factor = np.sum(fit_params)
            fit_params = fit_params / renormalization_factor
            # Also normalize the confidence intervals
            for i in range(len(confidence_intervals)):
                confidence_intervals[i] = (confidence_intervals[i][0] / renormalization_factor, confidence_intervals[i][1] / renormalization_factor)
                # Bound to a max of 1 and a min of 0
                confidence_intervals[i] = (max(0, confidence_intervals[i][0]), min(1, confidence_intervals[i][1]))
                
        output_dict = {
            FIT_PARAMETERS: fit_params,
            CONFIDENCE_INTERVALS: confidence_intervals
        }
        return output_dict


    def _gaussian_deconvolution(
        self,
        spectrum_to_deconvolute: Spectrum
        ) -> Spectra:
        """ Perform Gaussian deconvolution of a spectrum."""

        # TODO: Later we can implement a gaussian devoncolution.
        intensities_to_fit = spectrum_to_deconvolute.intensities
        frequencies = spectrum_to_deconvolute.frequencies
        gaussian_spectra = Spectra([])

        return gaussian_spectra

    def compute_spectrum_fitting(
        self,
        spectrum_to_deconvolute: Spectrum,
        alpha: float = 0.05,
        constrained_fit: bool = True,
        epsilon: float = 10e-5,
        max_iter: int = 10000,
        normalize_fit: bool = True,
    ) -> Dict[str, Any]:
        """ Compute the deconvolution of a single spectrum using the reference spectra.

        Args:
            spectrum_to_deconvolute (Spectrum): The spectrum to be deconvoluted.
            alpha (float, optional): Confidence level for the confidence intervals. Defaults to 0.05.
            constrained_fit (bool, optional): Whether to use constrained fitting (non-negative least squares). Defaults to True.
            epsilon (float, optional): Tolerance for convergence in the constrained fit. Defaults to 10e-5.
            max_iter (int, optional): Maximum number of iterations for the constrained fit. Defaults to 10000.
            normalize_fit (bool, optional): Whether to normalize the fit parameters. Defaults to True.

        Raises:
            ValueError: If no reference spectra have been provided.
            ValueError: If the reference spectra and the spectrum to deconvolute do not have the same frequency shape.

        Returns:
            Dict[str, Any]: A dictionary containing the fit parameters (fitted fractions if normalized) and confidence intervals.
                - "fit parameters": np.ndarray of fit parameters.
                - "confidence intervals": List of tuples representing confidence intervals for each parameter.
        """

        if not self._reference_spectra:
            raise ValueError("No reference spectra have been provided.")
        
        if self._reference_spectra[0].compare_spectra_shape(spectrum_to_deconvolute) is False:
            raise ValueError(
                "The reference spectra and the spectrum to deconvolute do not have the same frequency shape. "
                "Please ensure that the reference spectra and the spectrum to deconvolute have the same frequency shape."
            )

        self._adjust_reference_spectra(spectrum_to_deconvolute)
        return self._deconvolute(
            spectrum_to_deconvolute,
            alpha,
            constrained_fit,
            epsilon,
            max_iter,
            normalize_fit,
            ci_mode="fixed"  # Use fixed as default for backward compatibility
        )

    def compute_spectra_fitting(
        self,
        spectra_to_deconvolute: Spectra,
        alpha: float = 0.05,
        constrained_fit: bool = True,
        epsilon: float = 10e-5,
        max_iter: int = 10000,
        normalize_fit: bool = True,
        save_csv_file: Optional[str] = None,
        save_file_include_confidence_intervals: bool = True,
        ci_mode: str = "fixed",
        spectra_standard_deviations: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        """ Compute the deconvolution of multiple spectra using the reference spectra.

        Args:
            spectra_to_deconvolute (Spectra): A collection of spectra to be deconvoluted.
            alpha (float, optional): Confidence level for the confidence intervals. Defaults to 0.05.
            constrained_fit (bool, optional): Whether to use constrained fitting (non-negative least squares). Defaults to True.
            epsilon (float, optional): Tolerance for convergence in the constrained fit. Defaults to 10e-5.
            max_iter (int, optional): Maximum number of iterations for the constrained fit. Defaults to 10000.
            normalize_fit (bool, optional): Whether to normalize the fit parameters. Defaults to True.
            save_csv_file (str, optional): File path to save the deconvolution results as a CSV file. If None, no file will be saved. Defaults to None.
            save_file_include_confidence_intervals (bool, optional): Whether to include confidence intervals in the saved CSV file. Defaults to True.
            ci_mode (str, optional): Method for computing confidence intervals ("fixed" or "bootstrap"). Defaults to "fixed".
            spectra_standard_deviations (Optional[np.ndarray], optional): Standard deviations for each reference spectrum, required if ci_mode is "bootstrap". Defaults to None.

        Raises:
            ValueError: If no reference spectra have been provided.
            ValueError: If the reference spectra and the spectrum to deconvolute do not have the same frequency shape.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each containing the fit parameters (fitted fractions if normalized) and confidence intervals for each spectrum.
                - "fit parameters": np.ndarray of fit parameters.
                - "confidence intervals": List of tuples representing confidence intervals for each parameter.
        """

        if not self._reference_spectra:
            raise ValueError("No reference spectra have been provided.")
        
        if self._reference_spectra[0].compare_spectra_shape(spectra_to_deconvolute[0]) is False:
            raise ValueError(
                "The reference spectra and the spectrum to deconvolute do not have the same frequency shape. "
                "Please ensure that the reference spectra and the spectrum to deconvolute have the same frequency shape."
            )

        # Process the data if the frequency data is not consistent
        self._adjust_reference_spectra(spectra_to_deconvolute[0])
        fit_params_list = []
        for spectrum_to_deconvolute in spectra_to_deconvolute:
            fit_params_list.append(self._deconvolute(
                spectrum_to_deconvolute, alpha, constrained_fit, epsilon, max_iter, normalize_fit, ci_mode, spectra_standard_deviations))

        # Save the data to a csv file if desired
        if save_csv_file is not None:
            
            row_labels = self._reference_spectra_names
            if not row_labels or len(row_labels) != len(fit_params_list[0][FIT_PARAMETERS]):
                row_labels = [f"ref_{i}" for i in range(len(fit_params_list[0][FIT_PARAMETERS]))]

            column_labels = [s.name for s in spectra_to_deconvolute]

            if not save_file_include_confidence_intervals:
                data = np.array([fit[fit_params_list] for fit in fit_params_list]).T
                df = pd.DataFrame(data, index=row_labels, columns=column_labels)
                df.to_csv(save_csv_file)
            else:
                # Extract values, lower bounds, and upper bounds
                values = np.array([fit[FIT_PARAMETERS] for fit in fit_params_list]).T
                lowers = np.array([[fit[CONFIDENCE_INTERVALS][j][0] for fit in fit_params_list] for j in range(len(row_labels))])
                uppers = np.array([[fit[CONFIDENCE_INTERVALS][j][1] for fit in fit_params_list] for j in range(len(row_labels))])

                df = pd.DataFrame(values, index=row_labels, columns=column_labels)
                df_lower = pd.DataFrame(lowers, index=row_labels, columns=column_labels)
                df_upper = pd.DataFrame(uppers, index=row_labels, columns=column_labels)

                if save_csv_file.lower().endswith('.csv'):
                    base = os.path.splitext(save_csv_file)[0]
                else:
                    base = save_csv_file

                df.to_csv(f"{base}.csv")
                df_lower.to_csv(f"{base}_lower_bound.csv")
                df_upper.to_csv(f"{base}_upper_bound.csv")


        return fit_params_list
    

    def compute_spectra_fitting_as_spectra(
        self,
        spectra_to_deconvolute: Spectra,
        alpha: float = 0.05,
        constrained_fit: bool = True,
        epsilon: float = 10e-5,
        max_iter: int = 10000,
        normalize_fit: bool = True,
        save_csv_file: Optional[str] = None,
        save_file_include_confidence_intervals: bool = True,
        ci_mode: str = "fixed",
        spectra_standard_deviations: Optional[np.ndarray] = None,
    ) -> Tuple[Spectra, List[Dict[str, Any]]]:
        """ Compute the deconvolution of multiple spectra using the reference spectra and return as Spectra object,
            where each reference spectrum has a single frequency and its intensity is the fitted value.

        Args:
            spectra_to_deconvolute (Spectra): A collection of spectra to be deconvoluted.
            alpha (float, optional): Confidence level for the confidence intervals. Defaults to 0.05.
            constrained_fit (bool, optional): Whether to use constrained fitting (non-negative least squares). Defaults to True.
            epsilon (float, optional): Tolerance for convergence in the constrained fit. Defaults to 10e-5.
            max_iter (int, optional): Maximum number of iterations for the constrained fit. Defaults to 10000.
            normalize_fit (bool, optional): Whether to normalize the fit parameters. Defaults to True.
            save_csv_file (str, optional): File path to save the deconvolution results as a CSV file. If None, no file will be saved. Defaults to None.
            save_file_include_confidence_intervals (bool, optional): Whether to include confidence intervals in the saved CSV file. Defaults to True.
            ci_mode (str, optional): Method for computing confidence intervals ("fixed" or "bootstrap"). Defaults to "fixed".
            spectra_standard_deviations (Optional[np.ndarray], optional): Standard deviations for each reference spectrum, required if ci_mode is "bootstrap". Defaults to None.

        Raises:
            ValueError: If no reference spectra have been provided.
            ValueError: If the reference spectra and the spectrum to deconvolute do not have the same frequency shape.

        Returns:
            Spectra: A Spectra object with each reference spectrum having a single frequency and its intensity being the fitted value.
        """

        fit_params_list = self.compute_spectra_fitting(
            spectra_to_deconvolute,
            alpha,
            constrained_fit,
            epsilon,
            max_iter,
            normalize_fit,
            save_csv_file,
            save_file_include_confidence_intervals,
            ci_mode,
            spectra_standard_deviations 
        )
        deconvoluted_spectra = Spectra([])
        number_of_references = len(self._reference_spectra)
        frequencies_if_each_spectrum_had_1_frequency = np.arange(number_of_references)

        for i, fit_params in enumerate(fit_params_list):
            deconvoluted_spectrum = Spectrum(
                frequencies_if_each_spectrum_had_1_frequency,
                fit_params[FIT_PARAMETERS],
                name=spectra_to_deconvolute[i].name + "_deconvoluted"
            )
            deconvoluted_spectra.append(deconvoluted_spectrum)

        return deconvoluted_spectra, fit_params_list
    
    def compute_reference_spectra(
            self,
            composite_spectra: Spectra,
            matrix_of_pure_component_fractions: np.ndarray
    ):
        
        # check to ensure the matrix_of_pure_component_fractions is valid
        num_spectra = len(composite_spectra)
        mat_shape = matrix_of_pure_component_fractions.shape
        if mat_shape[0] != num_spectra:
            raise ValueError("The number of rows in matrix_of_pure_component_fractions must match the number of composite spectra.")
        if mat_shape[1] == 0:
            raise ValueError("The matrix_of_pure_component_fractions must have at least one column.")
        
        # solve the NNLS problem to
        # A = fractions (m×l), C = spectra (m×n)
        A = matrix_of_pure_component_fractions
        C = composite_spectra.data  # assuming Spectra.data is an (m×n) NumPy array

        # Solve for B (l×n) using least squares: minimize ||A B - C||
        B, residuals, rank, s = np.linalg.lstsq(A, C, rcond=None)

        # Wrap in Spectra object (same x-axis, columns correspond to pure components)
        pure_component_spectra = Spectra(
            [Spectrum(
                composite_spectra.frequencies,
                B[i, :],
                name=f"Pure_Component_{i+1}"
            ) for i in range(B.shape[0])]
        )


        return pure_component_spectra


if __name__ == '__main__':
    pass