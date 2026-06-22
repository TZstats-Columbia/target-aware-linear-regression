from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from estimators import hybrid_optimal_omega_star


METHODS = ("OLS", "MM", "Hybrid", "Calibration")


@dataclass(frozen=True)
class TheoryValues:
    scaled_beta_error: float
    normalized_excess_mse: float
    omega: float | None = None


def target_marginal_y(beta0: float, beta: np.ndarray, sigma_eps_sq: float) -> tuple[float, float]:
    return float(beta0), math.sqrt(float(beta @ beta) + float(sigma_eps_sq))


def source_second_moment(mu_sx: np.ndarray, sigma_sx: np.ndarray) -> np.ndarray:
    return sigma_sx + np.outer(mu_sx, mu_sx)


def augmented_source_second_moment(mu_sx: np.ndarray, sigma_sx: np.ndarray) -> np.ndarray:
    d = mu_sx.shape[0]
    q_s = np.empty((d + 1, d + 1), dtype=float)
    q_s[0, 0] = 1.0
    q_s[0, 1:] = mu_sx
    q_s[1:, 0] = mu_sx
    q_s[1:, 1:] = sigma_sx + np.outer(mu_sx, mu_sx)
    return q_s


def target_augmented_second_moment(dimension: int) -> np.ndarray:
    return np.eye(dimension + 1, dtype=float)


def ols_theory(mu_sx: np.ndarray, sigma_sx: np.ndarray) -> TheoryValues:
    sigma_inv = np.linalg.inv(sigma_sx)
    scaled_beta_error = float(np.trace(sigma_inv))
    normalized_excess_mse = float(1.0 + np.trace(sigma_inv) + mu_sx @ sigma_inv @ mu_sx)
    return TheoryValues(scaled_beta_error=scaled_beta_error, normalized_excess_mse=normalized_excess_mse)


def moment_matching_theory(
    beta: np.ndarray,
    sigma_eps_sq: float,
    mu_sx: np.ndarray,
    sigma_sx: np.ndarray,
) -> TheoryValues:
    q_s_given_k = source_second_moment(mu_sx, sigma_sx)
    q_inv = np.linalg.inv(q_s_given_k)
    v = beta
    kappa = float(v @ q_inv @ v)
    if kappa <= 0.0:
        return TheoryValues(scaled_beta_error=math.nan, normalized_excess_mse=math.nan)
    omega_matrix = (
        q_s_given_k
        - np.outer(v, v) / kappa
        + (sigma_eps_sq / 2.0) * np.outer(v, v) / (kappa**2)
    )
    middle = q_inv @ omega_matrix @ q_inv
    scaled_beta_error = float(np.trace(middle))
    return TheoryValues(
        scaled_beta_error=scaled_beta_error,
        normalized_excess_mse=scaled_beta_error,
    )


def calibration_theory(
    beta: np.ndarray,
    sigma_eps_sq: float,
    sigma_sx: np.ndarray,
) -> TheoryValues:
    beta_norm_sq = float(beta @ beta)
    sigma_inv = np.linalg.inv(sigma_sx)
    scaled_beta_error = float(
        np.trace(sigma_inv)
        - (beta @ sigma_inv @ beta) / beta_norm_sq
        + sigma_eps_sq / (2.0 * beta_norm_sq)
    )
    return TheoryValues(
        scaled_beta_error=scaled_beta_error,
        normalized_excess_mse=scaled_beta_error,
    )


def hybrid_middle_matrix(
    beta: np.ndarray,
    sigma_eps_sq: float,
    mu_sx: np.ndarray,
    sigma_sx: np.ndarray,
    sigma_ky: float,
    omega: float,
) -> np.ndarray:
    d = beta.shape[0]
    q_s = augmented_source_second_moment(mu_sx, sigma_sx)
    tilde_mu_k = np.zeros(d + 1, dtype=float)
    tilde_mu_k[0] = 1.0
    v_sigma_theta = np.zeros(d + 1, dtype=float)
    v_sigma_theta[1:] = beta

    q_matrix = q_s + omega * (
        np.outer(tilde_mu_k, tilde_mu_k)
        + np.outer(v_sigma_theta, v_sigma_theta) / (sigma_ky**2)
    )
    omega_matrix = q_s + (
        (omega**2) * sigma_eps_sq / (2.0 * sigma_ky**4)
    ) * np.outer(v_sigma_theta, v_sigma_theta)

    q_inv = np.linalg.inv(q_matrix)
    return q_inv @ omega_matrix @ q_inv


def hybrid_theory(
    beta: np.ndarray,
    sigma_eps_sq: float,
    mu_sx: np.ndarray,
    sigma_sx: np.ndarray,
    sigma_ky: float,
) -> TheoryValues:
    sigma_eps = math.sqrt(float(sigma_eps_sq))
    omega = hybrid_optimal_omega_star(
        beta=beta,
        sigma_eps=sigma_eps,
        mu_sx=mu_sx,
        sigma_sx=sigma_sx,
        sigma_ky=sigma_ky,
        mu_zero_tol=0.0,
    )
    middle = hybrid_middle_matrix(
        beta=beta,
        sigma_eps_sq=float(sigma_eps_sq),
        mu_sx=mu_sx,
        sigma_sx=sigma_sx,
        sigma_ky=float(sigma_ky),
        omega=omega,
    )
    p_beta = np.concatenate((np.zeros((beta.shape[0], 1), dtype=float), np.eye(beta.shape[0], dtype=float)), axis=1)
    scaled_beta_error = float(np.trace(p_beta @ middle @ p_beta.T))
    normalized_excess_mse = float(np.trace(target_augmented_second_moment(beta.shape[0]) @ middle))
    return TheoryValues(
        scaled_beta_error=scaled_beta_error,
        normalized_excess_mse=normalized_excess_mse,
        omega=omega,
    )


def all_theory_values(
    beta: np.ndarray,
    sigma_eps_sq: float,
    mu_sx: np.ndarray,
    sigma_sx: np.ndarray,
    sigma_ky: float,
) -> dict[str, TheoryValues]:
    return {
        "OLS": ols_theory(mu_sx=mu_sx, sigma_sx=sigma_sx),
        "MM": moment_matching_theory(
            beta=beta,
            sigma_eps_sq=sigma_eps_sq,
            mu_sx=mu_sx,
            sigma_sx=sigma_sx,
        ),
        "Hybrid": hybrid_theory(
            beta=beta,
            sigma_eps_sq=sigma_eps_sq,
            mu_sx=mu_sx,
            sigma_sx=sigma_sx,
            sigma_ky=sigma_ky,
        ),
        "Calibration": calibration_theory(
            beta=beta,
            sigma_eps_sq=sigma_eps_sq,
            sigma_sx=sigma_sx,
        ),
    }
