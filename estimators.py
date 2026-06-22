from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.polynomial import Polynomial
from scipy.optimize import brentq, minimize, minimize_scalar


_EPS = 1e-12


@dataclass
class EstimatorResult:
    name: str
    beta0: float
    beta: np.ndarray
    theta: np.ndarray
    rss: float
    objective: float
    sigma_eps_sq_hat: float | None = None
    omega: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def _as_1d_float_array(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    return array


def _as_2d_float_array(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional.")
    return array


def _validate_training_sample(x: Any, y: Any) -> tuple[np.ndarray, np.ndarray]:
    x_array = _as_2d_float_array(x, "x")
    y_array = _as_1d_float_array(y, "y")
    if x_array.shape[0] != y_array.shape[0]:
        raise ValueError("x and y must have the same number of observations.")
    if x_array.shape[0] == 0:
        raise ValueError("The training sample must be non-empty.")
    return x_array, y_array


def _augment_design(x: np.ndarray) -> np.ndarray:
    return np.concatenate((np.ones((x.shape[0], 1), dtype=float), x), axis=1)


def _q_s_matrix(mu_sx: np.ndarray, sigma_sx: np.ndarray) -> np.ndarray:
    d = mu_sx.shape[0]
    q_s = np.empty((d + 1, d + 1), dtype=float)
    q_s[0, 0] = 1.0
    q_s[0, 1:] = mu_sx
    q_s[1:, 0] = mu_sx
    q_s[1:, 1:] = sigma_sx + np.outer(mu_sx, mu_sx)
    return q_s


def _moment_matching_loss_normalized(
    x: np.ndarray,
    y: np.ndarray,
    beta0: float,
    beta: np.ndarray,
    mu_ky: float,
    sigma_ky: float,
) -> float:
    residual = y - beta0 - x @ beta
    sample_mse = float(np.mean(residual**2))
    total_scale = math.sqrt(max(float(beta @ beta) + sample_mse, _EPS))
    return (beta0 - mu_ky) ** 2 + (total_scale - sigma_ky) ** 2


def _selection_rss_normalized(x: np.ndarray, y: np.ndarray, beta: np.ndarray, mu_ky: float) -> float:
    residual = (y - mu_ky) - x @ beta
    return float(residual @ residual)


def _hybrid_loss_and_grad_normalized(
    theta: np.ndarray,
    x_aug: np.ndarray,
    y: np.ndarray,
    mu_ky: float,
    sigma_ky: float,
    omega: float,
) -> tuple[float, np.ndarray]:
    residual = y - x_aug @ theta
    sample_mse = float(np.mean(residual**2))
    beta = theta[1:]
    scale = math.sqrt(max(float(beta @ beta) + sample_mse, _EPS))

    loss = sample_mse
    loss += omega * (theta[0] - mu_ky) ** 2
    loss += omega * (scale - sigma_ky) ** 2

    grad_sample = -2.0 * (x_aug.T @ residual) / x_aug.shape[0]

    grad_mean = np.zeros_like(theta)
    grad_mean[0] = 2.0 * omega * (theta[0] - mu_ky)

    sigma_theta = np.zeros_like(theta)
    sigma_theta[1:] = beta
    variance_core = sigma_theta - (x_aug.T @ residual) / x_aug.shape[0]
    grad_variance = 2.0 * omega * ((scale - sigma_ky) / scale) * variance_core

    return float(loss), grad_sample + grad_mean + grad_variance


def hybrid_v_h(
    omega: float,
    beta: Any,
    sigma_eps: float,
    mu_sx: Any,
    sigma_sx: Any,
    sigma_ky: float,
) -> float:
    beta_array = _as_1d_float_array(beta, "beta")
    mu_sx_array = _as_1d_float_array(mu_sx, "mu_sx")
    sigma_sx_array = _as_2d_float_array(sigma_sx, "sigma_sx")
    d = beta_array.shape[0]

    q_s = _q_s_matrix(mu_sx_array, sigma_sx_array)
    tilde_mu_k = np.zeros(d + 1, dtype=float)
    tilde_mu_k[0] = 1.0
    v_sigma_theta = np.zeros(d + 1, dtype=float)
    v_sigma_theta[1:] = beta_array
    p_beta = np.concatenate((np.zeros((d, 1), dtype=float), np.eye(d, dtype=float)), axis=1)

    q_matrix = q_s + omega * (
        np.outer(tilde_mu_k, tilde_mu_k)
        + np.outer(v_sigma_theta, v_sigma_theta) / (sigma_ky**2)
    )
    omega_matrix = q_s + (
        (omega**2) * (sigma_eps**2) / (2.0 * sigma_ky**4)
    ) * np.outer(v_sigma_theta, v_sigma_theta)

    q_inv = np.linalg.inv(q_matrix)
    middle = q_inv @ omega_matrix @ q_inv
    return float(np.trace(p_beta @ middle @ p_beta.T))


def _candidate_omegas_from_quartic(
    beta: np.ndarray,
    sigma_eps: float,
    mu_sx: np.ndarray,
    sigma_sx: np.ndarray,
    sigma_ky: float,
) -> list[float]:
    d = beta.shape[0]
    q_s = _q_s_matrix(mu_sx, sigma_sx)
    q_s_inv = np.linalg.inv(q_s)

    tilde_mu_k = np.zeros(d + 1, dtype=float)
    tilde_mu_k[0] = 1.0
    v_sigma_theta = np.zeros(d + 1, dtype=float)
    v_sigma_theta[1:] = beta
    p_beta = np.concatenate((np.zeros((d, 1), dtype=float), np.eye(d, dtype=float)), axis=1)
    beta_projection = p_beta.T @ p_beta
    shared = q_s_inv @ beta_projection @ q_s_inv

    m_1 = float(tilde_mu_k @ q_s_inv @ tilde_mu_k)
    m_2 = float((tilde_mu_k @ q_s_inv @ v_sigma_theta) / sigma_ky)
    m_3 = float((v_sigma_theta @ q_s_inv @ v_sigma_theta) / (sigma_ky**2))

    n_1 = float(tilde_mu_k @ shared @ tilde_mu_k)
    n_2 = float((tilde_mu_k @ shared @ v_sigma_theta) / sigma_ky)
    n_3 = float((v_sigma_theta @ shared @ v_sigma_theta) / (sigma_ky**2))

    a_1 = -2.0 * (n_1 + n_3)
    a_2 = (
        -(n_1 + n_3) * (m_1 + m_3)
        - 3.0 * (m_1 * n_3 + m_3 * n_1 - 2.0 * m_2 * n_2)
        + (sigma_eps**2 / (2.0 * sigma_ky**2)) * n_3
    )
    a_3 = (
        -2.0 * (m_1 + m_3) * (m_1 * n_3 + m_3 * n_1 - 2.0 * m_2 * n_2)
        + (sigma_eps**2 / (sigma_ky**2)) * (m_1 * n_3 - m_2 * n_2)
    )
    a_4 = (
        -(m_1 * m_3 - m_2**2) * (m_1 * n_3 + m_3 * n_1 - 2.0 * m_2 * n_2)
        + (sigma_eps**2 / (2.0 * sigma_ky**2))
        * (m_1**2 * n_3 - 2.0 * m_1 * m_2 * n_2 + m_2**2 * n_1)
    )

    b_0 = a_1
    b_1 = 2.0 * a_2 - (m_1 + m_3) * a_1
    b_2 = 3.0 * a_3 - 3.0 * (m_1 * m_3 - m_2**2) * a_1
    b_3 = 4.0 * a_4 + (m_1 + m_3) * a_3 - 2.0 * (m_1 * m_3 - m_2**2) * a_2
    b_4 = 2.0 * (m_1 + m_3) * a_4 - (m_1 * m_3 - m_2**2) * a_3

    coefficients = np.array([b_0, b_1, b_2, b_3, b_4], dtype=float)
    while coefficients.size > 1 and abs(coefficients[-1]) <= 1e-14:
        coefficients = coefficients[:-1]
    if coefficients.size <= 1:
        return [0.0]

    raw_roots = np.roots(coefficients[::-1])
    candidates = [0.0]
    for root in raw_roots:
        if abs(root.imag) <= 1e-8 * max(1.0, abs(root.real)) and root.real >= -1e-10:
            candidates.append(float(max(root.real, 0.0)))

    unique_candidates: list[float] = []
    for candidate in sorted(candidates):
        if not unique_candidates or abs(candidate - unique_candidates[-1]) > 1e-7 * max(1.0, candidate):
            unique_candidates.append(candidate)
    return unique_candidates


def _omega_star_fallback(
    beta: np.ndarray,
    sigma_eps: float,
    mu_sx: np.ndarray,
    sigma_sx: np.ndarray,
    sigma_ky: float,
) -> float:
    def objective(omega: float) -> float:
        return hybrid_v_h(omega, beta, sigma_eps, mu_sx, sigma_sx, sigma_ky)

    upper = 1.0
    value_left = objective(0.0)
    value_right = objective(upper)
    while value_right < value_left and upper < 1e6:
        upper *= 2.0
        value_left = value_right
        value_right = objective(upper)

    result = minimize_scalar(objective, bounds=(0.0, upper), method="bounded")
    if not result.success:
        return 0.0
    return float(max(result.x, 0.0))


def hybrid_optimal_omega_star(
    beta: Any,
    sigma_eps: float,
    mu_sx: Any,
    sigma_sx: Any,
    sigma_ky: float,
    mu_zero_tol: float = 0.1,
) -> float:
    beta_array = _as_1d_float_array(beta, "beta")
    mu_sx_array = _as_1d_float_array(mu_sx, "mu_sx")
    sigma_sx_array = _as_2d_float_array(sigma_sx, "sigma_sx")

    if beta_array.shape[0] != mu_sx_array.shape[0]:
        raise ValueError("beta and mu_sx must have the same dimension.")
    if sigma_sx_array.shape != (beta_array.shape[0], beta_array.shape[0]):
        raise ValueError("sigma_sx must match beta dimension.")
    if sigma_eps < 0.0:
        raise ValueError("sigma_eps must be non-negative.")
    if sigma_ky <= 0.0:
        raise ValueError("sigma_ky must be positive.")
    if np.linalg.norm(beta_array) <= 1e-14:
        return 0.0

    try:
        mu_zero_score = float(mu_sx_array @ np.linalg.solve(sigma_sx_array, mu_sx_array))
    except np.linalg.LinAlgError:
        mu_zero_score = math.inf

    if mu_zero_score <= mu_zero_tol and sigma_eps > 1e-14:
        return float(2.0 * sigma_ky**2 / sigma_eps**2)

    try:
        candidates = _candidate_omegas_from_quartic(
            beta=beta_array,
            sigma_eps=float(sigma_eps),
            mu_sx=mu_sx_array,
            sigma_sx=sigma_sx_array,
            sigma_ky=float(sigma_ky),
        )
        scored = [
            (
                hybrid_v_h(
                    omega=candidate,
                    beta=beta_array,
                    sigma_eps=float(sigma_eps),
                    mu_sx=mu_sx_array,
                    sigma_sx=sigma_sx_array,
                    sigma_ky=float(sigma_ky),
                ),
                candidate,
            )
            for candidate in candidates
        ]
    except np.linalg.LinAlgError:
        return _omega_star_fallback(beta_array, float(sigma_eps), mu_sx_array, sigma_sx_array, float(sigma_ky))

    finite_scored = [(value, omega) for value, omega in scored if np.isfinite(value)]
    if not finite_scored:
        return _omega_star_fallback(beta_array, float(sigma_eps), mu_sx_array, sigma_sx_array, float(sigma_ky))
    return float(min(finite_scored, key=lambda item: item[0])[1])


def ols_estimator(x: Any, y: Any) -> EstimatorResult:
    x_array, y_array = _validate_training_sample(x, y)
    n_samples, n_features = x_array.shape
    if n_samples <= n_features + 1:
        raise ValueError("OLS requires n > d + 1 to estimate sigma_eps^2.")

    x_aug = _augment_design(x_array)
    theta_hat, _, rank, _ = np.linalg.lstsq(x_aug, y_array, rcond=None)
    if rank < n_features + 1:
        raise ValueError("The augmented design matrix is rank deficient.")

    residual = y_array - x_aug @ theta_hat
    rss = float(residual @ residual)
    sigma_eps_sq_hat = rss / (n_samples - n_features - 1)

    return EstimatorResult(
        name="OLS",
        beta0=float(theta_hat[0]),
        beta=np.asarray(theta_hat[1:], dtype=float),
        theta=np.asarray(theta_hat, dtype=float),
        rss=rss,
        objective=rss / n_samples,
        sigma_eps_sq_hat=float(sigma_eps_sq_hat),
    )


def calibration_estimator(
    x: Any,
    y: Any,
    mu_ky: float,
    sigma_ky: float,
    ols_result: EstimatorResult | None = None,
) -> EstimatorResult:
    x_array, y_array = _validate_training_sample(x, y)
    if sigma_ky <= 0.0:
        raise ValueError("sigma_ky must be positive.")

    ols_fit = ols_result if ols_result is not None else ols_estimator(x_array, y_array)
    beta_norm_sq = float(ols_fit.beta @ ols_fit.beta)
    target_signal_variance = max(float(sigma_ky**2) - float(ols_fit.sigma_eps_sq_hat), 0.0)
    scale = 0.0 if beta_norm_sq <= 1e-14 else math.sqrt(target_signal_variance / beta_norm_sq)

    beta0_hat = float(mu_ky)
    beta_hat = scale * ols_fit.beta
    theta_hat = np.concatenate(([beta0_hat], beta_hat))
    residual = y_array - beta0_hat - x_array @ beta_hat
    rss = float(residual @ residual)

    return EstimatorResult(
        name="Calibration",
        beta0=beta0_hat,
        beta=np.asarray(beta_hat, dtype=float),
        theta=np.asarray(theta_hat, dtype=float),
        rss=rss,
        objective=rss / x_array.shape[0],
        extra={"scale": float(scale), "target_signal_variance": target_signal_variance},
    )


def moment_matching_estimator(x: Any, y: Any, mu_ky: float, sigma_ky: float) -> EstimatorResult:
    x_array, y_array = _validate_training_sample(x, y)
    if sigma_ky <= 0.0:
        raise ValueError("sigma_ky must be positive.")

    n_samples, n_features = x_array.shape
    check_y = y_array - float(mu_ky)
    moment_xx = (x_array.T @ x_array) / n_samples
    moment_xy = (x_array.T @ check_y) / n_samples
    moment_yy = float((check_y @ check_y) / n_samples)

    a_matrix = np.eye(n_features, dtype=float) + moment_xx
    c_n = moment_yy - sigma_ky**2
    a_inv = np.linalg.inv(a_matrix)
    delta_n = float(moment_xy @ a_inv @ moment_xy - c_n)
    delta_tol = 1e-10 * max(1.0, abs(c_n), float(moment_xy @ a_inv @ moment_xy))

    def finalize(beta0: float, beta: np.ndarray, case_name: str, **extra: Any) -> EstimatorResult:
        theta = np.concatenate(([beta0], beta))
        residual = y_array - beta0 - x_array @ beta
        rss = float(residual @ residual)
        objective = _moment_matching_loss_normalized(
            x_array,
            y_array,
            beta0,
            beta,
            float(mu_ky),
            float(sigma_ky),
        )
        metadata = {
            "case": case_name,
            "delta_n": delta_n,
            "selection_rss": _selection_rss_normalized(x_array, y_array, beta, float(mu_ky)),
        }
        metadata.update(extra)
        return EstimatorResult(
            name="MM",
            beta0=float(beta0),
            beta=np.asarray(beta, dtype=float),
            theta=np.asarray(theta, dtype=float),
            rss=rss,
            objective=objective,
            extra=metadata,
        )

    if abs(delta_n) <= delta_tol:
        return finalize(float(mu_ky), a_inv @ moment_xy, "delta_zero")

    if delta_n > delta_tol:
        try:
            rho_n = float(1.0 / np.min(np.linalg.eigvalsh(a_matrix)))

            def q_of_lambda(lambda_value: float) -> float:
                beta_candidate = lambda_value * np.linalg.solve(
                    lambda_value * a_matrix - np.eye(n_features, dtype=float),
                    moment_xy,
                )
                return float(beta_candidate @ a_matrix @ beta_candidate - 2.0 * moment_xy @ beta_candidate + c_n)

            bracket_low = None
            for scale in (1e-6, 1e-8, 1e-10, 1e-12):
                candidate = rho_n + max(1.0, rho_n) * scale
                try:
                    value = q_of_lambda(candidate)
                except np.linalg.LinAlgError:
                    continue
                if np.isfinite(value) and value > 0.0:
                    bracket_low = candidate
                    break
            if bracket_low is None:
                raise RuntimeError("Failed to bracket the positive MM root.")

            bracket_high = max(bracket_low * 2.0, rho_n + 1.0)
            high_value = q_of_lambda(bracket_high)
            n_steps = 0
            while (not np.isfinite(high_value) or high_value > 0.0) and n_steps < 200:
                bracket_high = rho_n + 2.0 * (bracket_high - rho_n)
                high_value = q_of_lambda(bracket_high)
                n_steps += 1
            if not np.isfinite(high_value) or high_value > 0.0:
                raise RuntimeError("Failed to bracket the positive MM root.")

            lambda_hat = brentq(q_of_lambda, bracket_low, bracket_high, xtol=1e-12, rtol=1e-10, maxiter=500)
            beta_hat = lambda_hat * np.linalg.solve(
                lambda_hat * a_matrix - np.eye(n_features, dtype=float),
                moment_xy,
            )
            return finalize(float(mu_ky), beta_hat, "delta_positive", lambda_hat=float(lambda_hat), rho_n=rho_n)
        except (np.linalg.LinAlgError, RuntimeError):
            initial = np.zeros(n_features + 1, dtype=float)
            initial[0] = float(mu_ky)

            def objective_only(theta: np.ndarray) -> float:
                return _moment_matching_loss_normalized(
                    x_array,
                    y_array,
                    float(theta[0]),
                    theta[1:],
                    float(mu_ky),
                    float(sigma_ky),
                )

            fallback = minimize(
                objective_only,
                initial,
                method="Powell",
                options={"maxiter": 4000, "xtol": 1e-10, "ftol": 1e-10},
            )
            return finalize(
                float(fallback.x[0]),
                np.asarray(fallback.x[1:], dtype=float),
                "delta_positive_fallback",
                optimizer_success=bool(fallback.success),
            )

    bar_x = np.mean(x_array, axis=0)
    bar_y = float(np.mean(check_y))

    u_0 = float(moment_yy - moment_xy @ a_inv @ moment_xy)
    u_1 = float(2.0 * (moment_xy @ a_inv @ bar_x - bar_y))
    u_2 = float(1.0 - bar_x @ a_inv @ bar_x)
    u_poly = Polynomial([u_0, u_1, u_2])
    equation = ((Polynomial([0.0, 2.0]) + u_poly.deriv()) ** 2) * u_poly
    equation -= (sigma_ky**2) * (u_poly.deriv() ** 2)
    coefficients = np.asarray(equation.coef, dtype=float)
    while coefficients.size > 1 and abs(coefficients[-1]) <= 1e-14:
        coefficients = coefficients[:-1]

    roots = np.roots(coefficients[::-1]) if coefficients.size > 1 else np.array([], dtype=complex)
    candidates: list[tuple[float, np.ndarray, float, float]] = []
    for root in roots:
        if abs(root.imag) > 1e-8 * max(1.0, abs(root.real)):
            continue
        t_value = float(root.real)
        beta_candidate = a_inv @ (moment_xy - t_value * bar_x)
        beta0_candidate = float(mu_ky + t_value)
        objective = _moment_matching_loss_normalized(
            x_array,
            y_array,
            beta0_candidate,
            beta_candidate,
            float(mu_ky),
            float(sigma_ky),
        )
        selection_rss = _selection_rss_normalized(x_array, y_array, beta_candidate, float(mu_ky))
        candidates.append((objective, beta_candidate, beta0_candidate, selection_rss))

    if not candidates:
        initial = np.zeros(n_features + 1, dtype=float)
        initial[0] = float(mu_ky)

        def objective_only(theta: np.ndarray) -> float:
            return _moment_matching_loss_normalized(
                x_array,
                y_array,
                float(theta[0]),
                theta[1:],
                float(mu_ky),
                float(sigma_ky),
            )

        fallback = minimize(
            objective_only,
            initial,
            method="Powell",
            options={"maxiter": 4000, "xtol": 1e-10, "ftol": 1e-10},
        )
        return finalize(
            float(fallback.x[0]),
            np.asarray(fallback.x[1:], dtype=float),
            "delta_negative_fallback",
            optimizer_success=bool(fallback.success),
        )

    best_loss = min(item[0] for item in candidates)
    loss_tol = 1e-9 * max(1.0, abs(best_loss))
    admissible = [item for item in candidates if item[0] <= best_loss + loss_tol]
    _, beta_hat, beta0_hat, selection_rss = min(admissible, key=lambda item: item[3])
    return finalize(
        beta0_hat,
        beta_hat,
        "delta_negative",
        n_real_roots=len(candidates),
        selected_selection_rss=selection_rss,
    )


def hybrid_estimator(
    x: Any,
    y: Any,
    mu_ky: float,
    sigma_ky: float,
    omega: float,
    ols_result: EstimatorResult | None = None,
    maxiter: int = 1000,
    ftol: float = 1e-12,
    gtol: float = 1e-8,
    fallback: bool = False,
    accept_nonconverged: bool = True,
) -> EstimatorResult:
    x_array, y_array = _validate_training_sample(x, y)
    if sigma_ky <= 0.0:
        raise ValueError("sigma_ky must be positive.")
    if omega < 0.0:
        raise ValueError("omega must be non-negative.")
    if maxiter <= 0:
        raise ValueError("maxiter must be positive.")

    ols_fit = ols_result if ols_result is not None else ols_estimator(x_array, y_array)
    if omega <= 1e-14:
        return EstimatorResult(
            name="Hybrid",
            beta0=ols_fit.beta0,
            beta=ols_fit.beta.copy(),
            theta=ols_fit.theta.copy(),
            rss=ols_fit.rss,
            objective=ols_fit.objective,
            sigma_eps_sq_hat=ols_fit.sigma_eps_sq_hat,
            omega=float(omega),
            extra={
                "used_ols_shortcut": True,
                "optimizer_success": True,
                "accepted_nonconverged": False,
                "nit": 0,
                "nfev": 0,
                "njev": 0,
            },
        )

    x_aug = _augment_design(x_array)

    def objective(theta: np.ndarray) -> float:
        return _hybrid_loss_and_grad_normalized(theta, x_aug, y_array, float(mu_ky), float(sigma_ky), float(omega))[0]

    def gradient(theta: np.ndarray) -> np.ndarray:
        return _hybrid_loss_and_grad_normalized(theta, x_aug, y_array, float(mu_ky), float(sigma_ky), float(omega))[1]

    initial = ols_fit.theta.copy()
    initial_objective = objective(initial)
    result = minimize(
        objective,
        initial,
        jac=gradient,
        method="L-BFGS-B",
        options={"maxiter": int(maxiter), "ftol": float(ftol), "gtol": float(gtol)},
    )

    best_theta = initial
    best_objective = initial_objective
    optimizer_name = "L-BFGS-B"
    optimizer_success = bool(result.success)
    accepted_nonconverged = False

    result_is_usable = np.isfinite(result.fun) and result.fun <= best_objective + 1e-12
    if result_is_usable and (result.success or accept_nonconverged):
        best_theta = np.asarray(result.x, dtype=float)
        best_objective = float(result.fun)
        accepted_nonconverged = not bool(result.success)

    if fallback and not bool(result.success):
        fallback_result = minimize(
            objective,
            best_theta,
            method="Powell",
            options={"maxiter": 4000, "xtol": 1e-10, "ftol": 1e-10},
        )
        if np.isfinite(fallback_result.fun) and fallback_result.fun < best_objective:
            best_theta = np.asarray(fallback_result.x, dtype=float)
            best_objective = float(fallback_result.fun)
            optimizer_name = "Powell"
            optimizer_success = bool(fallback_result.success)
            accepted_nonconverged = False

    beta0_hat = float(best_theta[0])
    beta_hat = np.asarray(best_theta[1:], dtype=float)
    residual = y_array - beta0_hat - x_array @ beta_hat
    rss = float(residual @ residual)

    return EstimatorResult(
        name="Hybrid",
        beta0=beta0_hat,
        beta=beta_hat,
        theta=np.asarray(best_theta, dtype=float),
        rss=rss,
        objective=best_objective,
        omega=float(omega),
        extra={
            "optimizer": optimizer_name,
            "optimizer_success": optimizer_success,
            "accepted_nonconverged": accepted_nonconverged,
            "nit": int(getattr(result, "nit", 0) or 0),
            "nfev": int(getattr(result, "nfev", 0) or 0),
            "njev": int(getattr(result, "njev", 0) or 0),
            "initial_objective": float(initial_objective),
            "final_objective": float(best_objective),
            "message": str(getattr(result, "message", "")),
            "maxiter": int(maxiter),
            "ftol": float(ftol),
            "gtol": float(gtol),
            "fallback_enabled": bool(fallback),
        },
    )
