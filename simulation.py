from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.stats import f

from estimators import (
    EstimatorResult,
    calibration_estimator,
    hybrid_estimator,
    hybrid_optimal_omega_star,
    moment_matching_estimator,
    ols_estimator,
)
from theory import all_theory_values, target_marginal_y


DEFAULT_BETA0 = 1.0
DEFAULT_BETA = np.array([0.969861226039, -1.26828006482, 0.671442387258], dtype=float)
DEFAULT_N = 1000
DEFAULT_M = 100
DEFAULT_REPETITIONS = 1000
PAPER_REPETITIONS = 1_000_000

RESULT_COLUMNS = [
    "regime",
    "x_name",
    "x_value",
    "n",
    "m",
    "repetitions",
    "dimension",
    "beta0",
    "beta",
    "sigma_eps_sq",
    "mu_sx",
    "sigma_sx_upper",
    "mu_ky",
    "sigma_ky",
    "method",
    "method_family",
    "hybrid_maxiter",
    "hybrid_ftol",
    "hybrid_gtol",
    "fit_time_mean",
    "fit_time_std",
    "beta_l2_sq_mean",
    "beta_l2_sq_std",
    "scaled_beta_error_mean",
    "scaled_beta_error_std",
    "mse_mean",
    "mse_std",
    "normalized_excess_mse_mean",
    "normalized_excess_mse_std",
    "theory_scaled_beta_error",
    "theory_normalized_excess_mse",
    "theory_omega",
    "optimizer_success_rate",
    "accepted_nonconverged_rate",
    "nit_mean",
    "nfev_mean",
    "njev_mean",
]


@dataclass(frozen=True)
class ProblemConfig:
    regime: str
    x_name: str
    x_value: float
    n: int
    m: int
    repetitions: int
    beta0: float
    beta: np.ndarray
    sigma_eps_sq: float
    mu_sx: np.ndarray
    sigma_sx: np.ndarray


@dataclass(frozen=True)
class HybridConfig:
    method: str
    maxiter: int
    ftol: float
    gtol: float


@dataclass
class RunningStats:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2

    @property
    def std(self) -> float:
        if self.count <= 1:
            return 0.0
        return math.sqrt(self.m2 / (self.count - 1))


def new_method_stats() -> dict[str, RunningStats]:
    return {
        "fit_time": RunningStats(),
        "beta_l2_sq": RunningStats(),
        "scaled_beta_error": RunningStats(),
        "mse": RunningStats(),
        "normalized_excess_mse": RunningStats(),
        "optimizer_success": RunningStats(),
        "accepted_nonconverged": RunningStats(),
        "nit": RunningStats(),
        "nfev": RunningStats(),
        "njev": RunningStats(),
    }


def json_compact(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def upper_triangle(matrix: np.ndarray) -> np.ndarray:
    return np.asarray(matrix[np.triu_indices(matrix.shape[0])], dtype=float)


def upper_triangle_to_symmetric(values: Iterable[float], dimension: int) -> np.ndarray:
    values_array = np.asarray(list(values), dtype=float)
    expected = dimension * (dimension + 1) // 2
    if values_array.shape[0] != expected:
        raise ValueError(f"Expected {expected} upper-triangle values, got {values_array.shape[0]}.")
    matrix = np.zeros((dimension, dimension), dtype=float)
    rows, cols = np.triu_indices(dimension)
    matrix[rows, cols] = values_array
    matrix[cols, rows] = values_array
    return matrix


def parse_float_sequence(text: str) -> np.ndarray:
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("Expected a non-empty numeric sequence.")
    if cleaned.startswith("["):
        return np.asarray(json.loads(cleaned), dtype=float)
    return np.asarray([float(part) for part in cleaned.replace(";", ",").split(",") if part.strip()], dtype=float)


def beta_orthogonal_basis(beta: np.ndarray) -> np.ndarray:
    _, _, vh = np.linalg.svd(beta.reshape(1, -1), full_matrices=True)
    return vh[1:].T


def exact_normalized_test_design(rng: np.random.Generator, m: int, dimension: int) -> np.ndarray:
    if m <= dimension:
        raise ValueError("m must be larger than the dimension.")
    for _ in range(20):
        z = rng.normal(size=(m, dimension))
        z = z - np.mean(z, axis=0, keepdims=True)
        empirical_second_moment = (z.T @ z) / m
        eigenvalues, eigenvectors = np.linalg.eigh(empirical_second_moment)
        if np.min(eigenvalues) > 1e-12:
            inv_sqrt = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T
            return z @ inv_sqrt
    raise RuntimeError("Failed to build an exact normalized test design.")


def sample_training_data(
    rng: np.random.Generator,
    n: int,
    beta0: float,
    beta: np.ndarray,
    sigma_eps: float,
    mu_sx: np.ndarray,
    sigma_sx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x = rng.multivariate_normal(mean=mu_sx, cov=sigma_sx, size=n, check_valid="raise")
    eps = rng.normal(loc=0.0, scale=sigma_eps, size=n)
    return x, beta0 + x @ beta + eps


def sample_test_data(
    rng: np.random.Generator,
    m: int,
    beta0: float,
    beta: np.ndarray,
    sigma_eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    x_test = exact_normalized_test_design(rng, m=m, dimension=beta.shape[0])
    eps_test = rng.normal(loc=0.0, scale=sigma_eps, size=m)
    return x_test, beta0 + x_test @ beta + eps_test


def predict(beta0: float, beta: np.ndarray, x: np.ndarray) -> np.ndarray:
    return beta0 + x @ beta


def mu_zero_tol_95_quantile(n: int, dimension: int) -> float:
    return float((dimension / (n - dimension)) * f.ppf(0.95, dimension, n - dimension))


def covariance_geometry_grid() -> list[tuple[float, np.ndarray]]:
    upper_triangles = {
        0.5: [2.37406854276, -2.27755137676, 0.935415059918, 3.4612094186, -1.50569688734, 1.80475079293],
        1.0: [1.51933294152, -0.927380111179, 0.387023318909, 1.91727050039, -0.705606672013, 1.10815371444],
        1.5: [1.2301575131, -0.505359393797, 0.157426959461, 1.40019184, -0.403007400073, 1.01136930286],
        2.0: [1.10307495406, -0.242934395577, 0.11446009693, 1.19847274065, -0.218646251835, 0.92167027319],
        2.5: [1.00576032298, -0.110847179599, 0.071190416455, 1.07768123498, -0.070933963392, 0.963183023158],
        3.0: [1.59867754792, 0.419021698976, -0.0732710267, 1.22140854181, -0.187037430665, 0.752543002866],
        3.5: [1.34643356558, 0.512986177055, 0.262220644473, 1.36001830092, 0.208895804725, 0.872960668496],
        4.0: [1.20147500465, 0.19507873219, -0.283648512585, 0.896741393926, -0.004602202414, 1.15102146917],
        4.5: [1.26744703872, 0.342059732191, -0.221681043268, 0.972594126088, 0.083776699074, 1.14511749408],
        5.0: [1.2448766526, 0.385380906518, -0.203546785892, 1.01913171728, 0.235031934342, 1.33796123338],
        5.5: [1.27764580168, 0.648739759967, 0.167787732059, 1.42673827787, 0.727578507934, 1.67740944747],
        6.0: [1.55959627346, 0.573827352488, -0.446631840297, 1.13157290366, 0.364109308881, 1.83289690831],
        6.5: [2.68728266361, 1.79464932527, 0.174929322524, 2.22483156173, 0.738393497188, 1.60360604591],
        7.0: [2.0402821803, 1.06644751948, -0.313625771255, 1.76028150171, 0.975028165569, 2.72330629979],
        7.5: [5.32711843722, 4.95581379607, 2.24403276103, 6.12783259705, 3.66084164454, 4.07354245153],
        8.0: [3.73871414106, 1.96224021902, -1.15224445672, 2.96110937924, 2.05052628886, 5.91256942755],
        8.5: [12.433734523, 5.97008932934, -6.173199434, 5.25959359419, 0.644658868879, 10.4874737779],
    }
    return [
        (x_value, upper_triangle_to_symmetric(values, dimension=3))
        for x_value, values in upper_triangles.items()
    ]


def mean_mismatch_grid(beta: np.ndarray) -> list[tuple[float, np.ndarray]]:
    beta_unit = beta / np.linalg.norm(beta)
    perp_unit = beta_orthogonal_basis(beta)[:, 0]
    mu_norm = math.sqrt(3.0)
    grid: list[tuple[float, np.ndarray]] = []
    for rho in np.round(np.arange(0.0, 1.0 + 1e-12, 0.1), 10):
        mu = mu_norm * (math.sqrt(max(1.0 - float(rho), 0.0)) * beta_unit)
        mu += mu_norm * (math.sqrt(float(rho)) * perp_unit)
        grid.append((float(rho), mu))
    return grid


def standard_problem_configs(
    regimes: Iterable[str],
    n: int = DEFAULT_N,
    m: int = DEFAULT_M,
    repetitions: int = DEFAULT_REPETITIONS,
    beta0: float = DEFAULT_BETA0,
    beta: np.ndarray = DEFAULT_BETA,
) -> list[ProblemConfig]:
    requested = list(regimes)
    if "all" in requested:
        requested = ["noise", "covariance", "mean"]

    configs: list[ProblemConfig] = []
    dimension = beta.shape[0]
    zero_mu = np.zeros(dimension, dtype=float)
    identity = np.eye(dimension, dtype=float)

    if "noise" in requested:
        for sigma_eps_sq in np.arange(1.0, 20.0 + 1e-12, 1.0):
            configs.append(
                ProblemConfig(
                    regime="noise",
                    x_name="sigma_eps_sq",
                    x_value=float(sigma_eps_sq),
                    n=n,
                    m=m,
                    repetitions=repetitions,
                    beta0=beta0,
                    beta=beta.copy(),
                    sigma_eps_sq=float(sigma_eps_sq),
                    mu_sx=zero_mu.copy(),
                    sigma_sx=identity.copy(),
                )
            )

    if "covariance" in requested:
        for sigma_eps_sq in (2.0, 4.0, 8.0, 16.0):
            for beta_sigma_inv_beta, sigma_sx in covariance_geometry_grid():
                configs.append(
                    ProblemConfig(
                        regime="covariance",
                        x_name="beta_sigma_inv_beta",
                        x_value=float(beta_sigma_inv_beta),
                        n=n,
                        m=m,
                        repetitions=repetitions,
                        beta0=beta0,
                        beta=beta.copy(),
                        sigma_eps_sq=float(sigma_eps_sq),
                        mu_sx=zero_mu.copy(),
                        sigma_sx=sigma_sx.copy(),
                    )
                )

    if "mean" in requested:
        for sigma_eps_sq in (2.0, 4.0, 8.0, 16.0):
            for rho, mu_sx in mean_mismatch_grid(beta):
                configs.append(
                    ProblemConfig(
                        regime="mean",
                        x_name="rho",
                        x_value=float(rho),
                        n=n,
                        m=m,
                        repetitions=repetitions,
                        beta0=beta0,
                        beta=beta.copy(),
                        sigma_eps_sq=float(sigma_eps_sq),
                        mu_sx=mu_sx.copy(),
                        sigma_sx=identity.copy(),
                    )
                )

    unknown = set(requested).difference({"noise", "covariance", "mean"})
    if unknown:
        raise ValueError(f"Unknown standard regime(s): {', '.join(sorted(unknown))}")
    return configs


def pareto_problem_configs(
    n: int = DEFAULT_N,
    m: int = DEFAULT_M,
    repetitions: int = DEFAULT_REPETITIONS,
    beta0: float = DEFAULT_BETA0,
    beta: np.ndarray = DEFAULT_BETA,
) -> list[ProblemConfig]:
    dimension = beta.shape[0]
    return [
        ProblemConfig(
            regime="pareto",
            x_name="sigma_eps_sq",
            x_value=float(sigma_eps_sq),
            n=n,
            m=m,
            repetitions=repetitions,
            beta0=beta0,
            beta=beta.copy(),
            sigma_eps_sq=float(sigma_eps_sq),
            mu_sx=np.zeros(dimension, dtype=float),
            sigma_sx=np.eye(dimension, dtype=float),
        )
        for sigma_eps_sq in (1.0, 2.0, 3.0, 4.0, 5.0)
    ]


def default_pareto_hybrid_configs() -> list[HybridConfig]:
    return [
        HybridConfig(method="Hybrid_full", maxiter=1000, ftol=1e-12, gtol=1e-8),
        HybridConfig(method="Hybrid_tol_5e-04_5e-01", maxiter=1000, ftol=5e-4, gtol=5e-1),
        HybridConfig(method="Hybrid_tol_5e-05_5e-02", maxiter=1000, ftol=5e-5, gtol=5e-2),
        HybridConfig(method="Hybrid_tol_5e-06_5e-03", maxiter=1000, ftol=5e-6, gtol=5e-3),
    ]


def update_estimator_stats(
    stats: dict[str, RunningStats],
    estimator: EstimatorResult,
    beta: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    n: int,
    sigma_eps_sq: float,
    fit_time: float,
    optimizer_extra: dict[str, object] | None = None,
) -> None:
    beta_error = estimator.beta - beta
    beta_l2_sq = float(beta_error @ beta_error)
    mse = float(np.mean((predict(estimator.beta0, estimator.beta, x_test) - y_test) ** 2))

    stats["fit_time"].update(float(fit_time))
    stats["beta_l2_sq"].update(beta_l2_sq)
    stats["scaled_beta_error"].update(n * beta_l2_sq / sigma_eps_sq)
    stats["mse"].update(mse)
    stats["normalized_excess_mse"].update(n * (mse - sigma_eps_sq) / sigma_eps_sq)

    if optimizer_extra is not None:
        stats["optimizer_success"].update(1.0 if optimizer_extra.get("optimizer_success") else 0.0)
        stats["accepted_nonconverged"].update(1.0 if optimizer_extra.get("accepted_nonconverged") else 0.0)
        stats["nit"].update(float(optimizer_extra.get("nit", 0.0) or 0.0))
        stats["nfev"].update(float(optimizer_extra.get("nfev", 0.0) or 0.0))
        stats["njev"].update(float(optimizer_extra.get("njev", 0.0) or 0.0))


def timed_fit(callable_fit):
    start = time.perf_counter()
    result = callable_fit()
    return result, time.perf_counter() - start


def sample_omega_star(
    x_train: np.ndarray,
    ols: EstimatorResult,
    sigma_ky: float,
    mu_zero_tol: float,
) -> float:
    mu_sx_hat = np.mean(x_train, axis=0)
    x_centered = x_train - mu_sx_hat
    sigma_sx_hat = (x_centered.T @ x_centered) / x_train.shape[0]
    sigma_eps_hat = math.sqrt(max(float(ols.sigma_eps_sq_hat), 0.0))
    return hybrid_optimal_omega_star(
        beta=ols.beta,
        sigma_eps=sigma_eps_hat,
        mu_sx=mu_sx_hat,
        sigma_sx=sigma_sx_hat,
        sigma_ky=sigma_ky,
        mu_zero_tol=mu_zero_tol,
    )


def run_problem_config(
    config: ProblemConfig,
    rng: np.random.Generator,
    hybrid_configs: list[HybridConfig],
    include_standard_hybrid: bool,
    hybrid_fallback: bool,
    accept_nonconverged: bool,
    progress_every: int = 0,
) -> list[dict[str, object]]:
    sigma_eps = math.sqrt(config.sigma_eps_sq)
    mu_ky, sigma_ky = target_marginal_y(config.beta0, config.beta, config.sigma_eps_sq)
    theory = all_theory_values(
        beta=config.beta,
        sigma_eps_sq=config.sigma_eps_sq,
        mu_sx=config.mu_sx,
        sigma_sx=config.sigma_sx,
        sigma_ky=sigma_ky,
    )
    mu_zero_tol = mu_zero_tol_95_quantile(config.n, config.beta.shape[0])

    method_meta: dict[str, dict[str, object]] = {
        "OLS": {"method_family": "ols", "hybrid_maxiter": "", "hybrid_ftol": "", "hybrid_gtol": ""},
        "MM": {"method_family": "moment_matching", "hybrid_maxiter": "", "hybrid_ftol": "", "hybrid_gtol": ""},
        "Calibration": {"method_family": "calibration", "hybrid_maxiter": "", "hybrid_ftol": "", "hybrid_gtol": ""},
    }
    if include_standard_hybrid:
        method_meta["Hybrid"] = {
            "method_family": "hybrid",
            "hybrid_maxiter": 1000,
            "hybrid_ftol": 1e-12,
            "hybrid_gtol": 1e-8,
        }
    for hybrid_config in hybrid_configs:
        method_meta[hybrid_config.method] = {
            "method_family": "hybrid",
            "hybrid_maxiter": hybrid_config.maxiter,
            "hybrid_ftol": hybrid_config.ftol,
            "hybrid_gtol": hybrid_config.gtol,
        }
    stats_by_method = {method: new_method_stats() for method in method_meta}

    for repetition in range(1, config.repetitions + 1):
        x_train, y_train = sample_training_data(
            rng=rng,
            n=config.n,
            beta0=config.beta0,
            beta=config.beta,
            sigma_eps=sigma_eps,
            mu_sx=config.mu_sx,
            sigma_sx=config.sigma_sx,
        )
        x_test, y_test = sample_test_data(
            rng=rng,
            m=config.m,
            beta0=config.beta0,
            beta=config.beta,
            sigma_eps=sigma_eps,
        )

        ols, ols_time = timed_fit(lambda: ols_estimator(x_train, y_train))
        update_estimator_stats(
            stats_by_method["OLS"],
            ols,
            config.beta,
            x_test,
            y_test,
            config.n,
            config.sigma_eps_sq,
            fit_time=ols_time,
        )

        mm, mm_time = timed_fit(lambda: moment_matching_estimator(x_train, y_train, mu_ky=mu_ky, sigma_ky=sigma_ky))
        update_estimator_stats(
            stats_by_method["MM"],
            mm,
            config.beta,
            x_test,
            y_test,
            config.n,
            config.sigma_eps_sq,
            fit_time=mm_time,
        )

        cali, cali_time = timed_fit(
            lambda: calibration_estimator(x_train, y_train, mu_ky=mu_ky, sigma_ky=sigma_ky, ols_result=ols)
        )
        update_estimator_stats(
            stats_by_method["Calibration"],
            cali,
            config.beta,
            x_test,
            y_test,
            config.n,
            config.sigma_eps_sq,
            fit_time=ols_time + cali_time,
        )

        setup_start = time.perf_counter()
        omega_star = sample_omega_star(x_train=x_train, ols=ols, sigma_ky=sigma_ky, mu_zero_tol=mu_zero_tol)
        omega_setup_time = time.perf_counter() - setup_start

        if include_standard_hybrid:
            hybrid, hybrid_time = timed_fit(
                lambda: hybrid_estimator(
                    x_train,
                    y_train,
                    mu_ky=mu_ky,
                    sigma_ky=sigma_ky,
                    omega=omega_star,
                    ols_result=ols,
                    maxiter=1000,
                    ftol=1e-12,
                    gtol=1e-8,
                    fallback=hybrid_fallback,
                    accept_nonconverged=accept_nonconverged,
                )
            )
            update_estimator_stats(
                stats_by_method["Hybrid"],
                hybrid,
                config.beta,
                x_test,
                y_test,
                config.n,
                config.sigma_eps_sq,
                fit_time=ols_time + omega_setup_time + hybrid_time,
                optimizer_extra=hybrid.extra,
            )

        for hybrid_config in hybrid_configs:
            hybrid, hybrid_time = timed_fit(
                lambda cfg=hybrid_config: hybrid_estimator(
                    x_train,
                    y_train,
                    mu_ky=mu_ky,
                    sigma_ky=sigma_ky,
                    omega=omega_star,
                    ols_result=ols,
                    maxiter=cfg.maxiter,
                    ftol=cfg.ftol,
                    gtol=cfg.gtol,
                    fallback=hybrid_fallback,
                    accept_nonconverged=accept_nonconverged,
                )
            )
            update_estimator_stats(
                stats_by_method[hybrid_config.method],
                hybrid,
                config.beta,
                x_test,
                y_test,
                config.n,
                config.sigma_eps_sq,
                fit_time=ols_time + omega_setup_time + hybrid_time,
                optimizer_extra=hybrid.extra,
            )

        if progress_every and repetition % progress_every == 0:
            print(
                f"{config.regime} {config.x_name}={config.x_value:g}, "
                f"sigma_eps_sq={config.sigma_eps_sq:g}: {repetition}/{config.repetitions}"
            )

    rows = []
    base_row = {
        "regime": config.regime,
        "x_name": config.x_name,
        "x_value": config.x_value,
        "n": config.n,
        "m": config.m,
        "repetitions": config.repetitions,
        "dimension": int(config.beta.shape[0]),
        "beta0": float(config.beta0),
        "beta": json_compact(config.beta.tolist()),
        "sigma_eps_sq": float(config.sigma_eps_sq),
        "mu_sx": json_compact(config.mu_sx.tolist()),
        "sigma_sx_upper": json_compact(upper_triangle(config.sigma_sx).tolist()),
        "mu_ky": mu_ky,
        "sigma_ky": sigma_ky,
    }

    for method, stats in stats_by_method.items():
        theory_key = "Hybrid" if method.startswith("Hybrid") else method
        theory_values = theory[theory_key]
        meta = method_meta[method]
        is_hybrid = meta["method_family"] == "hybrid"
        rows.append(
            {
                **base_row,
                "method": method,
                "method_family": meta["method_family"],
                "hybrid_maxiter": meta["hybrid_maxiter"],
                "hybrid_ftol": meta["hybrid_ftol"],
                "hybrid_gtol": meta["hybrid_gtol"],
                "fit_time_mean": stats["fit_time"].mean,
                "fit_time_std": stats["fit_time"].std,
                "beta_l2_sq_mean": stats["beta_l2_sq"].mean,
                "beta_l2_sq_std": stats["beta_l2_sq"].std,
                "scaled_beta_error_mean": stats["scaled_beta_error"].mean,
                "scaled_beta_error_std": stats["scaled_beta_error"].std,
                "mse_mean": stats["mse"].mean,
                "mse_std": stats["mse"].std,
                "normalized_excess_mse_mean": stats["normalized_excess_mse"].mean,
                "normalized_excess_mse_std": stats["normalized_excess_mse"].std,
                "theory_scaled_beta_error": theory_values.scaled_beta_error,
                "theory_normalized_excess_mse": theory_values.normalized_excess_mse,
                "theory_omega": "" if theory_values.omega is None else theory_values.omega,
                "optimizer_success_rate": stats["optimizer_success"].mean if is_hybrid else "",
                "accepted_nonconverged_rate": stats["accepted_nonconverged"].mean if is_hybrid else "",
                "nit_mean": stats["nit"].mean if is_hybrid else "",
                "nfev_mean": stats["nfev"].mean if is_hybrid else "",
                "njev_mean": stats["njev"].mean if is_hybrid else "",
            }
        )
    return rows


def run_standard_experiments(
    regimes: Iterable[str],
    n: int,
    m: int,
    repetitions: int,
    seed: int | None,
    progress_every: int,
) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    configs = standard_problem_configs(regimes=regimes, n=n, m=m, repetitions=repetitions)
    for config in configs:
        rows.extend(
            run_problem_config(
                config=config,
                rng=rng,
                hybrid_configs=[],
                include_standard_hybrid=True,
                hybrid_fallback=True,
                accept_nonconverged=True,
                progress_every=progress_every,
            )
        )
    return rows


def run_pareto_experiments(
    n: int,
    m: int,
    repetitions: int,
    seed: int | None,
    progress_every: int,
) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for config in pareto_problem_configs(n=n, m=m, repetitions=repetitions):
        rows.extend(
            run_problem_config(
                config=config,
                rng=rng,
                hybrid_configs=default_pareto_hybrid_configs(),
                include_standard_hybrid=False,
                hybrid_fallback=False,
                accept_nonconverged=True,
                progress_every=progress_every,
            )
        )
    return rows


def write_result_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in RESULT_COLUMNS})
