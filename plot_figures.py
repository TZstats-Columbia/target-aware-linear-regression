from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd


METHOD_COLORS = {
    "OLS": "#1f77b4",
    "MM": "#ff7f0e",
    "Hybrid": "#2ca02c",
    "Calibration": "#d62728",
}
METHOD_ORDER = ["OLS", "Hybrid", "MM", "Calibration"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot paper figures from experiment CSV files.")
    parser.add_argument(
        "--figure",
        choices=("all", "noise", "covariance", "mean", "pareto"),
        default="all",
        help="Which figure family to generate.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results") / "experiment_results.csv",
        help="CSV produced by run_experiments.py --mode standard.",
    )
    parser.add_argument(
        "--pareto-input",
        type=Path,
        default=Path("results") / "pareto_results.csv",
        help="CSV produced by run_experiments.py --mode pareto.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("figures"), help="Directory for generated figures.")
    parser.add_argument("--show", action="store_true", help="Display figures after saving.")
    return parser.parse_args()


def save_figure(fig, output_dir: Path, stem: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / f"{stem}.png", output_dir / f"{stem}.pdf"]
    for path in paths:
        fig.savefig(path, dpi=300, bbox_inches="tight")
    return paths


def load_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input CSV: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"No rows found in {path}")
    return df


def style_axis(ax) -> None:
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.35)
    ax.tick_params(labelsize=11)


def method_label(method: str) -> str:
    if method == "Hybrid_full":
        return "Hybrid full"
    if method.startswith("Hybrid_tol_"):
        parts = method.replace("Hybrid_tol_", "").split("_")
        if len(parts) == 2:
            return f"Hybrid {parts[0]}/{parts[1]}"
    return method


def set_shared_ylim(axes, bottom: float = 0.0, padding: float = 0.05) -> None:
    flat_axes = np.asarray(axes, dtype=object).ravel()
    top = bottom
    for ax in flat_axes:
        ymin, ymax = ax.get_ylim()
        if np.isfinite(ymax):
            top = max(top, ymax)
    if math.isclose(top, bottom):
        top = bottom + 1.0
    for ax in flat_axes:
        ax.set_ylim(bottom, top * (1.0 + padding))


def draw_metric_series(ax, df: pd.DataFrame, x_col: str, y_col: str, theory_col: str) -> None:
    for method in METHOD_ORDER:
        method_df = df.loc[df["method"] == method].sort_values(x_col)
        if method_df.empty:
            continue
        color = METHOD_COLORS[method]
        ax.plot(
            method_df[x_col],
            method_df[theory_col],
            color=color,
            linestyle="--",
            linewidth=2.0,
            alpha=0.95,
        )
        ax.scatter(
            method_df[x_col],
            method_df[y_col],
            color=color,
            s=36,
            alpha=0.88,
            edgecolors="none",
            zorder=3,
        )


def add_standard_legends(beta_ax, mse_ax) -> None:
    from matplotlib.lines import Line2D

    method_handles = [
        Line2D([0], [0], marker="o", linestyle="None", color=METHOD_COLORS[method], markersize=7, label=method)
        for method in METHOD_ORDER
    ]
    style_handles = [
        Line2D([0], [0], marker="o", linestyle="None", color="black", markersize=7, label="MC avg"),
        Line2D([0], [0], linestyle="--", color="black", linewidth=2.0, label="Theory"),
    ]
    beta_ax.legend(handles=method_handles, frameon=False, fontsize=10, title="Method", title_fontsize=11)
    mse_ax.legend(handles=style_handles, frameon=False, fontsize=10)


def plot_noise(df: pd.DataFrame, output_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    plot_df = df.loc[df["regime"] == "noise"].copy()
    if plot_df.empty:
        raise ValueError("No noise-regime rows found.")

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.6), sharey=True)
    beta_ax, mse_ax = axes
    draw_metric_series(
        beta_ax,
        plot_df,
        x_col="sigma_eps_sq",
        y_col="scaled_beta_error_mean",
        theory_col="theory_scaled_beta_error",
    )
    draw_metric_series(
        mse_ax,
        plot_df,
        x_col="sigma_eps_sq",
        y_col="normalized_excess_mse_mean",
        theory_col="theory_normalized_excess_mse",
    )

    beta_ax.set_title(r"$n\|\hat{\beta}-\beta\|_2^2/\sigma_\varepsilon^2$", fontsize=13)
    mse_ax.set_title(r"$n(\mathrm{MSE}-\sigma_\varepsilon^2)/\sigma_\varepsilon^2$", fontsize=13)
    for ax in axes:
        style_axis(ax)
        ax.set_xlabel(r"$\sigma_\varepsilon^2$", fontsize=12)
        ax.set_xlim(0.0, 20.5)
        ax.set_xticks([0, 5, 10, 15, 20])
    mse_ax.tick_params(labelleft=False)
    add_standard_legends(beta_ax, mse_ax)
    set_shared_ylim(axes, bottom=0.0)
    fig.tight_layout()
    paths = save_figure(fig, output_dir, "noise_level")
    plt.close(fig)
    return paths


def plot_shift_grid(
    df: pd.DataFrame,
    regime: str,
    x_col: str,
    x_label: str,
    output_dir: Path,
    stem: str,
    sigma_levels: list[float],
) -> list[Path]:
    import matplotlib.pyplot as plt

    plot_df = df.loc[(df["regime"] == regime) & (df["sigma_eps_sq"].isin(sigma_levels))].copy()
    if plot_df.empty:
        raise ValueError(f"No {regime}-regime rows found for {sigma_levels}.")

    fig, axes = plt.subplots(len(sigma_levels), 2, figsize=(12.2, 3.6 * len(sigma_levels)), sharex="col", sharey=True)
    if len(sigma_levels) == 1:
        axes = np.asarray([axes])

    for row_index, sigma_eps_sq in enumerate(sigma_levels):
        panel_df = plot_df.loc[np.isclose(plot_df["sigma_eps_sq"], sigma_eps_sq)].copy()
        beta_ax, mse_ax = axes[row_index]
        draw_metric_series(
            beta_ax,
            panel_df,
            x_col=x_col,
            y_col="scaled_beta_error_mean",
            theory_col="theory_scaled_beta_error",
        )
        draw_metric_series(
            mse_ax,
            panel_df,
            x_col=x_col,
            y_col="normalized_excess_mse_mean",
            theory_col="theory_normalized_excess_mse",
        )
        for ax in (beta_ax, mse_ax):
            style_axis(ax)
        beta_ax.set_ylabel(rf"$\sigma_\varepsilon^2={sigma_eps_sq:g}$", fontsize=12)
        mse_ax.tick_params(labelleft=False)
        if row_index == 0:
            beta_ax.set_title(r"$n\|\hat{\beta}-\beta\|_2^2/\sigma_\varepsilon^2$", fontsize=13)
            mse_ax.set_title(r"$n(\mathrm{MSE}-\sigma_\varepsilon^2)/\sigma_\varepsilon^2$", fontsize=13)

    for ax in axes[-1]:
        ax.set_xlabel(x_label, fontsize=12)
    if regime == "mean":
        for ax in axes.ravel():
            ax.set_xlim(-0.05, 1.05)
            ax.set_xticks(np.linspace(0.0, 1.0, 6))
    if regime == "covariance":
        for ax in axes.ravel():
            ax.set_xlim(0.0, 9.0)

    add_standard_legends(axes[0, 0], axes[0, 1])
    set_shared_ylim(axes, bottom=0.0)
    fig.tight_layout()
    paths = save_figure(fig, output_dir, stem)
    plt.close(fig)
    return paths


def plot_covariance(df: pd.DataFrame, output_dir: Path) -> list[Path]:
    paths = []
    label = r"$\beta^\top\Sigma_{s,x}^{-1}\beta$"
    paths.extend(
        plot_shift_grid(
            df,
            regime="covariance",
            x_col="x_value",
            x_label=label,
            output_dir=output_dir,
            stem="covariance_geometry",
            sigma_levels=[2.0, 4.0, 8.0, 16.0],
        )
    )
    paths.extend(
        plot_shift_grid(
            df,
            regime="covariance",
            x_col="x_value",
            x_label=label,
            output_dir=output_dir,
            stem="covariance_geometry_main",
            sigma_levels=[2.0],
        )
    )
    return paths


def plot_mean(df: pd.DataFrame, output_dir: Path) -> list[Path]:
    paths = []
    label = r"$\|P_{\beta^\perp}\mu_{s,x}\|_2^2 / \|\mu_{s,x}\|_2^2$"
    paths.extend(
        plot_shift_grid(
            df,
            regime="mean",
            x_col="x_value",
            x_label=label,
            output_dir=output_dir,
            stem="mean_mismatch",
            sigma_levels=[2.0, 4.0, 8.0, 16.0],
        )
    )
    paths.extend(
        plot_shift_grid(
            df,
            regime="mean",
            x_col="x_value",
            x_label=label,
            output_dir=output_dir,
            stem="mean_mismatch_main",
            sigma_levels=[2.0],
        )
    )
    return paths


def pareto_method_order(methods: pd.Series) -> list[str]:
    method_set = set(methods.astype(str))
    ordered = [method for method in ["OLS", "Calibration", "MM", "Hybrid_full"] if method in method_set]
    hybrid_tol = sorted(method for method in method_set if method.startswith("Hybrid_tol_"))
    remaining = sorted(method_set.difference(ordered).difference(hybrid_tol))
    return ordered + hybrid_tol + remaining


def draw_pareto_panel(fig, ax, df: pd.DataFrame, y_col: str, y_label: str, cmap, norm, show_ylabel: bool) -> None:
    for _, sigma_df in df.groupby("sigma_eps_sq", sort=True):
        sigma_df = sigma_df.sort_values("log_fit_time_mean")
        if len(sigma_df) < 2:
            continue
        ax.plot(
            sigma_df["log_fit_time_mean"],
            sigma_df[y_col],
            color=cmap(norm(float(sigma_df["test_r_square_mean"].mean()))),
            alpha=0.25,
            linewidth=1.0,
            zorder=1,
        )

    ordered_methods = pareto_method_order(df["method"])
    marker_by_method = {method: MARKERS[index % len(MARKERS)] for index, method in enumerate(ordered_methods)}
    for method in ordered_methods:
        method_df = df.loc[df["method"] == method]
        ax.scatter(
            method_df["log_fit_time_mean"],
            method_df[y_col],
            c=method_df["test_r_square_mean"],
            cmap=cmap,
            norm=norm,
            marker=marker_by_method[method],
            s=52,
            alpha=0.92,
            edgecolors="white",
            linewidths=0.55,
            zorder=2,
        )

    style_axis(ax)
    ax.set_xlabel("log runtime per fit (seconds)", fontsize=12)
    if show_ylabel:
        ax.set_ylabel(y_label, fontsize=12)
    else:
        ax.set_ylabel("")
    return ordered_methods, marker_by_method


def plot_pareto(df: pd.DataFrame, output_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D

    plot_df = df.loc[df["regime"] == "pareto"].copy()
    if plot_df.empty:
        raise ValueError("No pareto-regime rows found.")
    plot_df = plot_df.replace([np.inf, -np.inf], np.nan)
    plot_df = plot_df.dropna(subset=["fit_time_mean", "scaled_beta_error_mean", "normalized_excess_mse_mean"])
    plot_df = plot_df.loc[
        (plot_df["fit_time_mean"] > 0.0)
        & (plot_df["scaled_beta_error_mean"] > 0.0)
        & (plot_df["normalized_excess_mse_mean"] > 0.0)
    ].copy()
    plot_df["log_fit_time_mean"] = np.log(plot_df["fit_time_mean"].astype(float))
    plot_df["neg_log_scaled_beta_error"] = -np.log(plot_df["scaled_beta_error_mean"].astype(float))
    plot_df["neg_log_normalized_excess_mse"] = -np.log(plot_df["normalized_excess_mse_mean"].astype(float))
    plot_df["test_r_square_mean"] = 1.0 - plot_df["mse_mean"].astype(float) / (plot_df["sigma_ky"].astype(float) ** 2)

    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=plot_df["test_r_square_mean"].min(), vmax=plot_df["test_r_square_mean"].max())
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5), constrained_layout=True)
    ordered_methods, marker_by_method = draw_pareto_panel(
        fig,
        axes[0],
        plot_df,
        y_col="neg_log_scaled_beta_error",
        y_label=r"$-\log(n\|\hat{\beta}-\beta\|_2^2/\sigma_\varepsilon^2)$",
        cmap=cmap,
        norm=norm,
        show_ylabel=True,
    )
    draw_pareto_panel(
        fig,
        axes[1],
        plot_df,
        y_col="neg_log_normalized_excess_mse",
        y_label=r"$-\log(n(\mathrm{MSE}-\sigma_\varepsilon^2)/\sigma_\varepsilon^2)$",
        cmap=cmap,
        norm=norm,
        show_ylabel=True,
    )
    axes[0].set_title("Beta error", fontsize=12)
    axes[1].set_title("Prediction error", fontsize=12)

    scalar_mappable = ScalarMappable(norm=norm, cmap=cmap)
    scalar_mappable.set_array([])
    colorbar = fig.colorbar(scalar_mappable, ax=axes, pad=0.02, shrink=0.92)
    colorbar.set_label(r"$R_{\mathrm{test}}^2$", fontsize=11)
    colorbar.ax.tick_params(labelsize=10)

    handles = [
        Line2D(
            [0],
            [0],
            marker=marker_by_method[method],
            linestyle="None",
            markerfacecolor="0.65",
            markeredgecolor="white",
            markeredgewidth=0.55,
            color="0.65",
            markersize=7,
            label=method_label(method),
        )
        for method in ordered_methods
    ]
    fig.legend(handles=handles, title="Method", frameon=False, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.12))

    paths = save_figure(fig, output_dir, "pareto_frontier")
    plt.close(fig)
    return paths


def main() -> None:
    args = parse_args()
    if not args.show:
        matplotlib.use("Agg")

    saved_paths: list[Path] = []
    needs_standard = args.figure in {"all", "noise", "covariance", "mean"}
    needs_pareto = args.figure in {"all", "pareto"}

    if needs_standard:
        standard_df = load_results(args.input)
        if args.figure in {"all", "noise"}:
            saved_paths.extend(plot_noise(standard_df, args.output_dir))
        if args.figure in {"all", "covariance"}:
            saved_paths.extend(plot_covariance(standard_df, args.output_dir))
        if args.figure in {"all", "mean"}:
            saved_paths.extend(plot_mean(standard_df, args.output_dir))
    if needs_pareto:
        pareto_df = load_results(args.pareto_input)
        saved_paths.extend(plot_pareto(pareto_df, args.output_dir))

    if args.show:
        import matplotlib.pyplot as plt

        plt.show()

    print("Saved figures:")
    for path in saved_paths:
        print(path)


if __name__ == "__main__":
    main()
