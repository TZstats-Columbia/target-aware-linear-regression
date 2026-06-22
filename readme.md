# Target-Aware Linear Regression Experiments

This folder contains the Python code needed to reproduce the numerical figures for the paper. It includes estimator implementations, Monte Carlo experiment drivers, theoretical curves, and plotting code.

## Dependencies

Use Python 3.10 or newer with:

```bash
pip install numpy scipy pandas matplotlib
```

## File Overview

- `estimators.py`: OLS, moment-matching (MM), hybrid-loss, and two-stage calibration estimators.
- `theory.py`: asymptotic coefficient-error and prediction-error formulas used for dashed theory curves.
- `simulation.py`: experiment grids, data generation, Monte Carlo aggregation, and CSV writing helpers.
- `run_experiments.py`: command-line entry point for running experiments.
- `plot_figures.py`: command-line entry point for plotting figures from generated CSV files.

## Quick Smoke Run

From this folder:

```bash
python run_experiments.py --mode standard --regime noise --repetitions 10 --output results/experiment_results.csv
python plot_figures.py --figure noise --input results/experiment_results.csv --output-dir figures
```

This produces `figures/noise_level.png` and `figures/noise_level.pdf`.

## Reproduce the Paper Figures

The paper uses `n=1000`, `m=100`, and `1000000` Monte Carlo repetitions per configuration. These runs can take a long time.

Run the standard Monte Carlo regimes:

```bash
python run_experiments.py --mode standard --regime all --paper-repetitions --output results/experiment_results.csv
```

Run the runtime-accuracy Pareto experiment:

```bash
python run_experiments.py --mode pareto --paper-repetitions --output results/pareto_results.csv
```

Generate all figures:

```bash
python plot_figures.py --figure all --input results/experiment_results.csv --pareto-input results/pareto_results.csv --output-dir figures
```

The plotting command writes:

- `noise_level.pdf`: noise-level experiment.
- `covariance_geometry_main.pdf`: covariance geometry at `sigma_eps_sq = 2`.
- `mean_mismatch_main.pdf`: mean mismatch at `sigma_eps_sq = 2`.
- `pareto_frontier.pdf`: accuracy-runtime Pareto frontier.
- `covariance_geometry.pdf`: covariance geometry across noise levels.
- `mean_mismatch.pdf`: mean mismatch across noise levels.

PNG copies are saved alongside the PDF files.

## Notes

The code writes generated CSV results and figures only when the commands above are run. No generated result files are included in this folder.
