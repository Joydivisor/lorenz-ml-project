from __future__ import annotations

import csv
from pathlib import Path

RESULTS_DIR = Path("results")
ROW_END = r" \\"


def rows(path: str) -> list[dict[str, str]]:
    with (RESULTS_DIR / path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def f(value: str, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def by_model(data: list[dict[str, str]], model: str) -> dict[str, str]:
    return next(row for row in data if (row.get("model") or row.get("") or "") == model)


def print_original_metrics() -> None:
    labels = {
        "Persistence": "Persistence Model",
        "Linear Regression": "线性回归",
        "Random Forest": "随机森林",
        "MLP": "MLP",
    }
    data = rows("model_metrics.csv")
    print("% original_metrics from results/model_metrics.csv")
    for row in data:
        model = row.get("", "")
        print(f"{labels.get(model, model)} & {f(row['RMSE'])} & {f(row['MAE'])} & {f(row['R2'])}{ROW_END}")
    print()


def print_horizon_metrics() -> None:
    data = rows("horizon_results.csv")
    wanted = [("10", "Random Forest"), ("10", "MLP"), ("100", "Random Forest"), ("100", "MLP"), ("500", "Random Forest"), ("500", "MLP")]
    print("% horizon_metrics from results/horizon_results.csv")
    for horizon, model in wanted:
        row = next(r for r in data if r["horizon"] == horizon and r["model"] == model)
        print(f"{horizon} & {model} & {f(row['RMSE_state'])} & {f(row['R2_mean'])}{ROW_END}")
    print()


def print_model_enhancement() -> None:
    data = rows("model_enhancement_results.csv")
    wanted = [
        ("10", "baseline-MLP", "原始 MLP"),
        ("10", "MLP-residual-relu-64x64x64", "Residual MLP ReLU 64-64-64"),
        ("10", "ESN-residual", "ESN residual"),
        ("100", "baseline-MLP", "原始 MLP"),
        ("100", "MLP-residual-relu-64x64x64", "Residual MLP ReLU 64-64-64"),
        ("100", "ESN-residual", "ESN residual"),
        ("500", "baseline-MLP", "原始 MLP"),
        ("500", "MLP-residual-relu-64x64x64", "Residual MLP ReLU 64-64-64"),
        ("500", "baseline-Random Forest", "Random Forest 基线"),
    ]
    print("% model_enhancement from results/model_enhancement_results.csv")
    for horizon, model, label in wanted:
        row = next(r for r in data if r["horizon"] == horizon and r["model"] == model)
        print(f"{horizon} & {label} & {f(row['RMSE_state'])} & {f(row['R2_mean'], 5)}{ROW_END}")
    print()


def print_sindy_true_vs_identified() -> None:
    coeffs = rows("sindy_coefficients.csv")
    lookup = {row[""]: row for row in coeffs}
    entries = [
        ("$dx/dt$", "$x$", -10.0, lookup["x"]["dx_dt"]),
        ("$dx/dt$", "$y$", 10.0, lookup["y"]["dx_dt"]),
        ("$dy/dt$", "$x$", 28.0, lookup["x"]["dy_dt"]),
        ("$dy/dt$", "$y$", -1.0, lookup["y"]["dy_dt"]),
        ("$dy/dt$", "$xz$", -1.0, lookup["xz"]["dy_dt"]),
        ("$dz/dt$", "$xy$", 1.0, lookup["xy"]["dz_dt"]),
        ("$dz/dt$", "$z$", -8.0 / 3.0, lookup["z"]["dz_dt"]),
    ]
    print("% sindy_true_vs_identified from results/sindy_coefficients.csv")
    for equation, term, true_value, identified in entries:
        identified_value = float(identified)
        print(f"{equation} & {term} & {true_value:.4f} & {identified_value:.4f} & {abs(identified_value - true_value):.4f}{ROW_END}")
    print()


def print_hybrid_metrics() -> None:
    data = rows("hybrid_metrics.csv")
    print("% hybrid_metrics from results/hybrid_metrics.csv")
    for model in ["Imperfect physics", "Pure ML Residual MLP", "Hybrid RF", "Hybrid MLP"]:
        row = by_model(data, model)
        print(f"{model} & {f(row['RMSE_state'])} & {f(row['R2_mean'], 10)}{ROW_END}")
    print()


def print_valid_prediction_time() -> None:
    data = rows("valid_prediction_time.csv")
    print("% valid_prediction_time from results/valid_prediction_time.csv")
    for model in ["Imperfect physics", "Pure ML Residual MLP", "Hybrid RF", "Hybrid MLP"]:
        row = by_model(data, model)
        print(f"{model} & {int(float(row['valid_steps']))} & {f(row['valid_physical_time'])} & {f(row['valid_lyapunov_time'])}{ROW_END}")
    print()


if __name__ == "__main__":
    print_original_metrics()
    print_horizon_metrics()
    print_model_enhancement()
    print_sindy_true_vs_identified()
    print_hybrid_metrics()
    print_valid_prediction_time()
