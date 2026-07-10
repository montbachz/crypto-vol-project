from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

STEPS = [
    ("Random-mask baselines", "src/evaluate_baselines.py"),
    ("Structured-mask baselines", "src/evaluate_structured_baselines.py"),
    ("Train ConvVAE", "src/train_convvae.py"),
    ("Structured ConvVAE evaluation", "src/evaluate_convvae_structured.py"),
    ("Compare structured results", "src/compare_structured_results.py"),
    ("ConvVAE residuals", "src/convvae_residuals.py"),
    ("IV mean-reversion regression", "src/test_iv_mean_reversion.py"),
    ("Signal breakdown", "src/analyze_signal_breakdown.py"),
    ("Filtered residual strategy", "src/filtered_residual_strategy.py"),
    ("Out-of-sample filtered strategy", "src/out_of_sample_filtered_strategy.py"),
    ("Lambda VaR risk summary", "src/lambda_var.py"),
    ("Final plots", "src/make_final_plots.py"),
]


def run_step(name: str, script_path: str) -> None:
    print("\n" + "=" * 90, flush=True)
    print(f"Running: {name}", flush=True)
    print(f"Script:  {script_path}", flush=True)
    print("=" * 90, flush=True)

    result = subprocess.run(
        [sys.executable, script_path],
        cwd=PROJECT_ROOT,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Step failed: {name}")


def main() -> None:
    for name, script_path in STEPS:
        run_step(name, script_path)

    print("\n" + "=" * 90, flush=True)
    print("All steps completed successfully.", flush=True)
    print("=" * 90, flush=True)


if __name__ == "__main__":
    main()
