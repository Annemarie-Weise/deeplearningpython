"""Aggregate experiment metrics across seeds for every model configuration."""

import argparse
import json
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "results" / "model_evaluation" / "summary.csv"


def parse_args() -> argparse.Namespace:
    """Parse input and output paths provided via the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-file", type=Path)
    parser.add_argument("--class-metrics-output-file", type=Path)
    return parser.parse_args()


def parse_class_metric(
    value: object,
    column_name: str,
) -> dict[str, float]:
    """Parse one JSON object containing class-specific metric values."""
    if not isinstance(value, str):
        raise ValueError(f"Missing or invalid value in '{column_name}'.")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in '{column_name}': {value}") from error

    # Normalize class identifiers and metric values obtained from JSON
    return {
        str(class_id): float(metric)
        for class_id, metric in parsed.items()
    }


def main() -> None:
    args = parse_args()
    output_file = args.output_file or args.input_file.with_name("summary_aggregated.csv")
    class_metrics_output_file = (
        args.class_metrics_output_file
        or output_file.with_name("precision_recall_aggregated.csv")
    )
    results = pd.read_csv(args.input_file, sep=";", decimal=",")

    # Aggregate all scalar metrics by dataset, model, and configuration
    group_columns = ["dataset", "model", "configuration"]

    # Automatically include every numeric result column except seed
    metric_columns = [
        column
        for column in results.select_dtypes(include="number").columns
        if column != "seed"
    ]

    grouped = results.groupby(group_columns, sort=True, dropna=False)
    aggregated = grouped.size().rename("number_of_runs").to_frame()

    # Calculate mean and sample standard deviation across seeds
    for metric in metric_columns:
        aggregated[f"{metric}_mean"] = grouped[metric].mean()
        aggregated[f"{metric}_std"] = grouped[metric].std(ddof=1)

    aggregated = aggregated.reset_index()

    # Convert JSON-encoded class metrics into long format
    class_metric_rows = []
    for _, row in results.iterrows():
        precision = parse_class_metric(row["precision_per_class"],"precision_per_class")
        recall = parse_class_metric(row["recall_per_class"],"recall_per_class")

        # Create one row for every class in current run
        for class_id in precision:
            class_metric_rows.append(
                {
                    "dataset": row["dataset"],
                    "model": row["model"],
                    "configuration": row["configuration"],
                    "seed": row["seed"],
                    "class_id": int(class_id),
                    "precision": precision[class_id],
                    "recall": recall[class_id]
                }
            )

    class_metrics = pd.DataFrame(class_metric_rows)
    class_group_columns = group_columns + ["class_id"]
    class_grouped = class_metrics.groupby(class_group_columns, sort=True, dropna=False)

    # Report means and sample standard deviations across seeds
    class_metrics_aggregated = class_grouped.agg(
        number_of_runs=("seed", "size"),
        precision_mean=("precision", "mean"),
        precision_std=("precision", lambda values: values.std(ddof=1)),
        recall_mean=("recall", "mean"),
        recall_std=("recall", lambda values: values.std(ddof=1))
    ).reset_index()

    output_file.parent.mkdir(parents=True, exist_ok=True)
    class_metrics_output_file.parent.mkdir(parents=True, exist_ok=True)
    aggregated.to_csv(output_file, sep=";", decimal=",", index=False)
    class_metrics_aggregated.to_csv(class_metrics_output_file, sep=";", decimal=",", index=False)

    print(f"Aggregated {len(results)} runs into {len(aggregated)} configurations.")
    print(f"Configuration results saved to: {output_file}")
    print(f"Class-specific precision and recall saved to: {class_metrics_output_file}")


if __name__ == "__main__":
    main()
