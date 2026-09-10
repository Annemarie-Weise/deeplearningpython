"""Evaluate synthetic-dataset parameter combinations against CiteSeer.

 1) Generate synthetic datasets for a predefined parameter grid and multiple random seeds
 2) Evaluate standardized logistic-regression classifier using each element's features + mean neighborhood features on
    both Synthetic and CiteSeer
 3) Save validation and test metrics, label-flip rates, class distributions, and similarity scores in CSV

The script reports candidate configurations but does not generate or save the
final synthetic dataset used by the model implementations.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import ToUndirected

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
PARAMETER_RESULTS = PROJECT_ROOT / "results" / "synthetic_parameter_results.csv"

# Fixed settings for the CiteSeer reference and synthetic datasets
CITESEER_REFERENCE_SEED = 42
DIM = 8
CLASS_PROBABILITY = 0.5

# Parameter values evaluated for synthetic dataset
SIGMA_VALUES = [0.5, 1.0, 1.5]
K_VALUES = [3, 5, 10]
EPSILON_VALUES = [0.05, 0.11, 0.21]
# DISTANCE_VALUES control coordinate-wise separation of class means
DISTANCE_VALUES = [1.0, 2.0, 3.0]
# Seeds used to repeat each synthetic parameter configuration
SEEDS = [0, 1, 2]


def parse_args() -> argparse.Namespace:
    """Parse paths for the CiteSeer data and parameter-search results."""
    parser = argparse.ArgumentParser(description="Search synthetic parameters using an 80/10/10 split.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-file", type=Path, default=PARAMETER_RESULTS)

    return parser.parse_args()


def build_neighbor_lists(edge_index: torch.Tensor, num_nodes: int) -> list[torch.Tensor]:
    """Build sorted, duplicate-free neighbor lists from an edge index.

    Self-loops are excluded. For undirected neighborhoods, the provided
    edge index must already contain edges in both directions.
    """
    neighbor_sets = [set() for _ in range(num_nodes)]

    for source, target in edge_index.t().tolist():
        if source != target:
            neighbor_sets[source].add(target)

    return [
        torch.tensor(sorted(values), dtype=torch.long)
        for values in neighbor_sets
    ]


def neighbor_mean(
    features: torch.Tensor,
    neighbors: torch.Tensor | list[torch.Tensor]
) -> torch.Tensor:
    """Calculate the mean feature vector of every neighborhood.

    Fixed-size neighborhoods are represented by one index tensor, while
    variable-size neighborhoods are represented by a list of index tensors.
    Empty neighborhoods are represented by a zero vector.
    """
    # Synthetic dataset: all neighborhoods contain the same number of points
    if isinstance(neighbors, torch.Tensor):
        return features[neighbors].mean(dim=1)

    # CiteSeer: neighborhood sizes depend on the node degree
    means = []
    zero_vector = torch.zeros(features.shape[1], dtype=features.dtype)

    for node_neighbors in neighbors:
        if len(node_neighbors) == 0:
            means.append(zero_vector)
        else:
            means.append(features[node_neighbors].mean(dim=0))

    return torch.stack(means)


def make_stratified_masks(
    labels: torch.Tensor,
    seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create reproducible stratified 80/10/10 boolean masks."""
    labels_np = labels.detach().cpu().numpy()
    indices = np.arange(len(labels_np))

    # Separate 80% training data from the remaining 20%
    train_indices, remaining_indices = train_test_split(
        indices,
        test_size=0.20,
        stratify=labels_np,
        random_state=seed,
    )

    # Divide remaining data equally into validation and test sets
    validation_indices, test_indices = train_test_split(
        remaining_indices,
        test_size=0.50,
        stratify=labels_np[remaining_indices],
        random_state=seed + 1,
    )

    # Convert the three index sets into boolean masks
    train_mask = np.zeros(len(labels_np), dtype=bool)
    validation_mask = np.zeros(len(labels_np), dtype=bool)
    test_mask = np.zeros(len(labels_np), dtype=bool)

    train_mask[train_indices] = True
    validation_mask[validation_indices] = True
    test_mask[test_indices] = True

    return train_mask, validation_mask, test_mask


def evaluate(
    features: torch.Tensor,
    labels: torch.Tensor,
    neighbors: torch.Tensor | list[torch.Tensor],
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
    test_mask: np.ndarray,
    seed: int
) -> dict[str, float]:
    """Train one simple point-plus-neighborhood classifier."""
    mean_features = neighbor_mean(features, neighbors)

    # Each sample consists of its own features and the mean neighbor features
    x = torch.cat([features, mean_features], dim=1).detach().cpu().numpy()
    y = labels.detach().cpu().numpy()

    # Standardize inputs before applying balanced logistic regression
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            random_state=seed,
            class_weight="balanced",
            solver="lbfgs"
        )
    )

    # Fit both the scaler and classifier using only the training data
    model.fit(x[train_mask], y[train_mask])

    # Evaluate the fitted pipeline independently on validation and test data
    validation_prediction = model.predict(x[validation_mask])
    test_prediction = model.predict(x[test_mask])

    return {
        "validation_accuracy": float(
            accuracy_score(y[validation_mask], validation_prediction)
        ),
        "validation_macro_f1": float(
            f1_score(
                y[validation_mask],
                validation_prediction,
                average="macro",
                zero_division=0
            )
        ),
        "test_accuracy": float(accuracy_score(y[test_mask], test_prediction)),
        "test_macro_f1": float(
            f1_score(
                y[test_mask],
                test_prediction,
                average="macro",
                zero_division=0
            )
        )
    }


def chance_adjusted(value: float, number_of_classes: int) -> float:
    """Rescale a metric relative to the approximate chance level 1/C.

    The chance level is mapped to 0, while a perfect score remains 1.
    """
    chance = 1.0 / number_of_classes
    return (value - chance) / (1.0 - chance)


def load_citeseer(data_dir: Path):
    """Load CiteSeer and create neighborhoods and a stratified 80/10/10 split."""
    # Load graph and convert all citation edges to undirected edges
    # -> predefined public split masks are not used in this parameter search
    dataset = Planetoid(root=str(data_dir), name="CiteSeer", split="public", transform=ToUndirected())
    graph = dataset[0]

    # Construct graph-based neighborhoods and independent stratified masks
    neighbors = build_neighbor_lists(graph.edge_index, graph.num_nodes)
    train_mask, validation_mask, test_mask = make_stratified_masks(graph.y, seed=CITESEER_REFERENCE_SEED)

    return (
        graph,
        neighbors,
        train_mask,
        validation_mask,
        test_mask,
        dataset.num_classes
    )


def generate_synthetic(
    n: int,
    dim: int,
    sigma: float,
    distance: float,
    k: int,
    epsilon: float,
    seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
    """Generate one synthetic candidate dataset for the parameter search.

    Returns the features, modified labels, neighborhood indices, label-flip
    rate, and proportion of modified labels belonging to class 1.
    """
    generator = torch.Generator().manual_seed(seed)

    # Generate approximately balanced original binary labels
    original_labels = (
        torch.rand(n, generator=generator) < CLASS_PROBABILITY
    ).long()

    # Place class means symmetrically around zero
    # -> distance parameter controls their coordinate-wise separation
    left_mean = torch.full((dim,), -distance / 2)
    right_mean = torch.full((dim,), distance / 2)
    means = torch.where(
        original_labels[:, None] == 0,
        left_mean,
        right_mean,
    )

    # Sample each point from its class-specific Gaussian distribution
    features = torch.normal(
        mean=means,
        std=sigma,
        generator=generator,
    ).float()

    # Find k neighbors and exclude the query point returned at distance zero
    knn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    neighbor_indices = knn.fit(features.numpy()).kneighbors(
        features.numpy(),
        return_distance=False,
    )[:, 1:]
    neighbors = torch.tensor(neighbor_indices, dtype=torch.long)

    # Flip labels whose original neighborhood class proportion is close to 0.5
    neighbor_label_mean = original_labels[neighbors].float().mean(dim=1)
    flip_mask = torch.abs(neighbor_label_mean - 0.5) < epsilon
    modified_labels = original_labels.clone()
    modified_labels[flip_mask] = 1 - modified_labels[flip_mask]

    # Record statistics used to filter unsuitable parameter configurations
    flip_rate = flip_mask.float().mean().item()
    class_one_rate = modified_labels.float().mean().item()

    return features, modified_labels, neighbors, flip_rate, class_one_rate


def main() -> None:
    """Evaluate the synthetic parameter grid against a CiteSeer reference."""
    args = parse_args()

    # Establish the CiteSeer reference performance
    (
        graph,
        citeseer_neighbors,
        citeseer_train,
        citeseer_validation,
        citeseer_test,
        number_of_classes
    ) = load_citeseer(args.data_dir)
    citeseer_results = evaluate(
        features=graph.x.float(),
        labels=graph.y,
        neighbors=citeseer_neighbors,
        train_mask=citeseer_train,
        validation_mask=citeseer_validation,
        test_mask=citeseer_test,
        seed=CITESEER_REFERENCE_SEED
    )

    # Rescale the metrics relative to the approximate chance level
    # -> Synthetic and CiteSeer have different numbers of classes
    citeseer_validation_adjusted_accuracy = chance_adjusted(citeseer_results["validation_accuracy"], number_of_classes)
    citeseer_validation_adjusted_f1 = chance_adjusted(citeseer_results["validation_macro_f1"], number_of_classes)
    citeseer_test_adjusted_accuracy = chance_adjusted(citeseer_results["test_accuracy"],number_of_classes)
    citeseer_test_adjusted_f1 = chance_adjusted(citeseer_results["test_macro_f1"],number_of_classes)

    print("\nCiteSeer reference: stratified 80/10/10 split")
    print("--------------------------------------------")
    print(f"nodes: {graph.num_nodes}")
    print(f"train nodes: {citeseer_train.sum()}")
    print(f"validation nodes: {citeseer_validation.sum()}")
    print(f"test nodes: {citeseer_test.sum()}")
    print(f"validation accuracy: {citeseer_results['validation_accuracy']:.4f}")
    print(f"validation macro-F1: {citeseer_results['validation_macro_f1']:.4f}")
    print(f"test accuracy: {citeseer_results['test_accuracy']:.4f}")
    print(f"test macro-F1: {citeseer_results['test_macro_f1']:.4f}")

    # Construct the complete Cartesian product of parameter values
    settings = [{"sigma": sigma, "distance": distance, "k": k,  "epsilon": epsilon, }
        for sigma, distance, k, epsilon in itertools.product(
            SIGMA_VALUES,
            DISTANCE_VALUES,
            K_VALUES,
            EPSILON_VALUES
        )
    ]

    # Evaluate every parameter configuration across all synthetic seeds
    rows = []
    for setting in settings:
        run_results = []
        for seed in SEEDS:
            features, labels, neighbors, flip_rate, class_one_rate = generate_synthetic(
                n=graph.num_nodes,
                dim=DIM,
                sigma=setting["sigma"],
                distance=setting["distance"],
                k=setting["k"],
                epsilon=setting["epsilon"],
                seed=seed
            )
            train_mask, validation_mask, test_mask = make_stratified_masks(labels, seed=seed,)
            result = evaluate(
                features=features,
                labels=labels,
                neighbors=neighbors,
                train_mask=train_mask,
                validation_mask=validation_mask,
                test_mask=test_mask,
                seed=seed
            )

            # Adjust the binary-classification metrics to same reference scale
            validation_adjusted_accuracy = chance_adjusted(result["validation_accuracy"],number_of_classes=2,)
            validation_adjusted_f1 = chance_adjusted(
                result["validation_macro_f1"],
                number_of_classes=2
            )
            test_adjusted_accuracy = chance_adjusted(
                result["test_accuracy"],
                number_of_classes=2
            )
            test_adjusted_f1 = chance_adjusted(
                result["test_macro_f1"],
                number_of_classes=2
            )

            # Only validation results are used to rank parameter settings
            validation_similarity_score = (
                abs(
                    validation_adjusted_accuracy
                    - citeseer_validation_adjusted_accuracy
                )
                + abs(validation_adjusted_f1 - citeseer_validation_adjusted_f1)
            )

            # Reported only as an independent final check
            test_similarity_score = (
                abs(test_adjusted_accuracy - citeseer_test_adjusted_accuracy)
                + abs(test_adjusted_f1 - citeseer_test_adjusted_f1)
            )

            run_results.append(
                {
                    **result,
                    "flip_rate": flip_rate,
                    "class_one_rate": class_one_rate,
                    "validation_similarity_score": validation_similarity_score,
                    "test_similarity_score": test_similarity_score
                }
            )

        # Average all metrics and dataset statistics across the synthetic seeds
        rows.append(
            {
                **setting,
                "validation_accuracy_mean": np.mean(
                    [r["validation_accuracy"] for r in run_results]
                ),
                "validation_macro_f1_mean": np.mean(
                    [r["validation_macro_f1"] for r in run_results]
                ),
                "test_accuracy_mean": np.mean(
                    [r["test_accuracy"] for r in run_results]
                ),
                "test_macro_f1_mean": np.mean(
                    [r["test_macro_f1"] for r in run_results]
                ),
                "flip_rate_mean": np.mean(
                    [r["flip_rate"] for r in run_results]
                ),
                "class_one_rate_mean": np.mean(
                    [r["class_one_rate"] for r in run_results]
                ),
                "validation_similarity_score_mean": np.mean(
                    [r["validation_similarity_score"] for r in run_results]
                ),
                "test_similarity_score_mean": np.mean(
                    [r["test_similarity_score"] for r in run_results]
                )
            }
        )
    results = pd.DataFrame(rows)

    # Keep configurations with non-trivial flip rate and acceptable class balance
    valid_results = results[
        results["flip_rate_mean"].between(0.10, 0.30)
        & results["class_one_rate_mean"].between(0.35, 0.65)
    ]

    # Rank valid configurations and fall back to all results if none are valid
    if valid_results.empty:
        print("\nNo setting satisfied the flip-rate and class-balance filters.")
        ranked = results.sort_values("validation_similarity_score_mean")
    else:
        ranked = valid_results.sort_values("validation_similarity_score_mean")

    # Save all evaluated configurations, including those excluded by filters
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    results.sort_values("validation_similarity_score_mean").to_csv(
        args.output_file,
        index=False
    )

    columns = [
        "sigma",
        "distance",
        "k",
        "epsilon",
        "validation_accuracy_mean",
        "validation_macro_f1_mean",
        "test_accuracy_mean",
        "test_macro_f1_mean",
        "flip_rate_mean",
        "class_one_rate_mean",
        "validation_similarity_score_mean",
        "test_similarity_score_mean"
    ]
    print("\nBest synthetic settings")
    print("-----------------------")
    print(ranked[columns].head(10).to_string(index=False))
    print(f"\nAll results saved to: {args.output_file}")
    print(
        "Settings are ranked by validation_similarity_score_mean; "
        "the test score is not used for parameter selection."
    )


if __name__ == "__main__":
    main()
