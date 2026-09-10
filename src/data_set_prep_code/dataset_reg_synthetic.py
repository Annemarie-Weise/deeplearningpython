"""Generate a synthetic dataset with neighborhood-dependent binary labels.

 1) Samples two Gaussian classes
 2) Constructs Euclidean k-nearest-neighbor neighborhoods
 3) Modifies labels in locally ambiguous regions
 4) Stores the generated data/metadata for use by all model implementations
"""

import argparse
from pathlib import Path
import torch
from sklearn.neighbors import NearestNeighbors

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC_DIR = PROJECT_ROOT / "data" / "SyntheticDataset"

def generate_data_points(n, dim, sigma, distance, class_probability, seed):
    """Sample labeled points from two isotropic Gaussian distributions."""
    # Use local generator to make dataset reproducible
    generator = torch.Generator().manual_seed(seed)

    # Assign class 1 with probability class_probability
    random_numbers = torch.rand(n, generator=generator)
    labels = torch.where(
        random_numbers < class_probability,
        torch.tensor(1),
        torch.tensor(0)
    )

    # Place the two class means symmetrically around zero
    left_mean = torch.full((dim,), -distance / 2)
    right_mean = torch.full((dim,), distance / 2)

    # Select corresponding class mean for every generated point
    labels_as_column = labels[:, None]
    is_class_zero = labels_as_column == 0
    means = torch.where(is_class_zero, left_mean, right_mean)

    # Sample each point from normal distribution
    features = torch.normal(mean=means, std=sigma, generator=generator)
    return features.float(), labels


def modify_labels(labels, neighbors, epsilon):
    """Flip labels when the class-1 proportion of a neighborhood is close to 0.5."""
    neighbor_labels = labels[neighbors]
    neighbor_mean = neighbor_labels.float().mean(dim=1)

    flip_mask = torch.abs(neighbor_mean - 0.5) < epsilon

    modified = labels.clone()
    modified[flip_mask] = 1 - modified[flip_mask]
    return modified, neighbor_mean, flip_mask


def parser():
    parser = argparse.ArgumentParser(description="Synthetischen Datensatz erzeugen")
    parser.add_argument("--n", type=int, default=3327)
    parser.add_argument("--dim", type=int, default=8)
    parser.add_argument("--sigma", type=float, default=1.5)
    parser.add_argument("--distance", type=float, default=1.0)
    parser.add_argument("--class-probability", type=float, default=0.5)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--epsilon", type=float, default=0.11)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir",type=Path, default=SYNTHETIC_DIR)
    return parser.parse_args()


def main():
    args = parser()
    if args.n <= args.k:
        raise ValueError("n has to be bigger then k")
    if not 0 <= args.epsilon <= 0.5:
        raise ValueError("epsilon has to be between 0 and 0.5")
    if not 0 <= args.class_probability <= 1:
        raise ValueError("class_probability must be between 0 and 1")
    if args.sigma <= 0:
        raise ValueError("sigma has to be greater than 0.")

    features, original_labels = generate_data_points(
        n=args.n,
        dim=args.dim,
        sigma=args.sigma,
        distance=args.distance,
        class_probability=args.class_probability,
        seed=args.seed
    )

    # Construct one Euclidean k-nearest-neighbor set for every point
    # -> k + 1 neighbors, because each point initially finds itself
    neighbor_model = NearestNeighbors(
        n_neighbors=args.k + 1,
        metric="euclidean"
    )
    neighbor_model.fit(features.numpy())
    _, neighbor_indices = neighbor_model.kneighbors(features.numpy())

    # Remove the query point itself, which has distance zero
    neighbor_indices = neighbor_indices[:, 1:]

    # Convert the NumPy array to PyTorch tensor
    neighbor_indices = torch.tensor(neighbor_indices)

    modified_labels, neighbor_mean, flip_mask = modify_labels(
        original_labels,
        neighbor_indices,
        args.epsilon
    )

    # Store model inputs together with information about label modification
    data = {
        "features": features,
        "original_labels": original_labels,
        "modified_labels": modified_labels,
        "neighbor_indices": neighbor_indices,
        "neighbor_label_mean": neighbor_mean,
        "flip_mask": flip_mask,
        "parameters": vars(args),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_file = args.output_dir / "synthetic_dataset.pt"
    torch.save(data, dataset_file)

    print("\nsummary")
    print("---------------")
    print(f"n: {len(data['original_labels'])}")
    print(f"original class 0: {(data['original_labels'] == 0).sum().item()}")
    print(f"original class 1: {(data['original_labels'] == 1).sum().item()}")
    print(f"new class 0: {(data['modified_labels'] == 0).sum().item()}")
    print(f"new class 1: {(data['modified_labels'] == 1).sum().item()}")
    print(f"flipped Labels: {data['flip_mask'].sum().item()}")
    print(f"percental amount of flips: {data['flip_mask'].float().mean().item():.2%}")
    print(f"\nsummary saved as: {dataset_file}")


if __name__ == "__main__":
    main()
