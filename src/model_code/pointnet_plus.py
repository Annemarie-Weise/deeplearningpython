"""Implement, train, and evaluate PointNet++ for node classification.

 1) Model treats every prepared node neighborhood as unordered point set
 2) Uses deterministic farthest point sampling, k-nearest-neighbor grouping, two hierarchical set-abstraction levels
    and global PointNet aggregation
 3) Combine resulting neighborhood representation with the current-node features before classification
 4) Save best validation checkpoint
"""

import argparse
from pathlib import Path

import torch
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

from torch import nn

# Reuse the same dataset format and split logic as other models
from deepset import create_splits, load_dataset, select_device, train_model, print_additional_results

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = PROJECT_ROOT / "results" / "standalone_checkpoints"


def farthest_point_sample(coordinates, num_centroids):
    """Select centroids using deterministic farthest point sampling.

    The point farthest from the set mean is selected first. Each subsequent
    centroid maximizes its minimum distance to the previously selected centroids.
    """
    num_points = coordinates.shape[0]
    if num_points == 0:
        raise ValueError("FPS requires at least one point.")
    if num_centroids < 1:
        raise ValueError("num_centroids must be at least 1.")

    # A point set cannot provide more centroids than it contains points
    num_centroids = min(num_centroids, num_points)
    centroid_indices = torch.empty(
        num_centroids,
        dtype=torch.long,
        device=coordinates.device
    )

    # Choose point farthest from set mean as deterministic start
    set_mean = coordinates.mean(dim=0, keepdim=True)
    farthest = ((coordinates - set_mean) ** 2).sum(dim=1).argmax()

    # Track each point's minimum squared distance to any selected centroid
    minimum_distances = torch.full(
        (num_points,),
        float("inf"),
        device=coordinates.device
    )

    for i_centroid in range(num_centroids):
        # Add the currently farthest point to the centroid set
        centroid_indices[i_centroid] = farthest
        centroid = coordinates[farthest].unsqueeze(0)

        # Squared distances preserve the ordering and avoid square roots
        distances = ((coordinates - centroid) ** 2).sum(dim=1)

        # Update distances to closest centroid selected so far
        minimum_distances = torch.minimum(minimum_distances, distances)
        farthest = minimum_distances.argmax()

    return centroid_indices


def query_knn(coordinates, centroid_coordinates, num_neighbors):
    """Return the indices of the nearest points for every centroid."""
    if num_neighbors < 1:
        raise ValueError("num_neighbors must be at least 1.")

    # Use all available points if requested neighborhood is larger
    num_neighbors = min(num_neighbors, coordinates.shape[0])

    # Calculate pairwise squared Euclidean distances
    #  -> Squaring does not change which points are nearest
    squared_distances = torch.cdist(centroid_coordinates,coordinates,p=2).square()

    # Select nearest points -> their internal ordering is not required
    return squared_distances.topk(k=num_neighbors, dim=1,largest=False,sorted=False).indices


class SharedMLP(nn.Module):
    """Point-wise MLP whose weights are shared across all points in a region."""

    def __init__(self, input_features, layer_features):
        super().__init__()

        # Build one Linear-ReLU pair for every requested output dimension
        layers = []
        previous_features = input_features

        for output_features in layer_features:
            layers.extend([nn.Linear(previous_features, output_features),nn.ReLU()])
            previous_features = output_features
        self.network = nn.Sequential(*layers)


    def forward(self, point_features):
        return self.network(point_features)


class PointNetSetAbstraction(nn.Module):
    """
    One PointNet++ set-abstraction level:
    sampling -> grouping -> local PointNet -> max pooling.
    """

    def __init__(
        self,
        num_centroids,
        num_neighbors,
        coordinate_features,
        input_features,
        mlp_features
    ):
        super().__init__()

        if num_centroids < 1:
            raise ValueError("num_centroids must be at least 1.")
        if num_neighbors < 1:
            raise ValueError("num_neighbors must be at least 1.")

        self.num_centroids = num_centroids
        self.num_neighbors = num_neighbors

        # Each grouped point contains relative coordinates and point features
        self.local_pointnet = SharedMLP(coordinate_features + input_features,mlp_features)


    def forward(
        self,
        metric_coordinates,
        local_coordinates,
        point_features
    ):
        """Create one feature vector for every sampled centroid."""
        # FPS and kNN are discrete index-selection operations -> they do not require an autograd graph
        with torch.no_grad():
            centroid_indices = farthest_point_sample(metric_coordinates.detach(), self.num_centroids)
            centroid_metric_coordinates = metric_coordinates.detach()[centroid_indices]
            group_indices = query_knn(metric_coordinates.detach(),centroid_metric_coordinates, self.num_neighbors)

        # Retain selected centroid coordinates for next level
        new_metric_coordinates = metric_coordinates[centroid_indices]
        new_local_coordinates = local_coordinates[centroid_indices]

        # Gather local-coordinate and feature vectors of every group
        grouped_local_coordinates = local_coordinates[group_indices]
        grouped_point_features = point_features[group_indices]

        # Express grouped coordinates relative to their respective centroid
        relative_coordinates = (grouped_local_coordinates - new_local_coordinates.unsqueeze(1))

        # Combine relative coordinates with corresponding point features
        grouped_input = torch.cat((relative_coordinates, grouped_point_features),dim=-1)

        # Encode every grouped point and pool over neighborhood dimension
        encoded_groups = self.local_pointnet(grouped_input)
        new_point_features = encoded_groups.max(dim=1).values

        return new_metric_coordinates, new_local_coordinates, new_point_features


class GlobalPointNetAbstraction(nn.Module):
    """Aggregate the final set into one permutation-invariant feature vector."""

    def __init__(
        self,
        coordinate_features,
        input_features,
        mlp_features
    ):
        super().__init__()

        # Each point is represented by relative coordinates and point features
        self.global_pointnet = SharedMLP(coordinate_features + input_features, mlp_features)


    def forward(self, local_coordinates, point_features):
        """Encode and globally aggregate all remaining points."""
        # Use mean coordinate as reference center of complete set
        centroid = local_coordinates.mean(dim=0, keepdim=True)

        # Express every coordinate relative to global reference center
        relative_coordinates = local_coordinates - centroid

        # Combine relative coordinates with corresponding point features
        pointnet_input = torch.cat((relative_coordinates, point_features), dim=-1)

        # Apply shared MLP independently to every remaining point
        encoded_points = self.global_pointnet(pointnet_input)

        # Pool over all points to obtain one permutation-invariant vector
        return encoded_points.max(dim=0).values


class PointNetPlusPlusClassifier(nn.Module):
    """Classify nodes using a PointNet++ encoder over their neighborhoods.

    The raw neighbor features define the metric used by deterministic FPS and
    kNN grouping. Learned projections provide compact local coordinates and
    point features. Two hierarchical set-abstraction levels and one global
    abstraction produce a neighborhood vector, which is concatenated with the
    current-node features before classification.
    """

    def __init__(
        self,
        node_dim,
        internal_features,
        n_class,
        coordinate_features=16,
        first_centroids=16,
        first_neighbors=8,
        second_centroids=4,
        second_neighbors=4
    ):
        super().__init__()

        if coordinate_features < 1:
            raise ValueError("coordinate_features must be at least 1.")
        if internal_features < 2:
            raise ValueError("internal_features must be at least 2.")
        self.internal_features = internal_features

        # Raw node features define metric for FPS and kNN
        # -> This projection creates compact coordinates used to calculate relative positions
        self.coordinate_projection = nn.Linear(node_dim, coordinate_features, bias=False)

        # Independently project every neighbor into point-feature space
        self.input_projection = nn.Sequential(nn.Linear(node_dim, internal_features), nn.ReLU())

        # Use smaller intermediate representation in first local PointNet
        reduced_features = max(8, internal_features // 2)

        # First abstraction level: sample and encode local regions of input points
        self.set_abstraction_1 = PointNetSetAbstraction(
            num_centroids=first_centroids,
            num_neighbors=first_neighbors,
            coordinate_features=coordinate_features,
            input_features=internal_features,
            mlp_features=[reduced_features,reduced_features,internal_features]
        )

        # Second abstraction level: group and encode first-level centroids
        self.set_abstraction_2 = PointNetSetAbstraction(
            num_centroids=second_centroids,
            num_neighbors=second_neighbors,
            coordinate_features=coordinate_features,
            input_features=internal_features,
            mlp_features=[internal_features,internal_features, internal_features]
        )

        # Aggregate all remaining centroid features into one neighborhood vector
        self.global_abstraction = GlobalPointNetAbstraction(
            coordinate_features=coordinate_features,
            input_features=internal_features,
            mlp_features=[internal_features,internal_features,internal_features]
        )

        # Classify concatenated current-node and neighborhood features
        self.rho = nn.Sequential(
            nn.Linear(node_dim + internal_features, internal_features),
            nn.ReLU(),
            nn.Linear(internal_features, n_class)
        )


    def encode_neighborhood(self, neighborhood_features):
        """Encode one non-empty neighborhood into one fixed-size vector."""
        # Keep original features as metric coordinates for FPS and kNN
        metric_coordinates = neighborhood_features

        # Create compact coordinates for relative-position encoding
        local_coordinates = self.coordinate_projection(neighborhood_features)

        # Create learned point features processed by local PointNets
        point_features = self.input_projection(neighborhood_features)

        # Sample, group, and encode first level of local regions
        (metric_coordinates, local_coordinates, point_features) = self.set_abstraction_1(
            metric_coordinates,
            local_coordinates,
            point_features
        )

        # Repeat set abstraction on centroids produced by first level
        (metric_coordinates,local_coordinates, point_features) = self.set_abstraction_2(
            metric_coordinates,
            local_coordinates,
            point_features
        )

        # Aggregate all remaining centroid features into one neighborhood vector
        return self.global_abstraction(local_coordinates, point_features)


    def forward(self, node_features, neighbor_indices, node_indices):
        """Compute class logits for the requested nodes."""
        predictions = []

        for i_node in node_indices.tolist():
            # Retrieve prepared neighborhood of current node
            neighbors = torch.as_tensor(neighbor_indices[i_node], dtype=torch.long, device=node_features.device)

            if len(neighbors) > 0:
                # Encode complete neighborhood using PointNet++
                neighborhood_vector = self.encode_neighborhood(node_features[neighbors])
            else:
                # Represent empty neighborhood by a zero vector
                neighborhood_vector = torch.zeros(self.internal_features,device=node_features.device)

            # Combine current-node and neighborhood representations
            combined = torch.cat(tensors=(node_features[i_node], neighborhood_vector))
            predictions.append(self.rho(combined))

        return torch.stack(predictions)


def main():
    """Configure, train, evaluate, and save one PointNet++ model."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--internal-features", type=int, default=128)
    parser.add_argument("--coordinate-features", type=int, default=16)
    parser.add_argument("--first-centroids", type=int, default=16)
    parser.add_argument("--first-neighbors", type=int, default=8)
    parser.add_argument("--second-centroids", type=int, default=4)
    parser.add_argument("--second-neighbors", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-dir", type=Path, default=CHECKPOINT_DIR)
    args = parser.parse_args()

    # Seed random number generators for reproducible model initialization
    torch.manual_seed(args.seed)
    device = select_device(args.device)

    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    # Load prepared dataset and create reproducible data splits
    features, labels, neighbors, dataset_file, dataset_name = load_dataset(args.dataset_path)
    splits = create_splits(labels, args.seed)

    features = features.to(device)
    labels = labels.to(device)

    # Construct PointNet++ with the selected hierarchy configuration
    model = PointNetPlusPlusClassifier(
        node_dim=features.shape[1],
        internal_features=args.internal_features,
        n_class=int(labels.max().item()) + 1,
        coordinate_features=args.coordinate_features,
        first_centroids=args.first_centroids,
        first_neighbors=args.first_neighbors,
        second_centroids=args.second_centroids,
        second_neighbors=args.second_neighbors
    ).to(device)

    # Train model and restore best validation checkpoint
    (test_loss, test_accuracy, best_epoch, best_validation_accuracy, test_results) = train_model(
        model=model,
        features=features,
        labels=labels,
        neighbors=neighbors,
        splits=splits,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=device
    )

    # Save best model state together with its configuration and results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_file = args.output_dir / f"pointnet_plus_plus_{dataset_name}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "splits": splits,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "best_epoch": best_epoch,
            "best_validation_accuracy": best_validation_accuracy,
            "internal_features": args.internal_features,
            "coordinate_features": args.coordinate_features,
            "first_centroids": args.first_centroids,
            "first_neighbors": args.first_neighbors,
            "second_centroids": args.second_centroids,
            "second_neighbors": args.second_neighbors,
            "detailed_results": test_results
        },
        model_file
    )

    print("\nPointNet++ summary")
    print("------------------")
    print(f"dataset: {dataset_file}")
    print(f"test loss: {test_loss:.4f}")
    print(f"test accuracy: {test_accuracy:.2%}")
    print_additional_results(test_results)
    print(f"model saved as: {model_file}")


if __name__ == "__main__":
    main()