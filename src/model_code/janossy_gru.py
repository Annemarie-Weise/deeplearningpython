"""Implement, train, and evaluate Janossy Pooling using a GRU.

 1) Model encodes randomly ordered node neighborhoods with a GRU
 2) Training uses one sampled permutation per neighborhood
 3) Validation and testing average predictions over multiple sampled permutations
 4) Save best validation checkpoint and detailed test metrics after training
"""

import argparse
import copy
import time
from pathlib import Path

import torch
from torch import nn

# Reuse the same dataset format and split logic as other models
from deepset import build_classification_metrics, create_splits, load_dataset, print_additional_results, select_device, \
    synchronize_device

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = PROJECT_ROOT / "results" / "standalone_checkpoints"


class JanossyGRUClassifier(nn.Module):
    """Classify nodes using GRU-based approximate Janossy pooling.

    During training, each neighborhood is encoded using one randomly sampled
    permutation. During inference, class probabilities are averaged over
    multiple random permutations.
    """

    def __init__(self,
        node_dim,
        internal_features,
        n_class,
        inference_permutations=20
    ):
        super().__init__()
        if inference_permutations < 1:
            raise ValueError("inference_permutations must be at least 1.")
        self.internal_features = internal_features

        # Default number of sampled permutations used during evaluation
        self.inference_permutations = inference_permutations

        # Maps every neighbor feature vector to GRU input space
        self.input_projection = nn.Sequential(nn.Linear(node_dim, internal_features), nn.ReLU())

        # Use final hidden state as permutation-sensitive encoding f*
        self.gru = nn.GRU(input_size=internal_features, hidden_size=internal_features, batch_first=True)

        # Classify concatenated current-node and neighborhood features
        self.rho = nn.Sequential(
            nn.Linear(node_dim + internal_features, internal_features),
            nn.ReLU(),
            nn.Linear(internal_features, n_class)
        )


    def encode_random_permutation(self, node_features, neighbors):
        """Encode one random ordering of a non-empty neighborhood."""
        # Sample a new ordering because GRU is permutation sensitiv
        permutation = torch.randperm(len(neighbors),device=node_features.device)
        permuted_neighbors = neighbors[permutation]

        # Project ordered neighbors and add batch dimension for the GRU
        # Shape: [batch_size=1, sequence_length, internal_features]
        neighbor_sequence = self.input_projection(node_features[permuted_neighbors]).unsqueeze(0)

        # Final hidden state represents complete ordered neighborhood
        _, final_hidden = self.gru(neighbor_sequence)

        # Select final GRU layer and single batch element
        return final_hidden[-1, 0]


    def forward(self, node_features, neighbor_indices, node_indices):
        """Compute class logits using one sampled permutation per neighborhood."""
        predictions = []

        for i_node in node_indices.tolist():
            # Retrieve prepared neighborhood of current node
            neighbors = torch.as_tensor(neighbor_indices[i_node], dtype=torch.long, device=node_features.device)

            if len(neighbors) > 0:
                # Encode one randomly ordered version of the neighborhood
                neighborhood_vector = self.encode_random_permutation(node_features, neighbors)
            else:
                # Represent an empty neighborhood by zero vector
                neighborhood_vector = torch.zeros(self.internal_features, device=node_features.device)

            # Combine current-node and neighborhood representations
            combined = torch.cat((node_features[i_node], neighborhood_vector))
            predictions.append(self.rho(combined))

        return torch.stack(predictions)


    @torch.no_grad()
    def predict_log_probabilities(
        self,
        node_features,
        neighbor_indices,
        node_indices,
        num_permutations=None
    ):
        """
        Average class probabilities over several random permutations.

        Log-probabilities are returned so that NLLLoss can be used directly.
        """
        self.eval()

        # Use configured default unless a different number is requested
        if num_permutations is None:
            num_permutations = self.inference_permutations

        if num_permutations < 1:
            raise ValueError("num_permutations must be at least 1.")
        sampled_probabilities = []
        for _ in range(num_permutations):
            # Each forward pass samples new ordering for every neighborhood
            logits = self.forward(node_features, neighbor_indices, node_indices)

            # Convert logits to probabilities before averaging permutations
            sampled_probabilities.append(torch.softmax(logits, dim=1))

        # Average across the sampled permutations
        mean_probabilities = torch.stack(sampled_probabilities, dim=0).mean(dim=0)

        # Avoid log(0) and return values that can be passed to NLLLoss
        return torch.log(mean_probabilities.clamp_min(1e-12))


def evaluate(
    model,
    features,
    labels,
    neighbors,
    indices,
    num_permutations
):
    """Evaluate loss and accuracy using permutation-averaged predictions."""
    model.eval()

    with torch.no_grad():
        # Average predictions across requested number of permutations
        log_probabilities = model.predict_log_probabilities(
            features,
            neighbors,
            indices,
            num_permutations=num_permutations
        )
        targets = labels[indices.to(labels.device)]

        # Model returns log-probabilities, so NLLLoss is used
        loss = nn.NLLLoss()(log_probabilities, targets).item()
        accuracy = (log_probabilities.argmax(dim=1) == targets).float().mean().item()

    return loss, accuracy


def evaluate_test_detailed(
    model,
    features,
    labels,
    neighbors,
    indices,
    num_permutations
):
    """Evaluate detailed test metrics and measure averaged inference time."""
    model.eval()

    # Synchronize before timing so pending CUDA operations are excluded
    synchronize_device(features.device)
    inference_start = time.perf_counter()

    with torch.no_grad():
        # Measured inference includes all sampled permutation passes
        log_probabilities = model.predict_log_probabilities(
            features,
            neighbors,
            indices,
            num_permutations=num_permutations
        )

    # Wait for inference to finish before stopping CUDA timer
    synchronize_device(features.device)
    inference_time_seconds = time.perf_counter() - inference_start

    # Metric calculations are intentionally excluded from inference time
    targets = labels[indices.to(labels.device)]
    predictions = log_probabilities.argmax(dim=1)
    test_loss = nn.NLLLoss()(log_probabilities, targets).item()

    return build_classification_metrics(
        targets=targets,
        predictions=predictions,
        neighbors=neighbors,
        indices=indices,
        test_loss=test_loss,
        inference_time_seconds=inference_time_seconds,
        num_classes=int(labels.max().item()) + 1
    )


def train_model(
    model,
    features,
    labels,
    neighbors,
    splits,
    epochs,
    learning_rate,
    inference_permutations,
    device
):
    """
    Train using one sampled ordering per neighborhood and training epoch.

    Validation averages predictions across multiple permutations. The model
    state with the highest validation accuracy is restored before testing.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    train_indices = splits["train"]

    # Track checkpoint with highest validation accuracy
    best_validation_accuracy = float("-inf")
    best_epoch = 0
    best_model_state = None

    # Measure complete training loop, including validation after each epoch
    synchronize_device(device)
    training_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        # Forward samples one new ordering for every training neighborhood
        logits = model(features, neighbors, train_indices)
        targets = labels[train_indices.to(device)]
        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()

        # Validation averages predictions over several sampled permutations
        _, validation_accuracy = evaluate(
            model=model,
            features=features,
            labels=labels,
            neighbors=neighbors,
            indices=splits["validation"],
            num_permutations=inference_permutations
        )

        # Store independent copy because model parameters change afterwards
        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())

        print(
            f"epoch {epoch:3d} | "
            f"loss: {loss.item():.4f} | "
            f"validation accuracy: {validation_accuracy:.2%}"
        )

    # Wait for pending CUDA operations before stopping timer
    synchronize_device(device)
    training_time_seconds = time.perf_counter() - training_start

    if best_model_state is None:
        raise RuntimeError("No validation checkpoint created.")

    # Restore best validation checkpoint instead of testing final epoch
    model.load_state_dict(best_model_state)
    print(
        f"Using model from epoch {best_epoch} "
        f"with validation accuracy {best_validation_accuracy:.2%}."
    )

    # Evaluate restored model once on independent test split
    test_results = evaluate_test_detailed(
        model=model,
        features=features,
        labels=labels,
        neighbors=neighbors,
        indices=splits["test"],
        num_permutations=inference_permutations
    )

    # Add checkpoint, model-complexity, and timing information
    test_results["best_epoch"] = best_epoch
    test_results["best_validation_accuracy"] = best_validation_accuracy
    test_results["trainable_parameters"] = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    test_results["training_time_seconds"] = training_time_seconds

    return (
        test_results["test_loss"],
        test_results["test_accuracy"],
        best_epoch,
        best_validation_accuracy,
        test_results
    )


def main():
    """Configure, train, evaluate, and save one Janossy GRU model."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--internal-features", type=int, default=128)
    parser.add_argument("--inference-permutations", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-dir", type=Path,default=CHECKPOINT_DIR)
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

    # Construct the GRU-based approximate Janossy Pooling model
    model = JanossyGRUClassifier(
        node_dim=features.shape[1],
        internal_features=args.internal_features,
        n_class=int(labels.max().item()) + 1,
        inference_permutations=args.inference_permutations
    ).to(device)

    # Train model and restore best validation checkpoint
    (test_loss,test_accuracy, best_epoch,best_validation_accuracy,test_results) = train_model(
        model=model,
        features=features,
        labels=labels,
        neighbors=neighbors,
        splits=splits,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        inference_permutations=args.inference_permutations,
        device=device
    )

    # Save best model state together with its configuration and results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_file = args.output_dir / f"janossy_gru_{dataset_name}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "splits": splits,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "internal_features": args.internal_features,
            "inference_permutations": args.inference_permutations,
            "detailed_results": test_results
        },
        model_file
    )

    print("\nJanossy GRU summary")
    print("-------------------")
    print(f"dataset: {dataset_file}")
    print(f"inference permutations: {args.inference_permutations}")
    print(f"test loss: {test_loss:.4f}")
    print(f"test accuracy: {test_accuracy:.2%}")
    print_additional_results(test_results)
    print(f"model saved as: {model_file}")


if __name__ == "__main__":
    main()