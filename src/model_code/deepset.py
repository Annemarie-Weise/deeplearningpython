"""Train and evaluate Deep Sets for neighborhood-based classification.

 1) Define the Deep Sets classifier adapted to the Synthetic and CiteSeer neighborhood inputs
 2) Define shared dataset loading, data splitting, device selection, training, checkpoint selection, and detailed
    test-evaluation functions used by the other model implementations as well
 3) Save model state with the best validation accuracy and its evaluation results in a checkpoint
"""

import argparse
import time
from pathlib import Path

import torch
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch import nn
import copy

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = PROJECT_ROOT / "results" / "standalone_checkpoints"


class DeepSetClassifier(nn.Module):
    """Classify nodes using their features and an invariant neighbor sum."""

    def __init__(self, node_dim, internal_features, n_class):
        super().__init__()
        self.internal_features = internal_features

        # Encode every neighbor independently using the shared phi network
        self.phi = nn.Sequential(
            nn.Linear(node_dim, internal_features),
            # non linear activation function to cover non linear features
            nn.ReLU(),
            nn.Linear(internal_features, internal_features),
            nn.ReLU(),
        )

        # Classify the concatenated current-node and neighborhood features
        self.rho = nn.Sequential(
            nn.Linear(node_dim + internal_features, internal_features),
            nn.ReLU(),
            nn.Linear(internal_features, n_class)
        )

    def forward(self, node_dim, i_neighbors, i_node):
        """Compute class logits for the requested nodes."""
        predictions = []

        for i_node in i_node.tolist():
            # Retrieve the stored neighborhood indices of the current node
            neighbors = torch.as_tensor(
                i_neighbors[i_node],
                dtype=torch.long,
                device=node_dim.device,
            )

            # Apply shared phi network + sum resulting neighbor representations to obtain permutation-invariant vector
            if len(neighbors) > 0:
                neighbor_sum = self.phi(node_dim[neighbors]).sum(dim=0)
            else:
                # Represent an empty neighborhood by a zero vector
                neighbor_sum = torch.zeros(
                    self.internal_features,
                    device=node_dim.device,
                )
            # Combine the current node features with its neighborhood representation
            predictions.append(self.rho(torch.cat((node_dim[i_node], neighbor_sum))))

        return torch.stack(predictions)


def create_splits(labels, seed):
    """Creates reproducible stratified 80/10/10 splits."""
    indices = list(range(len(labels)))

    # Use 80% of data for training and reserve 20% for validation and test
    train_indices, remaining_indices = train_test_split(
        indices,
        # --> 0.8 training
        test_size=0.20,
        random_state=seed,
        stratify=labels.numpy()
    )

    # Divide remaining 20% equally into 10% validation and 10% test
    validation_indices, test_indices = train_test_split(
        remaining_indices,
        # 0.5 von 0.2 for val, so 0.1/0.1
        test_size=0.50,
        random_state=seed,
        stratify=labels.numpy()[remaining_indices]
    )

    return {
        "train": torch.tensor(train_indices),
        "validation": torch.tensor(validation_indices),
        "test": torch.tensor(test_indices)
    }


def load_dataset(dataset_path):
    """Load a prepared Synthetic or CiteSeer dataset."""
    dataset_file = Path(dataset_path)

    if not dataset_file.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_file}"
        )

    # Load dataset on CPU before moving tensors to selected device
    data = torch.load(
        dataset_file,
        map_location="cpu",
        weights_only=False,
    )

    features = data["features"].float()
    neighbors = data["neighbor_indices"]

    # Synthetic models predict neighborhood-dependent modified labels
    if "modified_labels" in data:
        labels = data["modified_labels"].long()
        dataset_name = "synthetic"
    # CiteSeer models predict the original node classes
    elif "labels" in data:
        labels = data["labels"].long()
        dataset_name = "citeseer"
    else:
        raise KeyError(
            "Dataset contains neither 'modified_labels' nor 'labels'."
        )

    return features, labels, neighbors, dataset_file, dataset_name


def evaluate(model, features, labels, neighbors, indices, criterion):
    """Return loss and accuracy for the nodes specified by indices."""
    # Disable training-specific behavior and gradient calculation
    model.eval()

    with torch.no_grad():
        logits = model(features, neighbors, indices)
        targets = labels[indices.to(labels.device)]
        loss = criterion(logits, targets).item()
        accuracy = (logits.argmax(dim=1) == targets).float().mean().item()

    return loss, accuracy


def synchronize_device(device):
    """Wait for pending CUDA operations to obtain accurate timing results."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def build_classification_metrics(
        targets,
        predictions,
        neighbors,
        indices,
        test_loss,
        inference_time_seconds,
        num_classes
):
    """Collect overall, per-class, degree-specific, and timing metrics."""
    # Move predictions and targets to CPU for scikit-learn metrics
    targets_cpu = targets.detach().cpu()
    predictions_cpu = predictions.detach().cpu()
    class_ids = list(range(num_classes))

    # Calculate precision, recall, and support separately for every class
    precision, recall, _, support = precision_recall_fscore_support(
        targets_cpu.numpy(),
        predictions_cpu.numpy(),
        labels=class_ids,
        zero_division=0
    )

    # Determine neighborhood size + prediction correctness of each evaluated test node
    test_node_indices = indices.detach().cpu().tolist()
    degrees = torch.tensor(
        [len(neighbors[node_index]) for node_index in test_node_indices],
        dtype=torch.long,
    )
    correct_predictions = predictions_cpu == targets_cpu

    # Group nodes by neighborhood size for the degree-specific evaluation
    degree_masks = {
        "0": degrees == 0,
        "1-2": (degrees >= 1) & (degrees <= 2),
        "3-5": (degrees >= 3) & (degrees <= 5),
        "6-10": (degrees >= 6) & (degrees <= 10),
        ">10": degrees > 10
    }

    # Calculate accuracy separately for every non-empty degree group
    accuracy_by_degree = {}
    for degree_group, mask in degree_masks.items():
        node_count = int(mask.sum().item())
        accuracy_by_degree[degree_group] = {
            "count": node_count,
            "accuracy": (
                correct_predictions[mask].float().mean().item()
                if node_count > 0
                else None
            )
        }

    return {
        "test_loss": float(test_loss),
        "test_accuracy": correct_predictions.float().mean().item(),
        "macro_f1": f1_score(
            targets_cpu.numpy(),
            predictions_cpu.numpy(),
            labels=class_ids,
            average="macro",
            zero_division=0
        ),
        "weighted_f1": f1_score(
            targets_cpu.numpy(),
            predictions_cpu.numpy(),
            labels=class_ids,
            average="weighted",
            zero_division=0
        ),
        "precision_per_class": {
            str(class_id): float(precision[class_id])
            for class_id in class_ids
        },
        "recall_per_class": {
            str(class_id): float(recall[class_id])
            for class_id in class_ids
        },
        "support_per_class": {
            str(class_id): int(support[class_id])
            for class_id in class_ids
        },
        "accuracy_by_degree": accuracy_by_degree,
        "inference_time_seconds": inference_time_seconds,
        "inference_time_per_node_ms": (
                inference_time_seconds * 1000 / max(len(targets_cpu), 1)
        )
    }


def evaluate_test_detailed(
        model,
        features,
        labels,
        neighbors,
        indices,
        criterion
):
    """Evaluate the model on the test split and collect detailed metrics."""
    model.eval()

    # Synchronize before and after forward pass so CUDA execution is completed within the measured interval
    synchronize_device(features.device)
    inference_start = time.perf_counter()
    with torch.no_grad():
        logits = model(features, neighbors, indices)
    synchronize_device(features.device)
    inference_time_seconds = time.perf_counter() - inference_start

    # Calculate predictions and loss outside the timed inference interval
    targets = labels[indices.to(labels.device)]
    predictions = logits.argmax(dim=1)
    test_loss = criterion(logits, targets).item()

    return build_classification_metrics(
        targets=targets,
        predictions=predictions,
        neighbors=neighbors,
        indices=indices,
        test_loss=test_loss,
        inference_time_seconds=inference_time_seconds,
        num_classes=int(labels.max().item()) + 1
    )


def print_additional_results(results):
    """Print detailed evaluation metrics in a human-readable format."""
    print(f"macro F1: {results['macro_f1']:.4f}")
    print(f"weighted F1: {results['weighted_f1']:.4f}")
    print(
        f"best validation accuracy: "
        f"{results['best_validation_accuracy']:.2%}"
    )
    print(f"best epoch: {results['best_epoch']}")
    print(f"trainable parameters: {results['trainable_parameters']:,}")
    print(f"training time: {results['training_time_seconds']:.4f} s")
    print(
        f"inference time for test set: "
        f"{results['inference_time_seconds']:.6f} s"
    )
    print(
        f"inference time per node: "
        f"{results['inference_time_per_node_ms']:.6f} ms"
    )

    print("\nPrecision and recall per class")
    print("--------------------------------")
    for class_id in results["precision_per_class"]:
        print(
            f"class {class_id}: "
            f"precision={results['precision_per_class'][class_id]:.4f} | "
            f"recall={results['recall_per_class'][class_id]:.4f} | "
            f"support={results['support_per_class'][class_id]}"
        )

    print("\nAccuracy by node degree")
    print("-----------------------")
    for degree_group, degree_results in results["accuracy_by_degree"].items():
        accuracy = degree_results["accuracy"]
        accuracy_text = "n/a" if accuracy is None else f"{accuracy:.2%}"
        print(
            f"degree {degree_group:>4}: "
            f"accuracy={accuracy_text} | "
            f"nodes={degree_results['count']}"
        )


def select_device(device_name):
    """Select CPU or CUDA based on the requested mode and availability."""
    if device_name == "cpu":
        print("Training on CPU.")
        return torch.device("cpu")
    if device_name == "cuda":
        if torch.cuda.is_available():
            print("GPU available: training on GPU.")
            return torch.device("cuda")
        print("GPU not available: training on CPU.")
        return torch.device("cpu")

    # In auto mode, prefer CUDA when available and otherwise use CPU
    if torch.cuda.is_available():
        print("GPU available: training on GPU.")
        return torch.device("cuda")

    print("GPU not available: training on CPU.")
    return torch.device("cpu")


def train_model(
        model,
        features,
        labels,
        neighbors,
        splits,
        epochs,
        learning_rate,
        device
):
    """Train the model, restore the best checkpoint, and evaluate it on test data."""
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    train_indices = splits["train"]

    # Track checkpoint with highest validation accuracy
    best_validation_accuracy = float("-inf")
    best_epoch = 0
    best_model_state = None

    # Measure complete training process, including validation
    synchronize_device(device)
    training_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(features, neighbors, train_indices)
        targets = labels[train_indices.to(device)]
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        # Evaluate after every epoch for checkpoint selection
        _, validation_accuracy = evaluate(
            model, features, labels, neighbors, splits["validation"], criterion, )

        # Store an independent copy whenever validation accuracy improves
        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())

        print(
            f"epoch {epoch:3d} | "
            f"loss: {loss.item():.4f} | "
            f"validation accuracy: {validation_accuracy:.2%}"
        )

    # Synchronize pending CUDA operations before stopping the timer
    synchronize_device(device)
    training_time_seconds = time.perf_counter() - training_start

    if best_model_state is None:
        raise RuntimeError("No validation checkpoint created.")

    # Restore checkpoint selected using validation data
    model.load_state_dict(best_model_state)
    print(
        f"Using model from epoch {best_epoch} "
        f"with validation accuracy {best_validation_accuracy:.2%}."
    )

    # Evaluate selected checkpoint once on held-out test split
    test_results = evaluate_test_detailed(
        model,
        features,
        labels,
        neighbors,
        splits["test"],
        criterion
    )

    # Add checkpoint, model-complexity, and training-time information
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
    """Train and evaluate Deep Sets and save the resulting checkpoint."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=Path, required=True, )
    parser.add_argument("--internal-features", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=CHECKPOINT_DIR )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", )
    args = parser.parse_args()

    # Seed model initialization and create reproducible dataset splits
    torch.manual_seed(args.seed)
    # Use gpu if possible
    device = select_device(args.device)

    # Load the common dataset representation and create an 80/10/10 spli
    features, labels, neighbors, dataset_file, dataset_name = load_dataset(args.dataset_path)
    splits = create_splits(labels, args.seed)

    features = features.to(device)
    labels = labels.to(device)

    # Infer input and output dimensions from loaded dataset
    model = DeepSetClassifier(
        node_dim=features.shape[1],
        internal_features=args.internal_features,
        n_class=int(labels.max().item()) + 1
    ).to(device)
    (test_loss, test_accuracy, best_epoch, best_validation_accuracy, test_results) = (
        # Train model and evaluate best validation checkpoint
        train_model(model=model,
                    features=features,
                    labels=labels,
                    neighbors=neighbors,
                    splits=splits,
                    epochs=args.epochs,
                    learning_rate=args.learning_rate,
                    device=device
                    ))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_file = args.output_dir / f"deepset_{dataset_name}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "splits": splits,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "best_epoch": best_epoch,
            "best_validation_accuracy": best_validation_accuracy,
            "detailed_results": test_results
        },
        model_file
    )
    print("\nDeep Sets summary")
    print("-----------------")
    print(f"dataset: {dataset_file}")
    print(f"test loss: {test_loss:.4f}")
    print(f"test accuracy: {test_accuracy:.2%}")
    print_additional_results(test_results)
    print(f"model saved as: {model_file}")


if __name__ == "__main__":
    main()
