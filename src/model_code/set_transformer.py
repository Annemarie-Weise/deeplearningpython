import argparse
from pathlib import Path

import torch
from torch import nn

from deepset import create_splits, load_dataset, train_model, select_device, print_additional_results

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = PROJECT_ROOT / "results" / "standalone_checkpoints"

class MultiheadAttentionBlock(nn.Module):
    """Apply multihead attention, residual connections, and a row-wise feedforward network."""

    def __init__(self, internal_features, num_heads):
        super().__init__()

        # Multihead attention preserves internal feature dimension
        self.attention = nn.MultiheadAttention(
            embed_dim=internal_features,
            num_heads=num_heads,
            batch_first=True,
        )

        # Normalize after attention and feedforward residual connections
        self.normalization_1 = nn.LayerNorm(internal_features)
        # Second norm after feedfordward
        self.normalization_2 = nn.LayerNorm(internal_features)

        # Apply same feedforward network independently to every set element
        self.row_feedforward = nn.Sequential(
            nn.Linear(internal_features, internal_features),
            nn.ReLU(),
            nn.Linear(internal_features, internal_features),
        )


    def forward(self, query, key_value):
        """Apply attention followed by two residual and normalization steps."""
        # Use key_value as both keys and values in attention operation
        attention_output, _ = self.attention(
            query,
            key_value,
            key_value,
            need_weights=False,
        )

        # First residual connection: add  attention output to queries
        hidden = self.normalization_1(query + attention_output)
        # Second residual connection: add row-wise feedforward output
        return self.normalization_2(hidden + self.row_feedforward(hidden))


class SetAttentionBlock(nn.Module):
    """Apply permutation-equivariant self-attention to the elements of a set."""

    def __init__(self, internal_features, num_heads):
        super().__init__()
        self.mab = MultiheadAttentionBlock(internal_features, num_heads)


    def forward(self, set_features):
        return self.mab(set_features, set_features)


class InducedSetAttentionBlock(nn.Module):
    """Approximate set self-attention using a fixed number of learned inducing points."""

    def __init__(self, internal_features,num_heads,num_inducing_points,):
        super().__init__()

        # Learn one shared set of inducing points
        self.inducing_points = nn.Parameter(torch.empty(1, num_inducing_points, internal_features))

        # Initialize inducing points following the authors' implementation
        nn.init.xavier_uniform_(self.inducing_points)

        # First attend from inducing points to the input set
        self.inducing_attention = MultiheadAttentionBlock(internal_features, num_heads)
        # Then attend from input set to induced representation
        self.set_attention = MultiheadAttentionBlock(internal_features,num_heads)


    def forward(self, set_features):
        # Use same learned inducing points for every set in the batch
        inducing_points = self.inducing_points.expand(set_features.shape[0], -1, -1)

        # H = MAB(I, X): summarize the input through the inducing points
        induced_features = self.inducing_attention(inducing_points,set_features)

        # ISAB(X) = MAB(X, H): return one updated vector per input element
        return self.set_attention(set_features, induced_features)


class PoolingByMultiheadAttention(nn.Module):
    """Aggregate a variable-size set using learned seed vectors."""

    def __init__(self, internal_features, num_heads, num_seeds=1):
        super().__init__()

        # Learn one or more seed vectors that act as pooling queries
        self.seed_vectors = nn.Parameter(torch.empty(1, num_seeds, internal_features))
        nn.init.xavier_uniform_(self.seed_vectors)

        self.mab = MultiheadAttentionBlock(internal_features, num_heads)


    def forward(self, set_features):
        batch_size = set_features.shape[0]

        # Use same learned seed vectors for every set in the batch
        seed_vectors = self.seed_vectors.expand(batch_size, -1, -1)

        # PMA(X) = MAB(S, X): each seed aggregates information from the set
        return self.mab(seed_vectors, set_features)


class SetTransformerClassifier(nn.Module):
    """Classify nodes by encoding and pooling their unordered neighborhoods."""

    def __init__(
        self,
        node_dim,
        internal_features,
        n_class,
        num_heads,
        num_blocks,
        attention_block,
        num_inducing_points
    ):
        super().__init__()
        self.internal_features = internal_features

        # Feature dimension must be divisible among all attention heads
        if internal_features % num_heads != 0:
            raise ValueError(
                "internal_features must be divisible by num_heads."
            )

        # Project every neighboring node into internal feature space
        self.input_projection = nn.Sequential(nn.Linear(node_dim, internal_features), nn.ReLU())

        # Construct encoder from either SAB or ISAB blocks
        if attention_block == "sab":
            blocks = [SetAttentionBlock(internal_features, num_heads)
                for _ in range(num_blocks)
            ]
        else:
            blocks = [InducedSetAttentionBlock(internal_features,num_heads,num_inducing_points,                )
                for _ in range(num_blocks)
            ]
        self.encoder = nn.ModuleList(blocks)

        # One seed vector produces one representation per neighborhood
        self.pooling = PoolingByMultiheadAttention(internal_features, num_heads, num_seeds=1)

        # Classify concatenated current-node and neighborhood features
        self.rho = nn.Sequential(
            nn.Linear(node_dim + internal_features, internal_features),
            nn.ReLU(),
            nn.Linear(internal_features, n_class)
        )


    def forward(self, node_features, neighbor_indices, node_indices):
        """Compute class logits for the requested nodes."""
        predictions = []

        for i_node in node_indices.tolist():
            # Retrieve prepared neighborhood of current node
            neighbors = torch.as_tensor(neighbor_indices[i_node],dtype=torch.long,device=node_features.device,)

            if len(neighbors) > 0:
                # Project neighbor features and add a batch dimension
                neighbor_set = self.input_projection(node_features[neighbors]).unsqueeze(0)

                # Model interactions between neighborhood elements
                encoded_set = neighbor_set
                for block in self.encoder:
                    encoded_set = block(encoded_set)

                # Pool variable-size encoded set into one vector
                neighborhood_vector = self.pooling(encoded_set).squeeze(0).squeeze(0)
            else:
                # Represent an empty neighborhood by a zero vector
                neighborhood_vector = torch.zeros(self.internal_features, device=node_features.device)

            # Preserve current node features separately from its neighborhood
            combined = torch.cat((node_features[i_node], neighborhood_vector)            )
            predictions.append(self.rho(combined))

        return torch.stack(predictions)


def main():
    """Configure, train, evaluate, and save one Set Transformer model."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--internal-features", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-blocks", type=int, default=2)
    parser.add_argument(
        "--attention-block",
        choices=["sab", "isab"],
        default="sab",
    )
    parser.add_argument("--num-inducing-points", type=int, default=16)
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

    # Load the prepared dataset and create reproducible data splits
    features, labels, neighbors, dataset_file, dataset_name = load_dataset(args.dataset_path)
    splits = create_splits(labels, args.seed)

    features = features.to(device)
    labels = labels.to(device)

    # Construct the selected SAB- or ISAB-based model configuration
    model = SetTransformerClassifier(
        node_dim=features.shape[1],
        internal_features=args.internal_features,
        n_class=int(labels.max().item()) + 1,
        num_heads=args.num_heads,
        num_blocks=args.num_blocks,
        attention_block=args.attention_block,
        num_inducing_points=args.num_inducing_points
    ).to(device)

    # Train model and restore checkpoint with best validation accuracy
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
    model_file = args.output_dir / f"set_transformer_{dataset_name}_{args.attention_block}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "splits": splits,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "attention_block": args.attention_block,
            "internal_features": args.internal_features,
            "num_heads": args.num_heads,
            "num_blocks": args.num_blocks,
            "num_inducing_points": args.num_inducing_points,
            "best_epoch": best_epoch,
            "best_validation_accuracy": best_validation_accuracy,
            "detailed_results": test_results
        },
        model_file
    )

    print("\nSet Transformer summary")
    print("-----------------------")
    print(f"dataset: {dataset_file}")
    print(f"attention block: {args.attention_block.upper()}")
    print(f"test loss: {test_loss:.4f}")
    print(f"test accuracy: {test_accuracy:.2%}")
    print_additional_results(test_results)
    print(f"model saved as: {model_file}")


if __name__ == "__main__":
    main()