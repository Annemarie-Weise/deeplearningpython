"""Prepare the CiteSeer dataset for neighborhood-based node classification.

 1) Load CiteSeer through PyTorch Geometric
 2) Convert the citation graph to undirected graph
 3) Constructs neighborhood index set for every node
 4) Store prepared data for use by all model implementations
"""

import argparse
from pathlib import Path

import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import ToUndirected

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
CITESEER_DIR = DATA_DIR / "CiteSeer"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=CITESEER_DIR)
    args = parser.parse_args()

    # Load CiteSeer and convert directed citation edges to undirected edges
    # -> provided split masks are not stored because
    # -> shared training pipeline creates own stratified train/validation/test splits
    dataset = Planetoid(
        root=str(args.data_dir),
        name="CiteSeer",
        split="public",
        transform=ToUndirected(),
    )
    dataset = Planetoid(
        root=str(args.data_dir),
        name="CiteSeer",
        split="public",
        transform=ToUndirected()
    )

    graph = dataset[0]
    features = graph.x.float()
    labels = graph.y
    edge_index = graph.edge_index

    # Create one neighborhood set for every node
    source_nodes = edge_index[0]
    target_nodes = edge_index[1]
    neighbor_indices = []
    for node_index in range(graph.num_nodes):
        node_neighbors = []
        for edge_position in range(len(source_nodes)):
            # Check whether the edge starts at the current node
            if source_nodes[edge_position].item() == node_index:
                neighbor = target_nodes[edge_position].item()

                # Ignore self-loops and duplicated neighbors
                if neighbor != node_index and neighbor not in node_neighbors:
                    node_neighbors.append(neighbor)

        neighbor_indices.append(
            torch.tensor(node_neighbors, dtype=torch.int64)
        )

    # Store the common dataset representation
    data = {
        "features": features,
        "labels": labels,
        "edge_index": edge_index,
        "neighbor_indices": neighbor_indices,
        "num_classes": dataset.num_classes,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_file = args.output_dir / "citeseer_dataset.pt"
    torch.save(data, dataset_file)

    print("\nCiteseer summary")
    print("-----------------")
    print(f"nodes: {graph.num_nodes}")
    print(f"edges: {graph.num_edges}")
    print(f"features per node: {graph.num_node_features}")
    print(f"\ndataset saved as: {dataset_file}")


if __name__ == "__main__":
    main()
