import argparse
import csv
from pathlib import Path

import flwr as fl
import torch
from flwr.server.strategy import FedAvg

from src.flsys.utils.config import TrainConfig


def parse_args():
    """Parse command line arguments for server configuration."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--min-fit", type=int, default=2)
    parser.add_argument("--min-available", type=int, default=2)
    return parser.parse_args()


def main():
    """Start Flower federated learning server with FedAvg strategy."""
    args = parse_args()
    cfg = TrainConfig(rounds=args.rounds)

    # create directory for saving metrics
    Path("runs").mkdir(parents=True, exist_ok=True)
    metrics_path = Path("runs/metrics.csv")

    # initialise metrics CSV file with headers
    if not metrics_path.exists():
        with open(metrics_path, "w") as f:
            csv.writer(f).writerow(["round", "accuracy"])

    def evaluate(server_round, parameters, config):
        """Placeholder evaluation function - to be implemented."""
        return 0.0, {}

    # configure federated averaging strategy
    strategy = FedAvg(
        min_fit_clients=args.min_fit,
        min_available_clients=args.min_available,
        evaluate_fn=evaluate,
    )

    # start Flower server
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=cfg.rounds),
    )


if __name__ == "__main__":
    torch.set_num_threads(4)  # limit CPU threads for stability
    main()
