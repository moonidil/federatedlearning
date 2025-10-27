import argparse

import flwr as fl
import torch

from src.flsys.core.model import MnistNet
from src.flsys.core.train import evaluate, train_one_epoch
from src.flsys.data.mnist import get_partition_loaders
from src.flsys.utils.config import TrainConfig


def parse_args():
    """Parse command line arguments for client configuration."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--client-id", type=int, required=True)
    return parser.parse_args()


def main():
    """Start Flower client for federated learning."""
    args = parse_args()
    cfg = TrainConfig()
    device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # load client specific data partition
    train_loader, test_loader, n_samples = get_partition_loaders(
        client_id=args.client_id,
        n_clients=cfg.clients,
        alpha=cfg.non_iid_alpha,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
    )

    # initialise model
    model = MnistNet().to(device)

    class Client(fl.client.NumPyClient):
        """Flower client implementing federated learning interface."""

        def get_parameters(self, config):
            """Extract model parameters as NumPy arrays."""
            return [v.cpu().numpy() for _, v in model.state_dict().items()]

        def fit(self, parameters, config):
            """Train model on client data and return updated parameters."""
            # load server parameters into model
            state_dict = model.state_dict()
            for key, value in zip(state_dict.keys(), parameters, strict=True):
                state_dict[key] = torch.tensor(value)
            model.load_state_dict(state_dict, strict=True)

            # local training
            for _ in range(cfg.epochs):
                train_one_epoch(model, train_loader, device, cfg.lr, cfg.momentum)

            return self.get_parameters({}), n_samples, {}

        def evaluate(self, parameters, config):
            """Evaluate model on client test data."""
            # load server parameters into model
            state_dict = model.state_dict()
            for key, value in zip(state_dict.keys(), parameters, strict=True):
                state_dict[key] = torch.tensor(value)
            model.load_state_dict(state_dict, strict=True)

            # calculate accuracy
            accuracy = evaluate(model, test_loader, device)
            return 0.0, len(test_loader.dataset), {"accuracy": accuracy}

    # start Flower client
    fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=Client())


if __name__ == "__main__":
    main()
