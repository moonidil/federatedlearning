from pydantic import BaseModel


class TrainConfig(BaseModel):
    """Configuration for federated learning training parameters."""

    batch_size: int = 64  # local batch size to process before updating weights
    epochs: int = 1  # Number of local epochs per client
    lr: float = 0.01  # Learning rate
    momentum: float = 0.9  # SGD momentum
    rounds: int = 5  # Number of communication rounds
    clients: int = 3  # Number of clients participating
    non_iid_alpha: float = 0.5  # Controls data heterogeneity across clients
    seed: int = 42
    device: str | None = None
