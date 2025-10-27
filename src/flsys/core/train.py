import torch
from torch import nn, optim
from torch.utils.data import DataLoader


def train_one_epoch(model, loader: DataLoader, device, lr: float, momentum: float):
    """Train model for one epoch on client data."""
    model.train()
    opt = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    loss_fn = nn.CrossEntropyLoss()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()  # reset prev gradients
        loss = loss_fn(model(x), y)
        loss.backward()  # compute gradients
        opt.step()  # update weights


def evaluate(model, loader: DataLoader, device):
    """Evaluate model accuracy on dataset."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)  # predicted class
            correct += (pred == y).sum().item()  # count correct predictions
            total += y.size(0)
    return correct / total
