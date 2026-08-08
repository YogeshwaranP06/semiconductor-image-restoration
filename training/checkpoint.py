"""
Checkpoint Manager
==================

Save and load training checkpoints.

Project:
    Semiconductor Image Restoration
"""

from pathlib import Path
import torch


def save_checkpoint(
    model,
    optimizer,
    epoch: int,
    loss: float,
    path: str | Path,
):
    """
    Save model checkpoint.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }

    torch.save(checkpoint, path)


def load_checkpoint(
    model,
    optimizer,
    path: str | Path,
):
    """
    Load model checkpoint.

    Returns
    -------
    epoch, loss
    """

    checkpoint = torch.load(path, map_location="cpu")

    model.load_state_dict(checkpoint["model_state_dict"])

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    return (
        checkpoint["epoch"],
        checkpoint["loss"],
    )