"""
Training Engine
===============

Implements one training epoch and one validation epoch.

Project:
    Semiconductor Image Restoration
"""

import torch
from tqdm import tqdm


def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    epoch=None,
    total_epochs=None,
):
    """
    Train the model for one epoch.

    Returns
    -------
    float
        Average training loss.
    """

    model.train()

    running_loss = 0.0

    description = "Training"

    if epoch is not None and total_epochs is not None:
        description = f"Epoch {epoch}/{total_epochs} - Training"

    progress = tqdm(
        dataloader,
        desc=description,
        leave=False,
    )

    for batch in progress:

        noisy = batch["noisy"].to(
            device,
            non_blocking=True,
        )

        gt = batch["gt"].to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        prediction = model(noisy)

        loss = criterion(
            prediction,
            gt,
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        progress.set_postfix(
            loss=f"{loss.item():.6f}"
        )

    epoch_loss = running_loss / len(dataloader)

    return epoch_loss


@torch.no_grad()
def validate_one_epoch(
    model,
    dataloader,
    criterion,
    device,
    epoch=None,
    total_epochs=None,
):
    """
    Validate the model for one epoch.

    Returns
    -------
    float
        Average validation loss.
    """

    model.eval()

    running_loss = 0.0

    description = "Validation"

    if epoch is not None and total_epochs is not None:
        description = f"Epoch {epoch}/{total_epochs} - Validation"

    progress = tqdm(
        dataloader,
        desc=description,
        leave=False,
    )

    for batch in progress:

        noisy = batch["noisy"].to(
            device,
            non_blocking=True,
        )

        gt = batch["gt"].to(
            device,
            non_blocking=True,
        )

        prediction = model(noisy)

        loss = criterion(
            prediction,
            gt,
        )

        running_loss += loss.item()

        progress.set_postfix(
            loss=f"{loss.item():.6f}"
        )

    epoch_loss = running_loss / len(dataloader)

    return epoch_loss