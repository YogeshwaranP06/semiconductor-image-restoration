"""
Train Script
============

Main entry point for Semiconductor Image Restoration.

Author:
    Yogeshwaran
"""

from logging import config

import torch

from utils.config import load_config
from utils.seed import set_seed
from utils.logger import get_logger

from datasets.datamodule import create_dataloaders

from models.factory import create_model

from losses.factory import create_loss

from training.optimizer import create_optimizer
from training.scheduler import create_scheduler
from training.trainer import Trainer


def get_device(config):
    """
    Select training device.
    """

    device_name = config["training"]["device"]

    if device_name == "auto":

        if torch.cuda.is_available():
            return torch.device("cuda")

        return torch.device("cpu")

    return torch.device(device_name)


def main():

    # --------------------------------------------------
    # Configuration
    # --------------------------------------------------

    config = load_config("configs/default.yaml")

    # --------------------------------------------------
    # Reproducibility
    # --------------------------------------------------

    set_seed(config["training"]["seed"])

    # --------------------------------------------------
    # Logger
    # --------------------------------------------------

    logger = get_logger()

    # --------------------------------------------------
    # Device
    # --------------------------------------------------

    device = get_device(config)

    logger.info(f"Using device : {device}")

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------

    train_loader, val_loader = create_dataloaders(
    gt_dir=config["dataset"]["train_gt"],
    noisy_dir=config["dataset"]["train_noisy"],
    batch_size=config["training"]["batch_size"],
    num_workers=config["training"]["num_workers"],
    seed=config["training"]["seed"],
    debug=config["training"].get("debug", False),
    debug_train_samples=config["training"].get(
        "debug_train_samples", 32
    ),
    debug_val_samples=config["training"].get(
        "debug_val_samples", 8
    ),
)

    logger.info(
        f"Training Images : {len(train_loader.dataset)}"
    )

    logger.info(
        f"Validation Images : {len(val_loader.dataset)}"
    )

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    model = create_model(config)

    logger.info(
        f"Model : {config['model']['name']}"
    )

    # --------------------------------------------------
    # Loss
    # --------------------------------------------------

    criterion = create_loss(config)

    logger.info(
        f"Loss : {config['loss']['name']}"
    )

    # --------------------------------------------------
    # Optimizer
    # --------------------------------------------------

    optimizer = create_optimizer(
        model,
        config,
    )

    # --------------------------------------------------
    # Scheduler
    # --------------------------------------------------

    scheduler = create_scheduler(
        optimizer,
        config,
    )

    # --------------------------------------------------
    # Trainer
    # --------------------------------------------------

    trainer = Trainer(

        model=model,

        train_loader=train_loader,

        val_loader=val_loader,

        criterion=criterion,

        optimizer=optimizer,

        scheduler=scheduler,

        logger=logger,

        device=device,

        config=config,
    )

    trainer.fit()


if __name__ == "__main__":

    main()