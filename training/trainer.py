"""
Trainer
=======

Coordinates the complete training process.
"""

from pathlib import Path

from training.engine import (
    train_one_epoch,
    validate_one_epoch,
)

from training.checkpoint import save_checkpoint


class Trainer:

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        logger,
        device,
        config,
    ):

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.criterion = criterion

        self.optimizer = optimizer

        self.scheduler = scheduler

        self.logger = logger

        self.device = device

        self.config = config

        self.best_loss = float("inf")

        self.checkpoint_dir = Path(
            config["checkpoint"]["directory"]
        )

        self.checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.model.to(device)

    def fit(self):

        epochs = self.config["training"]["epochs"]

        self.logger.info("=" * 60)
        self.logger.info("Training Started")
        self.logger.info("=" * 60)

        for epoch in range(epochs):

            train_loss = train_one_epoch(
                self.model,
                self.train_loader,
                self.criterion,
                self.optimizer,
                self.device,
            )

            val_loss = validate_one_epoch(
                self.model,
                self.val_loader,
                self.criterion,
                self.device,
            )

            if self.scheduler is not None:
                self.scheduler.step()

            self.logger.info(
                f"Epoch [{epoch+1}/{epochs}] "
                f"Train: {train_loss:.6f} | "
                f"Val: {val_loss:.6f}"
            )

            if val_loss < self.best_loss:

                self.best_loss = val_loss

                save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch + 1,
                    loss=val_loss,
                    path=self.checkpoint_dir / "best_model.pth",
                )

                self.logger.info(
                    "Best model saved."
                )

        self.logger.info("=" * 60)
        self.logger.info("Training Finished")
        self.logger.info("=" * 60)