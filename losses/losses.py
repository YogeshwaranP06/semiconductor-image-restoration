%%writefile losses/losses.py

"""
Collection of loss functions for image restoration.

Project:
Semiconductor Image Restoration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    """
    Robust L1 Loss (Charbonnier Loss).
    """

    def __init__(self, epsilon: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, prediction, target):
        diff = prediction - target

        loss = torch.sqrt(
            diff * diff
            + self.epsilon * self.epsilon
        )

        return loss.mean()


class GradientLoss(nn.Module):
    """
    Gradient-based reconstruction loss.

    Penalizes differences between the horizontal
    and vertical image gradients of prediction
    and ground truth.
    """

    def __init__(self):
        super().__init__()

    def forward(self, prediction, target):

        pred_dx = (
            prediction[:, :, :, 1:]
            - prediction[:, :, :, :-1]
        )

        pred_dy = (
            prediction[:, :, 1:, :]
            - prediction[:, :, :-1, :]
        )

        target_dx = (
            target[:, :, :, 1:]
            - target[:, :, :, :-1]
        )

        target_dy = (
            target[:, :, 1:, :]
            - target[:, :, :-1, :]
        )

        loss_x = F.l1_loss(
            pred_dx,
            target_dx,
        )

        loss_y = F.l1_loss(
            pred_dy,
            target_dy,
        )

        return loss_x + loss_y


class GradientL1Loss(nn.Module):
    """
    Combined pixel-wise L1 and gradient loss.

    L = L1 + lambda_gradient * L_gradient
    """

    def __init__(
        self,
        lambda_gradient: float = 0.1,
    ):
        super().__init__()

        self.lambda_gradient = lambda_gradient

        self.l1 = nn.L1Loss()
        self.gradient = GradientLoss()

    def forward(self, prediction, target):

        l1_loss = self.l1(
            prediction,
            target,
        )

        gradient_loss = self.gradient(
            prediction,
            target,
        )

        total_loss = (
            l1_loss
            + self.lambda_gradient * gradient_loss
        )

        return total_loss
