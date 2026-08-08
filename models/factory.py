"""
Model Factory
=============

Creates models from the project configuration.
"""

from models.srdncnn import SRDnCNN


def create_model(config):
    """
    Create a model based on the configuration.
    """

    model_cfg = config["model"]
    model_name = model_cfg["name"].lower()

    if model_name == "srdncnn":
        return SRDnCNN(
            in_channels=model_cfg["in_channels"],
            out_channels=model_cfg["out_channels"],
            channels=model_cfg["num_features"],
            num_blocks=model_cfg["num_residual_blocks"],
            scale=model_cfg["upscale_factor"],
        )

    raise ValueError(f"Unknown model: {model_name}")