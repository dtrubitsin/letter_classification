import timm
import torch
import torch.nn as nn


class ViTFeatureExtractor(nn.Module):
    """Feature extractor using Vision Transformer (ViT) backbone from `timm`.

    Attributes:
        vit (nn.Module): Vision Transformer backbone with classification head removed.
    """

    def __init__(self, model_name: str = "vit_base_patch16_224", pretrained: bool = True) -> None:
        """
        Args:
            model_name (str): Name of the ViT model from the `timm` library.
            pretrained (bool): If True, loads pretrained weights.
        """
        super().__init__()
        self.vit = timm.create_model(model_name, pretrained=pretrained, num_classes=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W].

        Returns:
            torch.Tensor: Extracted features of shape [B, D].
        """
        return self.vit(x)


def compute_prototypes(
        features: torch.Tensor,
        labels: torch.Tensor,
        classes: list[int] | torch.Tensor
) -> torch.Tensor:
    """Computes class prototypes by averaging feature vectors of support samples.

    Args:
        features (torch.Tensor): Tensor of shape [N, D], feature vectors.
        labels (torch.Tensor): Tensor of shape [N], class labels for each feature.
        classes (list[int] | torch.Tensor): Unique class labels for which to compute prototypes.

    Returns:
        torch.Tensor: Tensor of shape [K, D] containing one prototype per class.
    """
    prototypes = []
    for cls in classes:
        cls_feat = features[labels == cls]
        prototype = cls_feat.mean(dim=0)
        prototypes.append(prototype)
    return torch.stack(prototypes)
