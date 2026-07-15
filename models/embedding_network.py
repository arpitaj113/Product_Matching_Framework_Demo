import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class EmbeddingNet(nn.Module):

    def __init__(self, embedding_dim=128, pretrained=True):
        super().__init__()

        # Load pretrained ResNet18
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)

        # Remove the final fully connected classification layer
        self.feature_extractor = nn.Sequential(
            *list(backbone.children())[:-1]
        )

        # Freeze the ResNet18 backbone
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # New embedding layer
        self.embedding = nn.Linear(
            backbone.fc.in_features,
            embedding_dim
        )

    def forward(self, x):

        # Extract features
        x = self.feature_extractor(x)

        # Flatten feature map
        x = torch.flatten(x, start_dim=1)

        # Project to embedding space
        x = self.embedding(x)

        # L2 normalization
        x = F.normalize(x, p=2, dim=1)

        return x