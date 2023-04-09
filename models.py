import torch
import torch.nn as nn
from torchvision.models.vision_transformer import vit_b_16


class VisionTransformer(torch.nn.Module):
    def __init__(self, num_classes: int):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes

        self.model = vit_b_16(pretrained=True)
        self.model.heads = nn.Identity()
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        output = self.classifier(self.model(x))
        return output
