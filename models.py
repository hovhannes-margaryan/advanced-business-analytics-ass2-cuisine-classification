import torch
import torch.nn as nn
from torchvision.models.vision_transformer import vit_b_16
from torchvision.models import convnext_tiny, resnet18, resnet50, alexnet, vgg16, vgg19


class VisionTransformer(torch.nn.Module):
    def __init__(self, num_classes: int):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes

        self.model = vit_b_16(pretrained=True)
        self.model.heads = nn.Identity()

        for name, param in self.model.named_parameters():
            param.requires_grad = False

        for param in self.model.encoder.layers[-1].mlp.parameters():
            param.requires_grad = True

        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        output = self.classifier(self.model(x))
        return output


class ConvNext(torch.nn.Module):
    def __init__(self, num_classes: int):
        super(ConvNext, self).__init__()
        self.num_classes = num_classes

        self.model = convnext_tiny(pretrained=True)
        self.model.classifier[2] = nn.Identity()

        for name, param in self.model.named_parameters():
            param.requires_grad = False

        self.classifier_layer = nn.Linear(768, num_classes)

    def forward(self, x):
        output = self.classifier_layer(self.model(x))
        return output


class ResNet(torch.nn.Module):
    def __init__(self, num_classes: int):
        super(ResNet, self).__init__()
        self.num_classes = num_classes

        self.model = resnet50(pretrained=True)
        self.model.fc = nn.Identity()
        for name, param in self.model.named_parameters():
            param.requires_grad = False
        self.classifier_layer = nn.Linear(2048, num_classes)

    def forward(self, x):
        output = self.classifier_layer(self.model(x))
        return output


class AlexNetCustom(torch.nn.Module):
    def __init__(self, num_classes: int):
        super(AlexNetCustom, self).__init__()
        self.num_classes = num_classes

        self.model = alexnet(pretrained=True)
        self.model.classifier[6] = nn.Identity()

        for name, param in self.model.named_parameters():
            param.requires_grad = False

        self.classifier_layer = nn.Linear(4096, num_classes)

    def forward(self, x):
        output = self.classifier_layer(self.model(x))
        return output


class VGG(torch.nn.Module):
    def __init__(self, num_classes: int):
        super(VGG, self).__init__()
        self.num_classes = num_classes

        self.model = vgg19(pretrained=True)
        self.model.classifier[6] = nn.Identity()

        for name, param in self.model.named_parameters():
            param.requires_grad = False

        self.classifier_layer = nn.Linear(4096, num_classes)

    def forward(self, x):
        output = self.classifier_layer(self.model(x))
        return output
