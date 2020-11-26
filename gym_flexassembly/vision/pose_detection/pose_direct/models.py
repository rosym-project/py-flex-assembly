import torch
import torchvision


class Resnet18Base(torch.nn.Module):

    def __init__(self, pretrained=True):
        super(Resnet18Base, self).__init__()

        resnet18 = torchvision.models.resnet18(pretrained=pretrained)
        # remove the classification layer
        self.backend = torch.nn.Sequential(*(list(resnet18.children())[:-1]))
        # save number of features output by backend
        self.backend_feature_number = resnet18.fc.in_features
        
        self.backend_parameters = self.parameters()

    def forward(self, data):
        return self.backend(data)


class RotationDetector(Resnet18Base):
    """
    A neural network to detect the rotation of a clamp
    """

    def __init__(self, init_backend=True):
        super(RotationDetector, self).__init__(init_backend)

        self.rotation = torch.nn.Linear(self.backend_feature_number, 3)
        self.rotation_parameters = self.rotation.parameters()

    def forward(self, data):
        features = super().forward(data)
        features = features.squeeze(dim=3).squeeze(dim=2)
        return self.rotation(features)


class TranslationDetector(Resnet18Base):
    """
    A neural network to detect the rotation of a clamp
    """

    def __init__(self, init_backend=True):
        super(RotationDetector, self).__init__(init_backend)

        self.translation = torch.nn.Linear(self.backend_feature_number, 3)
        self.translation_parameters = self.rotation.parameters()

    def forward(self, data):
        features = super().forward(data)
        features = features.squeeze(dim=3).squeeze(dim=2)
        return self.translation(features)

