import torch
import torchvision


class VGG16Backend(torch.nn.Module):

    def __init__(self, pretrained=True):
        super(VGG16Backend, self).__init__()

        vgg16 = torchvision.models.vgg16(pretrained=pretrained)
        vgg16_children = list(vgg16.children())

        self.backend_bot = torch.nn.Sequential(*vgg16_children[:-1])

        self.backend_top = vgg16_children[-1]
        self.backend_top = torch.nn.Sequential(*(list(self.backend_top.children())[:-1]))
        
        self.backend_parameters = self.parameters()
        self.backend_feature_number = 4096

    def forward(self, data):
        x = self.backend_bot(data)
        x = x.view(x.shape[0], 25088) # 25088 = 512 * 7 * 7
        return self.backend_top(x)


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
        return self.backend(data).squeeze(dim=3).squeeze(dim=2)


class RotationDetector(VGG16Backend):
    """
    A neural network to detect the rotation of a clamp
    """

    def __init__(self, init_backend=True):
        super(RotationDetector, self).__init__(init_backend)

        self.rotation = torch.nn.Linear(self.backend_feature_number, 3)
        self.rotation_parameters = self.rotation.parameters()

    def forward(self, data):
        features = super().forward(data)
        return self.rotation(features)


class TranslationDetector(VGG16Backend):
    """
    A neural network to detect the rotation of a clamp
    """

    def __init__(self, init_backend=True):
        super(TranslationDetector, self).__init__(init_backend)

        self.translation = torch.nn.Linear(self.backend_feature_number, 3)
        self.translation_parameters = self.translation.parameters()

    def forward(self, data):
        features = super().forward(data)
        return self.translation(features)

