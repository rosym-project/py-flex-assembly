import torch
import torchvision

class VGG16NBackend(torch.nn.Module):

    def __init__(self, n=3, before_pool=True, pretrained=True, channel_reduction=8):
        #TODO: this is only viable for n = 3 with channel reduction
        # else there are just to many connections for the fully connected part
        super(VGG16NBackend, self).__init__()

        self.feature_number = 4096

        vgg16 = torchvision.models.vgg16(pretrained=pretrained)
        convolution_layers = list(list(vgg16.children())[0].children())

        idx = self.find_nth_pooling_layer(n, convolution_layers, before_pool)
        out_channels = convolution_layers[idx - (0 if before_pool else 1) - 2].out_channels

        self.convolution_backend = torch.nn.Sequential(
                *convolution_layers[:idx],
                # reduce channel count
                torch.nn.Conv2d(out_channels, int(out_channels / channel_reduction), 3, padding=1),
                torch.nn.ReLU(inplace=True))

        input_shape = (3, 224, 224)
        output_shape = self.convolution_backend(torch.empty((1, *input_shape))).squeeze(0).shape
        self.linear_in_size = output_shape[0] * output_shape[1] * output_shape[2]

        self.linear_backend = torch.nn.Linear(in_features=self.linear_in_size, out_features=self.feature_number)

    def find_nth_pooling_layer(self, n, layers, before_pool):
        pooling_layer_count = 0
        idx = -1
        for i, layer in enumerate(layers):
            if not isinstance(layer, torch.nn.MaxPool2d):
                continue

            pooling_layer_count += 1
            if pooling_layer_count == n:
                idx = i
                break
        return idx if before_pool else idx + 1

    def forward(self, data):
        x = self.convolution_backend(data)
        x = x.view(x.shape[0], self.linear_in_size)
        return self.linear_backend(x)


class VGG16Backend(torch.nn.Module):

    def __init__(self, pretrained=True):
        super(VGG16Backend, self).__init__()

        vgg16 = torchvision.models.vgg16(pretrained=pretrained)
        vgg16_children = list(vgg16.children())

        self.convolution_backend = torch.nn.Sequential(*vgg16_children[:-1])

        self.linear_backend = vgg16_children[-1]
        self.linear_backend = torch.nn.Sequential(*(list(self.linear_backend.children())[:-1]))
        
        self.parameters = self.parameters()
        self.feature_number = 4096

    def forward(self, data):
        x = self.convolution_backend(data)
        x = x.view(x.shape[0], 25088) # 25088 = 512 * 7 * 7
        return self.linear_backend(x)


class Resnet18Backend(torch.nn.Module):

    def __init__(self, pretrained=True):
        super(Resnet18Backend, self).__init__()

        resnet18 = torchvision.models.resnet18(pretrained=pretrained)
        self.layers = torch.nn.Sequential(*(list(resnet18.children())[:-1]))
        self.feature_number = resnet18.fc.in_features
        self.parameters = self.parameters()

    def forward(self, data):
        return self.layers(data).squeeze(dim=3).squeeze(dim=2)


class RotationDetector(torch.nn.Module):
    """
    A neural network to detect the rotation of a clamp
    """

    def __init__(self, backend):
        super(RotationDetector, self).__init__()

        self.backend = backend

        self.rotation = torch.nn.Linear(self.backend.feature_number, 4)
        self.rotation_parameters = self.rotation.parameters()

    def forward(self, data):
        features = self.backend(data)
        return self.rotation(features)


class TranslationDetector(torch.nn.Module):
    """
    A neural network to detect the translation of a clamp
    """

    def __init__(self, backend):
        super(TranslationDetector, self).__init__()

        self.backend = backend

        self.translation = torch.nn.Linear(self.backend.feature_number, 3)
        self.translation_parameters = self.translation.parameters()

    def forward(self, data):
        features = self.backend(data)
        return self.translation(features)


backends = [Resnet18Backend, VGG16Backend, VGG16NBackend]
