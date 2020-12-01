import torch
import torchvision

class VGG16ToNBackend(torch.nn.Module):

    def __init__(self, n, before_pool=True, pretrained=True):
        #TODO: this is only viable for n = 3 with channel reduction
        # else there are just to many connections for the fully connected part
        super(VGG16ToNBackend, self).__init__()

        self.backend_feature_number = 4096

        vgg16 = torchvision.models.vgg16(pretrained=pretrained)
        vgg16_children = list(vgg16.children())
        convolutions_children = list(vgg16_children[0].children())

        nth_pooling_idx = self.find_nth_pooling_layer(n, convolutions_children, before_pool)
        self.backend_bot = torch.nn.Sequential(
                *convolutions_children[:nth_pooling_idx],
                # reduce channeld count
                torch.nn.Conv2d(256, 64, 3, padding=1),
                torch.nn.ReLU()
                )

        input_shape = (3, 224, 224)
        output_shape = self.backend_bot(torch.empty((1, *input_shape))).squeeze(0).shape
        self.linear_input_size = output_shape[0] * output_shape[1] * output_shape[2]

        self.backend_top = torch.nn.Linear(in_features=self.linear_input_size, out_features=self.backend_feature_number)

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
        x = self.backend_bot(data)
        x = x.view(x.shape[0], self.linear_input_size)
        return self.backend_top(x)


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


class RotationDetector(VGG16ToNBackend):
    """
    A neural network to detect the rotation of a clamp
    """

    def __init__(self, init_backend=True):
        super(RotationDetector, self).__init__(3, pretrained=init_backend)

        self.rotation = torch.nn.Linear(self.backend_feature_number, 3)
        self.rotation_parameters = self.rotation.parameters()

    def forward(self, data):
        features = super().forward(data)
        return self.rotation(features)


class TranslationDetector(VGG16ToNBackend):
    """
    A neural network to detect the rotation of a clamp
    """

    def __init__(self, init_backend=True):
        super(TranslationDetector, self).__init__(3, init_backend)

        self.translation = torch.nn.Linear(self.backend_feature_number, 3)
        self.translation_parameters = self.translation.parameters()

    def forward(self, data):
        features = super().forward(data)
        return self.translation(features)

