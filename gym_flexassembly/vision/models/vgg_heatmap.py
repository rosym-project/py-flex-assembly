import torch
import torchvision

class VGG_Heatmap(torch.nn.Module):

    def __init__(self, point_number, init_vgg=True):
        super(VGG_Heatmap, self).__init__()
        conv1 = torch.nn.Sequential()
        conv1.add_module('conv1_1', torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1))
        conv1.add_module('relu1_1', torch.nn.ReLU())
        conv1.add_module('conv1_2', torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1))
        conv1.add_module('relu1_2', torch.nn.ReLU())
        self.conv1 = conv1

        conv2 = torch.nn.Sequential()
        conv2.add_module('pool1', torch.nn.AvgPool2d(kernel_size=2, stride=2))
        conv2.add_module('conv2_1', torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1))
        conv2.add_module('relu2_1', torch.nn.ReLU())
        conv2.add_module('conv2_2', torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1))
        conv2.add_module('relu2_2', torch.nn.ReLU())
        self.conv2 = conv2

        conv3 = torch.nn.Sequential()
        conv3.add_module('pool2', torch.nn.AvgPool2d(kernel_size=2, stride=2))
        conv2.add_module('conv3_1', torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1))
        conv3.add_module('relu3_1', torch.nn.ReLU())
        conv2.add_module('conv3_2', torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1))
        conv3.add_module('relu3_2', torch.nn.ReLU())
        conv2.add_module('conv3_3', torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1))
        conv3.add_module('relu3_3', torch.nn.ReLU())
        self.conv3 = conv3

        self.conv_out = torch.nn.Conv2d(in_channels=256, out_channels=point_number, kernel_size=1, stride=1, padding=0)

        # up3 = torch.nn.Sequential()
        # up3.add_module('up_conv3_2', torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1))
        # up3.add_module('up_relu3_2', torch.nn.ReLU())
        # up3.add_module('up_conv3_1', torch.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1))
        # up3.add_module('up_relu3_1', torch.nn.ReLU())
        # up3.add_module('deconv2', torch.nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1))
        # self.up3 = up3
        
        # up2 = torch.nn.Sequential()
        # up2.add_module('up_conv2_2', torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1))
        # up2.add_module('up_relu2_2', torch.nn.ReLU())
        # up2.add_module('up_conv2_1', torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1))
        # up2.add_module('up_relu2_1', torch.nn.ReLU())
        # up2.add_module('deconv1', torch.nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1))
        # self.up2 = up2

        # up1 = torch.nn.Sequential()
        # up1.add_module('up_conv1_2', torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1))
        # up1.add_module('up_relu1_2', torch.nn.ReLU())
        # up1.add_module('up_conv1_1', torch.nn.Conv2d(in_channels=64, out_channels=point_number, kernel_size=3, stride=1, padding=1))
        # up1.add_module('up_relu1_1', torch.nn.ReLU())
        # self.up1 = up1

        if init_vgg:
            self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv_out(x)
        x = torch.nn.functional.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        # x = self.up3(x)
        # x = self.up2(x)
        # x = self.up1(x)
        return x

    def _initialize_weights(self):
        keys = list(torchvision.models.vgg16(pretrained=True).state_dict())
        self.conv1.conv1_1.weight.data.copy_(pre_train[keys[0]])
        self.conv1.conv1_2.weight.data.copy_(pre_train[keys[2]])
        self.conv2.conv2_1.weight.data.copy_(pre_train[keys[4]])
        self.conv2.conv2_2.weight.data.copy_(pre_train[keys[6]])
        self.conv3.conv3_1.weight.data.copy_(pre_train[keys[8]])
        self.conv3.conv3_2.weight.data.copy_(pre_train[keys[10]])
        self.conv3.conv3_3.weight.data.copy_(pre_train[keys[12]])

        self.conv1.conv1_1.bias.data.copy_(pre_train[keys[1]])
        self.conv1.conv1_2.bias.data.copy_(pre_train[keys[3]])
        self.conv2.conv2_1.bias.data.copy_(pre_train[keys[5]])
        self.conv2.conv2_2.bias.data.copy_(pre_train[keys[7]])
        self.conv3.conv3_1.bias.data.copy_(pre_train[keys[9]])
        self.conv3.conv3_2.bias.data.copy_(pre_train[keys[11]])
        self.conv3.conv3_3.bias.data.copy_(pre_train[keys[13]])

if __name__ == '__main__':
    m = VGG_Heatmap(5, False)
    x_0 = torch.rand((1, 3, 224, 224))
    x = m(x_0)

