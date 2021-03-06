import torch
import torchvision

class AbstractVGGHeatmap(torch.nn.Module):

    def __init__(self, point_number, init_vgg=True):
        super(AbstractVGGHeatmap, self).__init__()

        self.point_number = point_number

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
        conv3.add_module('conv3_1', torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1))
        conv3.add_module('relu3_1', torch.nn.ReLU())
        conv3.add_module('conv3_2', torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1))
        conv3.add_module('relu3_2', torch.nn.ReLU())
        conv3.add_module('conv3_3', torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1))
        conv3.add_module('relu3_3', torch.nn.ReLU())
        self.conv3 = conv3

        self.encoder = torch.nn.Sequential()
        self.encoder.add_module('conv1', self.conv1)
        self.encoder.add_module('conv2', self.conv2)
        self.encoder.add_module('conv3', self.conv3)

        if init_vgg:
            self._initialize_weights()

        self.decoder = self._get_decoder()

    def _get_decoder(self):
        pass

    def _forward_after_decoding(self, x):
        return x

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self._forward_after_decoding(x)
        return x

    def compute_loss(self, labels, predictions):
        """
        Loss computation from the re-implementation of the satcked hout
        glass network in pytorch: TODO: link
        """
        return ((labels - predictions) ** 2).mean()

    def _initialize_weights(self):
        pre_train = torchvision.models.vgg16(pretrained=True).state_dict()
        keys = list(pre_train.keys())
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


class VGGHeatmapSimple(AbstractVGGHeatmap):

    def __init__(self, point_number, init_vgg=True):
        super(VGGHeatmapSimple, self).__init__(point_number, init_vgg=init_vgg)

    def _get_decoder(self):
        decoder = torch.nn.Sequential()
        decoder.add_module('conv', torch.nn.Conv2d(in_channels=256, out_channels=self.point_number, kernel_size=1, stride=1, padding=0))
        decoder.add_module('activation', torch.nn.Sigmoid())
        return decoder
        
    def _forward_after_decoding(self, x):
        return torch.nn.functional.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)


class VGGHeatmapComplex(AbstractVGGHeatmap):

    def __init__(self, point_number, init_vgg=True):
        super(VGGHeatmapComplex, self).__init__(point_number, init_vgg=init_vgg)

    def _get_decoder(self):
        up3 = torch.nn.Sequential()
        up3.add_module('up_conv3_2', torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1))
        up3.add_module('up_relu3_2', torch.nn.ReLU())
        up3.add_module('up_conv3_1', torch.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1))
        up3.add_module('up_relu3_1', torch.nn.ReLU())
        up3.add_module('deconv2', torch.nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1))
        
        up2 = torch.nn.Sequential()
        up2.add_module('up_conv2_2', torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1))
        up2.add_module('up_relu2_2', torch.nn.ReLU())
        up2.add_module('up_conv2_1', torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1))
        up2.add_module('up_relu2_1', torch.nn.ReLU())
        up2.add_module('deconv1', torch.nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1))

        up1 = torch.nn.Sequential()
        up1.add_module('up_conv1_2', torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1))
        up1.add_module('up_relu1_2', torch.nn.ReLU())
        up1.add_module('up_conv1_1', torch.nn.Conv2d(in_channels=64, out_channels=self.point_number, kernel_size=3, stride=1, padding=1))
        up1.add_module('up_relu1_1', torch.nn.Sigmoid())

        decoder = torch.nn.Sequential()
        decoder.add_module('up3', up3)
        decoder.add_module('up2', up2)
        decoder.add_module('up1', up1)
        return decoder
        

if __name__ == '__main__':
    import argparse
    import time

    parser = argparse.ArgumentParser(description='time the model forward and backward passes',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--complex', action='store_true',
                        help='use the more complex heatmap model')
    parser.add_argument('--point_number', type=int, default=10,
                        help='the number of points the model outputs')
    parser.add_argument('--iterations', type=int, default=100,
                        help='the number of iterations computed')
    parser.add_argument('--input_resolution', type=int, default=512,
                        help='the input resolution of images put into the network')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='the batch size used')
    parser.add_argument('--device', type=str, default=None,
                        help='the device on which to perform the computation. \
                              If this is None cuda:0 is chosen if available, \
                              else cpu')
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = VGGHeatmapComplex(args.point_number, False) if args.complex else VGGHeatmapSimple(args.point_number, False)
    model = model.to(device)

    time_total_forward = 0
    time_total_backward = 0
    for i in range(args.iterations):
        print(f'Iteration {i+1}/{args.iterations}', end='\r')
        inputs = torch.rand((args.batch_size, 3, args.input_resolution, args.input_resolution),
                            device=device)
        outputs = torch.empty((args.batch_size, args.point_number, args.input_resolution, args.input_resolution),
                              device=device)

        time_start = time.time()
        predictions = model(inputs)
        time_forward = time.time()

        loss = model.compute_loss(outputs, predictions)
        loss.backward()
        time_backward = time.time()

        time_total_forward += time_forward - time_start
        time_total_backward += time_backward - time_start

    time_average_forward = (time_total_forward / args.iterations) * 1000
    time_average_backward = (time_total_backward / args.iterations) * 1000

    print(f'Average time forward {time_average_forward:.1f}ms')
    print(f'Average time backward {time_average_backward:.1f}ms')



