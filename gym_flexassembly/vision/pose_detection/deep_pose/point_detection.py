import argparse
import json
import os

import PIL
import PIL.Image
import PIL.ImageDraw
import torch
import torchvision
import matplotlib.pyplot as plt

import transforms as T


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Use device', device)

class PointDataset(torch.utils.data.Dataset):
    """
    Dataset which loads all png images from a directory and an annotation
    from a json file.
    """

    def __init__(self, image_dir, point_file, transforms):
        self.image_dir = image_dir
        self.images = [f for f in os.listdir(image_dir) if f.endswith('.png')]
        self.images.sort()

        with open(point_file, 'r') as f:
            self.point_data = json.loads(f.read())

        self.point_number = 0
        for image in self.images:
            if image not in self.point_data:
                raise Exception('Annotation does not contain points for image[' + image + ']!')

            for point in self.point_data[image]:
                self.point_number = max(point['id'] + 1, self.point_number)

        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_file = self.images[index]

        # load the image
        image = PIL.Image.open(os.path.join(self.image_dir, image_file)).convert('RGB')
        width, height = image.size

        # pre-process the annotation by setting points to (-1, -1) if they
        # are not annotated on the image
        annotation = {}
        for i in range(self.point_number):
            annotation[i] = torch.ones(2) * -1

            for point in self.point_data[image_file]:
                if point['id'] != i:
                    continue

                x = int(point['x'])
                y = int(point['y'])
                annotation[point['id']] = torch.tensor((x, y), dtype=torch.float)

        if self.transforms is not None:
            image, annotation = self.transforms(image, annotation)

        return image, annotation


class PointDetector(torch.nn.Module):
    """
    A neural network for point detection.
    """

    def __init__(self, point_number):
        super(PointDetector, self).__init__()
        self.point_number = point_number

        # use resnet 18 as a backend because it is small and fast
        resnet18 = torchvision.models.resnet18(pretrained=True)
        feature_number = resnet18.fc.in_features
        # remove the classification layer from the backend to use it as a feature extractor
        self.backend = torch.nn.Sequential(*(list(resnet18.children())[:-1]))

        # add an output layer for each point
        for i in range(point_number):
            self.add_module('point_' + str(i), torch.nn.Linear(feature_number, 2))
            self.add_module('visibility_' + str(i),
                            torch.nn.Sequential(torch.nn.Linear(feature_number, 1), torch.nn.Sigmoid()))

        self.point_loss = torch.nn.MSELoss()
        # self.visibility_loss = torch.nn.BCEWithLogitsLoss()
        self.visibility_loss = torch.nn.BCELoss()

        self.backend_parameters = []
        self.visibility_parameters = []
        self.point_parameters = []
        for name, params in self.named_parameters():
            if name.startswith('point'):
                self.point_parameters.append(params)
            elif name.startswith('visibility'):
                self.visibility_parameters.append(params)
            else:
                self.backend_parameters.append(params)

    def forward(self, data):
        features = self.backend(data)
        # features.shape is (#batch_size, #features, 1, 1)
        # squeeze the last two dimensions
        features = features.squeeze(dim=3)
        features = features.squeeze(dim=2)

        outputs = {'point': {}, 'visibility': {}}
        for i in range(self.point_number):
            point_layer = self._modules['point_' + str(i)]
            visibility_layer = self._modules['visibility_' + str(i)]
            outputs['point'][i] = point_layer(features)
            outputs['visibility'][i] = visibility_layer(features)
        return outputs

    def compute_loss(self, result, annotation):
        tensor_zero = torch.zeros(1).to(device)
        tensor_one = torch.ones(1).to(device)
        loss = {'point': 0,
                'visibility': 0 }
        for point_id in annotation:
            for b in range(annotation[point_id].shape[0]):
                # ignore the loss of points not annotated because they are not
                # visible in the image
                if annotation[point_id][b][0] == -1:
                    loss['visibility'] += self.visibility_loss(result['visibility'][point_id][b], tensor_zero)
                    continue

                loss['visibility'] += self.visibility_loss(result['visibility'][point_id][b], tensor_one)
                loss['point'] += self.point_loss(result['point'][point_id][b].to(device), annotation[point_id][b].to(device))
        return loss


def show_res(image_file='001.png', image_dir='test_data'):
    """
    Show all points detected on an image.
    """
    img = PIL.Image.open(os.path.join(image_dir, image_file)).convert('RGB')
    width, height = img.size
    draw = PIL.ImageDraw.Draw(img)

    inputs = torchvision.transforms.Resize((224, 224))(img)
    inputs = torchvision.transforms.ToTensor()(inputs)

    outputs = detector(inputs.unsqueeze(0))
    for point_id in outputs:
        point = outputs[point_id].squeeze(0)
        point[0] *= width
        point[1] *= height
        draw.ellipse((point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3), fill='blue')

    img.show()


def getEstimation(image_dir='test_data'):
    """
    Get estimation into json file in the same format as training json
    """
    diff_file = open('estimation.json', 'w')
    diff_file.write('{\n')

    images = os.listdir(image_dir)
    images.sort()

    for i in range(len(images)):
        image_file = images[i]
        print(image_file)
        diff_file.write('  \"' + image_file + '\": [')

        img = PIL.Image.open(os.path.join(image_dir, image_file)).convert('RGB')
        width, height = img.size

        inputs = torchvision.transforms.Resize((224, 224))(img)
        inputs = torchvision.transforms.ToTensor()(inputs)

        outputs = detector(inputs.unsqueeze(0))

        for point_id in outputs:
            point = outputs[point_id].squeeze(0)
            point[0] *= width
            point[1] *= height
            diff_file.write('\n    {\n')
            diff_file.write('      \"id\": ' + str(point_id) + ',\n')
            diff_file.write('      \"x\": ' + str(int(point[0].item())) + ',\n')
            diff_file.write('      \"y\": ' + str(int(point[1].item())) + '\n')

            if point_id == (len(outputs) - 1):
                diff_file.write('    }')
            else:
                diff_file.write('    },')

        if i == (len(images) - 1):
            diff_file.write('\n  ]\n')
        else:
            diff_file.write('\n  ],\n')
    diff_file.write('}')
    diff_file.close()


def train(detector, data_loader_train, data_loader_val=None, epochs=100):
    """
    Basic training script for pytorch.
    """

    # optimizer = torch.optim.SGD(detector.parameters(), lr=0.0001, momentum=0.95)
    optimizer = torch.optim.SGD([{'params': detector.backend_parameters, 'lr': 1e-6},
                                 {'params': detector.point_parameters, 'lr': 1e-4},
                                 {'params': detector.visibility_parameters, 'lr': 1e-4}],
                                lr=1e-4, momentum=0.95)

    for i in range(1, epochs + 1):
        epoch_loss = {'point': 0,
                      'visibility': 0}
        for inputs, annotations in data_loader_train:
            optimizer.zero_grad()

            inputs = inputs.to(device)

            outputs = detector(inputs)
            loss = detector.compute_loss(outputs, annotations)

            if type(loss) == type(0):
                # no image has the point annotated so skip the loss computation
                continue

            epoch_loss['point'] += loss['point']
            epoch_loss['visibility'] += loss['visibility']

            (loss['point'] + loss['visibility']).backward()
            optimizer.step()

        epoch_loss['point'] /= len(data_loader_train.dataset)
        epoch_loss['visibility'] /= len(data_loader_train.dataset)

        s = 'Epoch[' + str(i) + '/' + str(epochs) + '] Training:   Point-Loss[' + '{:.6f}'.format(epoch_loss['point']) + '], Visibility-Loss[' + '{:.6f}'.format(epoch_loss['visibility']) + ']'
        print(s)
        loss_file.write(s + '\n')
        if '{:.6f}'.format(epoch_loss['point']) == '0.000000':
            break

        if i % 10 == 0:
            path = './model/point_detector_' + str(i) + '.model'
            torch.save(detector.state_dict(), path)

            if data_loader_val != None:
                loss = compute_val_loss(detector, data_loader_val)
                s = 'Epoch[' + str(i) + '/' + str(epochs) + '] Validation: Point-Loss[' + '{:.6f}'.format(loss['point']) + '], Visibility-Loss[' + '{:.6f}'.format(loss['visibility']) + ']'
                print(s)
                loss_file.write(s + '\n')


    # save the trained model
    torch.save(detector.state_dict(), 'point_detector.model')


def compute_val_loss(detector, data_loader):
    """
    Computes the evaluation loss
    """
    detector = detector.eval()

    eval_loss = {'point': 0,
                 'visibility': 0}
    for inputs, annotations in data_loader:
        outputs = detector(inputs)
        loss = detector.compute_loss(outputs, annotations)
        eval_loss['point'] += loss['point']
        eval_loss['visibility'] += loss['visibility']

    eval_loss['point'] /= len(data_loader.dataset)
    eval_loss['visibility'] /= len(data_loader.dataset)

    detector = detector.train()

    return eval_loss


def get_transform(train, noise_std=3.37, flip_prob=0.5, discrete_rot=True, crop_size=(700, 700), target_size=(224, 224)):
    """
    Returns a pipeline of transformations that is used for data augmentation and preprocessing:
    @param train            whether the training or the validation transforms are required
                            (the validation pipeline doesn't include the random flips and the random crop)
    @param noise_std        the standard deviation of the gaussian noise
    @param flip_prob        the probability for random horizontal and vertical flips to occur
    @param discrete_rot     whether rotations are discrete or continuous
    @param crop_size        the size of the crop
    @param target_size      the size of the resized image (224x224 is the expected input for resnet)
    """
    transforms = []
    if train:
        transforms.append(T.GaussianNoise(noise_std))
        transforms.append(T.RandomHorizontalFlip(flip_prob))
        transforms.append(T.RandomVerticalFlip(flip_prob))
        transforms.append(T.RandomRotation(discrete_rot))
        transforms.append(T.RandomCrop(crop_size))
    else:
        transforms.append(T.CenterCrop(crop_size))
    transforms.append(T.NormalizePoints())
    transforms.append(T.Resize(target_size))
    transforms.append(T.ToTensor())

    return T.Compose(transforms)

def show_transforms(args, id):
    # load the dataset without any transformations
    dataset = PointDataset(args.data_train, os.path.join(args.data_train, 'data.json'), None)
    img, points = dataset.__getitem__(id)

    # plot the original image and points
    plt.subplot(2, 4, 1)
    plt.title("Original")
    plt.imshow(img)
    for p in points.values():
        if p[0] >= 0 and p[1] >= 0:
            plt.scatter(p[0], p[1], s=4, c='r')

    # list of transforms
    transforms = []
    transforms.append(T.GaussianNoise(3.37))
    transforms.append(T.RandomHorizontalFlip(1))
    transforms.append(T.RandomVerticalFlip(1))
    transforms.append(T.RandomRotation(False))
    transforms.append(T.RandomCrop((700, 700)))
    names = ["Gaussian Noise", "Horizontal Flip", "Vertial Flip", "Random Rotation", "Random Crop"]

    # plot the different transforms
    for i in range(len(transforms)):
        img, points = transforms[i](img, points)

        plt.subplot(2, 4, i + 2)
        plt.title(names[i])
        plt.imshow(img)
        for p in points.values():
            if p[0] >= 0 and p[1] >= 0:
                plt.scatter(p[0], p[1], s=4, c='r')

    # plot resized image with normalized points
    img, points = T.NormalizePoints()(img, points)
    img, points = T.Resize((244, 244))(img, points)

    plt.subplot(2, 4, 7)
    plt.title("Point Normalization & Resize")
    plt.imshow(img)
    for p in points.values():
        if p[0] >= 0 and p[1] >= 0:
            plt.scatter(int(p[0] * 244), int(p[1] * 244), s=4, c='r')

    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_train', type=str)
    parser.add_argument('data_val', type=str)
    parser.add_argument('-w', '--weights', type=str,
                        help='load initial weights from a file')
    args = parser.parse_args()

    # Create the network
    detector = PointDetector(10)
    detector.to(device)
    if args.weights:
        detector.load_state_dict(torch.load(args.weights, map_location=device))

    # # How to load its saved parameters:
    # # detector.load_state_dict(torch.load('point_detector.model'))
    # # detector = detector.eval()

    loss_file = open('loss.csv', 'w+')

    # creation of a data loader from a dataset
    dataset_train = PointDataset(args.data_train, os.path.join(args.data_train, 'data.json'), get_transform(train=True))
    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=16, shuffle=True)
    dataset_val = PointDataset(args.data_val, os.path.join(args.data_val, 'data.json'), get_transform(train=False))
    data_loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=16, shuffle=True)

    # training the detector
    detector = detector.train()
    train(detector, data_loader_train, data_loader_val=data_loader_val, epochs=50)

    # getEstimation()
    loss_file.close()
