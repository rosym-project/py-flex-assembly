import argparse
import json
import os
import math

import PIL
import PIL.Image
import torch
import torchvision
import numpy as np
import cv2 as cv

import transforms as T

class HeatmapDataset(torch.utils.data.Dataset):
    """
    Dataset which is created by loading all images from the data directory and creating a heatmap
    annotation from the corresponding json file.
    """

    def __init__(self, image_dir, point_file, transforms, std=1):
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

        # generate a 2d gaussian distribution
        width, height = PIL.Image.open(os.path.join(self.image_dir, self.images[0])).size
        x = np.arange(0, width * 2 - 1)
        y = np.arange(0, height * 2 - 1)

        mu_x = width - 1
        mu_y = height - 1

        x = 1 / (std * math.sqrt(2 * math.pi)) * np.exp(-1 * (x - mu_x) ** 2 / (2 * std ** 2))
        y = 1 / (std * math.sqrt(2 * math.pi)) * np.exp(-1 * (y - mu_y) ** 2 / (2 * std ** 2))

        self.gaussian = np.rot90(np.outer(x, y))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_file = self.images[index]

        # load the image
        image = PIL.Image.open(os.path.join(self.image_dir, image_file)).convert('RGB')
        width, height = image.size

        # create a list of empty heatmaps
        heatmaps = torch.tensor(())
        heatmaps = heatmaps.new_full([self.point_number, height, width], 0, dtype=torch.float)

        # copy the corresponding section from the gaussian distribution for each visible point
        for point in self.point_data[image_file]:
            i = int(point['id'])
            x = int(point['x'])
            y = int(point['y'])

            yFrom = height - 1 - y
            yTo = 2 * height - 1 - y
            xFrom = width - 1 - x
            xTo = 2 * width - 1 - x
            heatmaps[i, :, :] = torch.tensor(self.gaussian[yFrom:yTo, xFrom:xTo].copy())

        # apply the transforms
        if self.transforms is not None:
            image, heatmaps = self.transforms(image, heatmaps)

        return image, heatmaps


def get_transform(train, noise_std=3.37, flip_prob=0.5, discrete_rot=True, crop_size=(700, 700), target_size=(512, 512)):
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
    transforms.append(T.ToPILImage())
    if train:
        transforms.append(T.GaussianNoise(noise_std))
        transforms.append(T.RandomHorizontalFlip(flip_prob))
        transforms.append(T.RandomVerticalFlip(flip_prob))
        transforms.append(T.RandomCrop(crop_size))
        transforms.append(T.RandomRotation(discrete_rot))
    else:
        transforms.append(T.CenterCrop(crop_size))
    transforms.append(T.Resize(target_size))
    transforms.append(T.ToTensor())

    return T.Compose(transforms)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_train', type=str)
    parser.add_argument('data_val', type=str)
    parser.add_argument('-w', '--weights', type=str,
                        help='load initial weights from a file')
    args = parser.parse_args()

    # creation of a data loader from a dataset
    dataset_train = HeatmapDataset(args.data_train, os.path.join(args.data_train, 'data.json'), get_transform(train=True))
    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=16, shuffle=True)
    dataset_val = HeatmapDataset(args.data_val, os.path.join(args.data_val, 'data.json'), get_transform(train=False))
    data_loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=16, shuffle=True)

    import cv2 as cv
    for img, target in dataset_train:
        img = (img.numpy() * 255).astype(np.uint8)

        t_img = np.zeros((512, 512), np.uint8)
        for i in range(target.shape[0]):
            t_img = np.where(target[i] > 0, np.ones(t_img.shape) * 255, t_img)

        res = np.zeros((*img.shape[1:], img.shape[0]), np.uint8)
        res[:, :, 0] = img[2]
        res[:, :, 1] = img[1]
        res[:, :, 2] = img[0]

        for i in range(target.shape[0]):
            heatmap = np.where(target[i] > 0, np.ones(t_img.shape, dtype=np.uint8), np.zeros(t_img.shape, dtype=np.uint8))
            num_labels, _, _, centroids = cv.connectedComponentsWithStats(heatmap)
            if num_labels > 1:
                cv.circle(res, tuple(np.array(centroids[1]).round().astype(np.int)), 1, (0, 0, 255), thickness=-1)

        cv.imshow('Clamp Image', res)
        cv.imshow('Combined Heatmaps', t_img)
        if cv.waitKey(0) == ord('q'):
            break
