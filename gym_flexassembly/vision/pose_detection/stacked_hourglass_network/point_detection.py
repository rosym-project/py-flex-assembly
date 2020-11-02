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


class HeatmapGenerator():

    def __init__(self, point_number, output_size):
        self.output_size = output_size
        self.point_number = point_number
        self.sigma = min(self.output_size) / 64
        # self.sigma = 1
        self.gaussian = self.compute_gaussian()

    def compute_gaussian(self, sigma_multiplier : float=3.0):
        size = int(sigma_multiplier * self.sigma + 3)
        x = np.arange(size).reshape((1, size))
        x_t = np.transpose(x)
        mean = (sigma_multiplier / 2) * self.sigma + 1
        return np.exp(- ((x - mean) ** 2 + (x_t - mean) ** 2) / (2 * self.sigma ** 2))

    def add_point(self, point, heatmap):
        x, y = point
        if x < 0 or y < 0:
            return
        if x >= self.output_size[1] or y >= self.output_size[0]:
            return

        x_h = slice(max(x - int(self.gaussian.shape[1] / 2), 0),
                    min(x + int(self.gaussian.shape[1] / 2), heatmap.shape[1]))
        y_h = slice(max(y - int(self.gaussian.shape[0] / 2), 0),
                    min(y + int(self.gaussian.shape[0] / 2), heatmap.shape[0]))

        x_g = slice(int(self.gaussian.shape[1] / 2 - (x_h.stop - x_h.start) / 2),
                    int(self.gaussian.shape[1] / 2 + (x_h.stop - x_h.start) / 2))
        y_g = slice(int(self.gaussian.shape[0] / 2 - (y_h.stop - y_h.start) / 2),
                    int(self.gaussian.shape[0] / 2 + (y_h.stop - y_h.start) / 2))

        heatmap[y_h, x_h] = np.maximum(heatmap[y_h, x_h], self.gaussian[y_g, x_g])


    def __call__(self, object_keypoints):
        heatmaps = np.zeros((self.point_number, *self.output_size), np.float32)
        for keypoints in object_keypoints:
            for keypoint in keypoints:
                x, y = int(keypoint['x']), int(keypoint['y'])
                point_id = int(keypoint['id'])
                self.add_point((x, y), heatmaps[point_id])
        return heatmaps


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

        output_size = cv.imread(os.path.join(self.image_dir, self.images[0]), cv.IMREAD_GRAYSCALE).shape
        self.heatmap_generator = HeatmapGenerator(self.point_number, output_size)


    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_file = self.images[index]

        # load the image
        image = PIL.Image.open(os.path.join(self.image_dir, image_file)).convert('RGB')
        heatmaps = self.heatmap_generator([self.point_data[image_file]])
        heatmaps = torch.from_numpy(heatmaps)

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
    transforms.append(T.ImageNetNormalization())

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
            t_img = cv.addWeighted(t_img, 1.0, (target[i].detach().numpy() * 255).astype(np.uint8), 1.0, 0.0)

        # TODO: this should revert the image net normalization
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
