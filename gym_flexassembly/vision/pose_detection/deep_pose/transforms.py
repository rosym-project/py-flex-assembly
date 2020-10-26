import math

import numpy as np
import cv2 as cv
import torch
import torchvision
from PIL import Image

class GaussianNoise(object):
    def __init__(self, std):
        self.std = std

    def __call__(self, image, target):
        image = np.array(image).astype(np.int32)
        noise = np.random.normal(0, self.std, np.prod(image.shape))
        noise = np.reshape(noise, image.shape)

        noise = noise.astype(np.int32)
        image = image + noise
        image = np.minimum(image, np.full(image.shape, 255))
        image = np.maximum(image, np.zeros(image.shape))
        image = image.astype(np.uint8)

        return Image.fromarray(image), target


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if np.random.rand() < self.prob:
            image = torchvision.transforms.functional.hflip(image)
            width = image.size[0]

            for key in target.keys():
                if target[key][0] != -1:
                    target[key][0] = width - target[key][0] - 1

        return image, target


class RandomVerticalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if np.random.rand() < self.prob:
            image = torchvision.transforms.functional.vflip(image)
            height = image.size[1]

            for key in target.keys():
                if target[key][0] != -1:
                    target[key][1] = height - target[key][1] - 1

        return image, target


class RandomRotation(object):
    def __init__(self, discrete):
        # parameter discrete:
        # true: angles are chosen frome [0, 90, 180, 270]
        # false: angles are chosen frome [0, 1, 2, ..., 359]
        self.discrete = discrete

    def __call__(self, image, target):
        if self.discrete:
            angle = np.random.randint(0, 4) * 90
        else:
            angle = np.random.randint(0, 360)

        image = torchvision.transforms.functional.rotate(image, angle)

        # boundaries of the image
        lowerb = np.array([0, 0], dtype=np.float32)
        upperb = np.array(image.size, dtype=np.float32)

        angle = math.radians(-angle)
        rotation_matrix = torch.tensor([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
        offset = torch.tensor(image.size, dtype=torch.float) / 2.0
        for key in target.keys():
            if target[key][0] != -1:
                target[key] -= offset
                target[key] = torch.matmul(rotation_matrix, target[key])
                target[key] += offset

                # discard point if it is not inside the image
                if not cv.inRange(target[key].numpy(), lowerb, upperb).all():
                    target[key] = torch.ones(2) * -1

        return image, target


class RandomCrop(object):
    def __init__(self, size):
        self.dest_width = size[0]
        self.dest_height = size[1]

    def __call__(self, image, target):
        width, height = image.size

        left = np.random.randint(0, width - self.dest_width)
        top = np.random.randint(0, height - self.dest_height)

        image = torchvision.transforms.functional.crop(image, top, left, self.dest_height, self.dest_width)

        # boundaries of the crop
        lowerb = np.array([0, 0], dtype=np.float32)
        upperb = np.array([self.dest_width - 1, self.dest_height - 1], dtype=np.float32)

        for key in target.keys():
            if target[key][0] != -1:
                target[key] -= torch.tensor([left, top])

                # discard point if it is not inside the crop
                if not cv.inRange(target[key].numpy(), lowerb, upperb).all():
                    target[key] = torch.ones(2) * -1

        return image, target


class CenterCrop(object):
    def __init__(self, size):
        self.dest_width = size[0]
        self.dest_height = size[1]

    def __call__(self, image, target):
        width, height = image.size

        left = int(round((width - self.dest_width) / 2.0))
        top = int(round((height - self.dest_height) / 2.0))

        image = torchvision.transforms.functional.crop(image, top, left, self.dest_height, self.dest_width)

        # boundaries of the crop
        lowerb = np.array([0, 0], dtype=np.float32)
        upperb = np.array([self.dest_width - 1, self.dest_height - 1], dtype=np.float32)

        for key in target.keys():
            if target[key][0] != -1:
                target[key] -= torch.tensor([left, top])

                # discard point if it is not inside the crop
                if not cv.inRange(target[key].numpy(), lowerb, upperb).all():
                    target[key] = torch.ones(2) * -1

        return image, target


class NormalizePoints(object):
    def __call__(self, image, target):
        shape = torch.tensor(image.size)

        for key in target.keys():
            if target[key][0] != -1:
                target[key] /= shape

        return image, target


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = torchvision.transforms.functional.resize(image, self.size)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = torchvision.transforms.functional.to_tensor(image)
        return image, target

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target
