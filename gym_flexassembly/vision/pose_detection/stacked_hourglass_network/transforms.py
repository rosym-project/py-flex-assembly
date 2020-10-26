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

            for i in range(len(target)):
                target[i] = torchvision.transforms.functional.hflip(target[i])

        return image, target


class RandomVerticalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if np.random.rand() < self.prob:
            image = torchvision.transforms.functional.vflip(image)

            for i in range(len(target)):
                target[i] = torchvision.transforms.functional.vflip(target[i])

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

        for i in range(len(target)):
            target[i] = torchvision.transforms.functional.rotate(target[i], angle)

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

        for i in range(len(target)):
            target[i] = torchvision.transforms.functional.crop(target[i], top, left, self.dest_height, self.dest_width)

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

        for i in range(len(target)):
            target[i] = torchvision.transforms.functional.crop(target[i], top, left, self.dest_height, self.dest_width)

        return image, target


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = torchvision.transforms.functional.resize(image, self.size)

        for i in range(len(target)):
            target[i] = torchvision.transforms.functional.resize(target[i], self.size)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = torchvision.transforms.functional.to_tensor(image)
        target = torch.cat([torchvision.transforms.functional.to_tensor(t) for t in target])
        return image, target

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToPILImage(object):
    def __call__(self, image, target):
        image = self._to_pil_image(image)
        target = [self._to_pil_image(target[i]) for i in range(target.shape[0])]
        return image, target

    def _to_pil_image(self, val):
        if not isinstance(val, Image.Image):
            return torchvision.transforms.functional.to_pil_image(val)
        return val

