import csv
import os

import PIL
import torch
import torchvision


IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD  = [0.229, 0.224, 0.225]


class TranslationDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.data_csv = os.path.join(dataset_dir, 'data.csv')

        self.data = []
        with open(self.data_csv, 'r') as f:
            dict_reader = csv.DictReader(f)

            for row in dict_reader:
                sub_dict = dict((key, row[key] if key == 'image_name' else float(row[key])) for key in ['image_name', 'x', 'y', 'z'])
                self.data.append(sub_dict)

        self.transforms = torchvision.transforms.Compose([
            #TODO: maybe add color jitter?
            torchvision.transforms.CenterCrop(512),
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(IMAGE_NET_MEAN, IMAGE_NET_STD)
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]

        # load the image
        image = PIL.Image.open(os.path.join(self.dataset_dir, item['image_name'])).convert('RGB')
        image = self.transforms(image)

        translation = torch.tensor((item['x'], item['y'], item['z']))

        return image, translation


class RotationDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.data_csv = os.path.join(dataset_dir, 'data.csv')

        self.data = []
        with open(self.data_csv, 'r') as f:
            dict_reader = csv.DictReader(f)

            for row in dict_reader:
                #sub_dict = dict((key, row[key] if key == 'image_name' else float(row[key])) for key in ['image_name', 'roll', 'pitch', 'yaw'])
                sub_dict = dict((key, row[key] if key == 'image_name' else float(row[key])) for key in ['image_name', 'qx', 'qy', 'qz', 'qw'])
                self.data.append(sub_dict)

        self.transforms = torchvision.transforms.Compose([
            #TODO: maybe add color jitter?
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(IMAGE_NET_MEAN, IMAGE_NET_STD)
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]

        # load the image
        image = PIL.Image.open(os.path.join(self.dataset_dir, item['image_name'])).convert('RGB')
        image = self.transforms(image)

        rotation = torch.tensor((item['qx'], item['qy'], item['qz'], item['qw']))

        return image, rotation


if __name__ == '__main__':
    import argparse

    import numpy as np
    import cv2 as cv

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir')
    parser.add_argument('--translation', action='store_true')
    args = parser.parse_args()

    def revert_image_net_mean(img):
        if len(img.shape) != 3 and img.shape[0] != 3:
            raise ValueError(f'Expected RGB image in order (channels, height, width) but got {img.shape}')

        result = torch.empty(img.shape, dtype=img.dtype)
        for channel in range(img.shape[0]):
            result[channel] = (img[channel] * IMAGE_NET_STD[channel]) + IMAGE_NET_MEAN[channel]
        return result

    def to_opencv(img):
        if len(img.shape) != 3 and img.shape[0] != 3:
            return (img.detach().numpy() * 255).astype(np.uint8)
        _img = torchvision.transforms.functional.to_pil_image(img)
        return np.array(_img)[:, :, ::-1].copy()

    dataset = TranslationDataset(args.dataset_dir) if args.translation else RotationDataset(args.dataset_dir)
    for i in range(len(dataset)):
        img, rotation = dataset[i]

        print(f'{i:5d}: Rotation {rotation}')

        cv.imshow('Original', cv.imread(os.path.join(args.dataset_dir, dataset.data[i]['image_name'])))
        cv.imshow('Transformed', to_opencv(img))
        cv.imshow('Transformed without Notmalization', to_opencv(revert_image_net_mean(img)))
        if cv.waitKey(0) == ord('q'):
            break
