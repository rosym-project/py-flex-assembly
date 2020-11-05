import argparse
import os

import cv2 as cv
import numpy as np
import torch
import torchvision

from point_detection import get_transform, HeatmapDataset
from transforms import revert_image_net_mean, to_opencv
from vgg_heatmap import VGG_Heatmap

parser = argparse.ArgumentParser(description='Show the points detected by the vgg heatmap model for images in a dataset')
# parser.add_argument('weights', type=str)
parser.add_argument('data_dir', type=str, default='./data/flat')
args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

dataset = HeatmapDataset(args.data_dir, os.path.join(args.data_dir, 'data.json'), get_transform(train=False))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

detector = VGG_Heatmap(dataset.point_number, init_vgg=True)
# detector = VGG_Heatmap(dataset.point_number, init_vgg=False)
# detector.load_state_dict(torch.load(args.weights, map_location=device))
detector = detector.to(device)
detector.eval()

for i, (inputs, targets) in enumerate(dataloader):
    inputs = inputs.to(device)
    targets = targets.to(device)
    outputs = detector(inputs)

    loss = detector.compute_loss(targets, outputs)
    print(f'Loss {loss:.5f}')

    img = inputs.squeeze()
    target = targets.squeeze()
    output = outputs.squeeze()

    img = to_opencv(revert_image_net_mean(img))

    t_img = np.zeros((512, 512), np.uint8)
    for i in range(target.shape[0]):
        t_img = cv.addWeighted(t_img, 1.0, (target[i].detach().numpy() * 255).astype(np.uint8), 1.0, 0.0)

    o_img = np.zeros((512, 512), np.uint8)
    for i in range(output.shape[0]):
        o = output[i].detach().numpy()
        o = o - o.min()
        # print(i, o.min())
        o = o / o.max()
        # print(i, o.min(), o.max())
        o_img = cv.addWeighted(o_img, 1.0, (o * 255).astype(np.uint8), 1.0, 0.0)
        # print('Out:', i)
        # cv.imshow('Out', (o * 255).astype(np.uint8))
        # cv.waitKey(0)

    # TODO: this should revert the image net normalization
    # res = np.zeros((*img.shape[1:], img.shape[0]), np.uint8)
    # res[:, :, 0] = img[2]
    # res[:, :, 1] = img[1]
    # res[:, :, 2] = img[0]
    # res = (res * 255).astype(np.uint8)

    # for i in range(target.shape[0]):
        # heatmap = np.where(target[i] > 0, np.ones(t_img.shape, dtype=np.uint8), np.zeros(t_img.shape, dtype=np.uint8))
        # num_targets, _, _, centroids = cv.connectedComponentsWithStats(heatmap)
        # if num_targets > 1:
            # cv.circle(res, tuple(np.array(centroids[1]).round().astype(np.int)), 1, (0, 0, 255), thickness=-1)

    cv.imshow('Clamp Image', img)
    cv.imshow('Combined Heatmaps', t_img)
    cv.imshow('Combined Outputs', o_img)
    if cv.waitKey(0) == ord('q'):
        break

