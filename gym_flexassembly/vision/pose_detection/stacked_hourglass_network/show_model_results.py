import argparse
import os

import cv2 as cv
import numpy as np
import torch
import torchvision

from point_detection import get_transform, HeatmapDataset
from transforms import revert_image_net_mean, to_opencv
from vgg_heatmap import VGGHeatmapComplex, VGGHeatmapSimple

parser = argparse.ArgumentParser(description='Show the points detected by the vgg heatmap model for images in a dataset')
parser.add_argument('weights', type=str)
parser.add_argument('data_dir', type=str, default='./data/flat')
args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

dataset = HeatmapDataset(args.data_dir, get_transform(train=False))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

def get_model_cls(weights):
    model_clss = dict([(model_cls.__name__, model_cls) for model_cls in [VGGHeatmapComplex, VGGHeatmapSimple]])
    for name, cls in model_clss.items():
        if name in weights:
            return cls
    raise ValueError(f'Cannot find model class for weights {weights}')

detector = get_model_cls(args.weights)(dataset.point_number, init_vgg=False)
detector.load_state_dict(torch.load(args.weights, map_location=device))
detector = detector.to(device)
detector.eval()

def combine_imgs(imgs):
    res = np.zeros(imgs[0].shape, np.uint8)
    for img in imgs:
        res = cv.addWeighted(res, 1.0, img, 1.0, 0.0)
    return res

def add_out_row(out, imgs, row, offset=10):
    row_idx_from = row * imgs[0].shape[0] + row * offset
    row_idx_to   = row_idx_from + imgs[0].shape[0]

    for i, img in enumerate(imgs):
        column_idx_from = i * (offset + img.shape[1])
        column_idx_to   = column_idx_from + img.shape[1]
        out[row_idx_from:row_idx_to, column_idx_from:column_idx_to] = img
    return out

def first_component(stats):
    return 1

def largest_component(stats):
    areas = stats[1:]
    areas = list(map(lambda x: (x[0] + 1, x[1][cv.CC_STAT_AREA]), enumerate(stats)))
    areas.sort(key=lambda area: -area[1])
    return areas[0][0]

def create_point_img(img, component_chooser, threshold=0):
    res = np.zeros(img.shape, np.uint8)
    
    _, heatmap = cv.threshold(img, threshold, 255, cv.THRESH_BINARY)
    num_labels, label_img, stats, centroids = cv.connectedComponentsWithStats(heatmap)
    if num_labels == 1:
        return res

    label = component_chooser(stats)
    cv.circle(res, tuple(np.array(centroids[label]).round().astype(np.int)), 3, 255, thickness=-1)
    return res

for i, (inputs, targets) in enumerate(dataloader):
    inputs = inputs.to(device)
    targets = targets.to(device)
    outputs = detector(inputs)

    loss = detector.compute_loss(targets, outputs)
    print(f'Loss {loss:.5f}')

    img = inputs.squeeze()
    targets = targets.squeeze()
    outputs = outputs.squeeze()

    img = to_opencv(revert_image_net_mean(img))

    offset = 10
    shape = (targets.shape[1] * 4 + (3 * offset), targets.shape[2] * (targets.shape[0] + 1) + targets.shape[0] * offset)
    out = np.ones(shape, np.uint8) * 127

    targets = [to_opencv(target) for target in targets]
    targets.append(combine_imgs(targets))
    target_points = [create_point_img(target, first_component) for target in targets[:-1]]
    target_points.append(combine_imgs(target_points))

    outputs = [to_opencv(output) for output in outputs]
    outputs.append(combine_imgs(outputs))
    output_points = [create_point_img(output, largest_component) for output in outputs[:-1]]
    output_points.append(combine_imgs(output_points))

    add_out_row(out, targets, 0, offset=offset)
    add_out_row(out, target_points, 1, offset=offset)
    add_out_row(out, outputs, 2, offset=offset)
    add_out_row(out, output_points, 3, offset=offset)

    # t_img = np.zeros((512, 512), np.uint8)
    # for i in range(target.shape[0]):
        # as_numpy = (target[i].detach().numpy() * 255).astype(np.uint8)
        # t_img = cv.addWeighted(t_img, 1.0, as_numpy, 1.0, 0.0)

        # out[:target.shape[1], (i * offset) + (i * target.shape[2]):(i * offset) + ((i + 1) * (target.shape[2]))] = as_numpy

        # heatmap = np.where(target[i] > 0, np.ones(t_img.shape, dtype=np.uint8), np.zeros(t_img.shape, dtype=np.uint8))
        # num_labels, _, _, centroids = cv.connectedComponentsWithStats(heatmap)
        # if num_labels > 1:
            # cv.circle(img, tuple(np.array(centroids[1]).round().astype(np.int)), 1, (0, 0, 255), thickness=-1)
    # out[:target.shape[1], (target.shape[0] * offset) + (target.shape[0] * target.shape[2]):] = t_img

    # o_img = np.zeros((512, 512), np.uint8)
    # for i in range(output.shape[0]):
        # as_numpy = (output[i].detach().numpy() * 255).astype(np.uint8)
        # o_img = cv.addWeighted(o_img, 1.0, as_numpy, 1.0, 0.0)

        # out[target.shape[1]:, (i * target.shape[2]):((i + 1) * target.shape[2])] = as_numpy

        # heatmap = output[i].detach().numpy()
        # heatmap = np.where(heatmap < 0.5, np.zeros(heatmap.shape), heatmap)
        # heatmap = np.where(heatmap > 0, np.ones(o_img.shape, dtype=np.uint8), np.zeros(o_img.shape, dtype=np.uint8))
        # # cv.imshow('Heatmap ' + str(i), heatmap * 255)
        # # cv.waitKey(0)
        # num_labels, _, _, centroids = cv.connectedComponentsWithStats(heatmap)
        # # print(f'Regions {num_labels - 1} for output {i}')
        # for j in range(1, num_labels):
            # cv.circle(img, tuple(np.array(centroids[j]).round().astype(np.int)), 3, (0, 255, 0), thickness=-1)
    # out[target.shape[1]:, (target.shape[0] * offset) + (target.shape[0] * target.shape[2]):] = o_img

    cv.namedWindow('Outputs', cv.WINDOW_NORMAL)
    cv.resizeWindow('Outputs', 1920, 1080)
    cv.imshow('Outputs', out)
    # cv.imshow('Clamp Image', img)
    # cv.imshow('Combined Heatmaps', t_img)
    # cv.imshow('Combined Outputs', (output[2].detach().numpy() * 255).astype(np.uint8))
    # cv.imshow('Combined Outputs', o_img)
    if cv.waitKey(0) == ord('q'):
        break

