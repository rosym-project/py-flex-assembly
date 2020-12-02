import argparse

import cv2 as cv
import numpy as np
import PIL
import torch

import models

def expand_to_range(left, right, new_size, lower, upper):
    """
    Compute new left and right so that right - left = new_size and
    left >= lower and right <= upper.
    """
    old_size = right - left
    half = int((new_size - old_size) / 2)

    if left - half < lower:
        right = right + half - (left - half)
        left = lower
    elif right + half > upper:
        left = left - half - (right + half - upper)
        right = upper
    else:
        left = left - half
        right = right + half

    return left, right


DEFAULT_LOWER_HSV = np.array([75, 60, 20])
DEFAULT_UPPER_HSV = np.array([120, 255, 255])
def crop_image(img, lower_hsv=DEFAULT_LOWER_HSV, upper_hsv=DEFAULT_UPPER_HSV):
    """
    Crop a rectangular region from an image containing
    a clamp.
    """
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    mask = cv.inRange(hsv, lower_hsv, upper_hsv)
    mask = cv.dilate(mask, np.ones((3, 3), np.uint8), iterations=5)
    mask = cv.erode(mask, np.ones((3, 3), np.uint8), iterations=5)

    _, _, stats, _ = cv.connectedComponentsWithStats(mask, connectivity=8)
    stats = list(stats)[1:]
    if len(stats) > 1:
        pass
        #raise ValueError(f'WARNING: Detected multiple ({len(stats)}) crop locations')
    stats.sort(key=lambda stat: -stat[cv.CC_STAT_AREA])
    stat = stats[0]

    width,  height = stat[cv.CC_STAT_WIDTH], stat[cv.CC_STAT_HEIGHT]
    top,    left   = stat[cv.CC_STAT_TOP],   stat[cv.CC_STAT_LEFT]
    bottom, right  = top + height,           left + width

    if width < height:
        left, right = expand_to_range(left, right, height, 0, img.shape[1])
    else:
        top, bottom = expand_to_range(top, bottom, width, 0, img.shape[0])

    return img[top:bottom, left:right], mask


class Crop_Clamp:

    def __init__(self, lower_hsv=DEFAULT_LOWER_HSV, upper_hsv=DEFAULT_UPPER_HSV):
        self.lower_hsv = lower_hsv
        self.upper_hsv = upper_hsv

    def __call__(self, img):
        return crop_image(img, lower_hsv=self.lower_hsv, upper_hsv=self.upper_hsv)


def opencv_to_PIL(img):
    return PIL.Image.fromarray(cv.cvtColor(img, cv.BGR2RGB))


class OpenCVToPil:

    def __call__(self, img):
        return opencv_to_PIL(img)


def get_rotation_transform(train=False):
    #TODO: add noise in case of training_mode
    return torchvision.transforms.Compose([
        OpenCVToPil(),
        Crop_Clamp(),
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(IMAGE_NET_MEAN, IMAGE_NET_STD)
    ])


def pre_process_rotation(img):
    return get_rotation_transform()(img).unsqueeze(dim=0)


def load_model_parser(model_type='rotation', description='', parser=None):
    if parser is None:
        _parser = argparse.ArgumentParser(description=description,
                                          formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    else:
        _parser = parser

    
    backend_names = list(map(lambda backend_cls: backend_cls.__name__, models.backends))
    _parser.add_argument(f'--{model_type}_backend', type=str, default=models.backends[-1].__name__,
            help=f'set the backend used by the model. Backends: {backend_names}')
    _parser.add_argument(f'--{model_type}_weights', type=str, default=None,
             help='set a weights file to restore the model weights. If None the backend is still loaded with pretrained weights')

    return _parser


def load_model(args, device, model_type='rotation'):
    backend_cls_name = getattr(args, f'{model_type}_backend')
    weights = getattr(args, f'{model_type}_weights')

    backend = getattr(models, backend_cls_name)(pretrained=(weights == None))
    model = models.RotationDetector(backend) if model_type == 'rotation' else models.TranslationDetector(backend)

    if weights is not None:
        model.load_state_dict(torch.load(weights, map_location=device), strict=True)

    model = model.to(device)
    return model

if __name__ == '__main__':
    parser = load_model_parser()
    args = parser.parse_args()
    print(args)

    model = load_model(args, torch.device('cpu'))
    # import os

    # in_dir = 'pose_datasets/large'
    # out_dir = 'pose_datasets/large_cropped'
    # image_files = [f for f in os.listdir(in_dir) if f.endswith('.png')] 
    # image_files.sort()

    # for image_file in image_files:
        # img = cv.imread(os.path.join(in_dir, image_file), cv.IMREAD_COLOR)

        # img, mask = crop_image(img)

        # print('Write: ' + os.path.join(out_dir, image_file))
        # cv.imwrite(os.path.join(out_dir, image_file), img)

        # # cv.imshow('Image', img)
        # # cv.imshow('Mask', mask)
        # # if cv.waitKey(0) == ord('q'):
            # # break
        # # try:
            # # img = crop_image(img)

            # # cv.imshow('Image', img)
            # # cv.waitKey(0)
        # # except ValueError as e:
            # # print(e)
            # # cv.imshow('Fail', img)
            # # cv.waitKey(0)

