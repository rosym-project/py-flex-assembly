import argparse

import cv2 as cv
import numpy as np
import PIL
import torch
import torchvision

import gym_flexassembly.vision.pose_detection.pose_direct.models as models

IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD  = [0.229, 0.224, 0.225]

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

    return img[top:bottom, left:right]


class CropClamp:

    def __init__(self, lower_hsv=DEFAULT_LOWER_HSV, upper_hsv=DEFAULT_UPPER_HSV):
        self.lower_hsv = lower_hsv
        self.upper_hsv = upper_hsv

    def __call__(self, img):
        return crop_image(img, lower_hsv=self.lower_hsv, upper_hsv=self.upper_hsv)


def revert_image_net_mean(img):
    if len(img.shape) != 3 and img.shape[0] != 3:
        raise ValueError(f'Expected RGB image in order (channels, height, width) but got {img.shape}')

    result = torch.empty(img.shape, dtype=img.dtype)
    for channel in range(img.shape[0]):
        result[channel] = (img[channel] * IMAGE_NET_STD[channel]) + IMAGE_NET_MEAN[channel]
    return result

def opencv_to_PIL(img):
    return PIL.Image.fromarray(cv.cvtColor(img, cv.COLOR_BGR2RGB))


def torch_to_opencv(img):
    if len(img.shape) != 3 and img.shape[0] != 3:
        return (img.detach().numpy() * 255).astype(np.uint8)
    _img = torchvision.transforms.functional.to_pil_image(img)
    return np.array(_img)[:, :, ::-1].copy()

class OpenCVToPil:

    def __call__(self, img):
        return opencv_to_PIL(img)


def get_rotation_transform(train=False):
    #TODO: add noise in case of training_mode
    return torchvision.transforms.Compose([
        CropClamp(),
        OpenCVToPil(),
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(IMAGE_NET_MEAN, IMAGE_NET_STD)
    ])


def pre_process_rotation(img):
    return get_rotation_transform()(img).unsqueeze(dim=0)


def get_translation_transform(train=False):
    #TODO: add noise in case of training_mode
    return torchvision.transforms.Compose([
        OpenCVToPil(),
        torchvision.transforms.CenterCrop(512),
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(IMAGE_NET_MEAN, IMAGE_NET_STD)
    ])


def pre_process_translation(img):
    return get_translation_transform()(img).unsqueeze(dim=0)


def load_model_parser(model_type='rotation', description='', parser=None):
    if parser is None:
        _parser = argparse.ArgumentParser(description=description,
                                          formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    else:
        _parser = parser

    backend_arg = '--backend' if model_type == '' else f'--{model_type}_backend'
    weights_arg = '--weights' if model_type == '' else f'--{model_type}_weights'

    backend_names = list(map(lambda backend_cls: backend_cls.__name__, models.backends))
    _parser.add_argument(backend_arg, type=str, default=models.backends[-1].__name__,
            help=f'set the backend used by the model. Backends: {backend_names}')
    _parser.add_argument(weights_arg, type=str, default=None,
             help='set a weights file to restore the model weights. If None the backend is still loaded with pretrained weights')

    return _parser


def load_model(args, device, model_type='rotation'):
    try:
        backend_cls_name = getattr(args, f'{model_type}_backend')
    except AttributeError:
        backend_cls_name = getattr(args, 'backend')

    try:
        weights = getattr(args, f'{model_type}_weights')
    except AttributeError:
        weights = getattr(args, 'weights')

    backend = getattr(models, backend_cls_name)(pretrained=(weights == None))
    model = models.RotationDetector(backend) if model_type == 'rotation' else models.TranslationDetector(backend)

    if weights is not None:
        model.load_state_dict(torch.load(weights, map_location=device), strict=True)

    model = model.to(device)
    return model

class QuatLossModule(torch.nn.Module):
    def forward(self, inputs, targets):
        # Todo: remove normalization after implementing unit quaternions
        s = 1 / torch.norm(inputs, dim=1)
        i = inputs * s[:, None]
        # s = 1 / torch.norm(target, dim=1)
        # t = target * s[:, None]

        batch_size = inputs.shape[0]
        #TODO: this should always be 4 for quaternions right?
        # vec_length = inputs[0].shape[0]
        dots = torch.bmm(i.view(batch_size, 1, 4), targets.view(batch_size, 4, 1))

        res = torch.sum(torch.acos(torch.abs(dots)))
        return res

class QuatLossFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, target):
        # Todo: remove normalization after implementing unit quaternions
        s = 1 / torch.norm(input, dim=1)
        input *= s[:, None]
        s = 1 / torch.norm(target, dim=1)
        target *= s[:, None]

        ctx.save_for_backward(input, target)
        batch_size = input.shape[0]
        vec_length = input[0].shape[0]

        dots = torch.bmm(input.view(batch_size, 1, vec_length), target.view(batch_size, vec_length, 1))

        res = torch.sum(torch.acos(torch.abs(dots)))
        return res

    @staticmethod
    def backward(ctx, grad_output):
        input, target = ctx.saved_tensors

        batch_size = input.shape[0]
        vec_length = input[0].shape[0]

        dots = torch.bmm(input.view(batch_size, 1, vec_length), target.view(batch_size, vec_length, 1))
        dots = dots.flatten()

        sgn = torch.sign(dots)
        # since abs() isn't differentiable at x=0, those values need to be replaced manually
        sgn[sgn == 0] = 1

        abs = torch.abs(dots)
        # the values in abs have a range of [0, 1] since the quaternions have unit length
        # since 1/sqrt(1-x^2) is undefined for x=1, the range needs to be changed to [0, 1[
        # Todo: determine epsilon
        abs = torch.minimum(abs, torch.full(abs.shape, 1 - 1e-6))

        s = sgn * -1 / torch.sqrt(1 - abs ** 2)
        grad = target * s[:, None]
        return grad, None

if __name__ == '__main__':
    backend = models.Resnet18Backend()
    detector = models.RotationDetector(backend)

    lossf = QuatLossModule()
