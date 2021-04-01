import argparse
import os

import cv2 as cv
import torch
import torchvision

import gym_flexassembly.vision.pose_detection.pose_direct.datasets as datasets
import gym_flexassembly.vision.pose_detection.pose_direct.util as util


def run_epoch(detector, data_loader, loss_function, optim, device, train=True):
    detector = detector.train() if train else detector.eval()

    i = 0
    epoch_loss = 0
    for inputs, annotations in data_loader:
        print(f'Iteration {i}/{len(data_loader)}', end='\r')
        i += 1

        inputs = inputs.to(device)
        annotations = annotations.to(device)

        if train:
            optim.zero_grad()

        inputs = inputs.to(device)
        outputs = detector(inputs)
        loss = loss_function(outputs, annotations) # needed for torch.nn.Module
        #loss = loss_function.apply(outputs, annotations) # needed for torch.autograd.Function

        epoch_loss += loss.detach() / data_loader.batch_size

        if train:
            regularization = 0.5 * torch.norm(outputs)
            (regularization + loss).backward()
            optim.step()

    return epoch_loss / len(data_loader)


parser = util.load_model_parser(model_type='', description='train a pose direct model')
parser.add_argument('--data', required=True, type=str,
                    help='location of the data set')
parser.add_argument('--translation', action='store_true',
                    help='if a translation or totation model should be trained')
args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Use device {device}')

model_type = 'translation' if args.translation else 'rotation'
detector = util.load_model(args, device, model_type)
detector = detector.eval()

dataset = datasets.TranslationDataset(args.data) if args.translation else datasets.RotationDataset(args.data)

loss_function = torch.nn.MSELoss() if args.translation else util.QuatLossModule()

transforms = util.get_translation_transform() if args.translation else util.get_rotation_transform()
if not args.translation:
    transforms = torchvision.transforms.Compose(transforms.transforms[1:])

total_loss = 0
for i, (image, targets) in enumerate(dataset):
    imagefile = dataset.data[i]['image_name']
    print(f'Process image {imagefile}', end='\r')

    outputs = detector(image.unsqueeze(dim=0))

    _inputs = cv.imread(os.path.join(dataset.dataset_dir, imagefile), cv.IMREAD_COLOR)
    _inputs = transforms(_inputs)
    _outputs = detector(_inputs.unsqueeze(dim=0))

    if not (outputs == _outputs).all():
        print(f'ERROR: outputs not equal {outputs} != {_outputs}')
        break

    targets = targets.unsqueeze(dim=0)
    total_loss += loss_function(outputs, targets).detach().detach()
total_loss = total_loss / len(dataset)
print(f'Total loss: {total_loss:.6f}')

detector = detector.train()
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
l = run_epoch(detector, dataloader, loss_function, None, device, train=False)
print(' ' * 100, end='\r')
print(f'Other loss: {l:.6f}')
