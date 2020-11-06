import argparse
import os
import time
import sys
from enum import Enum

class Phase(Enum):
    TRAIN = 1
    VALIDATION = 2

import torch
from loguru import logger

from vgg_heatmap import VGGHeatmapComplex, VGGHeatmapSimple
from point_detection import get_transform, HeatmapDataset

parser = argparse.ArgumentParser(description='train a vgg heatmap model',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# dataset params
parser.add_argument('--dir_train', type=str, required=True,
                    help='the directory of the training data')
parser.add_argument('--dir_val', type=str, required=True,
                    help='the directory of the validation data')
# model type
parser.add_argument('--complex', action='store_true',
                    help='use the more complex vgg heatmap model')
# training params
parser.add_argument('--batch_size', type=int, default=8,
                    help='the batch size used for training/validation')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs to train')
parser.add_argument('--snapshot_epoch', type=int, default=10,
                    help='every snapshot_epoch the current weights are saved')
parser.add_argument('--snapshot_dir', type=str, default='./snapshots',
                    help='snapshots are saved to this directory')
parser.add_argument('--validation_epoch', type=int, default=1,
                    help='every validation_epoch the validation loss is computed')
# continue training
parser.add_argument('--weights', type=str,
                    help='load weights to continue training from')
parser.add_argument('--starting_epoch', type=int, default=0,
                    help='the epoch at which to start training (should be defined when training is continued)')
# logging
parser.add_argument('--logfile', type=str, default='traing.log',
                    help='the file to which the training progress is logged')
args = parser.parse_args()

logger.add(args.logfile, level='DEBUG')
logger.debug(f'Args: {args}')

datasets = { Phase.TRAIN: HeatmapDataset(args.dir_train, get_transform(train=True)),
             Phase.VALIDATION: HeatmapDataset(args.dir_val, get_transform(train=False)) }

def get_dataloader(phase, dataset):
    return torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=(phase == Phase.TRAIN))
dataloaders = dict(map(lambda phase: (phase, get_dataloader(phase, datasets[phase])), datasets))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
def load_model(complex_type, weights):
    init_vgg = weights is None
    point_number = datasets[Phase.TRAIN].point_number
    if complex_type:
        model = VGGHeatmapComplex(point_number, init_vgg=init_vgg)
    else:
        model = VGGHeatmapSimple(point_number, init_vgg=init_vgg)

    if weights:
        model.load_state_dict(torch.load(args.weights, map_location=device))

    model = model.float()
    model = model.to(device)
    return model
detector = load_model(args.complex, args.weights)
optimizer = torch.optim.SGD([{'params': detector.encoder.parameters(), 'lr': 1e-4, 'momentum': 0.95},
                             {'params': detector.decoder.parameters(), 'lr': 1e-3, 'momentum': 0.90}])
# lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, 0.1)

def run_epoch(detector, dataloader, train):
    logger.trace(f'Run epoch train={train}')
    detector = detector.train() if train else detector.eval()

    epoch_loss = 0
    for i, (inputs, annotations) in enumerate(dataloader):
        print(f'Epoch iteration: {i} / {len(dataloaders[phase])}', end='\r')
        if train:
            optimizer.zero_grad()

        inputs = inputs.to(device)
        annotations = annotations.to(device)

        outputs = detector(inputs)
        loss = detector.compute_loss(outputs, annotations)

        epoch_loss += loss.detach()

        if train:
            loss.backward()
            optimizer.step()
    return epoch_loss / len(dataloaders[phase])


def log_loss(iteration, phase, loss):
    logger.info(f'Epoch {iteration}/{args.epochs} - {phase.name} - Loss: {loss:.4f}')

def save_weights(iteration, detector):
    path = os.path.join(args.snapshot_dir, f'{detector.__class__.__name__}_{i}.model')
    logger.debug(f'Save weights at {path}')
    torch.save(detector.state_dict(), path)

# compute initial training and validation loss
for phase in Phase:
    log_loss(0, phase, run_epoch(detector, dataloaders[phase], False))

# perform training
for i in range(args.starting_epoch + 1, args.epochs + 1):
    loss = run_epoch(detector, dataloaders[Phase.TRAIN], True)
    log_loss(i, Phase.TRAIN, loss)

    if i % args.validation_epoch == 0:
        with torch.no_grad():
            loss = run_epoch(detector, dataloaders[Phase.VALIDATION], False)
        log_loss(i, Phase.VALIDATION, loss)

    if i % args.snapshot_epoch == 0:
        save_weights(i, detector)

    # lr_scheduler.step()

# if the model was not saved through snapshot dir save it now
if i % args.snapshot_epoch != 0:
    save_weights(i, detector)

