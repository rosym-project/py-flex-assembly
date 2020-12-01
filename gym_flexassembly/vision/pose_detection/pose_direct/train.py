import argparse
import importlib

from loguru import logger
import torch

import models
from models import RotationDetector, TranslationDetector
from datasets import RotationDataset, TranslationDataset


def print_loss(loss, iteration, epochs, logger, train=True):
    t = 'TRAIN' if train else 'VAL' 
    print(f'Epoch {iteration:{len(str(epochs))}d} / {epochs} : Loss={loss:.6f} : {t}')

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
        loss = loss_function(outputs, annotations)

        epoch_loss += loss.detach()

        if train:
            loss.backward()
            optim.step()

    return epoch_loss / len(data_loader)


parser = argparse.ArgumentParser()
parser.add_argument('dataset_train')
parser.add_argument('dataset_val')
parser.add_argument('--translation', action='store_true')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--momentum', type=float, default=0.95)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--output_file', type=str, default='detector.pth')
parser.add_argument('--logfile', type=str, default='train.log',
                    help='the file to which the training progress is logged')
parser.add_argument('--backend', type=str, default=models.backends[1].__name__)
args = parser.parse_args()

logger.add(args.logfile, level='DEBUG')
logger.info(f'Args: {args}')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
logger.debug('Use device', device)

backend = getattr(models, args.backend)()
detector = TranslationDetector(backend) if args.translation else RotationDetector(backend)
detector = detector.to(device)

dataset_train = TranslationDataset(args.dataset_train) if args.translation else RotationDataset(args.dataset_train)
data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
dataset_val = TranslationDataset(args.dataset_val) if args.translation else RotationDataset(args.dataset_val)
data_loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=True)

optim = torch.optim.SGD(detector.parameters(), lr=args.lr, momentum=args.momentum)
loss_function = torch.nn.MSELoss()

for i in range(1, args.epochs + 1):
    epoch_loss_train = run_epoch(detector, data_loader_train, loss_function, optim, device, train=True) 
    print_loss(epoch_loss_train, i, args.epochs, logger, True)

    epoch_loss_train = run_epoch(detector, data_loader_val, loss_function, optim, device, train=False) 
    print_loss(epoch_loss_train, i, args.epochs, logger, False)

detector_type = 'translation' if args.translation else 'rotation'
torch.save(detector.state_dict(), f'{detector_type}_{args.output_file}')
