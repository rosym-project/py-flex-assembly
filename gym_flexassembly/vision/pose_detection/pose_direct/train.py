import argparse
import importlib

from loguru import logger
import torch

import gym_flexassembly.vision.pose_detection.pose_direct.datasets as datasets
import gym_flexassembly.vision.pose_detection.pose_direct.util as util


def print_loss(loss, iteration, epochs, logger, train=True):
    phase = 'TRAIN' if train else 'VAL' 
    logger.info(f'Epoch={iteration}/{epochs}, Loss={loss:.6f}, Phase={phase}')


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


parser = util.load_model_parser(model_type='', description='train a pose direct model')
parser.add_argument('--data_train', required=True, type=str,
                    help='location of the training data set')
parser.add_argument('--data_val', required=True, type=str,
                    help='location of the validation data set')
parser.add_argument('--translation', action='store_true',
                    help='if a translation or totation model should be trained')
parser.add_argument('--batch_size', type=int, default=16,
                    help='the batch size')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='the learning rate')
parser.add_argument('--momentum', type=float, default=0.95,
                    help='the momentum')
parser.add_argument('--epochs', type=int, default=30,
                    help='the number of epochs to train')
parser.add_argument('--output_file', type=str, default='detector',
                    help='the name of the file into which the trained weights are saved')
parser.add_argument('--logfile', type=str, default='train.log',
                    help='the file to which the training progress is logged')
args = parser.parse_args()

logger.add(args.logfile, level='DEBUG')
logger.info(f'Args: {args}')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
logger.debug(f'Use device {device}')

model_type = 'translation' if args.translation else 'rotation'
detector = util.load_model(args, device, model_type)

dataset_train = datasets.TranslationDataset(args.data_train) if args.translation else datasets.RotationDataset(args.data_train)
data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
dataset_val = datasets.TranslationDataset(args.data_val) if args.translation else datasets.RotationDataset(args.data_val)
data_loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=True)

optim = torch.optim.SGD(detector.parameters(), lr=args.lr, momentum=args.momentum)
loss_function = torch.nn.MSELoss()

for i in range(1, args.epochs + 1):
    epoch_loss_train = run_epoch(detector, data_loader_train, loss_function, optim, device, train=True) 
    print_loss(epoch_loss_train, i, args.epochs, logger, True)

    epoch_loss_train = run_epoch(detector, data_loader_val, loss_function, optim, device, train=False) 
    print_loss(epoch_loss_train, i, args.epochs, logger, False)

detector_type = 'translation' if args.translation else 'rotation'
torch.save(detector.state_dict(), f'{detector_type}_{args.output_file}.pth')
