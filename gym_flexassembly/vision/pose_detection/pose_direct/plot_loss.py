import argparse
import math

import matplotlib.pyplot as plt

def parse_line(line):
    _line = line.split('-')[-1].strip()
    epoch, loss, phase = _line.split(',')
    
    epoch = int(epoch.split('=')[1].split('/')[0])
    loss = float(loss.split('=')[1])
    phase = phase.split('=')[1]

    return epoch, loss, phase

def parse_log(logfile, loss_type=None):
    data = {}
    with open(logfile, mode='r') as f:
        next(f)
        next(f)
        for line in f:
            epoch, loss, phase = parse_line(line)

            if loss_type == 'translation':
                # convert loss into distance in cm
                loss = 100 * math.sqrt(loss)

            phase_data = data.get(phase, {})

            epochs = phase_data.get('epochs', [])
            epochs.append(epoch)
            phase_data['epochs'] = epochs

            losses = phase_data.get('loss', [])
            losses.append(loss)
            phase_data['loss'] = losses

            data[phase] = phase_data
    return data

def filter_data(epochs, loss, start_epoch=1, end_epoch=-1):
    if end_epoch == -1:
        end_epoch = epochs[-1]

    _epochs = []
    _loss = []
    for epoch, loss in zip(epochs, loss):
        if start_epoch <= epoch <= end_epoch:
            _epochs.append(epoch)
            _loss.append(loss)
    return _epochs, _loss


    

parser = argparse.ArgumentParser()
parser.add_argument('logfiles', nargs='+')
parser.add_argument('--loss_type', type=str, default=None,
                    help='set the loss type to convert the loss into a human readable measure')
parser.add_argument('--start_epoch', type=int, default=1)
parser.add_argument('--end_epoch', type=int, default=-1)
args = parser.parse_args()

data= {}
for logfile in args.logfiles:
    data[logfile.split('.')[0]] = parse_log(logfile, args.loss_type)

for log in data:
    _data = data[log]
    for phase in _data:
        epochs, loss = filter_data(_data[phase]['epochs'], _data[phase]['loss'], args.start_epoch, args.end_epoch)
        plt.plot(epochs, loss, label=f'{log} - {phase}')
plt.xlabel('Epochs')
if args.loss_type == 'translation':
    plt.ylabel('Distance in cm')
else:
    plt.ylabel('MSE Loss')
plt.legend()
plt.show()
