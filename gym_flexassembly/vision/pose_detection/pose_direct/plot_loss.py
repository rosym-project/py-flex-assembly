import argparse

import matplotlib.pyplot as plt

def parse_line(line):
    _line = line.split('-')[-1].strip()
    epoch, loss, phase = _line.split(',')
    
    epoch = int(epoch.split('=')[1].split('/')[0])
    loss = float(loss.split('=')[1])
    phase = phase.split('=')[1]

    return epoch, loss, phase

def parse_log(logfile):
    data = {}
    with open(logfile, mode='r') as f:
        next(f)
        next(f)
        for line in f:
            epoch, loss, phase = parse_line(line)

            phase_data = data.get(phase, {})

            epochs = phase_data.get('epochs', [])
            epochs.append(epoch)
            phase_data['epochs'] = epochs

            losses = phase_data.get('loss', [])
            losses.append(loss)
            phase_data['loss'] = losses

            data[phase] = phase_data
    return data

parser = argparse.ArgumentParser()
parser.add_argument('logfiles', nargs='+')
args = parser.parse_args()

data= {}
for logfile in args.logfiles:
    data[logfile.split('.')[0]] = parse_log(logfile)

for log in data:
    _data = data[log]
    for phase in _data:
        plt.plot(_data[phase]['epochs'], _data[phase]['loss'], label=f'{log} - {phase}')
plt.legend()
plt.show()
