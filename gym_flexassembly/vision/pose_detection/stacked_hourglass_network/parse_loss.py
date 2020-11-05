import argparse

parser = argparse.ArgumentParser()
parser.add_argument('log_file', type=str, default='train.log')
args = parser.parse_args()

with open(args.log_file, mode='r') as f:
    lines = [line.strip() for line in f.readlines()]

lines = list(filter(lambda line: 'INFO' in line, lines))
lines = list(map(lambda line: '-'.join(line.split('-')[3:]), lines))

data = {}
for l in lines:
    epoch, phase, loss = l.split('-')

    epoch = int(epoch.strip().split(' ')[1].split('/')[0])
    phase = phase.strip().lower()
    loss = float(loss.strip().split(' ')[-1])

    if phase not in data:
        data[phase] = {'epoch': [], 'loss': []}

    data[phase]['epoch'].append(epoch)
    data[phase]['loss'].append(epoch)
