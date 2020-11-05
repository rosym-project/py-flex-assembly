import os
import time
import sys

import torch
from loguru import logger

from vgg_heatmap import VGG_Heatmap
from point_detection import get_transform, HeatmapDataset

logger.add('train.log', level='INFO')

dir_data_train = os.path.join('data', 'reduced')
dir_data_val = os.path.join('data', 'reduced')
dataset_train = HeatmapDataset(dir_data_train, os.path.join(dir_data_train, 'data.json'), get_transform(train=True))
data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=2, shuffle=True)

dataset_val = HeatmapDataset(dir_data_val, os.path.join(dir_data_val, 'data.json'), get_transform(train=False))
data_loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=2, shuffle=True)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

detector = VGG_Heatmap(dataset_train.point_number)
detector = detector.float()
detector = detector.to(device)

optimizer = torch.optim.SGD([{'params': detector.encoder.parameters(), 'lr': 1e-4, 'momentum': 0.95},
                             {'params': detector.decoder.parameters(), 'lr': 1e-3, 'momentum': 0.90}])
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, 0.1)


epochs = 500
for i in range(1, epochs + 1):
    detector = detector.train()
    epoch_loss = 0
    since = time.time()
    for inputs, annotations in data_loader_train:
        optimizer.zero_grad()

        inputs = inputs.to(device)
        annotations = annotations.to(device)

        outputs = detector(inputs)
        loss = detector.compute_loss(outputs, annotations)

        epoch_loss += loss.detach()

        loss.backward()
        optimizer.step()
    to = time.time() - since
    # logger.debug(f'Epoch time {to:.3f}s avg {1000 * to / len(data_loader_train):.1f}ms')
    lr_scheduler.step()

    epoch_loss = epoch_loss / len(data_loader_train)
    logger.info(f'Epoch {i}/{epochs} - Training - Loss: {epoch_loss:.4f}')

    if i % 50 == 0:
        path = f'./snapshots/point_detector_{i}.model'
        torch.save(detector.state_dict(), path)

        detector = detector.eval()
        with torch.no_grad():
            loss = 0
            for inputs, annotations in data_loader_train:
                inputs = inputs.to(device)
                annotations = annotations.to(device)

                outputs = detector(inputs)
                loss += detector.compute_loss(outputs, annotations).detach()
            logger.info(f'Epoch {i}/{epochs} - Validation - Loss: {loss:.4f}')

# save the trained model
# torch.save(detector.state_dict(), 'point_detector.model')

