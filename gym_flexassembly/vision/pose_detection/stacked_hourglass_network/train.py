import os
import time

import torch

from models.vgg_heatmap import VGG_Heatmap
from point_detection import get_transform, HeatmapDataset


dir_data = './data'
dataset_train = HeatmapDataset(dir_data, os.path.join(dir_data, 'data.json'), get_transform(train=True))
data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=1, shuffle=True)

dataset_val = HeatmapDataset(dir_data, os.path.join(dir_data, 'data.json'), get_transform(train=False))
data_loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=True)


detector = VGG_Heatmap(dataset_train.point_number)
optimizer = torch.optim.SGD(detector.parameters(), lr=1e-4, momentum=0.95)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

epochs = 5
for i in range(1, epochs + 1):
    detector = detector.train()
    epoch_loss = 0
    since = time.time()
    for inputs, annotations in data_loader_train:
        optimizer.zero_grad()

        inputs = inputs.to(device)

        outputs = detector(inputs)
        loss = detector.compute_loss(outputs, annotations)

        epoch_loss += loss

        loss.backward()
        optimizer.step()
    to = time.time() - since
    print(f'Epoch time {to:.3f} avg {to / len(data_loader_train):.3f}')

    epoch_loss = epoch_loss / len(data_loader_train)
    print(f'Epoch {i}/{epochs} - Training - Loss: {epoch_loss:.4f}')

    if i % 1 == 0:
        path = f'./model/point_detector_{i}.model'
        torch.save(detector.state_dict(), path)

        detector = detector.eval()
        loss = 0
        for inputs, annotations in data_loader_train:
            outputs = detector(inputs)
            loss += detector.compute_loss(outputs, annotations)
        print(f'Epoch {i}/{epochs} - Validation - Loss: {loss:.4f}')

# save the trained model
# torch.save(detector.state_dict(), 'point_detector.model')

