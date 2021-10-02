import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd

from datasetGenerator import CustomDataset
from model import AnagnorModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

traindataset = CustomDataset()
trainloader = torch.utils.data.DataLoader(traindataset, batch_size=8,
                                         shuffle=False, num_workers=2)


net = AnagnorModel().to(device)

import torch.optim as optim

criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i%100==99:
            torch.save(net.state_dict(), "./checkpoints/model_{epoch}_i.pth")

        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')
torch.save(net.state_dict(), "./final_model.pth")