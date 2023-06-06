import datetime
import gc
from typing import List, Mapping

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(4, 4)
        self.conv2 = nn.Conv2d(6, 16, 4)
        self.fc1 = nn.Linear(16 * 13 * 13, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 4)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def get_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    lr = 0.01
    momentum = 0.5
    decay = 0.01
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=momentum, weight_decay=decay
    )
    return optimizer


def train_model(
    model: nn.Module,
    loss_func: torch.nn.modules.loss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    dataloaders: Mapping[str, DataLoader],
    early_stop=10,
    num_epochs=5,
) -> nn.Module:
    start_time = datetime.datetime.now().replace(microsecond=0)
    model = model.to(device)

    # number of epochs to train the model
    valid_loss_min = np.Inf  # track change in validation loss
    early_stop_cnt = 0
    last_epoch_loss = np.Inf
    globaliter = 0

    for epoch in range(1, num_epochs + 1):
        globaliter += 1
        # keep track of training and validation loss
        train_loss = 0.0
        valid_loss = 0.0

        model.train()
        train_corrects = 0

        for data, target in dataloaders["train"]:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            # calculate the batch loss
            _, preds = torch.max(output, 1)
            loss = loss_func(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)
            train_corrects += torch.sum(preds == target.data)

        train_loss = train_loss / len(dataloaders["train"].dataset)
        train_acc = (train_corrects.double() * 100) / len(dataloaders["train"].dataset)

        # validate the model
        model.eval()
        val_corrects = 0
        for data, target in dataloaders["val"]:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, preds = torch.max(output, 1)
            loss = loss_func(output, target)
            valid_loss += loss.item() * data.size(0)
            val_corrects += torch.sum(preds == target.data)

        # calculate average losses
        valid_loss = valid_loss / len(dataloaders["val"].dataset)
        valid_acc = (val_corrects.double() * 100) / len(dataloaders["val"].dataset)

        # print training/validation statistics
        print(
            "Epoch: {} \tTraining Loss:  {:.6f} \tValidation Loss:  {:.6f}".format(
                epoch, train_loss, valid_loss
            )
        )
        print(
            "\t\tTraining Acc:  {:.3f} \t\tValidation Acc:  {:.3f}".format(
                train_acc, valid_acc
            )
        )

        if valid_loss <= valid_loss_min:
            print(
                "\t\tValidation loss decreased ({:.6f} --> {:.6f}).".format(
                    valid_loss_min, valid_loss
                )
            )

            valid_loss_min = valid_loss
        elif valid_loss == np.nan:
            print("Model Loss: NAN")

        if (last_epoch_loss < valid_loss) and last_epoch_loss != np.Inf:
            early_stop_cnt += 1
            if early_stop_cnt == early_stop:
                print("-" * 50 + "\nEarly Stopping Hit\n" + "-" * 50)
                break
            else:
                print(
                    "-" * 50
                    + f"\n\t\tEarly Stopping Step: {early_stop_cnt}/{early_stop}\n"
                    + "-" * 50
                )
        else:
            early_stop_cnt = 0
            last_epoch_loss = valid_loss

    print(
        f"Training Completed with best model having loss of {round(valid_loss_min,6)}"
    )
    del data, target
    gc.collect()
    end_time = datetime.datetime.now().replace(microsecond=0)
    print(f"Time Taken: {end_time-start_time}")
    return model


def predict(
    model: nn.Module, device: torch.device, dataloader: DataLoader
) -> npt.NDArray:
    model.eval()
    predictions: List[npt.NDArray] = []
    for data, target in dataloader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        predictions.append(
            torch.nn.functional.softmax(output, dim=1).cpu().detach().numpy()
        )
    return np.concatenate(predictions)
