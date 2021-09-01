import os
import sys
from copy import copy, deepcopy
from typing import List, Dict, Tuple, Union

import numpy as np
import torch
import torchvision
from PIL import Image

from lightly.active_learning.scorers import ScorerClassification
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm import tqdm



class TorchImageDataset(Dataset):
    def __init__(self, image_paths: List[str], label_names: List[str], transform: object = None,
                 labels: List[str] = None):
        """

        :param image_paths:
            one path for each image
        :param label_names:
            an ordered list of possible labels
        :param transform:
        :param labels:
            one label for each image. Each label must be in the label_names.
        """
        if transform is None:
            transform = T.Compose([
                T.Resize((360, 117)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        self.transform = transform
        self.image_paths = image_paths
        self.labels = labels

        self.label_names = label_names
        self.label_name_to_index = {name: i for i, name in enumerate(label_names)}

    def __getitem__(self, index: int) -> Union[Tuple[torch.Tensor, int], torch.Tensor]:
        image_path = self.image_paths[index]
        image_pil = Image.open(image_path)
        image_torch: torch.Tensor = self.transform(image_pil)
        if self.labels:
            return image_torch, self.label_name_to_index[self.labels[index]]
        else:
            return image_torch

    def __len__(self):
        return len(self.image_paths)


class ClassificationModel():
    def __init__(self, num_classes: int = 3, no_epochs=5, **kwargs):
        # don't forget to initialize base class...
        self.model = torchvision.models.resnet18(pretrained=False, progress=True, num_classes=num_classes)
        self.no_epochs = no_epochs

        self.model_is_trained = False

    def save_on_disk(self, model_path: str="./weather_classifier.pth"):
        to_save = {"model": self.model, "label_names": self.label_names}
        torch.save(to_save, model_path)


    def load_from_disk(self, model_path: str="./weather_classifier.pth"):
        saved_data = torch.load(model_path)
        self.label_names = saved_data["label_names"]
        self.model = saved_data["model"]
        self.model_is_trained = True

    def fit(self, image_paths: List[str], image_labels: List[str], label_names: List[str] = None):
        print("STARTING FITTING")
        if label_names is None:
            self.label_names = sorted(list(set(image_labels)))

        dataset = TorchImageDataset(image_paths=image_paths, label_names=self.label_names, labels=image_labels)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

        pbar = tqdm(range(self.no_epochs), file=sys.stdout)
        for epoch in pbar:  # loop over the dataset multiple times
            running_loss = 0.0
            total_samples = 0
            correct = 0
            for i, data in enumerate(dataloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                correct += (predicted == labels).sum().item()
            text = f'epoch: {epoch} loss: {running_loss / total_samples:.6f} accuracy: {correct/total_samples:.3f}'
            tqdm.write(text)

        self.model_is_trained = True
        print("FINISHED FITTING")

    def predict(self, image_paths: List[str]) -> Tuple[List[str], ScorerClassification]:
        print("STARTING PREDICTION")

        if not self.model_is_trained:
            raise ValueError

        dataset = TorchImageDataset(image_paths, self.label_names)

        #self.model.eval()
        dataloader = DataLoader(dataset, batch_size=16)
        predictions = []
        self.model.eval()
        with torch.no_grad():
            for x in tqdm(dataloader):
                pred = self.model(x)
                predictions.append(pred)

        print("PUTTING TOGETHER RETURN VALUES")

        # flatten over batches
        predictions = [i for sublist in predictions for i in sublist]
        predictions = torch.stack(predictions, dim=0)
        predictions = torch.nn.functional.softmax(predictions, dim=1)

        predicted_classes_int = torch.argmax(predictions, dim=1)
        predicted_classes_str = [self.label_names[i] for i in predicted_classes_int]
        scorer = ScorerClassification(predictions)

        print("FINISHED PREDICTION")

        return predicted_classes_str, scorer
