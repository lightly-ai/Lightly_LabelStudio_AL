from typing import List

import numpy as np
import torch
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class WeatherDataset(Dataset):
    def __init__(
        self,
        image_data: List[Image.Image],
        target: List[int],
        label_encoder: LabelEncoder,
        transform: bool = None,
    ) -> None:
        self.image_data = image_data
        self.target = torch.LongTensor(label_encoder.transform(target))
        self.transform = transform

    def __getitem__(self, index):
        x = self.image_data[index]
        y = self.target[index]
        if self.transform:
            x = Image.fromarray(
                np.uint8(np.array(self.image_data[index]))
            )  # Memory Efficient way
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.image_data)


def get_dataloader(
    label_classes: List[str],
    X: List[Image.Image],
    y: List[int],
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    le = LabelEncoder().fit(label_classes)
    dataset = WeatherDataset(X, y, label_encoder=le, transform=transforms.ToTensor())

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        shuffle=shuffle,
    )
