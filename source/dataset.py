from typing import List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class WeatherDataset(Dataset):
    def __init__(
        self,
        image_data: List[Image.Image],
        target: List[int],
        transform: bool = None,
    ) -> None:
        label_classes = {"cloudy": 0, "rain": 1, "shine": 2, "sunrise": 3}
        self.image_data = image_data
        self.target = torch.LongTensor([label_classes[t] for t in target])
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
    X: List[Image.Image],
    y: List[int],
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    dataset = WeatherDataset(X, y, transform=transforms.ToTensor())

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        shuffle=shuffle,
    )
