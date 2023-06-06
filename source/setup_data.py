import json
import os
import pathlib
import re
from typing import Dict, List

import numpy as np

SEED = 42
np.random.seed(SEED)


def setup_data(data_dir_str: str) -> None:
    """Splits the full dataset into a training set and a validation set.

    The training set will have images that will be used to train the model. Even if
    they are already labelled, we will use them as unlabelled data and use Lightly
    to label them. The validation set will be used to evaluate the model's performance.
    """
    data_dir = pathlib.Path(data_dir_str)
    files = os.listdir(data_dir)
    train_set_path = pathlib.Path("train_set")
    val_set_path = pathlib.Path("val_set")
    train_set_path.mkdir()
    val_set_path.mkdir()

    dataset: Dict[str, List[pathlib.PosixPath]] = {}
    train_set = []
    val_set = []
    pattern = re.compile("([a-z]+)[0-9]+")

    for file in files:
        filepath = data_dir / file
        regex_match = pattern.match(filepath.stem)
        category = regex_match.group(1)

        if dataset.get(category) is None:
            dataset[category] = []
        dataset[category].append(filepath)

    for category, samples in dataset.items():
        sample_idx = list(range(len(samples)))
        train_idx = np.random.choice(
            sample_idx, int(len(sample_idx) * 0.8), replace=False
        )
        val_idx = [idx for idx in sample_idx if idx not in train_idx]

        for idx in train_idx:
            filepath: pathlib.PosixPath = samples[idx]
            new_path = train_set_path / filepath.name
            os.rename(filepath, new_path)
            train_set.append({"path": str(new_path), "label": category})
        for idx in val_idx:
            filepath = samples[idx]
            new_path = val_set_path / filepath.name
            os.rename(filepath, new_path)
            val_set.append({"path": str(new_path), "label": category})

    with open("full_train.json", "w") as f:
        json.dump(train_set, f)
    with open("val.json", "w") as f:
        json.dump(val_set, f)


setup_data("./dataset2/")
