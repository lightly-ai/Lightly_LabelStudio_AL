import json
import os
import pathlib
from typing import List, Tuple

import numpy as np
import numpy.typing as npt
from label_studio import read_label_studio_annotation_file
from PIL import Image

IMAGE_SIZE = 224  # Resize images


def prepare_training_data(annotation_filepath: str) -> None:
    """Collects labels and filenames from LabelStudio output files.

    Images still stays in directory `train_set`. `train.json` only contains paths to
    samples to be used for training. For instance,
    [{"path": "/path/image1.png", "label": "cloudy"}]

    `train.json` will be picked up by the scripts for model training to load the
    actual images.
    """
    samples = []
    root = pathlib.Path("train_set")
    for filename, label in read_label_studio_annotation_file(annotation_filepath):
        samples.append({"path": str(root / filename), "label": label})

    with open("train.json", "w") as f:
        json.dump(samples, f)


def load_data(sample_json_path: str) -> Tuple[List[Image.Image], List[str], List[str]]:
    """Loads image data.

    Paths to samples to be used for training are loaded from the json file created in
    `prepare_training_data`.
    """
    with open(sample_json_path) as f:
        sample_list = json.load(f)

    all_images, all_labels = [], []
    filenames = []

    for sample in sample_list:
        all_images.append(
            Image.open(sample["path"])
            .convert("RGB")
            .resize((IMAGE_SIZE, IMAGE_SIZE), resample=Image.LANCZOS)
        )
        all_labels.append(sample["label"])
        filenames.append(pathlib.Path(sample["path"]).name)
    return all_images, all_labels, filenames


def dump_lightly_predictions(filenames: List[str], predictions: npt.NDArray) -> None:
    """Dumps model predictions in the Lightly Prediction format.

    Each input image has its own prediction file. The filename is `<image_name>.json`.
    """
    root = pathlib.Path("lightly_predictions")
    os.mkdir(root)
    for filename, prediction in zip(filenames, predictions):
        with open(str(root / pathlib.Path(filename).stem) + ".json", "w") as f:
            pred_list = prediction.tolist()
            # Normalise probabilities again because of precision loss in `to_list`.
            pred_sum = sum(pred_list)
            json.dump(
                {
                    "file_name": filename,
                    "predictions": [
                        {
                            "category_id": int(np.argmax(prediction)),
                            "probabilities": [p / pred_sum for p in pred_list],
                        }
                    ],
                },
                f,
            )
