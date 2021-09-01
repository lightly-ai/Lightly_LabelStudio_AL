import json
from typing import Dict, Tuple, List


def read_label_element(label_element: Dict) -> Tuple[str, str]:
    filepath = "/" + label_element["data"]["image"].split("?d=")[-1]
    label = label_element["annotations"][0]["result"][0]["value"]["choices"][0]
    return filepath, label

def read_LabelStudio_label_file(filepath: str) -> Tuple[List[str],List[str]]:
    # read the label file
    with open(filepath, 'r') as json_file:
        data = json.load(json_file)
    filepaths, labels = zip(* [read_label_element(label_element) for label_element in data])
    return filepaths, labels