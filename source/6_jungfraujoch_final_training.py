import os

from classification_model import ClassificationModel
from read_LabelStudio_label_file import read_LabelStudio_label_file

if __name__ == "__main__":
    path_labeled_set = os.environ["WEATHER_DIR_LABELED"]
    label_file = os.path.join(path_labeled_set, "export", "weather_labels_iter1_45.json")


    filepaths, labels = read_LabelStudio_label_file(label_file)

    # train a classifier on the labeled datasetc
    classifier = ClassificationModel(no_epochs=20)
    classifier.fit(image_paths=filepaths, image_labels=labels)

    classifier.save_on_disk()





