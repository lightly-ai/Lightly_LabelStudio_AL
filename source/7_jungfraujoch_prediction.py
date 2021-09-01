import os
import tempfile
from urllib.request import urlretrieve

from classification_model import ClassificationModel

if __name__ == "__main__":


    # load the classifier from disk
    classifier = ClassificationModel(no_epochs=3)
    classifier.load_from_disk()

    # save the current image from Jungfraujoch on disk
    url_current_image = "http://webcam.switch.ch/jungfraujoch/pano/Jungfraujoch_pano.jpg"
    fd, filename = tempfile.mkstemp()
    urlretrieve(url_current_image, filename)

    # predict on the current image
    predicted_classes_str, scorer = classifier.predict(image_paths=[filename])
    predicted_class = predicted_classes_str[0]

    print(f"Current weather at Jungfraujoch: {predicted_class}")
    print(f"URL of image: {url_current_image}")





