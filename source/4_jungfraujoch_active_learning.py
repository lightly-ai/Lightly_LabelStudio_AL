import os

from lightly.active_learning.agents import ActiveLearningAgent
from lightly.active_learning.config import SamplerConfig
from lightly.api import ApiWorkflowClient
from lightly.data import LightlyDataset
from lightly.openapi_generated.swagger_client import SamplingMethod

from classification_model import ClassificationModel
from read_LabelStudio_label_file import read_LabelStudio_label_file

if __name__ == "__main__":
    path_full_dataset = os.environ["WEATHER_DIR_RAW"]
    path_labeled_set = os.environ["WEATHER_DIR_LABELED"]
    label_file = os.path.join(path_labeled_set, "export", "weather_labels_iter0_30.json")

    # 1. Prepare the active learning agent to make sure that your credentials are correct
    lighty_webapp_token = os.environ["LIGHTLY_TOKEN"]
    lighty_webapp_dataset_id = os.environ["LIGHTLY_DATASET_ID_WEATHER"]
    api_workflow_client = ApiWorkflowClient(token=lighty_webapp_token, dataset_id=lighty_webapp_dataset_id)
    al_agent = ActiveLearningAgent(api_workflow_client=api_workflow_client, preselected_tag_name="Coreset_30")

    # 2. read the label file
    filepaths, labels = read_LabelStudio_label_file(label_file)

    # 3. Define the image classification model
    classifier = ClassificationModel(no_epochs=20)

    # 4. Fit the classifier on the labeled images
    classifier.fit(image_paths=filepaths, image_labels=labels)

    # 5. Predict with the classifier on the complete dataset
    image_filenames_full_dataset = LightlyDataset(path_full_dataset).get_filenames()
    image_paths_full_dataset = [os.path.join(path_full_dataset, filename) for filename in image_filenames_full_dataset]
    predicted_classes_str, scorer = classifier.predict(image_paths=image_paths_full_dataset)

    # 6. Use the active learning sampler "CORAL" via the Lightly API to sample until we have 45 samples
    sampler_config = SamplerConfig(method=SamplingMethod.CORAL, n_samples=45, name="CORAL_45")
    al_agent.query(sampler_config=sampler_config, al_scorer=scorer)

    # 7. Download the new samples via the CLI and copy the corresponding images to the labeled folder.
    print("Use the following command to download the new tag via the CLI:")
    cli_command = f"lightly-download token={lighty_webapp_token} dataset_id={lighty_webapp_dataset_id} " \
                  f"tag_name=CORAL_45 input_dir={path_full_dataset} output_dir={path_labeled_set} " \
                  f"exclude_parent_tag=True"
    print(cli_command)
 