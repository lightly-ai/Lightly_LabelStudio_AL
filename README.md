## Tutorial
This tutorial demonstrates a complete workflow of training a machine learning model with the aid of Active Learning using [Lightly](https://www.lightly.ai) and [LabelStudio](https://labelstud.io).

Assume that we have a new unlabelled dataset and we want to train a new model. We do not want to label all samples because not all of them are valuable. Lightly can help select a good subset of samples to kick off labelling and model training. The loop is as follows:

1. Lightly chooses a subset of the unlabelled samples.
1. This subset is labelled using LabelStudio.
1. A machine learning model is trained on the labeled data and generates predictions for the entire dataset.
1. Lightly consumes predictions and performs Active Learning to choose the next batch of samples to be labelled.
1. This new batch of samples is labelled in LabelStudio.
1. The machine learning model is re-trained on the enriched labelled dataset and achieves better performance.


Let's get started!

## 0. Installation and Requirements
Make sure you have an account for the [Lightly Web App](https://app.lightly.ai). 
You also need to know your API token which is shown under your `USERNAME` -> `Preferences`.

Clone this repo and install all python package requirements in the `requirements.txt` file, e.g. with pip.
```bash
git clone https://github.com/lightly-ai/Lightly_LabelStudio_AL.git
cd Lightly_LabelStudio_AL
pip install -r requirements.txt
```


## 1. Prepare data
We want to train a classifier predicting the weather displayed in an image. We use this dataset: [Multi-class Weather Dataset for Image Classification](https://data.mendeley.com/datasets/4drtyfjtfy/1). Download the dataset (zip file) from the [here](https://data.mendeley.com/public-files/datasets/4drtyfjtfy/files/a03e6097-f7fb-4e1a-9c6a-8923c6a0d3e0/file_downloaded) to this directory.

After downloading and extracting the zip file, you will see the extracted directory as follows:

```
dataset2
├── cloudy1.jpg
├── cloudy2.jpg
├── cloudy3.jpg
├── cloudy4.jpg
...
```

Here we have images in 4 weather conditions: `cloudy`, `rain`, `shine`, and `sunrise`.

#### 1.1 Split dataset
To compare results between iterations, we first split the entire dataset into a full training set and a validation set. The training set will be used to select samples and the validation set will be used as "new data" to evaluate the model's performance.

Run the script below to split the dataset:
```sh
python source/setup_data.py
```

After this, you will find the following files and directories in the current directory:
* `train_set`: Directory that contains all samples to be used for training the model. Here we pretend that these samples are all unlabelled.
* `val_set`: Directory that contains all samples to be used for model validation. Samples are labelled.
* `full_train.json`: JSON file that records paths to all files in `train_set`.
* `val.json`: JSON file that records paths and labels of all files in `val_set`.

These will be used in the following steps.

#### 1.2 Upload training samples to cloud storage
In this tutorial, samples are stored in the cloud, and Lightly Worker will read the samples from the cloud datasource. For details, please refer to [Set Up Your First Dataset](https://docs.lightly.ai/docs/set-up-your-first-dataset). Here we use Amazon S3 as an example.

Under your S3 bucket, create two directories: `data` and `lightly`. We will upload all training samples to `data`. For example, run the [AWS CLI tool](https://aws.amazon.com/cli/):
```sh
aws s3 sync train_set s3://<bucket>/data
```

After uploading the samples, your S3 bucket should look like
```
s3://bucket/
├── lightly/
└── data/
    ├── cloudy1.jpg
    ├── cloudy2.jpg
    ├── ...
```

## 2. Select the first batch of samples for labelling

Now, with all unlabelled data samples in your training dataset, we want to select a good subset, label them, and train our classification model with them. Lightly can do this selection for you in a simple way. The script [run_first_selection.py](./source/run_first_selection.py) does the job for you. You need to first setup Lightly Worker on your machine and put the correct configuration values in the script. Please refer to [Install Lightly](https://docs.lightly.ai/docs/install-lightly) and [Set Up Your First Dataset](https://docs.lightly.ai/docs/set-up-your-first-dataset) for more details.

Run the script after your worker is ready:

```sh
python source/run_first_selection.py
```

In this script, Lightly Worker first creates a dataset named `weather-classification`, selects 30 samples based on embeddings of the training samples, and records them in this dataset. These 30 samples are the ones that we are going to label in the first round. You can see the selected samples in the [Web App](https://app.lightly.ai/).

![First selection.](tutorial/images/init-selection.png)

## 3. Label the selected samples to train a classifier

We do this using the labelling tool **LabelStudio**, which as a browser-based tool hosted on your machine.
You have already installed it and can run it from the command line. It needs access to your local files. We will first download the selected samples, import them in LabelStudio, label them, and finally export the annotations.

#### 3.1 Download the selected samples

We can download the selected samples from the Lightly Platform. The [download_samples.py](./source/download_samples.py) script does everything for you and downloads the samples to a local directory called `samples_for_labelling`.

```sh
python source/export_filenames.py
```

Lightly Worker created a tag for the selected samples. This script pulls information about samples in this tag and downloads the samples.

#### 3.2 Run LabelStudio

Now we can launch LabelStudio.

```sh
export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true && label-studio start
```

You should see it in your browser. Create an account and login.

#### 3.3 Configure Storage

Create a new project called "weather-classification".
Then, head to `Settings` -> `Cloud Storage` -> `Add Source Storage` -> `Storage Type`: `Local files`.
Set the `Absolute local path` to the absolute path of directory `samples_for_labelling`.
Enable the option `Treat every bucket object as a source file`.
Then click `Add Storage`. It will show you that you have added a storage.
Now click on `Sync Storage` to finally load the 30 images.

![Configuration of local file input.](tutorial/images/ls-add-storage.png)

##### 3.4 Configure the labelling interface

In the `Settings` -> `Labelling Interface` in the `Code` insert
```xml
<View>
    <Image name="image" value="$image"/>
        <Choices name="choice" toName="image">
        <Choice value="cloudy"/>
        <Choice value="rain"/>
        <Choice value="shine" />
        <Choice value="sunrise" />
    </Choices>
</View>
```
![Configuration of Labeling Interface.](tutorial/images/ls-interface.png)

It tells LabelStudio that there is an image classification task with 4 distinct choices.

If you want someone else to help you labelling, you can go to `Settings`->`Instructions` and add some instructions.

#### 3.5 Labelling

Now if you click on your project again, you see 30 tasks and the corresponding images.
Click on `Label All Tasks` and get those 30 images labeled.
You can use the keys `1`, `2`, `3`, `4` as hotkeys to be faster.

#### 3.6 Export labels

Export the labels via `Export` and in the format `JSON-MIN`.
Rename the file to `annotation-0.json` and place that in the root directory of this repository.

## 4. Train a model and prepare for active learning

We can train a classification model with the 30 labelled samples. The [train_model_1.py](./source/train_model_1.py) script loads samples from `annotation-0.json` and performs this task.

```sh
python source/train_model_1.py
```

The following steps are performed in this script:
* Load the annotations and the labelled images.
* Load the validation set.
* Train a simple model as in [model.py](./source/model.py).
* Make predictions for all samples for training, including unlabelled samples.
* Dump the predictions in [Lightly Prediction format](https://docs.lightly.ai/docs/prediction-format#prediction-format) into directory `lightly_predictions`.

We can see that the model performance is not good:
```
Training Acc:  60.000           Validation Acc:  19.027
```

It is okay for now. We will improve this. Predictions will be used for active learning.

#### Upload predictions to datasource

Lightly Worker also does active learning for you based on predictions. It consumes predictions stored in the datasource. Now we need to place the predictions we just acquired in the datasource. For detailed information, please refer to [Predictions Folder Structure](https://docs.lightly.ai/docs/prediction-format#predictions-folder-structure). Here we still use AWS S3 bucket as an example.

In the `lightly` directory you created earlier in your S3 bucket, you will have a subdirectory `.lightly/predictions` where predictions are kept. You need the following additional files. You can create these files directly by copying the code blocks below.

##### tasks.json
```json
["weather-classification"]
```

We only have one task here and we name it as `weather-classification`.

##### schema.json
```json
{
    "task_type": "classification",
    "categories": [
        {
            "id": 0,
            "name": "cloudy"
        },
        {
            "id": 1,
            "name": "rain"
        },
        {
            "id": 2,
            "name": "shine"
        },
        {
            "id": 3,
            "name": "sunrise"
        }
    ]
}
```

Place these files in the `lightly` directory in your bucket along with predictions in directory `lightly_prediction`.
After uploading these files, your S3 bucket should look like
```
s3://bucket/
├── lightly/
│   └── .lightly/
│       └── predictions/
│           ├── tasks.json
│           └── weather-classification/
│               ├── schema.json
│               ├── cloudy1.json
│               ├── cloudy2.json
│               ├── ...
└── data/
    ├── cloudy1.jpg
    ├── cloudy2.jpg
    ├── ...
```

where files like `cloudy1.json` and `cloudy2.json` are prediction files in `lightly_prediction`.

## 5. Select and label new samples

With the predictions, Lightly Worker can perform active learning and select new samples for us. The [run_second_selection.py](./source/run_second_selection.py) script does the job.

```sh
python source/run_second_selection.py
```

This time, Lightly Worker goes through all training samples again and selects another 30 samples based on active learning scores computed from the predictions we uploaded in the previous step. For more details, please refer to [Selection Scores](https://docs.lightly.ai/docs/selection#scores) and [Active Learning Scorer](https://docs.lightly.ai/docs/active-learning-scorers).

You can see the results in the Web App.

![Second selection.](./tutorial/images/second-selection.png)


#### Label new samples

You can repeat step 3 to label new samples. To import new samples, go to `Settings` -> `Cloud Storage` and then click `Sync Storage` on the Source Cloud Storage you created earlier. A message `Synced 30 task(s)` should show up.

![Sync Storage.](./tutorial/images/ls-sync-storage.png)

Then, you can go back to the project page and label the new samples. After finishing annotating the samples, export the annotations again. Rename the file to `annotation-1.json` and place that in the root directory of this repository.

## 6. Train a new model with the new samples
Very similar to the script in step 4, script [train_model_2.py](source/train_model_2.py) loads samples from `annotation-1.json` and trains the classification model again with all 60 labelled samples now.

```sh
python source/train_model_2.py
```

The model indeed does better this time on the validation set:
```
Training Acc:  90.000           Validation Acc:  44.248
```
