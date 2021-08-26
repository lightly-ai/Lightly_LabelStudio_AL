We want to train a classifier predicting whether a webcam targeting a mountain 
is showing sunny, cloudy, or very cloudy weather.
To gather our dataset, we use the webcam at Jungfraujoch targeting the moutain Jungfrau in Switzerland.


## 1. Download the images
First, we decide in which directory we want to save the dataset and export the path to it as environment variable.
Then we download the images from the webcam hoster using a simple python script.
We download the webcam image at 13:00 for every day in 2020.
It downloads 365 images at a medium but sufficient resolution with a total size of 28MB.
If some URLs are down and a few less images are downloaded, the tutorial works nonetheless the same.
If you want to see the beauty of the swiss alps in full resolution, go to https://www.switch.ch/cam/jungfraujoch/.

```bash
# Set the directory the images should be downloaded to.
export WEATHER_DIR_RAW=path/to/dataset/weather_raw
# Download the images to the directory you just specified.
python source/1_scrape_junfraujoch.py
```

   
## 2. Analyze and subsample the dataset.

First, let's analyze the dataset and its distribution of data using the Lightly webapp.
You only need to use the `lightly-magic` command posted in the last step and
put in your token from the Lightly Webapp.
This will also create embeddings, which are later used for sampling diverse subsets of the dataset.
Furthermore, it will upload the images and embedings to the Lightly Platform.

---
**NOTE**

If you want to train an embedding model on this dataset instead of relying on a pretrained model, 
set the `trainer.max_epochs` to e.g. 100.
However, we strongly recommend doing this only when having a CUDA-GPU available.

---

![Terminal output of lightly-magic command.](images/jungfrau_lightly_magic.jpg)

In the Lightly Webapp head to the `Embedding` view and choose the UMAP embeddings.
It is clearly visible, that there is one distinct cluster.
Inspecting it shows that these images are mostly showing very cloudy weather.

![Embedding view of the dataset.](images/jungfrau_embedding_view.png)

We want to train our classifier without needing to label all 365 images. E.g. we don't want to label very similar images.
Furthermore, we want to ensure that all areas of the sample space are covered.
Both of it can be achieved by using the CORESET sampler to choose a diverse subset of only 30 images to be labelled.
Use it by clicking on `Create Sampling`, set the amount of datapoints to 30
and specify the name of the new tag to be created as `Coreset_30`.

![Sampling configuration for 30 samples with coreset.](images/jungfrau_sampling_configuration.png)

The embedding view after the sampling shows that the 30 labeled points are evenly spaced out in the sample space.

---
**NOTE**

The coreset sampler chooses the samples evenly spaced out in the 32-dimensional space.
This does not necessarily translate into being evenly spaced out after the dimensionality
reduction to 2 dimensions.

---


Last, we want to copy these 30 samples to a new directory on our local disk.
First, you need to decide where to copy these samples to and export the path as environment variable.

Next, head to the `Download` sections and use the first of the provided CLI commands to
copy the images without needing to download them.
Don't forget to replace the `input_dir` and `output_dir` by `input_dir=$WEATHER_DIR_RAW output_dir=$WEATHER_DIR_LABELLED`

```bash
# We need to define a directory where the labelled images are copied to.
export WEATHER_DIR_LABELLED=path/to/dataset/weather_labelled
# Download the filenames of the sample
lightly-download token=MY_TOKEN dataset_id=DATASET_ID tag_name='Coreset_30' input_dir=$WEATHER_DIR_RAW output_dir=$WEATHER_DIR_LABELLED
```

![Terminal output of lightly-download command.](images/jungfrau_lightly_download.png)

## 3. Label a subset of images to train a classifier.

We do this using the labelling tool **LabelStudio**, which as a browser-based tool hosted on your machine.
You have alread installed it and can run it from the command line. It needs access to your local files.

#### 3.1 Install and run Label Studio

```bash
export LABEL_STUDIO_BASE_DATA_DIR=$WEATHER_DIR_LABELED export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true && label-studio start
```

#### 3.2 Configure Storage

First lets copy the path where the labelled images to your clipboard: Copy the output from
```bash
echo $WEATHER_DIR_LABELLED
```

Open label studio in your browser and login. Create a new project called e.g. "Weather Jungfrau".
Then, head to the `Settings` -> `Cloud Storage` -> `Add Source Storage` -> `Storage Type`: `Local files`.
Set the `Absolute local path` to the path you just copied and the file filter to `.*jpg` .
Set the toggle button `Treat every bucket object as a source file`.
Then click `Add Storage`. It will show you that you have added a storage.
Now click on `Sync Storage` to finally load the 30 images.

![Configuration of local file input.](images/jungfrau_labelstudio_add_storage.png)

##### 3.3 Configure the labelling interface

In the `Settings` -> `Labelling Interface` in the `Code` insert
```xml
<View>
    <Image name="image" value="$image"/>
        <Choices name="choice" toName="image">
        <Choice value="Sunny"/>
        <Choice value="Cloudy" />
        <Choice value="Very Cloudy" />
    </Choices>
</View>
```
![Configuration of Labeling Interface.](images/jungfrau_labelstudio_labelling_interface.png)

It tells LabelStudio that there is an image classification task with 3 distinct choices.

#### 3.4 Add labelling instructions

If you want someone else to help you labelling, you can go to `Settings`->`Instructions` and add e.g. the following instructions:
```
The prominent mountain in the image middle is the Jungfrau.
Sunny: The Jungfrau and the background mountains are well visible.
Cloudy: The Jungfrau is visible, but the background is mostly covered by clouds.
Very Cloudy: Nearly nothing is visible, not even the Jungfrau.
```

#### 3.5 Labelling

Now if you click on your project again, you see 30 tasks and the corresponding images.
Click on `Label All Tasks` and get those 30 images labeled.
You can use the keys `1`, `2`, `3` as hotkeys to be faster.

#### 3.6 Export of Labels

Export the labels via `Export` and in the format `JSON-MIN`.
Ignore the downloaded file. You should find the labels already as a .json file in your `$WEATHER_DIR_LABELED/export`.
Rename the file to `weather_labels_iter0_30.json`.

## 4. Train a model and do active learning

Next we will take the exported labels an train an image classification model on them.
We use a resnet18 as backbone for the classifier.
After training the model, we use it to predict on the full set of images.
The predictions are stored in a lightly `ClassificationScorer` to calculate active learning scores.
These scores are needed for sampling another 15 images until we have 45 images.
We use the CORAL sampler, which combines CORESET and active learning to choose samples
which have both a high prediction uncertainty
and are different to each other and already chosen samples.

```bash
# WEATHER_DIR_RAW and WEATHER_DIR_LABELED must already by set.
# This script uses the label file named `weather_labels_iter0_30.json`
# We also need to set the environment variables for the Lightly Webapp connection.
# They are the same as the ones used in the lightly-download command at the end of step 2.
export LIGHTLY_TOKEN=MY_TOKEN
export LIGHTLY_DATASET_ID_WEATHER=DATASET_ID
python source/4_jungfraujoch_active_learning.py
```

With the just printed CLI command, you can download the filenames of the newly chosen images
and copy them to the directory with the labeled images. Execute this CLI command.


## 5. Label the additional 15 images.
If you have closed it, open LabelStudio again.
```bash
export LABEL_STUDIO_BASE_DATA_DIR=$WEATHER_DIR_LABELED export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true && label-studio start
```
In LabelStudio head again to `Settings` -> `Cloud Storage` and click on "Sync Storage".
Then you should see 15 more tasks in the project.
You see that they are different from each other and are mostly on the decision border
and thus harder to classify. Label all of them.

Export the labels again via `Export` and in the format `JSON-MIN`.
Ignore the downloaded file. You should find the labels already as a .json file in your `$WEATHER_DIR_LABELED/export`.
Rename the file to `weather_labels_iter1_45.json`.

___
**NOTE**

If you want to do more active learning loops, repeat steps 4 and 5.
Don't forget to change the sampling config to choose more images and have another tag name.
Furthermore, you need to update the name of the .json file with the exported images.

___


## 6. Train a model on the new labels.
We train the model on all images labeled in the previous active learning iterations.
Then we save the model, so that we can reuse it later.

```bash
# WEATHER_DIR_LABELED must already by set
# This uses the label file named `weather_labels_iter1_45.json`
python source/6_jungfraujoch_final_training.py
```

## 7. Apply the model on new images.
We load the model just saved on disk and use it to predict the current weather at Jungfraujoch
by using the most current webcam image.

```bash
python source/7_jungfraujoch_prediction.py
```
       