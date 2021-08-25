## Tutorial
This tutorial combines Lightly and LabelStudio for showing a complete workflow of creating a ML model.
1. It starts with collecting unlabelled data. 
2. Then it uses Lightly to choose a subset of the unlabelled to be labelled.
3. This subset is labelled with the help of LabelStudio.
4. A machine learning model is trained on the labeled data and Active Learning is used to choose the next batch to be labelled.
5. This batch is labelled again in LabelStudio.
6. The machine learning model is trained on the updated labelled dataset.
7. Finally, the model is used to predict on completely new data. 

After installing the requirements, you can follow the full tutorial [here](tutorial/tutorial.md).

## Installation and Requirements
Make sure you have an account for the [Lightly Web App](https://app.lightly.ai). 
You also need to know your API token which is shown under your `USERNAME` -> `Preferences`.

Install all python package requirements in the `requirements.txt` file, e.g. with pip.
```bash
pip install -r requirements.txt