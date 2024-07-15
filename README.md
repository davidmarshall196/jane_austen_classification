# Tech test Solution

## Overview

Given the dataset [here](./data/gutenberg-paragraphs.json), we would like you to create a model which can identify which paragraphs have been written by Jane Austen.

## Set Up
Clone the repo and create a new virtual environment to run the code in:

```
# Clone the repository
git clone https://github.com/davidmarshall196/jane_austen_classification.git

# Create a new virtual environment
python3 -m venv jane_austen_venv

# Activate the virtual environment
source jane_austen_venv/bin/activate

# Navigate inside the repo and install requirements
```
pip install -r requirements.txt
```

I recommend also creating a Python kernel to run the notebooks with:
```
python -m ipykernel install --user --name=jane_austen_venv --display-name="Jane Austen Env"
```

There are 3 notebooks which can be used to train/save a model, make predictions on unseen data, and run code quality checks.

To make predictions via Flask, simply run the main.py app:
```
python src/main.py
```
and then make predictions to the endpoint:
```
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"paragraph": "Sample input text"}'
```

## Modelling
Modelling: the model used is distilbert-base-uncased, a good language model for clssification tasks. It is pre-trained on a large amount of data but distilled into a smaller, faster format meaning results and performance are generally good. Preprocessing is generally not required.

## Results
With the current training process, training takes less than 2 minutes and evaluation about the same (classification matrix can be seen in the logs). This achieves 89% accuracy on the test set. Currently this is set to use 10% of the overall dataset with 0.5% used for training, 0.5% for validation and 9% for testing. Increasing the data used increases accuracy to 99%. F1 score has been chosen as a balance between precision and recall.

## Good Points
* Model Accuracy: while training takes a while with the volume used to achieve it, 99% accuracy (F1) is a very high accuracy.
* Logging: I am happy with the logging used in the project. I have kept the logs in the repo for viewing.

## Improvements
EDA: I did very little in the way of EDA. While there is only 1 column, I would like to look at things like:
* The books the texts come from. Perhaps the model only performs well on 1 or 2 of Jane Austen's novels
* The impact of NLP techniques including lemmatisation, stemming, stop word removal etc
* TF-IDF of some kind (perhaps combined with a simple classification model) to identify the words commonly used by Austen

Dockerfile: Depending on the use case, for production, the project could use a Dockerfile which would allow easier deployment, particularly around consistency of environment and dependencies.

Unit and integration tests: I created a couple of sample unit tests but I think it could benefit from more, including some integration tests to test the components and modelling together.

Training time: in order to achieve high accuracy (99%), a lot of data is utilised which leads to training time being high. Further experiments are needed to reduce this (e.g. Epochs, batch size, amount of data etc.)

