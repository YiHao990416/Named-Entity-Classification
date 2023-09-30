Named Entity Recognition With Distillbert Model

This README provides an overview of the Python code for training a Named Entity Recognition (NER) model using the DistilBERT architecture particularly the pretrained DistillBertForTokenClasification . NER is a natural language processing task that involves identifying and classifying named entities in text.

Table of Contents

    Project Description
    Installation
    Usage
    Dependencies
    Contributing
    License

Project Description

This Python code repository contains the implementation of a DistilBERT-based NER model. The code performs the following tasks:

    Imports necessary libraries and sets up GPU usage if available.
    Defines a tokenizer using the DistilBERT pre-trained model.
    Reads and loads training, testing, and validation data from JSON files into dataframes.
    Preprocesses the labels by adding "-100" at the beginning to represent "CLS".
    Defines a custom dataset class (NERdataset) and data collation function (collate_fn) for token classification.
    Defines the NER model (DistilbertNER) using the DistilBERT pre-trained model.
    Sets parameters for dataset and dataloaders.
    Implements training function (train) to train the NER model.
    Utilizes accuracy, F1-score to measure the performance of the model

Installation

To run this code, follow these installation steps:

    Clone the repository:

    bash

git clone [repository_url]
cd [project_directory]

Install the required dependencies:

bash

pip install pandas torch transformers scikit-learn

Download the DistilBERT pre-trained model using Hugging Face Transformers:

python

    from transformers import DistilBertTokenizer, DistilBertForTokenClassification
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertForTokenClassification.from_pretrained("distilbert-base-uncased")

    Place your training, testing, and validation data in JSON format (e.g., train.json, test.json, valid.json) within the project directory.

Usage

To use this code:

    Ensure that you have installed the required dependencies and downloaded the DistilBERT pre-trained model.
    Modify the dataset parameters and training parameters as needed for your specific task.
    Run the code by executing the Python script.

bash

python your_script.py

Dependencies

This code requires the following Python libraries and packages:

    pandas
    torch
    transformers (Hugging Face Transformers library)
    scikit-learn (for evaluation metrics)

You can install these dependencies using pip as mentioned in the installation section.
Contributing

Contributions to this project are welcome. To contribute, follow these steps:

    Fork the repository.
    Create a new branch: git checkout -b feature-name.
    Make your changes and commit them: git commit -m 'Add some feature'.
    Push to the branch: git push origin feature-name.
    Open a pull request.
Reference
    https://towardsdatascience.com/custom-named-entity-recognition-with-bert-cf1fd4510804
    
Please ensure that you follow the code of conduct and contribution guidelines when contributing to this project.
License
This project is licensed under the MIT License - see the LICENSE file for details
