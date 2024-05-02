"""
This module is designed to work with datasets using pandas, build and evaluate models using
TensorFlow Decision Forests, and manage file paths with Path from pathlib. It also includes logging
capabilities for tracking the flow and debugging.

Imports:
    pandas (pd): Provides data structures and data analysis tools.
    tensorflow_decision_forests (tfdf): Offers a suite of decision forest algorithms for machine
    learning.
    pathlib.Path: Used for manipulating filesystem paths in an object-oriented way.
    logging: Used for tracking events that happen when the software runs, which can be helpful for
    debugging.
"""
from pathlib import Path
import logging
import pandas as pd
import tensorflow_decision_forests as tfdf


def load_data(data_path):
    """
    Load training or validation data from a CSV file.
    """
    try:
        data = pd.read_csv(data_path)
        logging.info("Data loaded from {data_path}")
        return data
    except FileNotFoundError as e:
        logging.error("Failed to load data from {data_path}. Error: %s", e)
        return None


def prepare_dataset(data, label_column):
    """
    Converts a Pandas DataFrame to a TensorFlow dataset.
    """
    try:
        dataset = tfdf.keras.pd_dataframe_to_tf_dataset(data,
                                                        label=label_column,
                                                        task=tfdf.keras.Task.REGRESSION)
        logging.info("Dataset prepared for training.")
        return dataset
    except FileNotFoundError as e:
        logging.error("Failed to convert data to TensorFlow dataset. Error: %s", e)
        return None


def train_model(train_dataset, valid_dataset):
    """
    Configure and train a TensorFlow Decision Forests model.
    """
    try:
        model = tfdf.keras.GradientBoostedTreesModel(task=tfdf.keras.Task.REGRESSION)
        model.compile(metrics=["mse"])
        logging.info("Model compiled and training started.")
        model.fit(train_dataset, validation_data=valid_dataset, epochs=10)
        return model
    except FileNotFoundError as e:
        logging.error("Failed to train model. Error: %s", e)
        return None


def evaluate_model(model, dataset):
    """
    Evaluate the trained model using the validation dataset.
    """
    try:
        results = model.evaluate(dataset, return_dict=True)
        logging.info("Model evaluation results: %s", model)
        return results
    except FileNotFoundError as e:
        logging.error("Failed to evaluate model. Error: %s", e)
        return None


def save_model(model, model_path):
    """
    Save the trained model.
    """
    try:
        model.save(model_path)
        logging.info("Model saved to %s", model_path)
    except FileNotFoundError as e:
        logging.error("Failed to save model. Error: %s", e)


def main(train_data_path, validation_data_path, model_save_path):
    """
    Main execution function that handles the workflow for training and evaluating a machine
    learning model, and then saving the trained model.

    Parameters:
        train_data_path (str): File path to the training data.
        validation_data_path (str): File path to the validation data.
        model_save_path (str): File path where the trained model will be saved.

    This function utilizes extensive logging to provide visibility into the process flow and status.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    train_data = load_data(train_data_path)
    validation_data = load_data(validation_data_path)

    train_dataset = prepare_dataset(train_data, "SalePrice")
    validation_dataset = prepare_dataset(validation_data, "SalePrice")

    model = train_model(train_dataset, validation_dataset)

    if model is not None:
        evaluate_model(model, validation_dataset)
        save_model(model, model_save_path)


if __name__ == "__main__":
    TRAIN_DATA_PATH = Path("../data/train_data.csv")
    VALIDATION_DATA_PATH = Path("../data/validation_data.csv")
    MODEL_SAVE_PATH = Path("../models/trained_model")

    main(TRAIN_DATA_PATH, VALIDATION_DATA_PATH, MODEL_SAVE_PATH)
