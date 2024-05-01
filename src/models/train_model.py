import pandas as pd
import tensorflow_decision_forests as tfdf
from pathlib import Path
import logging

def load_data(data_path):
    """
    Load training or validation data from a CSV file.
    """
    try:
        data = pd.read_csv(data_path)
        logging.info(f"Data loaded from {data_path}")
        return data
    except Exception as e:
        logging.error(f"Failed to load data from {data_path}. Error: {e}")
        return None

def prepare_dataset(data, label_column):
    """
    Converts a Pandas DataFrame to a TensorFlow dataset.
    """
    try:
        dataset = tfdf.keras.pd_dataframe_to_tf_dataset(data, label=label_column, task=tfdf.keras.Task.REGRESSION)
        logging.info("Dataset prepared for training.")
        return dataset
    except Exception as e:
        logging.error("Failed to convert data to TensorFlow dataset. Error: {e}")
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
    except Exception as e:
        logging.error("Failed to train model. Error: {e}")
        return None

def evaluate_model(model, dataset):
    """
    Evaluate the trained model using the validation dataset.
    """
    try:
        results = model.evaluate(dataset, return_dict=True)
        logging.info(f"Model evaluation results: {results}")
        return results
    except Exception as e:
        logging.error("Failed to evaluate model. Error: {e}")
        return None

def save_model(model, model_path):
    """
    Save the trained model.
    """
    try:
        model.save(model_path)
        logging.info(f"Model saved to {model_path}")
    except Exception as e:
        logging.error(f"Failed to save model. Error: {e}")

def main(train_data_path, validation_data_path, model_save_path):
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
