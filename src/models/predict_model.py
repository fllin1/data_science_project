import pandas as pd
import tensorflow_decision_forests as tfdf
from pathlib import Path
import logging

def load_model(model_path):
    """
    Load the saved TensorFlow Decision Forest model.
    """
    try:
        model = tfdf.keras.models.load_model(model_path)
        logging.info("Model loaded successfully.")
        return model
    except Exception as e:
        logging.error(f"Failed to load model. Error: {e}")
        return None

def load_data(data_path):
    """
    Load new data for prediction from a CSV file.
    """
    try:
        data = pd.read_csv(data_path)
        logging.info("Data loaded successfully.")
        return data
    except Exception as e:
        logging.error(f"Failed to load data. Error: {e}")
        return None

def make_predictions(model, data):
    """
    Use the loaded model to make predictions on the provided data.
    """
    if model is not None and data is not None:
        try:
            # Assuming the model expects a TensorFlow dataset
            prediction_data = tfdf.keras.pd_dataframe_to_tf_dataset(data, task=tfdf.keras.Task.REGRESSION)
            predictions = model.predict(prediction_data)
            logging.info("Predictions made successfully.")
            return predictions
        except Exception as e:
            logging.error(f"Failed to make predictions. Error: {e}")
            return None
    else:
        return None

def save_predictions(predictions, output_path):
    """
    Save the predictions to a CSV file.
    """
    try:
        pd.DataFrame(predictions, columns=['Predicted_Value']).to_csv(output_path, index=False)
        logging.info(f"Predictions saved to {output_path}")
    except Exception as e:
        logging.error(f"Failed to save predictions. Error: {e}")

def main(model_path, data_path, output_path):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    model = load_model(model_path)
    data = load_data(data_path)
    predictions = make_predictions(model, data)
    if predictions is not None:
        save_predictions(predictions, output_path)

if __name__ == "__main__":
    MODEL_PATH = Path("../models/trained_model.pkl")
    DATA_PATH = Path("../data/new_data.csv")
    OUTPUT_PATH = Path("../data/predictions.csv")
    main(MODEL_PATH, DATA_PATH, OUTPUT_PATH)
