"""
Ce module importe des bibliothèques essentielles pour le traitement de données, la modélisation
prédictive, la gestion des chemins de fichiers, et la journalisation des opérations.

- `pandas` (pd): Fournit des structures de données puissantes et des fonctions d'analyse de données.
- `tensorflow_decision_forests` (tfdf): Intègre des modèles de forêts décisionnelles dans
  l'écosystème TensorFlow, permettant la construction, l'entraînement et l'évaluation de modèles de
  machine learning basés sur des arbres de décision avec une intégration profonde aux
  fonctionnalités de TensorFlow.
- `Path` de `pathlib`: Manipulation des chemins de fichiers, rendant la lecture, l'écriture et
  l'organisation des fichiers.
- `logging`: Permet de configurer la journalisation à différents niveaux de détails (debug, info,
  warning, error), crucial pour le débogage et le suivi de l'état des applications en production.

Ces bibliothèques sont intégrées pour faciliter le développement de processus automatisés de
  traitement et d'analyse de données, ainsi que pour le suivi et la journalisation robuste des
  processus d'exécution.
"""
from pathlib import Path
import logging
import pandas as pd
import tensorflow_decision_forests as tfdf


def load_model(model_path):
    """
    Load the saved TensorFlow Decision Forest model.
    """
    try:
        model = tfdf.keras.models.load_model(model_path)
        logging.info("Model loaded successfully.")
        return model
    except FileNotFoundError as e:
        logging.error("Failed to load model. Error: %s", e)
        return None


def load_data(data_path):
    """
    Load new data for prediction from a CSV file.
    """
    try:
        data = pd.read_csv(data_path)
        logging.info("Data loaded successfully.")
        return data
    except FileNotFoundError as e:
        logging.error("Failed to load data. Error: %s", e)
        return None


def make_predictions(model, data):
    """
    Use the loaded model to make predictions on the provided data.
    """
    if model is not None and data is not None:
        try:
            # Assuming the model expects a TensorFlow dataset
            prediction_data = tfdf.keras.pd_dataframe_to_tf_dataset(data,
                                                                    task=tfdf.keras.Task.REGRESSION)
            predictions = model.predict(prediction_data)
            logging.info("Predictions made successfully.")
            return predictions
        except FileNotFoundError as e:
            logging.error("Failed to make predictions. Error: %s", e)
            return None
    else:
        return None


def save_predictions(predictions, output_path):
    """
    Save the predictions to a CSV file.
    """
    try:
        pd.DataFrame(predictions, columns=['Predicted_Value']).to_csv(output_path, index=False)
        logging.info("Predictions saved to %s", output_path)
    except FileNotFoundError as e:
        logging.error("Failed to save predictions. Error: %s", e)


def main(model_path, data_path, output_path):
    """
    Loads a model, makes predictions on provided data, and saves the predictions.

    Parameters:
        model_path (str): Path to the pre-trained machine learning model.
        data_path (str): Path to the data file on which predictions are to be made.
        output_path (str): Path where the prediction results will be saved.

    This function integrates the model loading, prediction, and saving process into a seamless
    pipeline, facilitated by detailed logging at each step.
    """
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
