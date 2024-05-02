"""
Ces imports permettent d'utiliser les fonctionnalités de la bibliothèque TensorFlow Decision 
Forests et de gérer les chemins de recherche des modules Python.
"""
import sys
import tensorflow_decision_forests as tfdf
sys.path.append('../src/data')
import make_dataset as md
sys.path.append('../src/visualization')
import plot as pl


def get_model(dataset_df):
    """
    Crée et entraîne un modèle de forêt aléatoire TensorFlow Decision Forests à partir du dataset 
    fourni.

    Args:
        dataset_df (pandas.DataFrame): Le DataFrame contenant le dataset.

    Returns:
        tfdf.keras.Model: Le modèle entraîné.
    """
    train_ds_pd, valid_ds_pd = md.split_dataset(dataset_df)
    label = 'SalePrice'
    train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_ds_pd,
                                                     label=label,
                                                     task=tfdf.keras.Task.REGRESSION)
    rf = tfdf.keras.RandomForestModel(task=tfdf.keras.Task.REGRESSION)
    rf.fit(x=train_ds)
    return rf


def evaluate_logs(dataset_df):
    """
    Évalue les journaux d'entraînement du modèle de forêt aléatoire TensorFlow Decision Forests
    et génère des visualisations pour évaluer la performance du modèle.

    Args:
        dataset_df (pandas.DataFrame): Le DataFrame contenant le dataset.
    """
    rf = get_model(dataset_df)
    logs = rf.make_inspector().training_logs()
    return pl.evaluate_model(logs)


def plot_inspector(dataset_df):
    """
    Génère des visualisations basées sur l'inspecteur du modèle de forêt aléatoire TensorFlow 
    Decision Forests.

    Args:
        dataset_df (pandas.DataFrame): Le DataFrame contenant le dataset.
    """
    rf = get_model(dataset_df)
    inspector = rf.make_inspector()
    return pl.variable_weight(inspector)
