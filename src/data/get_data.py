"""
- `os` : Fournit des fonctions pour interagir avec le système d'exploitation, notamment pour la
  gestion des chemins de fichiers et des variables d'environnement.
- `yaml` : Permet de sérialiser et de désérialiser des données au format YAML, utile pour la
  configuration et le stockage de données structurées de manière lisible.
- `s3fs` : Offre une interface pour interagir avec Amazon S3 comme avec un système de fichiers,
  facilitant la lecture et l'écriture de fichiers dans le cloud.
- `pandas` : Propose des structures de données et des outils pour l'analyse et la manipulation 
  efficaces de grandes quantités de données, idéal pour le traitement de données tabulaires.
"""
import os
import yaml
import s3fs
import pandas as pd


def import_yaml_config(config_path: str) -> dict:
    """
    Importe la configuration depuis un fichier YAML.

    Args:
    config_path (str): Chemin vers le fichier de configuration YAML.

    Returns:
    dict: Dictionnaire contenant les configurations.
    """
    configuration = {}
    if os.path.exists(config_path):
        with open(config_path, mode='r', encoding='utf-8') as file:
            configuration = yaml.safe_load(file)
    else:
        raise FileNotFoundError("Attention: Le chemin vers le fichier config.yaml n'est pas \
            le bon.")
    return configuration


config = import_yaml_config("/home/onyxia/work/data_science_project/config/config.yaml")
# config = import_yaml_config("../config/config.yaml") # This might not work

os.environ["AWS_ACCESS_KEY_ID"] = config.get("AWS_ACCESS_KEY_ID")
os.environ["AWS_SECRET_ACCESS_KEY"] = config.get("AWS_SECRET_ACCESS_KEY")
os.environ["AWS_SESSION_TOKEN"] = config.get("AWS_SESSION_TOKEN")
os.environ["AWS_DEFAULT_REGION"] = config.get("AWS_DEFAULT_REGION")

fs = s3fs.S3FileSystem(
    client_kwargs={'endpoint_url': 'https://'+'minio.lab.sspcloud.fr'},
    key=os.environ["AWS_ACCESS_KEY_ID"],
    secret=os.environ["AWS_SECRET_ACCESS_KEY"],
    token=os.environ["AWS_SESSION_TOKEN"])


def get_train_data():
    """
    Charge les données d'entraînement à partir d'un fichier CSV stocké sur un système de fichiers
    distant.

    Utilise l'interface de fichiers fournie par la variable `fs` (supposée être une instance de
    S3FileSystem) pour ouvrir et lire le fichier 'flin/raw/train.csv' en mode binaire.

    Returns:
        pandas.DataFrame: Un DataFrame contenant les données d'entraînement chargées du fichier CSV.
    """
    with fs.open("flin/diffusion/train.csv", mode="rb") as file_in:
        dataset_df = pd.read_csv(file_in, sep=",")
    return dataset_df


def get_test_data():
    """
    Charge les données de test à partir d'un fichier CSV stocké sur un système de fichiers distant.

    Utilise l'interface de fichiers fournie par la variable `fs` pour ouvrir et lire le fichier
    'flin/raw/test.csv' en mode binaire.

    Returns:
        pandas.DataFrame: Un DataFrame contenant les données de test chargées du fichier CSV.
    """
    with fs.open('flin/diffusion/test.csv', mode="rb") as file_in:
        test_data = pd.read_csv(file_in, sep=",")
    return test_data


def get_processed_train_data():
    """
    Charge les données d'entraînement traitées à partir d'un fichier CSV stocké sur un système de
    fichiers distant.

    Utilise l'interface de fichiers fournie par la variable `fs` pour ouvrir et lire le fichier
    'flin/processed/train.csv' en mode binaire. Ces données sont présumées être déjà traitées.

    Returns:
        pandas.DataFrame: Un DataFrame contenant les données d'entraînement traitées chargées du
        fichier CSV.
    """
    with fs.open('flin/diffusion/train_processed.csv', mode="rb") as file_in:
        train_data = pd.read_csv(file_in, sep=",")
    return train_data


def get_processed_test_data():
    """
    Charge les données de test traitées à partir d'un fichier CSV stocké sur un système de fichiers
    distant.

    Utilise l'interface de fichiers fournie par la variable `fs` pour ouvrir et lire le fichier
    'flin/processed/test.csv' en mode binaire. Ces données sont présumées être déjà traitées.

    Returns:
        pandas.DataFrame: Un DataFrame contenant les données de test traitées chargées du fichier
        CSV.
    """
    with fs.open('flin/diffusion/test_processed.csv', mode="rb") as file_in:
        test_data = pd.read_csv(file_in, sep=",")
    return test_data


def get_processed_val_data():
    """
    Charge les données de test traitées à partir d'un fichier CSV stocké sur un système de fichiers
    distant.

    Utilise l'interface de fichiers fournie par la variable `fs` pour ouvrir et lire le fichier
    'flin/processed/test.csv' en mode binaire. Ces données sont présumées être déjà traitées.

    Returns:
        pandas.DataFrame: Un DataFrame contenant les données de test traitées chargées du fichier
        CSV.
    """
    with fs.open('flin/diffusion/val_processed.csv', mode="rb") as file_in:
        val_data = pd.read_csv(file_in, sep=",")
    return val_data
