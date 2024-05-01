import os
import yaml
import s3fs
import pandas as pd
import numpy as np


def import_yaml_config(config_path: str) -> dict:
    """
    Importe la configuration depuis un fichier YAML.

    Args:
    config_path (str): Chemin vers le fichier de configuration YAML.

    Returns:
    dict: Dictionnaire contenant les configurations.
    """
    config = {}
    if os.path.exists(config_path):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
    else:
        print(f"Attention: Le fichier {config_path} n'existe pas.")
    return config


CONFIG_PATH = 'config.yaml'
config = import_yaml_config(CONFIG_PATH)

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
    with fs.open("flin/diffusion/train.csv", mode="rb") as file_in:
        dataset_df = pd.read_csv(file_in, sep=",")
    return dataset_df


def get_test_data():
    with fs.open('flin/diffusion/test.csv', mode="rb") as file_in:
        test_data = pd.read_csv(file_in, sep=",")
    return test_data


def get_submission_data():
    with fs.open('flin/diffusion/sample_submission.csv', mode="rb") as file_in:
        sample_submission_df = pd.read_csv(file_in, sep=",")
    return sample_submission_df


def split_dataset(dataset, test_ratio=0.30):
    test_indices = np.random.rand(len(dataset)) < test_ratio
    return dataset[~test_indices], dataset[test_indices]
