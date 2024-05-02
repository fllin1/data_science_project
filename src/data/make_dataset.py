# Vous pouvez exécuter ce .py en entrant dans le terminal : python src/data/make_dataset.py ../
# -*- coding: utf-8 -*-
"""
Ce module importe diverses bibliothèques utiles pour gérer les interactions avec le système
d'exploitation, la journalisation, la manipulation de chemins de fichiers, la gestion
d'environnements et la manipulation de données numériques.

- `os` : Fournit des fonctions pour interagir avec le système d'exploitation.
- `logging` : Permet de configurer la journalisation à différents niveaux de détails (debug, info
    warning, error).
- `Path` de `pathlib` : Offre une approche orientée objet pour la gestion des chemins de fichiers.
- `click` : Utilisé pour créer des interfaces en ligne de commande.
- `find_dotenv` et `load_dotenv` de `dotenv` : Chargent les variables d'environnement à partir d'un
    fichier .env pour le développement sécurisé des applications.
- `np` (numpy) : Propose des structures de données et des fonctions pour la manipulation avancée de
    grandes tableaux numériques.
- `gd` (get_data) : Module personnalisé pour charger ou traiter les données spécifiques au projet.

Ces importations sont essentielles pour les applications qui nécessitent une interaction avancée
avec le système d'exploitation, la gestion des données d'environnement, la manipulation de données
et la création d'interfaces utilisateurs en ligne de commande.
"""

import os
import logging
from pathlib import Path
import click
from dotenv import find_dotenv, load_dotenv
import numpy as np
import get_data as gd


def split_dataset(dataset, test_ratio=0.30):
    """
    Divise un dataset en deux sous-ensembles, l'un pour l'entraînement et l'autre pour les tests,
    basé sur le ratio spécifié.

    Parameters:
        dataset (DataFrame): Le DataFrame à diviser.
        test_ratio (float, optional): La proportion du dataset à utiliser pour le test. Par défaut
        à 0.30.

    Returns:
        tuple: Deux DataFrames, le premier pour les données d'entraînement et le second pour les
        données de test.
    """
    test_indices = np.random.rand(len(dataset)) < test_ratio
    return dataset[~test_indices], dataset[test_indices]


def process_data(data):
    """
    Traite les données en effectuant des opérations de nettoyage ou de préparation.
    Par exemple, supprime la colonne 'Id'.

    Parameters:
        data (DataFrame): Le DataFrame à traiter.

    Returns:
        DataFrame: Le DataFrame traité, avec certaines colonnes modifiées ou supprimées selon les
        besoins.
    """
    # Example data processing steps
    processed_data = data.copy()
    # Your data processing steps here...
    processed_data = processed_data.drop('Id', axis=1)
    return processed_data


def save_data(train_data, test_data, val_data, output_filepath):
    """
    Sauvegarde les données d'entraînement, de test, et de validation dans des fichiers CSV
    spécifiés.

    Parameters:
        train_data (DataFrame): Les données d'entraînement à sauvegarder.
        test_data (DataFrame): Les données de test à sauvegarder.
        val_data (DataFrame): Les données de validation à sauvegarder.
        output_filepath (str): Le chemin du répertoire où les fichiers doivent être sauvegardés.

    Returns:
        None: Les fichiers sont écrits à l'emplacement spécifié.
    """
    # Example of saving processed data to CSV files
    train_data.to_csv(os.path.join(output_filepath, 'train_processed.csv'), index=False)
    test_data.to_csv(os.path.join(output_filepath, 'test_processed.csv'), index=False)
    val_data.to_csv(os.path.join(output_filepath, 'val_processed.csv'), index=False)


@click.command()
@click.argument('output_filepath', type=click.Path())
def main(output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    # Load raw data
    logger.info('loading raw data')
    raw_train_df = gd.get_train_data()
    raw_val_df = gd.get_test_data()

    # Perform data processing steps
    logger.info('performing data processing')
    train_df = process_data(raw_train_df)
    val_df = process_data(raw_val_df)

    # Split data into train and test sets
    logger.info('splitting dataset into train and test sets')
    train_df, test_df = split_dataset(train_df)

    # Save processed data
    logger.info('saving processed data')
    save_data(train_df, test_df, val_df, output_filepath)


if __name__ == '__main__':
    LOG_FMT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=LOG_FMT)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
