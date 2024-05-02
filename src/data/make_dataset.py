# -*- coding: utf-8 -*-
import os
import click
import logging
import pandas as pd
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import numpy as np
import get_data as gd

def split_dataset(dataset, test_ratio=0.30):
    test_indices = np.random.rand(len(dataset)) < test_ratio
    return dataset[~test_indices], dataset[test_indices]

def process_data(data):
    # Example data processing steps
    processed_data = data.copy()
    # Your data processing steps here...
    processed_data = processed_data.drop('Id', axis=1)
    return processed_data

def save_data(train_data, test_data, val_data, output_filepath):
    # Example of saving processed data to CSV files
    train_data.to_csv(os.path.join(output_filepath, 'train_df.csv'), index=False)
    test_data.to_csv(os.path.join(output_filepath, 'test_df.csv'), index=False)
    val_data.to_csv(os.path.join(output_filepath, 'val_df.csv'), index=False)

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
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
