import pandas as pd
import numpy as np
import warnings
from imblearn.over_sampling import SMOTE
from feature_engine.encoding import OneHotEncoder
from utils import read_csv_in_directory
from config import paths
from joblib import load, dump

warnings.filterwarnings('ignore')


def handle_class_imbalance(transformed_data: pd.DataFrame, transformed_labels: pd.Series):
    """
    Handle class imbalance using SMOTE.

    Args:
        transformed_data (pd.DataFrame): The transformed data.
        transformed_labels (pd.Series): The transformed labels.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: A tuple containing the balanced data and
            balanced labels.
    """
    # Adjust k_neighbors parameter for SMOTE
    # set k_neighbors to be the smaller of two values:
    #       1 and,
    #       the number of instances in the minority class minus one
    k_neighbors = min(
        1, sum(transformed_labels == min(transformed_labels.value_counts().index)) - 1
    )
    smote = SMOTE(k_neighbors=k_neighbors, random_state=0)
    balanced_data, balanced_labels = smote.fit_resample(
        transformed_data, transformed_labels
    )
    return balanced_data, balanced_labels


def impute_with_probability(data: pd.DataFrame, column: str):
    value_counts = data[column].value_counts()
    counts = value_counts.values
    index = value_counts.index

    probabilities = counts / counts.sum()

    imputation = np.random.choice(index.tolist(), size=data[column].isna().sum(), p=probabilities)
    missing = data[(data[column].isna())]
    missing[column] = imputation
    data.update(missing)


def impute_with_mode(data: pd.DataFrame, column: str):
    data[column] = data[column].fillna(data[column].mode()[0])
    return data


def preprocess(data_dir: str = paths.TRAIN_DIR, training: bool = True) -> pd.DataFrame:
    """
    Performs data preprocessing.
    Args:
        data_dir (str): Path to the directory containing the data.
        training (bool): Indicates whether preprocessing is for training or testing.

    Returns: (pd.DataFrame) The preprocessed data.
    """
    df = read_csv_in_directory(data_dir)
    if training:
        target = df['driver_factor']
        df = df.drop(columns='driver_factor')

    df[(df['deformed'].isna()) & (df['impact1'] == 'Non-Collision')]['deformed'] = \
        df[(df['deformed'].isna()) & (df['impact1'] == 'Non-Collision')]['deformed'].fillna('No damage')

    df = impute_with_mode(df, 'deformed')
    df = impute_with_mode(df, 'impact1')
    df = impute_with_mode(df, 'weather')

    df['numoccs'].fillna(df['numoccs'].mean(), inplace=True)

    if training:
        df['driver_factor'] = target
    df.dropna(inplace=True)
    if training:
        target = df['driver_factor']
        df = df.drop(columns='driver_factor')

    if training:
        encoder = OneHotEncoder(top_categories=6, drop_last=True, drop_last_binary=True)
        encoder.fit(df)
        dump(encoder, paths.ENCODER_FILE)
    else:
        encoder = load(paths.ENCODER_FILE)

    df = encoder.transform(df)

    if training:
        df['driver_factor'] = target
        x = df.drop(columns='driver_factor')
        y = df['driver_factor']
        x, y = handle_class_imbalance(x, y)
        x['driver_factor'] = y
        df = x
    return df
