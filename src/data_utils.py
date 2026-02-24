"""Utility functions for loading and cleaning data."""

import pandas as pd
import numpy as np


def load_csv(filepath):
    """Load a CSV file into a pandas DataFrame."""
    return pd.read_csv(filepath)


def save_csv(df, filepath, index=False):
    """Save a DataFrame to a CSV file."""
    df.to_csv(filepath, index=index)


def drop_missing(df, threshold=0.5):
    """Drop columns with missing value proportion above the threshold."""
    min_count = int((1 - threshold) * len(df))
    return df.dropna(thresh=min_count, axis=1)


def remove_duplicates(df):
    """Remove duplicate rows from a DataFrame."""
    return df.drop_duplicates()


def describe_data(df):
    """Return basic descriptive statistics of a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    dict
        Dictionary with keys 'shape', 'dtypes', and 'describe'.
    """
    return {
        "shape": df.shape,
        "dtypes": df.dtypes,
        "describe": df.describe(),
    }
