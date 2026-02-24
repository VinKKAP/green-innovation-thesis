"""Feature engineering utilities for the green innovation thesis."""

import pandas as pd
import numpy as np


def encode_categoricals(df, columns):
    """One-hot encode specified categorical columns."""
    return pd.get_dummies(df, columns=columns, drop_first=True)


def normalise(df, columns):
    """Min-max normalise specified numeric columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    columns : list of str
        Column names to normalise. Columns where min equals max are left unchanged.

    Returns
    -------
    pd.DataFrame
        DataFrame with specified columns scaled to [0, 1].
    """
    df = df.copy()
    for col in columns:
        col_min = df[col].min()
        col_max = df[col].max()
        if col_max != col_min:
            df[col] = (df[col] - col_min) / (col_max - col_min)
    return df


def create_lag_features(df, column, lags):
    """Create lag features for a given column.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame (should be sorted by time).
    column : str
        Column name to generate lags for.
    lags : list of int
        Lag periods to create (e.g., [1, 2, 3]).

    Returns
    -------
    pd.DataFrame
        DataFrame with additional ``<column>_lag<n>`` columns.
        The first ``max(lags)`` rows will contain NaN values.
    """
    df = df.copy()
    for lag in lags:
        df[f"{column}_lag{lag}"] = df[column].shift(lag)
    return df


def select_features(df, target, exclude=None):
    """Return feature matrix X and target vector y."""
    exclude = exclude or []
    feature_cols = [c for c in df.columns if c != target and c not in exclude]
    return df[feature_cols], df[target]
