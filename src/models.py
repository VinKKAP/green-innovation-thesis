"""Model definitions and training helpers for the green innovation thesis."""

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


def build_model(model_type="linear"):
    """Return an untrained model of the specified type.

    Parameters
    ----------
    model_type : str
        One of 'linear', 'ridge', 'random_forest', 'gradient_boosting'.

    Returns
    -------
    sklearn estimator
        An untrained scikit-learn regressor instance.

    Examples
    --------
    >>> model = build_model('random_forest')
    >>> model.fit(X_train, y_train)
    """
    models = {
        "linear": LinearRegression(),
        "ridge": Ridge(),
        "random_forest": RandomForestRegressor(random_state=42),
        "gradient_boosting": GradientBoostingRegressor(random_state=42),
    }
    if model_type not in models:
        raise ValueError(f"Unknown model type '{model_type}'. Choose from {list(models.keys())}.")
    return models[model_type]


def evaluate_model(model, X_test, y_test):
    """Evaluate a trained model and return a dict of metrics.

    Parameters
    ----------
    model : sklearn estimator
        A fitted scikit-learn regressor.
    X_test : array-like of shape (n_samples, n_features)
        Test feature matrix.
    y_test : array-like of shape (n_samples,)
        True target values.

    Returns
    -------
    dict
        Dictionary with keys 'r2' (R² score) and 'rmse' (root mean squared error).
    """
    y_pred = model.predict(X_test)
    return {
        "r2": r2_score(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
    }
