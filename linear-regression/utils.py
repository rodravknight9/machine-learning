"""
Utilities for the linear regression notebook.
"""
import numpy as np


def load_regression_data():
    """
    Load regression data (diabetes dataset from sklearn).
    Returns X of shape (n_features, n_samples) and y of shape (1, n_samples).
    """
    try:
        from sklearn.datasets import load_diabetes
    except ImportError:
        raise ImportError("scikit-learn is required. Install with: pip install scikit-learn")

    data = load_diabetes()
    # data.data is (442, 10), data.target is (442,)
    X = np.array(data.data, dtype=np.float64).T   # (10, 442)
    y = np.array(data.target, dtype=np.float64).reshape(1, -1)   # (1, 442)
    return X, y
