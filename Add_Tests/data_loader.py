import numpy as np

def load_data():
    # Dummy dataset for testing
    X = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [2, 3, 4, 5]])
    y = np.array([0, 1, 0])
    return X, y