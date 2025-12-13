from Add_Tests.data_loader import load_data
from Add_Tests.train import train_model
import numpy as np

def test_data_loading():
    X, y = load_data()
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert len(X) == len(y)

def test_model_training():
    X, y = load_data()
    model = train_model(X, y)
    assert hasattr(model, "predict")
    preds = model.predict(X)
    assert len(preds) == len(y)

def test_shape_validation():
    X, y = load_data()
    assert X.shape[1] == 4