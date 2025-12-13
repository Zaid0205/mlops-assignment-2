from Add_Tests.model import get_model

def train_model(X, y):
    model = get_model()
    model.fit(X, y)
    return model