import pickle

from .models import BlueMathModel


def load_model(model_path: str) -> BlueMathModel:
    """Loads a BlueMathModel from a file."""

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    return model
