import pickle

from .models import BlueMathModel


def load_model(model_path: str) -> BlueMathModel:
    """
    Loads a BlueMathModel from a file.

    Parameters
    ----------
    model_path : str
        The path to the model file.

    Returns
    -------
    BlueMathModel
        The loaded BlueMathModel.
    """

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    return model
