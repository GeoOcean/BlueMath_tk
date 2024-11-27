from abc import abstractmethod
from ..core.models import BlueMathModel


class BaseInterpolation(BlueMathModel):
    """
    Base class for all interpolation BlueMath models.
    This class provides the basic structure for all interpolation models.

    Methods
    -------
    fit(*args, **kwargs)
    predict(*args, **kwargs)
    fit_predict(*args, **kwargs)
    """

    @abstractmethod
    def __init__(self):
        super().__init__()

    @abstractmethod
    def fit(self, *args, **kwargs):
        """
        Fits the model to the data.

        Parameters
        ----------
        *args : list
            Positional arguments.
        **kwargs : dict
            Keyword arguments.
        """

        pass

    @abstractmethod
    def predict(self, *args, **kwargs):
        """
        Predicts the interpolated data given a dataset.

        Parameters
        ----------
        *args : list
            Positional arguments.
        **kwargs : dict
            Keyword arguments.
        """

        pass

    @abstractmethod
    def fit_predict(self, *args, **kwargs):
        """
        Fits the model to the subset and predicts the interpolated dataset.

        Parameters
        ----------
        *args : list
            Positional arguments.
        **kwargs : dict
            Keyword arguments.
        """

        pass
