from abc import abstractmethod
from typing import List
import pandas as pd
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


class InterpolationComparator:
    """
    Class for comparing interpolation models.
    """

    def __init__(self, list_of_models: List[BaseInterpolation]) -> None:
        """
        Initializes the InterpolationComparator class.
        """

        self.list_of_models = list_of_models

    def fit(
        self,
        subset_data: pd.DataFrame,
        target_data: pd.DataFrame,
    ) -> None:
        """
        Fits the clustering models.
        """

        for model in self.list_of_models:
            model.fit(
                subset_data=subset_data,
                target_data=target_data,
            )
