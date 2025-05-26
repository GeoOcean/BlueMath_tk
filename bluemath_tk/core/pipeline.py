from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr


class BlueMathPipeline:
    """
    This is the BlueMathPipeline class!
    """

    def __init__(self, steps: List[Tuple[str, Any]]):
        """
        Initialize the BlueMathPipeline.

        Parameters
        ----------
        steps : List[Tuple[str, Any]]
            A list of tuples where each tuple contains the name of the step,
            the model instance and the method to apply.
        """

        self.steps = steps

    def fit(
        self,
        data: Union[np.ndarray, pd.DataFrame, xr.Dataset],
        fit_params: Dict[str, Dict[str, Any]] = {},
    ):
        """
        Fit the pipeline models.

        Parameters
        ----------
        data : Union[np.ndarray, pd.DataFrame, xr.Dataset]
            The input data to fit the models.
        fit_params : Dict[str, Dict[str, Any]], optional
            A dictionary of parameters to pass to the fit method of each model.
            The keys should be the names of the steps, and the values should be
            dictionaries of parameters for the corresponding model's fit method.
        """

        for name, model in self.steps:
            params = fit_params.get(name, {})
            if hasattr(model, "fit"):
                model.fit(data, **params)
