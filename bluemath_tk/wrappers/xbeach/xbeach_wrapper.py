from .._base_wrappers import BaseModelWrapper


class XBeachModelWrapper(BaseModelWrapper):
    """
    Wrapper for the XBeach model.
    https://xbeach.readthedocs.io/en/latest/

    Attributes
    ----------
    default_parameters : dict
        The default parameters type for the wrapper.
    available_launchers : dict
        The available launchers for the wrapper.
    """

    default_parameters = {
        "spectra": str,
    }

    available_launchers = {
        "geoocean-cluster": "launchXbeach.sh",
    }

    def __init__(
        self,
        templates_dir: str,
        model_parameters: dict,
        output_dir: str,
        templates_name: dict = "all",
        debug: bool = True,
    ) -> None:
        """
        Initialize the SWASH model wrapper.
        """

        super().__init__(
            templates_dir=templates_dir,
            model_parameters=model_parameters,
            output_dir=output_dir,
            templates_name=templates_name,
            default_parameters=self.default_parameters,
        )
        self.set_logger_name(
            name=self.__class__.__name__, level="DEBUG" if debug else "INFO"
        )
