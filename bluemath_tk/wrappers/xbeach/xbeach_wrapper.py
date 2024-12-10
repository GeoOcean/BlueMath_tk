from .._base_wrappers import BaseModelWrapper


class XBeachModelWrapper(BaseModelWrapper):
    """
    Wrapper for the XBeach model.
    https://xbeach.readthedocs.io/en/latest/

    Attributes
    ----------
    xbeach_exec : str
        The XBeach executable path.
    default_parameters : dict
        The default parameters type for the model.

    Methods
    -------
    set_xbeach_exec(xbeach_exec: str) -> None
        Set the XBeach executable path.
    run_model() -> None
        Run the XBeach model for the specified case.
    """

    default_parameters = {
        "spectra": str,
    }

    def __init__(
        self,
        templates_dir: str,
        templates_name: dict,
        model_parameters: dict,
        output_dir: str,
    ) -> None:
        """
        Initialize the SWASH model wrapper.

        Parameters
        ----------
        templates_dir : str
            The directory where the templates are stored.
        templates_name : list
            The names of the templates.
        model_parameters : dict
            The parameters to be used in the templates.
        output_dir : str
            The directory where the output files will be saved.
        """

        super().__init__(
            templates_dir=templates_dir,
            templates_name=templates_name,
            model_parameters=model_parameters,
            output_dir=output_dir,
            default_parameters=self.default_parameters,
        )
        self.set_logger_name(self.__class__.__name__)
        self._xbeach_exec = None

    @property
    def xbeach_exec(self) -> str:
        return self._xbeach_exec

    def set_xbeach_exec(self, xbeach_exec: str) -> None:
        self._xbeach_exec = xbeach_exec

    def run_model(self) -> None:
        pass
