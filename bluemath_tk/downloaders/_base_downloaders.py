from abc import abstractmethod
from ..core.models import BlueMathModel


class BaseDownloader(BlueMathModel):
    """
    Abstract class for BlueMath downloaders.

    Attributes
    ----------
    base_path_to_download : str
        The base path to download the data.
    debug : bool, optional
        If True, the logger will be set to DEBUG level. Default is True.
    check : bool, optional
        If True, just file checking is required. Default is False.

    Methods
    -------
    download_data(*args, **kwargs)
        Downloads the data. This method must be implemented in the child class.

    Notes
    -----
    - This class is an abstract class and should not be instantiated.
    - The download_data method must be implemented in the child class.
    """

    def __init__(
        self, base_path_to_download: str, debug: bool = True, check: bool = False
    ) -> None:
        """
        The constructor for BaseDownloader class.

        Parameters
        ----------
        base_path_to_download : str
            The base path to download the data.
        debug : bool, optional
            If True, the logger will be set to DEBUG level. Default is True.
        check : bool, optional
            If True, just file checking is required. Default is False.

        Raises
        ------
        ValueError
            If base_path_to_download is not a string.
            If debug is not a boolean.
            If check is not a boolean.

        Notes
        -----
        - The logger will be set to INFO level.
        - If debug is True, the logger will be set to DEBUG level.
        """

        super().__init__()
        if not isinstance(base_path_to_download, str):
            raise ValueError("base_path_to_download must be a string")
        self._base_path_to_download: str = base_path_to_download
        if not isinstance(debug, bool):
            raise ValueError("debug must be a boolean")
        self._debug: bool = debug
        if not isinstance(check, bool):
            raise ValueError("check must be a boolean")
        self._check: bool = check

    @property
    def base_path_to_download(self) -> str:
        return self._base_path_to_download

    @property
    def debug(self) -> bool:
        return self._debug

    @property
    def check(self) -> bool:
        return self._check

    @abstractmethod
    def download_data(self, *args, **kwargs) -> None:
        pass
