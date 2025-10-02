import json
import os
from typing import List, Union

import xarray as xr
from ecmwf.opendata import Client

from .._base_downloaders import BaseDownloader


class ECMWFDownloader(BaseDownloader):
    """
    This is the main class to download data from the ECMWF.

    Attributes
    ----------
    product : str
        The product to download data from. Currently only OpenData is supported.
    product_config : dict
        The configuration for the product to download data from.
    client : ecmwf.opendata.Client
        The client to interact with the ECMWF API.

    Examples
    --------
    .. jupyter-execute::

        from bluemath_tk.downloaders.ecmwf.ecmwf_downloader import ECMWFDownloader

        ecmwf_downloader = ECMWFDownloader(
            product="OpenData",
            base_path_to_download="/path/to/ECMWF/",  # Will be created if not available
            check=True,
        )
        dataset = ecmwf_downloader.download_data(
            load_data=False,
            param=["msl"],
            step=[0, 240],
            type="fc",
        )
        print(dataset)
    """

    products_configs = {
        "OpenData": json.load(
            open(
                os.path.join(
                    os.path.dirname(__file__), "OpenData", "OpenData_config.json"
                )
            )
        )
    }

    def __init__(
        self,
        product: str,
        base_path_to_download: str,
        model: str = "ifs",
        resolution: str = "0p25",
        debug: bool = True,
        check: bool = True,
    ) -> None:
        """
        This is the constructor for the ECMWFDownloader class.

        Parameters
        ----------
        product : str
            The product to download data from. Currently only OpenData is supported.
        base_path_to_download : str
            The base path to download the data to.
        model : str, optional
            The model to download data from. Default is "ifs".
        resolution : str, optional
            The resolution to download data from. Default is "0p25".
        debug : bool, optional
            Whether to run in debug mode. Default is True.
        check : bool, optional
            Whether to just check the data. Default is True.

        Raises
        ------
        ValueError
            If the product configuration is not found.
        """

        super().__init__(
            base_path_to_download=base_path_to_download, debug=debug, check=check
        )
        self._product = product
        self._product_config = self.products_configs.get(product)
        if self._product_config is None:
            raise ValueError(f"{product} configuration not found")
        self.set_logger_name(
            f"ECMWFDownloader-{product}", level="DEBUG" if debug else "INFO"
        )
        if not self.check:
            if model not in self.product_config["datasets"]["forecast_data"]["models"]:
                raise ValueError(f"Model {model} not supported for {self.product}")
            if (
                resolution
                not in self.product_config["datasets"]["forecast_data"]["resolutions"]
            ):
                raise ValueError(
                    f"Resolution {resolution} not supported for {self.product}"
                )
            self._client = Client(
                source="ecmwf",
                model=model,
                resol=resolution,
                preserve_request_order=False,
                infer_stream_keyword=True,
            )
            self.logger.info("---- DOWNLOADING DATA ----")
        else:
            self.logger.info("---- CHECKING DATA ----")

        # Set the model and resolution parameters
        self.model = model
        self.resolution = resolution

    @property
    def product(self) -> str:
        return self._product

    @property
    def product_config(self) -> dict:
        return self._product_config

    @property
    def client(self) -> Client:
        return self._client

    def list_datasets(self) -> List[str]:
        """
        Lists the datasets available for the product.

        Returns
        -------
        List[str]
            The list of datasets available for the product.
        """

        return list(self.product_config["datasets"].keys())

    def download_data(
        self, load_data: bool = False, *args, **kwargs
    ) -> Union[str, xr.Dataset]:
        """
        Downloads the data for the product.

        Parameters
        ----------
        load_data : bool, optional
            Whether to load the data into an xarray.Dataset. Default is False.
        *args
            The arguments to pass to the download function.
        **kwargs
            The keyword arguments to pass to the download function.

        Returns
        -------
        Union[str, xr.Dataset]
            The path to the downloaded file if load_data is False, otherwise the xarray.Dataset.

        Raises
        ------
        ValueError
            If the product is not supported.
        """

        if self.product == "OpenData":
            downloaded_file_path = self.download_data_open_data(*args, **kwargs)
            if load_data:
                return xr.open_dataset(downloaded_file_path, engine="cfgrib")
            else:
                return downloaded_file_path
        else:
            raise ValueError(f"Download for product {self.product} not supported")

    def download_data_open_data(
        self,
        force: bool = False,
        **kwargs,
    ) -> str:
        """
        Downloads the data for the OpenData product.

        Parameters
        ----------
        force : bool, optional
            Whether to force the download. Default is False.
        **kwargs
            The keyword arguments to pass to the download function.

        Returns
        -------
        str
            The path to the downloaded file.
        """

        if "param" in kwargs:
            variables = kwargs["param"]
        else:
            variables = []
        if "step" in kwargs:
            steps = kwargs["step"]
            if not isinstance(steps, list):
                steps = [steps]
        else:
            steps = []
        if "type" in kwargs:
            type = kwargs["type"]
        else:
            type = "fc"

        output_grib_file = os.path.join(
            self.base_path_to_download,
            self.product,
            self.model,
            self.resolution,
            f"{'_'.join(variables)}_{'_'.join(str(step) for step in steps)}_{type}.grib2",
        )
        if not self.check:
            os.makedirs(os.path.dirname(output_grib_file), exist_ok=True)

        if self.check or not force:
            if os.path.exists(output_grib_file):
                self.logger.debug(f"{output_grib_file} already downloaded")
            else:
                if self.check:
                    self.logger.debug(f"{output_grib_file} not downloaded")
                else:
                    self.logger.debug(f"Downloading: {output_grib_file}")
                    self.client.retrieve(
                        target=output_grib_file,
                        **kwargs,
                    )
        else:
            self.logger.debug(f"Downloading: {output_grib_file}")
            self.client.retrieve(
                target=output_grib_file,
                **kwargs,
            )

        return output_grib_file
