import json
import os

from ecmwf.opendata import Client

from .._base_downloaders import BaseDownloader
from .._download_result import DownloadResult


class ECMWFDownloader(BaseDownloader):
    """
    This is the main class to download data from the ECMWF.

    Examples
    --------
    >>> downloader = ECMWFDownloader(
    ...     product="OpenData",
    ...     base_path_to_download="./ecmwf_data",
    ...     model="ifs",
    ...     resolution="0p25"
    ... )
    >>> result = downloader.download_data(
    ...     dataset="forecast_data",
    ...     param=["msl"],
    ...     step=[0, 240],
    ...     type="fc",
    ...     force=False,
    ...     dry_run=False
    ... )
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
    ) -> None:
        """
        Initialize the ECMWFDownloader.

        Parameters
        ----------
        product : str
            The product to download data from. Currently only OpenData is supported.
        base_path_to_download : str
            Base path where downloaded files will be stored.
        model : str, optional
            The model to download data from (e.g., "ifs", "aifs"). Default is "ifs".
        resolution : str, optional
            The resolution to download data from (e.g., "0p25"). Default is "0p25".
        debug : bool, optional
            If True, sets logger to DEBUG level. Default is True.

        Raises
        ------
        ValueError
            If the product configuration is not found, or if model/resolution are not supported.
        """

        super().__init__(
            product=product, base_path_to_download=base_path_to_download, debug=debug
        )

        self._product_config = self.products_configs.get(product)
        if self._product_config is None:
            available_products = list(self.products_configs.keys())
            raise ValueError(
                f"{product} configuration not found. Available products: {available_products}"
            )

        self.set_logger_name(
            f"ECMWFDownloader-{product}", level="DEBUG" if debug else "INFO"
        )

        # Validate model and resolution
        if model not in self.product_config["datasets"]["forecast_data"]["models"]:
            raise ValueError(f"Model {model} not supported for {self.product}")
        if (
            resolution
            not in self.product_config["datasets"]["forecast_data"]["resolutions"]
        ):
            raise ValueError(
                f"Resolution {resolution} not supported for {self.product}"
            )
        # Always initialize client (will skip API calls in dry_run mode)
        self._client = Client(
            source="ecmwf",
            model=model,
            resol=resolution,
            preserve_request_order=False,
            infer_stream_keyword=True,
        )
        self.logger.info(f"---- ECMWF DOWNLOADER INITIALIZED ({product}) ----")

        # Set the model and resolution parameters
        self.model = model
        self.resolution = resolution

    @property
    def product_config(self) -> dict:
        """
        Product configuration dictionary loaded from config file.

        Returns
        -------
        dict
            Product configuration dictionary.
        """
        return self._product_config

    @property
    def client(self) -> Client:
        """
        ECMWF OpenData client (initialized with model and resolution).

        Returns
        -------
        Client
            ECMWF OpenData client instance.
        """
        return self._client

    def download_data(
        self,
        dry_run: bool = True,
        *args,
        **kwargs,
    ) -> DownloadResult:
        """
        Download data for the product.

        Routes to product-specific download methods based on the product type.

        Parameters
        ----------
        dry_run : bool, optional
            If True, only check what would be downloaded without actually downloading.
            Default is True.
        *args
            Arguments passed to product-specific download method.
        **kwargs
            Keyword arguments passed to product-specific download method.

        Returns
        -------
        DownloadResult
            Result with information about downloaded, skipped, and error files.

        Raises
        ------
        ValueError
            If the product is not supported.
        """

        if self.product == "OpenData":
            return self.download_data_open_data(dry_run=dry_run, *args, **kwargs)
        else:
            raise ValueError(f"Download for product {self.product} not supported")

    def download_data_open_data(
        self,
        dataset: str,
        force: bool = False,
        dry_run: bool = True,
        **kwargs,
    ) -> DownloadResult:
        """
        Download data for the OpenData product.

        Downloads files based on the specified parameters. Files are saved to:
        base_path_to_download/product/dataset/model/resolution/filename.grib2

        Parameters
        ----------
        dataset : str
            The dataset to download (e.g., "forecast_data").
            Use list_datasets() to see available datasets.
        force : bool, optional
            Force re-download even if file exists. Default is False.
        dry_run : bool, optional
            If True, only check what would be downloaded. Default is True.
        **kwargs
            Keyword arguments passed to the ECMWF client retrieve method
            (e.g., param, step, type).

        Returns
        -------
        DownloadResult
            Result with all downloaded files and download statistics.

        Raises
        ------
        ValueError
            If dataset is not found.
        """

        # Validate dataset
        if dataset not in self.list_datasets():
            raise ValueError(
                f"Dataset '{dataset}' not found. Available: {self.list_datasets()}"
            )

        result = self.create_download_result()

        try:
            # Extract parameters from kwargs
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

            # Construct output file path: base_path/product/dataset/model/resolution/filename.grib2
            output_grib_file = os.path.join(
                self.base_path_to_download,
                self.product,
                dataset,
                self.model,
                self.resolution,
                f"{'_'.join(variables)}_{'_'.join(str(step) for step in steps)}_{type}.grib2",
            )

            # Skip if file already exists (unless force=True)
            if not force and os.path.exists(output_grib_file):
                result.add_skipped(output_grib_file, "Already downloaded")
                return self.finalize_download_result(result)

            # Handle dry run: record as skipped without actual download
            if dry_run:
                result.add_skipped(output_grib_file, "Would download (dry run)")
                return self.finalize_download_result(result)

            # Attempt to download the file
            try:
                # Create local directory structure if needed
                os.makedirs(os.path.dirname(output_grib_file), exist_ok=True)

                # Download the file
                self.logger.debug(f"Downloading: {output_grib_file}")
                self.client.retrieve(
                    target=output_grib_file,
                    **kwargs,
                )

                result.add_downloaded(output_grib_file)
                self.logger.info(f"Downloaded: {output_grib_file}")

            except Exception as e:
                result.add_error(output_grib_file, e)
                self.logger.error(f"Error downloading {output_grib_file}: {e}")

            return self.finalize_download_result(result)

        except Exception as e:
            result.add_error("download_operation", e)
            return self.finalize_download_result(result)
