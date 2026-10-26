"""
Downloader for NASA Earthdata collections (PO.DAAC and the other NASA DAACs).

Products (configs in ``earthdata/<PRODUCT>/<PRODUCT>_config.json``):

- SWOT: Surface Water and Ocean Topography mission, KaRIn L2 products.

Granules are found with the NASA CMR search API (public, no login) and
downloaded over HTTPS. Downloads redirect to the Earthdata login
(urs.earthdata.nasa.gov), which reads the credentials from ``~/.netrc``::

    machine urs.earthdata.nasa.gov login <user> password <password>

Each dataset of a product is one CMR collection, given by its short name.
"""

import netrc
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import List, Optional, Sequence, Union

import pandas as pd
import requests

from .._base_downloaders import BaseDownloader, load_config
from .._download_result import DownloadResult
from .._transfer import (
    file_checksum,
    http_download,
    http_get,
    make_session,
    retry_call,
)

_HERE = os.path.dirname(__file__)
URS_HOST = "urs.earthdata.nasa.gov"
CMR_PAGE_SIZE = 2000
GRANULE_COLUMNS = [
    "granule_ur",
    "filename",
    "url",
    "size",
    "checksum_algorithm",
    "checksum",
    "start_time",
    "end_time",
    "revision_date",
]

TimeLike = Union[str, datetime, pd.Timestamp]


def cmr_time_range(start: Optional[TimeLike], end: Optional[TimeLike]) -> str:
    """
    CMR ``temporal`` parameter for a time range.

    Strings are read at their own resolution, so ``end="2024-06"`` covers the
    whole of June 2024 and ``end="2024-06-15"`` the whole day.

    Parameters
    ----------
    start, end : str or datetime, optional
        First and last time (both included). None leaves that side open.

    Returns
    -------
    str
        ``"<start>,<end>"`` in ISO 8601 UTC, with empty sides when open.
    """

    def to_period(value: TimeLike) -> pd.Period:
        if isinstance(value, str):
            return pd.Period(value)
        return pd.Period(pd.Timestamp(value), freq="s")

    parts = []
    if start is not None:
        parts.append(to_period(start).start_time.strftime("%Y-%m-%dT%H:%M:%SZ"))
    else:
        parts.append("")
    if end is not None:
        parts.append(to_period(end).end_time.strftime("%Y-%m-%dT%H:%M:%SZ"))
    else:
        parts.append("")

    return ",".join(parts)


def parse_granule(item: dict) -> List[dict]:
    """
    Data files of one CMR granule (UMM-JSON item).

    Parameters
    ----------
    item : dict
        One element of ``items`` in a ``granules.umm_json`` response.

    Returns
    -------
    List[dict]
        One row per HTTPS data file ("GET DATA" link), with the columns in
        GRANULE_COLUMNS. Size and checksum come from the granule's
        ``DataGranule`` information and are None if it does not list the file.
    """

    umm = item["umm"]
    files_info = {
        f.get("Name"): f
        for f in umm.get("DataGranule", {}).get("ArchiveAndDistributionInformation", [])
    }
    time_range = umm.get("TemporalExtent", {}).get("RangeDateTime", {})

    rows = []
    for link in umm.get("RelatedUrls", []):
        url = link.get("URL", "")
        if link.get("Type") != "GET DATA" or not url.startswith("https://"):
            continue
        filename = url.rsplit("/", 1)[-1]
        info = files_info.get(filename, {})
        checksum = info.get("Checksum", {})
        rows.append(
            {
                "granule_ur": umm.get("GranuleUR"),
                "filename": filename,
                "url": url,
                "size": info.get("SizeInBytes"),
                "checksum_algorithm": checksum.get("Algorithm"),
                "checksum": checksum.get("Value"),
                "start_time": time_range.get("BeginningDateTime"),
                "end_time": time_range.get("EndingDateTime"),
                "revision_date": item.get("meta", {}).get("revision-date"),
            }
        )

    return rows


class EarthdataDownloader(BaseDownloader):
    """
    Downloader for NASA Earthdata collections, found through the CMR search API.

    Files are saved to base_path_to_download/<local_dir of the dataset>, where
    ``local_dir`` may use ``{year}`` and ``{month}`` of the granule start time.
    Existing files are compared with the size listed in CMR, transfers are
    resumable and checked against the CMR checksum, and several files are
    downloaded in parallel (``max_workers``).

    Examples
    --------
    >>> downloader = EarthdataDownloader(
    ...     product="SWOT",
    ...     base_path_to_download="/lustre/geocean/DATA/NASA",
    ... )
    >>> granules = downloader.search_granules(
    ...     "L2_LR_SSH_WINDWAVE_D", start="2024-06", end="2024-06",
    ...     bbox=(2.5, 51.0, 7.5, 54.0),
    ... )
    >>> result = downloader.download_data(
    ...     "L2_LR_SSH_WINDWAVE_D", start="2024-06", end="2024-06",
    ...     bbox=(2.5, 51.0, 7.5, 54.0), dry_run=False,
    ... )
    """

    products_configs = {
        "SWOT": load_config(_HERE, "SWOT", "SWOT_config.json"),
    }

    def __init__(
        self,
        product: str,
        base_path_to_download: str,
        max_workers: int = 4,
        debug: bool = True,
        **kwargs,
    ) -> None:
        """
        Initialize the EarthdataDownloader.

        Parameters
        ----------
        product : str
            The product to download data from (e.g. "SWOT").
        base_path_to_download : str
            Base path where downloaded files will be stored.
        max_workers : int, optional
            Number of files downloaded in parallel. Default is 4.
        debug : bool, optional
            If True, sets logger to DEBUG level. Default is True.
        **kwargs
            Passed to BaseDownloader (e.g., retries, retry_backoff).
        """

        super().__init__(
            product=product,
            base_path_to_download=base_path_to_download,
            debug=debug,
            **kwargs,
        )

        self.max_workers = max_workers
        self._local = threading.local()
        self._exclude_attributes.append("_local")

        self.set_logger_name(
            f"EarthdataDownloader-{product}", level="DEBUG" if debug else "INFO"
        )
        self.logger.info(f"---- EARTHDATA DOWNLOADER INITIALIZED ({product}) ----")

    @property
    def session(self) -> requests.Session:
        """
        HTTP session of the current thread.

        requests reads ``~/.netrc`` for the Earthdata login host when a download
        redirects there; the session then keeps the login cookies.

        Returns
        -------
        requests.Session
            Session with retries for connection errors and 5xx responses.
        """
        if not hasattr(self._local, "session"):
            self._local.session = make_session()
        return self._local.session

    def search_granules(
        self,
        dataset: str,
        start: Optional[TimeLike] = None,
        end: Optional[TimeLike] = None,
        bbox: Optional[Sequence[float]] = None,
        updated_since: Optional[TimeLike] = None,
    ) -> pd.DataFrame:
        """
        Data files of a dataset, from the CMR granule search.

        Parameters
        ----------
        dataset : str
            The dataset name. Use list_datasets() to see available datasets.
        start, end : str or datetime, optional
            Time range (both included, see cmr_time_range). Granules that
            overlap it are returned. Default is the whole dataset.
        bbox : Sequence[float], optional
            (west, south, east, north) in degrees, longitudes in -180/180.
        updated_since : str or datetime, optional
            Only granules added or changed in CMR since then.

        Returns
        -------
        pd.DataFrame
            One row per file (columns GRANULE_COLUMNS), sorted by start time.
            Files that do not match the dataset ``filename_pattern`` are left out.
        """

        dataset_config = self._get_dataset_config(dataset)
        if start is None:
            start = dataset_config.get("start")
        params = {
            "short_name": dataset_config["short_name"],
            "page_size": CMR_PAGE_SIZE,
            "sort_key": "start_date",
        }
        if start is not None or end is not None:
            params["temporal"] = cmr_time_range(start, end)
        if bbox is not None:
            if len(bbox) != 4:
                raise ValueError("bbox must be (west, south, east, north)")
            params["bounding_box"] = ",".join(str(float(v)) for v in bbox)
        if updated_since is not None:
            params["updated_since"] = pd.Timestamp(updated_since).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )

        rows = []
        search_after = None
        while True:
            headers = {"CMR-Search-After": search_after} if search_after else {}

            def get_page() -> requests.Response:
                response = http_get(
                    self.session,
                    self.product_config["cmr_url"],
                    params=params,
                    headers=headers,
                )
                response.json()  # retry truncated responses too
                return response

            response = retry_call(
                get_page,
                retries=self.retries,
                backoff=self.retry_backoff,
                logger=self.logger,
            )
            items = response.json().get("items", [])
            for item in items:
                rows.extend(parse_granule(item))
            search_after = response.headers.get("CMR-Search-After")
            if not items or not search_after:
                break

        granules = pd.DataFrame(rows, columns=GRANULE_COLUMNS)
        pattern = dataset_config.get("filename_pattern")
        if pattern:
            keep = granules["filename"].str.match(pattern)
            if (~keep).any():
                self.logger.info(
                    f"{(~keep).sum()} files do not match {pattern} and are left out"
                )
            granules = granules[keep]
        granules = granules.drop_duplicates("filename").sort_values("start_time")
        self.logger.info(
            f"{len(granules)} files in {dataset_config['short_name']}"
            f" ({granules['size'].sum() / 1e9:,.1f} GB)"
        )

        return granules.reset_index(drop=True)

    def get_local_path(self, dataset: str, filename: str, start_time: TimeLike) -> str:
        """
        Local path of a file.

        Parameters
        ----------
        dataset : str
            The dataset name.
        filename : str
            The file name.
        start_time : str or datetime
            Start time of the granule, used to fill ``{year}`` and ``{month}``
            of the dataset ``local_dir``.

        Returns
        -------
        str
            The local path.
        """

        dataset_config = self._get_dataset_config(dataset)
        start_time = pd.Timestamp(start_time)
        local_dir = dataset_config["local_dir"].format(
            year=start_time.year, month=start_time.month
        )

        return os.path.normpath(
            os.path.join(self.base_path_to_download, local_dir, filename)
        )

    def download_data(
        self,
        dataset: str,
        start: Optional[TimeLike] = None,
        end: Optional[TimeLike] = None,
        bbox: Optional[Sequence[float]] = None,
        updated_since: Optional[TimeLike] = None,
        force: bool = False,
        verify_existing: bool = True,
        dry_run: bool = True,
    ) -> DownloadResult:
        """
        Download the files of a dataset that match a search.

        The search arguments are those of search_granules. The same call can be
        repeated to keep a dataset up to date: complete files are skipped.

        Parameters
        ----------
        dataset : str
            The dataset to download. Use list_datasets() to see available datasets.
        start, end : str or datetime, optional
            Time range (both included). Default is the whole dataset.
        bbox : Sequence[float], optional
            (west, south, east, north) in degrees.
        updated_since : str or datetime, optional
            Only granules added or changed in CMR since then.
        force : bool, optional
            Download again even if the file exists. Default is False.
        verify_existing : bool, optional
            Compare the size of existing files with the CMR size and download
            them again if they differ. Default is True.
        dry_run : bool, optional
            If True, only check what would be downloaded. Default is True.

        Returns
        -------
        DownloadResult
            Result with information about downloaded, skipped, and error files.
            The CMR size of each file is in metadata["remote_sizes"].
        """

        granules = self.search_granules(dataset, start, end, bbox, updated_since)

        return self.download_granules(
            dataset,
            granules,
            force=force,
            verify_existing=verify_existing,
            dry_run=dry_run,
        )

    def download_granules(
        self,
        dataset: str,
        granules: pd.DataFrame,
        force: bool = False,
        verify_existing: bool = True,
        dry_run: bool = True,
    ) -> DownloadResult:
        """
        Download the files of a table returned by search_granules.

        Parameters
        ----------
        dataset : str
            The dataset the files belong to.
        granules : pd.DataFrame
            Files to download (output of search_granules, possibly filtered).
        force, verify_existing, dry_run : bool, optional
            As in download_data.

        Returns
        -------
        DownloadResult
            Result with information about downloaded, skipped, and error files.
        """

        if not dry_run:
            self.check_credentials()

        self.logger.info(f"Checking {len(granules)} files of {self.product}/{dataset}")
        result = self.create_download_result()
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(
                    self._download_file,
                    row,
                    self.get_local_path(dataset, row.filename, row.start_time),
                    force,
                    verify_existing,
                    dry_run,
                )
                for row in granules.itertuples(index=False)
            ]
            for future in futures:
                result.merge(future.result())

        return self.finalize_download_result(result)

    @staticmethod
    def check_credentials() -> None:
        """
        Check that ``~/.netrc`` has an entry for the Earthdata login.

        Raises
        ------
        ValueError
            If ``~/.netrc`` is missing or has no ``urs.earthdata.nasa.gov`` entry.
        """

        try:
            entry = netrc.netrc().authenticators(URS_HOST)
        except (FileNotFoundError, netrc.NetrcParseError) as e:
            raise ValueError(f"Cannot read ~/.netrc: {e}") from e
        if entry is None:
            raise ValueError(
                f"~/.netrc has no entry for {URS_HOST}: add "
                f"'machine {URS_HOST} login <user> password <password>' (chmod 600)"
            )

    def _get_dataset_config(self, dataset: str) -> dict:
        if dataset not in self.list_datasets():
            raise ValueError(
                f"Dataset '{dataset}' not found. Available: {self.list_datasets()}"
            )
        return self.product_config["datasets"][dataset]

    def _download_file(
        self,
        granule,
        output_file: str,
        force: bool,
        verify_existing: bool,
        dry_run: bool,
    ) -> DownloadResult:
        """
        Download a single file, comparing an existing file with the CMR size.
        """

        result = DownloadResult()
        size = None if pd.isna(granule.size) else int(granule.size)
        if size is not None:
            result.metadata["remote_sizes"] = {output_file: size}

        if os.path.exists(output_file) and not force:
            local_size = os.path.getsize(output_file)
            if not verify_existing or size is None or local_size == size:
                result.add_skipped(output_file, "Already downloaded")
                return result
            message = f"Local size {local_size} differs from CMR size {size}"
            if dry_run:
                result.add_skipped(
                    output_file, f"{message}, would download again (dry run)"
                )
                return result
            self.logger.warning(f"{message}, downloading again: {output_file}")
            force = True

        def fetch(part_file: str) -> None:
            http_download(
                self.session, granule.url, part_file, expected_size=size, resume=True
            )
            if isinstance(granule.checksum, str) and isinstance(
                granule.checksum_algorithm, str
            ):
                checksum = file_checksum(part_file, granule.checksum_algorithm)
                if checksum != granule.checksum.lower():
                    os.remove(part_file)
                    raise IOError(
                        f"{granule.checksum_algorithm} of {granule.filename} is "
                        f"{checksum}, CMR says {granule.checksum}"
                    )

        self.fetch_file(
            output_file, fetch, result, force=force, dry_run=dry_run, resumable=True
        )

        return result
