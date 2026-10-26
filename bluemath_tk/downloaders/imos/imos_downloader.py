"""
Downloader for IMOS data in the public AODN bucket on AWS S3 (no account needed).

Products (configs in ``imos/<PRODUCT>/<PRODUCT>_config.json``):

- SRS: Satellite Remote Sensing facility, e.g. the multi-mission altimeter
  wave/wind archive (``Wave-Wind-Altimetry-DM00``).

Each dataset is a prefix of the bucket. Files are listed with the S3
ListObjectsV2 API and mirrored locally with the same folders below the prefix.
IMOS rewrites whole files when it releases new data, so a file is downloaded
again when its size changes or the bucket copy is newer than the local one.
"""

import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import List, Optional, Sequence, Tuple, Union
from xml.etree import ElementTree

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
S3_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
FILE_COLUMNS = ["key", "path", "url", "size", "etag", "last_modified"]

TimeLike = Union[str, datetime, pd.Timestamp]


def parse_listing(xml: bytes) -> Tuple[List[dict], List[str], Optional[str]]:
    """
    One page of an S3 ListObjectsV2 answer.

    Parameters
    ----------
    xml : bytes
        Body of the answer.

    Returns
    -------
    Tuple[List[dict], List[str], Optional[str]]
        Objects (``key``, ``size``, ``etag`` without quotes, ``last_modified``
        as a UTC timestamp), common prefixes (sub-folders, when a delimiter is
        used) and the continuation token (None on the last page).
    """

    root = ElementTree.fromstring(xml)
    objects = [
        {
            "key": c.findtext("s3:Key", namespaces=S3_NS),
            "size": int(c.findtext("s3:Size", namespaces=S3_NS)),
            "etag": c.findtext("s3:ETag", default="", namespaces=S3_NS).strip('"'),
            "last_modified": pd.Timestamp(
                c.findtext("s3:LastModified", namespaces=S3_NS)
            ),
        }
        for c in root.findall("s3:Contents", S3_NS)
    ]
    prefixes = [
        p.findtext("s3:Prefix", namespaces=S3_NS)
        for p in root.findall("s3:CommonPrefixes", S3_NS)
    ]
    truncated = root.findtext("s3:IsTruncated", default="false", namespaces=S3_NS)
    token = root.findtext("s3:NextContinuationToken", namespaces=S3_NS)

    return objects, prefixes, token if truncated == "true" else None


class IMOSDownloader(BaseDownloader):
    """
    Downloader for IMOS datasets in the public AODN S3 bucket.

    Files are saved to base_path_to_download/<local_dir of the dataset>/<path
    below the dataset prefix>. Existing files are skipped when their size
    matches and they are not older than the bucket copy; downloaded files get
    the bucket modification time, are checked against the S3 ETag (MD5 of
    single-part uploads) and several are downloaded in parallel
    (``max_workers``; files are small, so many workers help).

    Examples
    --------
    >>> downloader = IMOSDownloader(
    ...     product="SRS",
    ...     base_path_to_download="/lustre/geocean/DATA/IMOS",
    ...     max_workers=16,
    ... )
    >>> downloader.list_subfolders("Wave-Wind-Altimetry-DM00")  # missions
    >>> result = downloader.download_data(
    ...     "Wave-Wind-Altimetry-DM00", subfolders=["JASON-3"], dry_run=True
    ... )
    """

    products_configs = {
        "SRS": load_config(_HERE, "SRS", "SRS_config.json"),
    }

    def __init__(
        self,
        product: str,
        base_path_to_download: str,
        max_workers: int = 16,
        debug: bool = True,
        **kwargs,
    ) -> None:
        """
        Initialize the IMOSDownloader.

        Parameters
        ----------
        product : str
            The product to download data from (e.g. "SRS").
        base_path_to_download : str
            Base path where downloaded files will be stored.
        max_workers : int, optional
            Number of files downloaded (and folders listed) in parallel.
            Default is 16.
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
            f"IMOSDownloader-{product}", level="DEBUG" if debug else "INFO"
        )
        self.logger.info(f"---- IMOS DOWNLOADER INITIALIZED ({product}) ----")

    @property
    def session(self) -> requests.Session:
        """
        HTTP session of the current thread.

        Returns
        -------
        requests.Session
            Session with retries for connection errors and 5xx responses.
        """
        if not hasattr(self._local, "session"):
            self._local.session = make_session()
        return self._local.session

    def list_subfolders(self, dataset: str) -> List[str]:
        """
        Sub-folders of a dataset (e.g. the missions of the altimeter archive).

        Parameters
        ----------
        dataset : str
            The dataset name.

        Returns
        -------
        List[str]
            Folder names directly below the dataset prefix.
        """

        prefix = self._get_dataset_config(dataset)["prefix"]
        _, prefixes = self._list(prefix, delimiter="/")

        return [p[len(prefix) :].rstrip("/") for p in prefixes]

    def list_files(
        self,
        dataset: str,
        subfolders: Optional[Sequence[str]] = None,
        modified_since: Optional[TimeLike] = None,
    ) -> pd.DataFrame:
        """
        Files of a dataset, from the bucket listing.

        Parameters
        ----------
        dataset : str
            The dataset name. Use list_datasets() to see available datasets.
        subfolders : Sequence[str], optional
            Only these sub-folders (e.g. missions). Default is all. They are
            listed in parallel.
        modified_since : str or datetime, optional
            Only files modified in the bucket since then (UTC).

        Returns
        -------
        pd.DataFrame
            One row per file (columns FILE_COLUMNS; ``path`` is relative to
            the dataset prefix), sorted by path. Files that do not match the
            dataset ``filename_pattern`` are left out.
        """

        dataset_config = self._get_dataset_config(dataset)
        prefix = dataset_config["prefix"]
        if subfolders is None:
            subfolders = self.list_subfolders(dataset)
        prefixes = [f"{prefix}{s.strip('/')}/" for s in subfolders]

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            listings = list(executor.map(lambda p: self._list(p)[0], prefixes))

        files = pd.DataFrame(
            [o for objects in listings for o in objects],
            columns=["key", "size", "etag", "last_modified"],
        )
        files["path"] = files["key"].str.slice(len(prefix))
        files["url"] = self.product_config["bucket_url"] + files["key"]
        pattern = dataset_config.get("filename_pattern")
        if pattern:
            keep = files["key"].str.rsplit("/", n=1).str[-1].str.match(pattern)
            if (~keep).any():
                self.logger.info(
                    f"{(~keep).sum()} files do not match {pattern} and are left out"
                )
            files = files[keep]
        if modified_since is not None:
            since = pd.Timestamp(modified_since)
            since = since.tz_localize("UTC") if since.tz is None else since
            files = files[files["last_modified"] >= since]
        files = files[FILE_COLUMNS].sort_values("path").reset_index(drop=True)
        self.logger.info(
            f"{len(files)} files in {prefix} ({files['size'].sum() / 1e9:,.1f} GB)"
        )

        return files

    def get_local_path(self, dataset: str, path: str) -> str:
        """
        Local path of a file.

        Parameters
        ----------
        dataset : str
            The dataset name.
        path : str
            Path of the file below the dataset prefix (``path`` of list_files).

        Returns
        -------
        str
            The local path.
        """

        local_dir = self._get_dataset_config(dataset)["local_dir"]

        return os.path.normpath(
            os.path.join(self.base_path_to_download, local_dir, path)
        )

    def download_data(
        self,
        dataset: str,
        subfolders: Optional[Sequence[str]] = None,
        modified_since: Optional[TimeLike] = None,
        force: bool = False,
        verify_existing: bool = True,
        dry_run: bool = True,
    ) -> DownloadResult:
        """
        Download (or update) the files of a dataset.

        The same call keeps a local copy up to date: complete files that did
        not change in the bucket are skipped.

        Parameters
        ----------
        dataset : str
            The dataset to download. Use list_datasets() to see available datasets.
        subfolders : Sequence[str], optional
            Only these sub-folders (e.g. missions). Default is all.
        modified_since : str or datetime, optional
            Only files modified in the bucket since then.
        force : bool, optional
            Download again even if the file is up to date. Default is False.
        verify_existing : bool, optional
            Compare existing files with the bucket (size and modification
            time) and download them again if they differ. Default is True.
        dry_run : bool, optional
            If True, only check what would be downloaded. Default is True.

        Returns
        -------
        DownloadResult
            Result with information about downloaded, skipped, and error files.
            The size of each file is in metadata["remote_sizes"].
        """

        files = self.list_files(dataset, subfolders, modified_since)

        return self.download_files(
            dataset,
            files,
            force=force,
            verify_existing=verify_existing,
            dry_run=dry_run,
        )

    def download_files(
        self,
        dataset: str,
        files: pd.DataFrame,
        force: bool = False,
        verify_existing: bool = True,
        dry_run: bool = True,
    ) -> DownloadResult:
        """
        Download the files of a table returned by list_files.

        Parameters
        ----------
        dataset : str
            The dataset the files belong to.
        files : pd.DataFrame
            Files to download (output of list_files, possibly filtered).
        force, verify_existing, dry_run : bool, optional
            As in download_data.

        Returns
        -------
        DownloadResult
            Result with information about downloaded, skipped, and error files.
        """

        self.logger.info(f"Checking {len(files)} files of {self.product}/{dataset}")
        result = self.create_download_result()
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(
                    self._download_file,
                    row,
                    self.get_local_path(dataset, row.path),
                    force,
                    verify_existing,
                    dry_run,
                )
                for row in files.itertuples(index=False)
            ]
            for future in futures:
                result.merge(future.result())

        return self.finalize_download_result(result)

    def _get_dataset_config(self, dataset: str) -> dict:
        if dataset not in self.list_datasets():
            raise ValueError(
                f"Dataset '{dataset}' not found. Available: {self.list_datasets()}"
            )
        return self.product_config["datasets"][dataset]

    def _list(
        self, prefix: str, delimiter: Optional[str] = None
    ) -> Tuple[List[dict], List[str]]:
        """All objects (and common prefixes) below a prefix, every page."""

        params = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if delimiter:
            params["delimiter"] = delimiter
        objects, prefixes, token = [], [], None
        while True:
            page = dict(params, **({"continuation-token": token} if token else {}))
            response = retry_call(
                lambda: http_get(
                    self.session, self.product_config["bucket_url"], params=page
                ),
                retries=self.retries,
                backoff=self.retry_backoff,
                logger=self.logger,
            )
            page_objects, page_prefixes, token = parse_listing(response.content)
            objects.extend(page_objects)
            prefixes.extend(page_prefixes)
            if token is None:
                return objects, prefixes

    def _download_file(
        self,
        row,
        output_file: str,
        force: bool,
        verify_existing: bool,
        dry_run: bool,
    ) -> DownloadResult:
        """
        Download a single file unless the local copy is complete and up to date.
        """

        result = DownloadResult()
        result.metadata["remote_sizes"] = {output_file: int(row.size)}
        modified = row.last_modified.timestamp()

        if os.path.exists(output_file) and not force:
            local_size = os.path.getsize(output_file)
            local_time = os.path.getmtime(output_file)
            if not verify_existing or (
                local_size == row.size and local_time >= modified
            ):
                result.add_skipped(output_file, "Already downloaded")
                return result
            reason = (
                "Updated in the bucket"
                if local_size == row.size
                else (f"Local size {local_size} differs from bucket size {row.size}")
            )
            if dry_run:
                result.add_skipped(
                    output_file, f"{reason}, would download again (dry run)"
                )
                return result
            self.logger.debug(f"{reason}, downloading again: {output_file}")
            force = True

        def fetch(part_file: str) -> None:
            http_download(
                self.session,
                row.url,
                part_file,
                expected_size=int(row.size),
                resume=True,
            )
            # The ETag is the MD5 of single-part uploads ("<md5>-<parts>" otherwise)
            if re.fullmatch(r"[0-9a-f]{32}", row.etag or ""):
                checksum = file_checksum(part_file, "MD5")
                if checksum != row.etag:
                    os.remove(part_file)
                    raise IOError(
                        f"MD5 of {row.key} is {checksum}, ETag says {row.etag}"
                    )
            os.utime(part_file, (modified, modified))

        self.fetch_file(
            output_file, fetch, result, force=force, dry_run=dry_run, resumable=True
        )

        return result
