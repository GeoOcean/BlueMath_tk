"""
Low-level transfer helpers shared by all downloaders.

These functions are deliberately independent from any data provider, so every
downloader gets the same behaviour for retries, resumable HTTP transfers and
file validation.
"""

import logging
import os
import time
from typing import Callable, Optional, Tuple, Type, TypeVar

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

PART_SUFFIX = ".part"
CHUNK_SIZE = 8 * 1024 * 1024

T = TypeVar("T")


class RemoteFileNotFound(Exception):
    """Raised when a remote file does not exist (e.g. not published yet)."""


def retry_call(
    func: Callable[[], T],
    retries: int = 3,
    backoff: float = 30.0,
    logger: Optional[logging.Logger] = None,
    no_retry: Tuple[Type[BaseException], ...] = (RemoteFileNotFound,),
) -> T:
    """
    Call ``func`` until it succeeds, waiting ``backoff * 2**n`` seconds between attempts.

    Parameters
    ----------
    func : Callable[[], T]
        Function without arguments to call.
    retries : int, optional
        Maximum number of attempts. Default is 3.
    backoff : float, optional
        Base waiting time in seconds. Default is 30.
    logger : logging.Logger, optional
        Logger used to report failed attempts.
    no_retry : Tuple[Type[BaseException], ...], optional
        Exceptions that are raised immediately, without retrying.

    Returns
    -------
    T
        Whatever ``func`` returns.
    """

    for attempt in range(1, retries + 1):
        try:
            return func()
        except no_retry:
            raise
        except Exception as e:
            if attempt == retries:
                raise
            wait = backoff * 2 ** (attempt - 1)
            if logger is not None:
                logger.warning(
                    f"Attempt {attempt}/{retries} failed ({type(e).__name__}: {e}). "
                    f"Retrying in {wait:.0f}s"
                )
            time.sleep(wait)


def make_session(retries: int = 5, backoff: float = 2.0) -> requests.Session:
    """
    Create a requests session that retries connection errors and 429/5xx responses.

    Parameters
    ----------
    retries : int, optional
        Number of retries at the HTTP level. Default is 5.
    backoff : float, optional
        urllib3 backoff factor. Default is 2.

    Returns
    -------
    requests.Session
        Configured session.
    """

    retry = Retry(
        total=retries,
        backoff_factor=backoff,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("HEAD", "GET"),
        raise_on_status=False,
    )
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    return session


def http_get(session: requests.Session, url: str, **kwargs) -> requests.Response:
    """
    GET ``url`` raising RemoteFileNotFound on 404 and HTTPError on other failures.
    """

    kwargs.setdefault("timeout", (30, 300))
    response = session.get(url, **kwargs)
    if response.status_code == 404:
        response.close()
        raise RemoteFileNotFound(url)
    response.raise_for_status()

    return response


def remote_size(session: requests.Session, url: str) -> Optional[int]:
    """
    Size in bytes of a remote file (HEAD request), or None if the server does not say.

    Raises
    ------
    RemoteFileNotFound
        If the server answers 404, or reports an empty file (THREDDS answers
        HEAD requests for missing files with 200 and Content-Length: 0).
    """

    response = session.head(url, allow_redirects=True, timeout=(30, 120))
    if response.status_code == 404:
        raise RemoteFileNotFound(url)
    response.raise_for_status()
    content_length = response.headers.get("Content-Length")
    if content_length is None:
        return None
    if int(content_length) == 0:
        raise RemoteFileNotFound(url)

    return int(content_length)


def http_download(
    session: requests.Session,
    url: str,
    target: str,
    expected_size: Optional[int] = None,
    resume: bool = True,
) -> None:
    """
    Stream ``url`` into ``target``, resuming a partial ``target`` with a Range request.

    Parameters
    ----------
    session : requests.Session
        Session used for the request.
    url : str
        Remote URL.
    target : str
        Local file to write (usually a ``.part`` file).
    expected_size : int, optional
        Expected final size in bytes. If given, the download is checked against it.
    resume : bool, optional
        Whether to continue an existing partial ``target``. Default is True.

    Raises
    ------
    RemoteFileNotFound
        If the server answers 404.
    IOError
        If the final size does not match ``expected_size``.
    """

    offset = os.path.getsize(target) if resume and os.path.exists(target) else 0
    if expected_size is not None:
        if offset == expected_size:
            return
        if offset > expected_size:
            offset = 0

    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with http_get(session, url, headers=headers, stream=True) as response:
        # The server may ignore the Range header and send the full file
        if offset and response.status_code != 206:
            offset = 0
        with open(target, "ab" if offset else "wb") as f:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                f.write(chunk)

    if expected_size is not None:
        size = os.path.getsize(target)
        if size != expected_size:
            raise IOError(
                f"Incomplete download of {url}: {size} of {expected_size} bytes"
            )


def validate_file(path: str) -> None:
    """
    Check that a downloaded file is readable, based on its magic bytes.

    NetCDF/HDF5 files are opened with netCDF4 and GRIB files must end with the
    GRIB end marker. Other formats are only checked to be non-empty.

    Raises
    ------
    IOError
        If the file is empty or cannot be read.
    """

    if os.path.getsize(path) == 0:
        raise IOError(f"Empty file: {path}")

    with open(path, "rb") as f:
        magic = f.read(8)

    if magic.startswith(b"CDF") or magic.startswith(b"\x89HDF"):
        import netCDF4

        try:
            netCDF4.Dataset(path).close()
        except Exception as e:
            raise IOError(f"Corrupt NetCDF file {path}: {e}") from e

    elif magic.startswith(b"GRIB"):
        with open(path, "rb") as f:
            f.seek(-4, os.SEEK_END)
            if f.read(4) != b"7777":
                raise IOError(f"Truncated GRIB file: {path}")
