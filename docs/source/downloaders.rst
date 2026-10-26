Data downloaders
================

``bluemath_tk.downloaders`` downloads climate and ocean data from several
providers with a common interface: every downloader can do a dry run first,
skips files that are already complete, retries failed transfers, and reports
what happened in a :class:`~bluemath_tk.downloaders.DownloadResult`.

Providers and products
----------------------

.. list-table::
   :header-rows: 1
   :widths: 15 25 15 45

   * - Provider
     - Class
     - Products
     - Notes
   * - ``csiro``
     - ``CSIRODownloader``
     - WHACS, CAWCR
     - WAVEWATCH III hindcasts from the CSIRO THREDDS server. Full monthly files
       (resumable, parallel) or time series at given points (OPeNDAP).
   * - ``copernicus``
     - ``CopernicusDownloader``
     - ERA5, CERRA
     - Copernicus Climate Data Store (``cdsapi``). Parallel requests.
   * - ``earthdata``
     - ``EarthdataDownloader``
     - SWOT
     - NASA Earthdata collections (PO.DAAC, ...), found with the CMR search API
       and downloaded over HTTPS with an Earthdata login (``~/.netrc``).
       Parallel, resumable, checked against the CMR size and checksum.
   * - ``imos``
     - ``IMOSDownloader``
     - SRS
     - IMOS data in the public AODN bucket on AWS S3 (no account), e.g. the
       multi-mission altimeter wave/wind archive. Mirrors a bucket prefix,
       re-downloads files that changed, checked against the S3 ETag (MD5).
   * - ``noaa``
     - ``NOAADownloader``
     - NDBC
     - NDBC buoy bulk parameters, wave spectra and directional spectra.
   * - ``ecmwf``
     - ``ECMWFDownloader``
     - OpenData
     - ECMWF IFS/AIFS real-time forecasts (``ecmwf-opendata``), from the ECMWF
       portal or its AWS, Azure and Google mirrors.

The Copernicus and ECMWF downloaders need the optional dependencies:

.. code-block:: bash

   pip install bluemath-tk[downloaders]

Package layout
--------------

Each provider is a subpackage with a single downloader module and one folder per
product, which only holds the product configuration (servers, datasets,
variables, file names):

.. code-block:: text

   bluemath_tk/downloaders/
   ├── _base_downloaders.py      # BaseDownloader: routing, fetch_file, results
   ├── _download_result.py       # DownloadResult
   ├── _transfer.py              # retries, resumable HTTP, file validation
   └── csiro/
       ├── __init__.py           # exports CSIRODownloader
       ├── csiro_downloader.py
       ├── CAWCR/CAWCR_config.json
       └── WHACS/WHACS_config.json

Downloaders are imported from their provider subpackage:

.. code-block:: python

   from bluemath_tk.downloaders.csiro import CSIRODownloader
   from bluemath_tk.downloaders.copernicus import CopernicusDownloader

Common behaviour
----------------

All downloaders share these rules:

``dry_run``
   Defaults to ``True``: nothing is downloaded and the result lists what would
   be. Pass ``dry_run=False`` to download.

Complete files only
   Each file is written to ``<file>.part``, checked (NetCDF files must open,
   GRIB files must end with the GRIB end marker, HTTP transfers must match the
   remote size) and then renamed. A file without ``.part`` is always complete,
   so an interrupted run never leaves a file that looks finished.

Re-runs are cheap
   Existing files are skipped (``force=True`` downloads them again). CSIRO files
   are also compared with the remote size (``verify_existing=True``), and
   interrupted CSIRO transfers resume where they stopped.

Retries
   Each file is tried ``retries`` times (default 3), waiting ``retry_backoff``
   seconds (default 30, doubled each time). Files that are not published yet
   are reported as skipped, not as errors, so the same call can be repeated to
   keep a dataset up to date.

Results
   Every call returns a :class:`~bluemath_tk.downloaders.DownloadResult`:

   .. code-block:: python

      print(result.summary())      # counts, skip reasons, size to download, errors
      result.downloaded_files      # list of paths
      result.has_errors
      result.save("run.json")      # keep a log of the run

CSIRO: WHACS and CAWCR
----------------------

Full monthly files, one per variable for WHACS:

.. code-block:: python

   from bluemath_tk.downloaders.csiro import CSIRODownloader

   downloader = CSIRODownloader(
       product="WHACS",
       base_path_to_download="/lustre/geocean/DATA/CSIRO",
       max_workers=4,                     # parallel downloads
   )
   downloader.list_datasets()             # ['global-smc']
   downloader.list_variables("global-smc")

   result = downloader.download_data(
       dataset="global-smc",
       variables=["phs0", "ptp0", "pdir0"],
       start="2020-01",
       end="2023-12",
       dry_run=True,
   )
   print(result.summary())                # includes the total size to download

CAWCR has two datasets, ``gridded`` (all variables in one file per month) and
``spec`` (spectra at output stations). Their station list and variable names
changed in June 2013; the downloader stores both periods in separate folders
and renames the old variables to the new ones (``Efth`` → ``efth``, ``depth`` →
``dpt``, ...).

Time series at one or more points are extracted through OPeNDAP, without
downloading the full files. One file is written per month and, by default, all
months are combined into one file:

.. code-block:: python

   downloader = CSIRODownloader(product="CAWCR", base_path_to_download="./data")
   result = downloader.download_point(
       dataset="spec",
       lon=-4.0,
       lat=44.0,
       start="2013-01",
       end="2013-12",
       dry_run=False,
   )
   result.metadata["combined_file"]

The nearest model point is used; its distance is stored in the file attributes
(``nearest_point_distance_km``) and a warning is logged above 25 km. OPeNDAP
extraction is slow for long periods (the server reads whole files), so for many
years it is faster to download the files and read them locally.

Copernicus: ERA5 and CERRA
--------------------------

The CDS key is read from ``~/.cdsapirc`` (or the ``CDSAPI_KEY`` environment
variable) unless ``api_key`` is given.

.. code-block:: python

   from bluemath_tk.downloaders.copernicus import CopernicusDownloader

   downloader = CopernicusDownloader(
       product="CERRA",
       base_path_to_download="/lustre/geocean/DATA/Copernicus",
       max_workers=4,                     # CDS requests in parallel
   )
   result = downloader.download_data(
       variables=["2m_temperature", "mean_sea_level_pressure"],
       years=range(2020, 2026),
       months=range(1, 13),
       file_period="month",
       dry_run=True,
   )

Files are saved to
``<base>/<product>/<dataset>/<type>/<product_type>/<variable>/``. The
``file_period`` argument decides how requests are split:

- ``"month"``: one file per variable and month, ``<name>_<YYYY>_<MM>.nc``.
- ``"year"``: one file per variable and year, ``<name>_<YYYY>.nc``.
- ``"auto"`` (default): yearly files when all 12 months are requested,
  monthly files otherwise.

Yearly files with the previous naming (``<name>_<YYYY>_01_02_..._12.nc``) are
recognised and not downloaded again.

NASA Earthdata: SWOT
--------------------

Each dataset is one NASA CMR collection. The search is public; downloads need
a free Earthdata account in ``~/.netrc`` (``chmod 600``)::

   machine urs.earthdata.nasa.gov login <user> password <password>

.. code-block:: python

   from bluemath_tk.downloaders.earthdata import EarthdataDownloader

   downloader = EarthdataDownloader(
       product="SWOT",
       base_path_to_download="/lustre/geocean/DATA/NASA",
       max_workers=4,                     # parallel downloads
   )
   downloader.list_datasets()             # ['L2_LR_SSH_WINDWAVE_D']

   # Table of files (URL, size, checksum, start/end time), sorted by time
   granules = downloader.search_granules(
       "L2_LR_SSH_WINDWAVE_D",
       start="2024-06",                   # strings are read at their resolution:
       end="2024-06",                     # this is the whole of June 2024
       bbox=(2.5, 51.0, 7.5, 54.0),       # west, south, east, north
   )

   result = downloader.download_data(
       "L2_LR_SSH_WINDWAVE_D",
       start="2024-06",
       end="2024-06",
       bbox=(2.5, 51.0, 7.5, 54.0),
       dry_run=True,
   )
   print(result.summary())                # includes the total size to download

Files are saved to ``<base>/SWOT/L2_LR_SSH_WINDWAVE_D/<YYYY-MM>/``, the month
of the granule start time. ``updated_since="2026-10-01"`` only lists granules
added or changed in CMR since then, which keeps a local copy up to date
cheaply. ``download_granules(dataset, granules)`` downloads a table returned by
``search_granules``, for example after filtering it.

The downloader mirrors what CMR lists. For SWOT this means that a pass can be
there several times (product counter ``_01``, ``_02``, ... and processing
versions such as ``PGD0`` reprocessed / ``PID0`` forward processing); choosing
one version per pass is left to whoever reads the files. Files that do not
match the dataset ``filename_pattern`` (a few granules of other SWOT products
are listed in the WindWave collection) are left out.

To add another NASA collection, add a dataset with its CMR ``short_name`` to a
product configuration (or a new ``<PRODUCT>/<PRODUCT>_config.json``).

IMOS: altimeter wave and wind archive
-------------------------------------

The IMOS multi-mission altimeter archive (Ribal and Young 2019, updated by
IMOS) has one file per mission and 1x1 degree cell, 1985 to present, in the
public AODN bucket. No account is needed.

.. code-block:: python

   from bluemath_tk.downloaders.imos import IMOSDownloader

   downloader = IMOSDownloader(
       product="SRS",
       base_path_to_download="/lustre/geocean/DATA/IMOS",
       max_workers=32,                    # files are small (~300 kB): many workers
   )
   downloader.list_subfolders("Wave-Wind-Altimetry-DM00")   # the missions

   result = downloader.download_data(
       "Wave-Wind-Altimetry-DM00",
       subfolders=["JASON-3", "SWOT"],    # default: all missions
       dry_run=True,
   )
   print(result.summary())

Files mirror the bucket below the dataset prefix:
``<base>/SRS/Surface-Waves/Wave-Wind-Altimetry-DM00/<MISSION>/<20x20 folder>/<file>``.
IMOS rewrites whole files at each release, so a file is downloaded again when
its size changes or the bucket copy is newer than the local one (downloaded
files get the bucket modification time). ``modified_since`` only lists files
changed since a date; ``list_files`` returns the table of files (path, URL,
size, ETag, modification time) and ``download_files`` downloads such a table.

NOAA and ECMWF
--------------

.. code-block:: python

   from bluemath_tk.downloaders.noaa import NOAADownloader

   downloader = NOAADownloader(product="NDBC", base_path_to_download="./data")
   downloader.list_data_types()
   result = downloader.download_data(
       data_type="wave_spectra", buoy_id="41001", years=[2022, 2023], dry_run=False
   )

.. code-block:: python

   from bluemath_tk.downloaders.ecmwf import ECMWFDownloader

   downloader = ECMWFDownloader(
       product="OpenData",
       base_path_to_download="./ecmwf",
       source="aws",              # mirror; "ecmwf" is rate limited
   )
   result = downloader.download_data(
       dataset="forecast_data", param=["msl"], step=[0, 24], type="fc", dry_run=False
   )

ECMWF files are saved per forecast run,
``<base>/OpenData/forecast_data/<model>/<resolution>/<YYYYMMDDHH>/``. Without
``date`` the latest run is used; ``date=-1, time=12`` selects yesterday's 12 UTC
run.

Adding a product or a provider
------------------------------

A new product of an existing provider only needs a configuration file:

1. Create ``<provider>/<PRODUCT>/<PRODUCT>_config.json``, following the files of
   the other products of that provider.
2. Add it to the ``products_configs`` dictionary of the downloader class.
3. If the product needs its own logic, add a ``download_data_<product>``
   method (snake case: ``OpenData`` → ``download_data_open_data``);
   ``download_data`` routes to it.

A new provider is a new subpackage with the same layout. Its class inherits
from :class:`~bluemath_tk.downloaders.BaseDownloader`, declares
``products_configs``, and writes every file through ``fetch_file``, which
handles skipping, dry runs, ``.part`` files, validation and retries:

.. code-block:: python

   def download_data_myproduct(self, ..., force=False, dry_run=True):
       result = self.create_download_result()
       for url, output_file in files:
           def fetch(part_file, url=url):
               http_download(self.session, url, part_file)
           self.fetch_file(output_file, fetch, result, force=force, dry_run=dry_run)
       return self.finalize_download_result(result)

The ``tests/downloaders/test_structure.py`` test checks that every provider
follows this layout and that ``products_configs`` matches the product folders.

Automated downloads at GeoOcean
-------------------------------

On the GeoOcean cluster, download jobs for ``/lustre/geocean/DATA`` live in
``/lustre/geocean/DATA/DownloadMe``: one small script and Slurm job per dataset,
which only says *what* to download, plus their logs. See its ``README.md``.
