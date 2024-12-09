#!/usr/bin/env python

"""
Usage: era5_download.py YEARS MONTHS [--var=<var>] [--base-path=<base-path>] [--token=<token>] [--debug] [--check] [--force]
era5_download.py (-h | --help | --version)
era5_download.py --list-vars
era5_download.py --list-datasets
era5_download.py --show-info

This script downloads ERA5 reanalysis data for multiple variables and saves them as NetCDF files.

Arguments:
  YEARS                     Years of the data to download, separated by commas
                              To download a single year, use the format YYYY
                              To download a range of years, use the format YYYY-YYYY
  MONTHS                    Months of the data to download, separated by commas
                              To download a single month, use the format MM
                              To download a range of months, use the format MM-MM

Options:
  --var=<var>               List of variable values separated by commas.
  --base-path=<base-path>   Base path to save the data [default: /lustre/geocean/DATA/Copernicus/].
  --token=<token>           Token for the Climate Data Store API.
  --debug                   Debug mode (it is recommended to use this mode).
  --check                   If just file checking is required.
  --force                   Overwrite the data in case it exists.
  --list-vars               Return a list with the available variables.
  --list-datasets           Return a list with the available datasets.
  --show-info               Show information about the variables.

Examples:
    era5_download.py 2014,2016 1,10 --var tp,sst --debug
    era5_download.py 2014,2016 1,10 --var tp,sst --debug --check
    era5_download.py --list-vars
    era5_download.py --list-datasets
    era5_download.py --show-info
"""

from docopt import docopt
from bluemath_tk.downloaders.copernicus.copernicus_downloader import (
    CopernicusDownloader,
)

# Parse the command line arguments
args = docopt(__doc__)
print(args)

# Create the CopernicusDownloader object
copernicus_downloader = CopernicusDownloader(
    product="ERA5",
    token=args["--token"],
    base_path_to_download=args["--base-path"] or "/lustre/geocean/DATA/Copernicus/",
    debug=args["--debug"],
    check=args["--check"],
)

if args["--list-vars"]:
    print(copernicus_downloader.list_variables())
    exit()
if args["--list-datasets"]:
    print(copernicus_downloader.list_datasets())
    exit()
if args["--show-info"]:
    print(copernicus_downloader.show_markdown_table())
    exit()

years = (
    args["YEARS"].split(",")
    if "-" not in args["YEARS"]
    else list(
        range(int(args["YEARS"].split("-")[0]), int(args["YEARS"].split("-")[1]) + 1)
    )
)
months = (
    args["MONTHS"].split(",")
    if "-" not in args["MONTHS"]
    else list(
        range(int(args["MONTHS"].split("-")[0]), int(args["MONTHS"].split("-")[1]) + 1)
    )
)
# Ocean variables are used by default, as right now, DestinE program is used for atmosphere variables
variables = (
    args["--var"].split(",")
    if args["--var"]
    else copernicus_downloader.list_variables(type="ocean")
)

output = copernicus_downloader.download_data(
    variables=variables,
    years=years,
    months=months,
    force=args["--force"],
)

print(output)
