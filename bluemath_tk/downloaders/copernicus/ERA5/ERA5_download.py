#!/usr/bin/env python

"""
Usage: era5_download.py YEARS MONTHS [--var=<var>] [--debug] [--check] [--force]
era5_download.py (-h | --help | --version)
era5_download.py --list-vars
era5_download.py --list-datasets

This script downloads ERA5 reanalysis data for multiple variables and saves them as NetCDF files.

Arguments:
  YEARS                     Years of the data to download, separated by commas
                              To download a single year, use the format YYYY
  MONTHS                    Months of the data to download, separated by commas
                              To download a single month, use the format MM

Options:
  --debug                   Debug mode
  --var=<var>               List of variable values separated by commas.
  --check                   Check the data of the month.
  --force                   Overwrite the data of the month in case it exists.
  --list-vars               Return a list with the available variables.
  --list-datasets           Return a list with the available datasets.

Examples:
    era5_download.py 2014,2016 1 --debug
    era5_download.py 2014,2016 1 --var tp,sst
    era5_download.py --list-vars
    era5_download.py --list-datasets
    era5_download.py 2014,2016 1 --check
"""

from docopt import docopt
from bluemath_tk.downloaders.copernicus.copernicus_downloader import (
    CopernicusDownloader,
)

# Parse the command line arguments
args = docopt(__doc__)

# Create the CopernicusDownloader object
copernicus_downloader = CopernicusDownloader(
    product="ERA5",
    base_path_to_download="/home/tausiaj/DATA/Copernicus/",
    debug=args["--debug"],
)

if args["--list-vars"]:
    print(copernicus_downloader.list_variables())
    exit()
if args["--list-datasets"]:
    print(copernicus_downloader.list_datasets())
    exit()

years = args["YEARS"].split(",")
months = args["MONTHS"].split(",")
if args["--var"]:
    variables = args["--var"].split(",")
else:
    raise ValueError("The --var argument is mandatory")

print(variables, years, months)

copernicus_downloader.download_data(
    variables=variables,
    years=years,
    months=months,
    force=args["--force"],
)
