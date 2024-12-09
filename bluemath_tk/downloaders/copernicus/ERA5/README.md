# Reanalysis ERA5

For updated documentation please go [here](https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation).

ERA5 is the fifth generation ECMWF atmospheric reanalysis of the global climate covering the period from January 1940 to present. ERA5 is produced by the Copernicus Climate Change Service (C3S) at ECMWF.

ERA5 provides hourly estimates for a large number of atmospheric, ocean-wave and land-surface quantities. An uncertainty estimate is sampled by an underlying 10-member ensemble at three-hourly intervals.

**There is a script scheduled to run periodically to download the latest month's data.**

## Dataset Characteristics

The Reanalysis ERA5 single levels dataset has the following characteristics:

- Data type: Gridded
- Projection: Regular latitude-longitude grid
- Horizontal coverage: Global
- Horizontal resolution:
  - Reanalysis: 0.25° x 0.25° (atmosphere), 0.5° x 0.5° (ocean waves)
  - Mean, spread and members: 0.5° x 0.5° (atmosphere), 1° x 1° (ocean waves)
- Temporal coverage: 1940 to present
- Temporal resolution: Hourly
- File format: NetCDF
- Update frequency: Daily

* geopotencial at 500Hpa is obtained from Reanalysis ERA5 pressure levels: https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels?tab=overview

## Downloaded Dataset Variables

These are the variables currently available for download. If a new variable is required, it MUST be added to the `ERA5_config.json` file, and then to the code or crontab downloading the files.

| name    | long_name                                           | units   | type  |
| ------- | --------------------------------------------------- | ------- | ----- |
| swh     | Significant height of combined wind waves and swell | m       | ocean |
| pp1d    | Peak wave period                                    | s       | ocean |
| mwp     | Mean wave period                                    | s       | ocean |
| mwd     | Mean wave direction                                 | degrees | ocean |
| shww    | Significant height of wind waves                    | m       | ocean |
| mpww    | Mean period of wind waves                           | s       | ocean |
| mdww    | Mean direction of wind waves                        | degrees | ocean |
| dwww    | Wave spectral directional width for wind waves      | degrees | ocean |
| p140121 | Significant wave height of first swell partition    | m       | ocean |
| p140124 | Significant wave height of second swell partition   | m       | ocean |
| p140127 | Significant wave height of third swell partition    | m       | ocean |
| p140123 | Mean wave period of first swell partition           | s       | ocean |
| p140126 | Mean wave period of second swell partition          | s       | ocean |
| p140129 | Mean wave period of third swell partition           | s       | ocean |
| p140122 | Mean wave direction of first swell partition        | degrees | ocean |
| p140125 | Mean wave direction of second swell partition       | degrees | ocean |
| p140128 | Mean wave direction of third swell partition        | degrees | ocean |

These variables represent different atmospheric, ocean-wave, and land-surface quantities that are provided by the ERA5 dataset. Each variable has a corresponding path that specifies where the data is located within the dataset.

For more information and to access the dataset, click [here](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview).

## Scripts used to download data

To download the Reanalysis ERA5 single levels dataset, you can use the following scripts:

1. Python script: `ERA5_download.py`

This script downloads ERA5 reanalysis data for multiple variables and saves them as NetCDF files.

```
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
```

2. Bash scripts: `bash_ERA5_download.sh and qsub_ERA5_download.sh` in [gitlab](https://gitlab.com/geoocean/datahub/datahub-scripts/-/tree/main/ERA5/launchers?ref_type=heads).

This script creates and submits jobs to download several datasets simultaneously.

```Exception: Requests using the API temporally limited to 20 to restrict the activity of abusive users. Please visit copernicus-support.ecmwf.int for further information.```

For more information on how to use the scripts and download the Reanalysis ERA5 single levels dataset, please refer to the [documentation](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview).

## Issues with data download limits

To ensure the protection of CDS resources, limits have been set for data downloads. You can find more information about these limits [here](https://cds.climate.copernicus.eu/live/limits).

To adhere to these limits, we have developed the `download_ERA5.sh` script and configured the cron of the user responsible for data downloads to initiate periodic downloads. To prevent user-specific limits, we have set up cron to periodically update the credentials of the user performing the data downloads.

```
# VA TODO 2 HORAS ANTES. Si quieremos ejecutar a las 10, tenemos que poner las 8.

0 0 * * * /usr/bin/cp /home/grupos/geocean/valvanuz/.cds/laura /home/grupos/geocean/valvanuz/.cdsapirc
0 6 * * * /usr/bin/cp /home/grupos/geocean/valvanuz/.cds/gmail1 /home/grupos/geocean/valvanuz/.cdsapirc
0 12 * * * /usr/bin/cp /home/grupos/geocean/valvanuz/.cds/valva /home/grupos/geocean/valvanuz/.cdsapirc
0 18 * * * /usr/bin/cp /home/grupos/geocean/valvanuz/.cds/isra /home/grupos/geocean/valvanuz/.cdsapirc
30 10 21 * * /usr/bin/bash -l /home/grupos/geocean/valvanuz/data_lustre/datahub-scripts/ERA5/download_ERA5.sh 1941 1941
0 12 21 * * /usr/bin/bash -l /home/grupos/geocean/valvanuz/data_lustre/datahub-scripts/ERA5/download_ERA5.sh 1940 1945
0 18 21 * * /usr/bin/bash -l /home/grupos/geocean/valvanuz/data_lustre/datahub-scripts/ERA5/download_ERA5.sh 1946 1950
```