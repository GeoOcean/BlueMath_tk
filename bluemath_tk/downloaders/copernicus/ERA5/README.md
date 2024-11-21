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

In the table, below, all paths start in `/lustre/geocean/DATA/Copernicus/ERA5/source/`.

| name    | path                                                            | units         | source                          |
| ------- | --------------------------------------------------------------- | ------------- | ------------------------------- |
| geo500  | pressure/reanalysis/geo500/$year/monthly_files.nc               | m2 s-2        | reanalysis-era5-pressure-levels |
| u10     | atmosphere/reanalysis/u10/$year/monthly_files.nc                | m s-1         | reanalysis-era5-single-levels   |
| v10     | WINDS/ERA5_05/10m_v_component_of_wind                           | wind          | reanalysis-era5-single-levels   |
| sst     | SST/ERA5_05/sea_surface_temperature                             | temperature   | reanalysis-era5-single-levels   |
| tp      | PRECIPITATION/ERA5_05/total_precipitation                       | precipitation | reanalysis-era5-single-levels   |
| msl     | PRESSURE/ERA5_05/mean_sea_level_pressure                        | pressure      | reanalysis-era5-single-levels   |
| wmb     | WAVES/ERA5_05/model_bathymetry                                  | waves         | reanalysis-era5-single-levels   |
| mdww    | WAVES/ERA5_05/mean_direction_of_wind_waves                      | waves         | reanalysis-era5-single-levels   |
| mpww    | WAVES/ERA5_05/mean_period_of_wind_waves                         | waves         | reanalysis-era5-single-levels   |
| mwp     | WAVES/ERA5_05/mean_wave_period                                  | waves         | reanalysis-era5-single-levels   |
| shww    | WAVES/ERA5_05/significant_height_of_wind_waves                  | waves         | reanalysis-era5-single-levels   |
| dwww    | WAVES/ERA5_05/wave_spectral_directional_width_for_wind_waves    | waves         | reanalysis-era5-single-levels   |
| pp1d    | WAVES/ERA5_05/peak_wave_period                                  | waves         | reanalysis-era5-single-levels   |
| swh     | WAVES/ERA5_05/significant_height_of_combined_wind_waves_and_    | waves         | reanalysis-era5-single-levels   |
| p140122 | WAVES/ERA5_05/mean_wave_direction_of_first_swell_partition      | wave          | reanalysis-era5-single-levels   |
| p140125 | WAVES/ERA5_05/mean_wave_direction_of_second_swell_partition     | wave          | reanalysis-era5-single-levels   |
| p140128 | WAVES/ERA5_05/mean_wave_direction_of_third_swell_partition      | wave          | reanalysis-era5-single-levels   |
| p140123 | WAVES/ERA5_05/mean_wave_period_of_first_swell_partition         | wave          | reanalysis-era5-single-levels   |
| p140126 | WAVES/ERA5_05/mean_wave_period_of_second_swell_partition        | wave          | reanalysis-era5-single-levels   |
| p140129 | WAVES/ERA5_05/mean_wave_period_of_third_swell_partition         | wave          | reanalysis-era5-single-levels   |
| p140121 | WAVES/ERA5_05/significant_wave_height_of_first_swell_partition  | wave          | reanalysis-era5-single-levels   |
| p140124 | WAVES/ERA5_05/significant_wave_height_of_second_swell_partition | wave          | reanalysis-era5-single-levels   |
| p140127 | WAVES/ERA5_05/significant_wave_height_of_third_swell_partition  | wave          | reanalysis-era5-single-levels   |

These variables represent different atmospheric, ocean-wave, and land-surface quantities that are provided by the ERA5 dataset. Each variable has a corresponding path that specifies where the data is located within the dataset.

For more information and to access the dataset, click [here](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview).

## Scripts used for download data

To download the Reanalysis ERA5 single levels dataset, you can use the following scripts:

1. Python script: `ERA5_download_month.py`

This script downloads ERA5 reanalysis data for multiple variables and saves them as NetCDF files.

```
Usage: ERA5_download_month.py YEAR MONTH [--var=<var>] [--dbg] [--check] [--force] 
ERA5_download_month.py (-h | --help | --version)
ERA5_download_month.py --list-vars

This script downloads ERA5 reanalysis data for multiple variables and saves them as NetCDF files.

Arguments:
  YEAR                      Year of the data to download   
  MONTH                     Month of the data to download

Options:
  --dbg                     Debug mode
  --var=<var>               List of variable values separated by commas.
  --check                   Check the data of the month.
  --force                   Overwrite the data of the month in case it exists.
  --list-vars               Return a list with the available variables.

Examples:
    ERA5_download_month.py 2014 1 --dbg
    ERA5_download_month.py 2014 1 --var tp,sst
    ERA5_download_month.py --list-vars
    ERA5_download_month.py 2014 1 --check
```

2. Bash scripts: `bash_ERA5_download.sh and qsub_ERA5_download.sh`

This script creates and submits jobs to download several datasets simultaneously.

```Exception: Requests using the API temporally limited to 20 to restrict the activity of abusive users. Please visit copernicus-support.ecmwf.int for further information.```

For more information on how to use the scripts and download the Reanalysis ERA5 single levels dataset, please refer to the [documentation](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview).

## Scripts used for extract data

```Usage: ERA5_extract.py --lat=<lat> --long=<long> --output_file=<output_file> [--nc_pattern=<nc_pattern>] [--var=<var>] [--dbg]
ERA5_extract.py (-h | --help | --version)
ERA5_extract.py --list-vars

This script downloads ERA5 reanalysis data for multiple variables and saves them as NetCDF files.

Options:
    --lat=<lat>               Latitude of the point location or tuple of the bounding box.
    --long=<long>             Longitude of the point location or tuple of the bounding box.
    --dbg                     Debug mode
    --var=<var>               List of variable values separated by commas. If empty, all variables will be extracted.
    --list-vars               Return a list with the available variables.
    --nc_pattern=<nc_pattern>              Pattern to match the netCDF files to be extracted.
    --output_file=<output_file>  Path to save the extracted data as a new netCDF file.

Examples:
    ERA5_extract.py --lat 41,42 --long 2,3 --var tp --nc_pattern "*1979*.nc" --output_file twl_1979.nc --dbg
    ERA5_extract.py --lat 41.05 --long 2.98 --var tp --nc_pattern "*1979*.nc" --output_file twl_1979.nc
    ERA5_extract.py --lat 41,42 --long 2,3 --nc_pattern "*202405.nc" --output_file twl_1979.nc 
    ERA5_extract.py --list-vars
```

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