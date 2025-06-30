# CloudBench Dataset

This document outlines the structure and access details for the raw and
post-processed output from 10,000 Large-Eddy Simulations (LES) driven by a
General Circulation Model (GCM), as part of the CloudBench dataset.

## Data Access

The raw simulation output is stored in the `zarr` format and is publicly
available through Google Cloud Storage. Post-processed statistics for all
simulations are consolidated into a single netCDF file.

  * **Raw Simulation Output:** `https://storage.googleapis.com/cloudbench-simulation-output`
  * **Post-processed Statistics:** `https://storage.googleapis.com/cloudbench-statistics`

-----

## Directory Structure

The raw output from each LES is organized into its own directory within the
Google Cloud Storage bucket. The directory path follows this structure:

```
[SITE_ID]/[MONTH]/[EXPERIMENT]/
```

For example:

```
cloudbench-simulation-output/
├── 0/
│   ├── 1/
│   │   ├── amip/
│   │   │   ├── parameters.json
│   │   │   ├── sounding.csv
│   │   │   └── data.zarr
│   │   └── amip-p4k/
│   │       └── ...
│   └── 4/
│       └── ...
└── 1/
    └── ...
```

### Path Components

  * **`SITE_ID`**: A unique integer from `0` to `499` that identifies a specific geolocation.
  * **`MONTH`**: An integer from `1` to `12` indicating the simulated month.
  * **`EXPERIMENT`**: A string identifying the GCM experiment configuration used
  to drive the LES. Possible values are:
      * `amip`
      * `amip-p4k`
      * `amip-4xco2`
      * `amip-p4k-2xco2`
      * `amip-p4k-4xco2`

-----

## File Descriptions

Each experiment directory contains the following files:

  * `parameters.json`: Contains pertinent metadata and a summary of the
  parameters used in the simulation.
  * `data.zarr`: Contains the raw simulation output. This data is partitioned
  (chunked in xarray) by time and vertical domain to allow for efficient loading
  and processing.
  * `sounding.csv`: Contains the atmospheric soundings extracted from the GCM
  site, which are used to initialize the simulation and compute large-scale
  forcings.

-----

## Data Variables

The following is an exhaustive table of all variables contained in both the raw
(`data.zarr`) and post-processed (`netCDF`) data sources.

| Variable                        | Dimensions   | Description                                                                                                                                                                        | Units        |
| :-----------------------------: | :----------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :----------: |
| T                               | (t, x, y, z) | Air temperature.                                                                                                                                                                   | K            |
| asr                             | (t)          | Absorbed shortwave radiation. The net shortwave radiation absorbed (incoming - outgoing) at the top of atmosphere (TOA).                                                           | W m<sup>-2</sup>        |
| cloud\_cover                    | (t)          | The fraction of grid columns that have at least one cloudy pixel (i.e. q\_c \> 1e-6).                                                                                              | unitless     |
| cloud\_fraction                 | (t, z)       | The fraction of grid cells, per level, that have at least one cloudy pixel (i.e. q\_c \> 1e-6.)                                                                                    | unitless     |
| cre\_lw                         | (t)          | Longwave cloud radiative effect. The difference between clear sky and cloudy sky longwave outgoing radiation at TOA. Positive values suggest a net warming effect due to clouds.   | W m<sup>-2</sup>        |
| cre\_sw                         | (t)          | Shortwave cloud radiative effect. The difference between cloudy sky and clear sky absorbed shortwave radiation at TOA. Positive values suggest a net warming effect due to clouds. | W m<sup>-2</sup>        |
| extended\_rad\_flux\_lw         | (t, x, y, z) | Longwave radiative fluxes in the extended grid above the LES domain.                                                                                                               | W m<sup>-2</sup>        |
| extended\_rad\_flux\_lw\_clear  | (t, x, y, z) | Clear-sky longwave radiative fluxes in the extended grid above the LES domain, used for computing the cloud radiative effect (CRE).                                                | W m<sup>-2</sup>        |
| extended\_rad\_flux\_sw         | (t, x, y, z) | Shortwave radiative fluxes in the extended grid above the LES domain.                                                                                                              | W m<sup>-2</sup>        |
| extended\_rad\_flux\_sw\_clear  | (t, x, y, z) | Clear-sky shortwave radiative fluxes in the extended grid above the LES domain, used for computing the cloud radiative effect (CRE).                                               | W m<sup>-2</sup>        |
| iwp                             | (t)          | ice water path                                                                                                                                                                     | kg m<sup>-2</sup>       |
| lwp                             | (t)          | liquid water path                                                                                                                                                                  | kg m<sup>-2</sup>       |
| olr                             | (t)          | outgoing longwave radiation at the TOA.                                                                                                                                            | W m<sup>-2</sup>        |
| p                               | (t, x, y, z) | Hydrodynamic pressure                                                                                                                                                              | Pa           |
| p\_ref                          | (t, x, y, z) | Pressure field                                                                                                                                                                     | Pa           |
| q\_c                            | (t, x, y, z) | Condensed phase specific humidity                                                                                                                                                  | kg kg<sup>-1</sup>      |
| q\_r                            | (t, x, y, z) | Rain droplet mass fraction.                                                                                                                                                        | kg kg<sup>-1</sup>      |
| q\_s                            | (t, x, y, z) | Snow mass fraction.                                                                                                                                                                | kg kg<sup>-1</sup>      |
| q\_t                            | (t, x, y, z) | Total water specific humidity.                                                                                                                                                     | kg kg<sup>-1</sup>      |
| q\_t\_diffusive\_flux\_z        | (t, x, y, z) | Vertical diffusive flux of total specific humidity                                                                                                                                 | kg m<sup>-2</sup> s-1   |
| q\_t\_les\_tendency             | (t, x, y, z) | The tendency of total specific humidity due to LES dynamics (convection + diffusion).                                                                                              | kg kg<sup>-1</sup> s-1  |
| q\_t\_microphysics\_source      | (t, x, y, z) | Total water specific humidity tendency from microphysical sources (autoconversion, accretion, evaporation)                                                                         | kg kg<sup>-1</sup> s<sup>-1</sup>  |
| rad\_flux\_lw                   | (t, x, y, z) | The net (upwelling - downwelling) longwave radiative fluxes.                                                                                                                       | W m<sup>-2</sup>        |
| rad\_flux\_lw\_clear            | (t, x, y, z) | Clear-sky longwave radiative fluxes. This is used to compute the longwave cloud radiative effect.                                                                                  | W m<sup>-2</sup>        |
| rad\_flux\_lw\_up               | (t, x, y, z) | The upwelling component of the longwave radiative flux. This is used to decompose the net fluxes into upwelling and downwelling components.                                        | W m<sup>-2</sup>        |
| rad\_flux\_sw                   | (t, x, y, z) | The net (upwelling - downwelling) shortwave fluxes at the grid cell interfaces.                                                                                                    | W m<sup>-2</sup>        |
| rad\_flux\_sw\_clear            | (t, x, y, z) | Clear-sky shortwave radiative fluxes. This is used to compute the shortwave cloud radiative effect (CRE).                                                                          | W m<sup>-2</sup>        |
| rad\_heat\_src                  | (t, x, y, z) | The radiative heating rate.                                                                                                                                                        | K s<sup>-1</sup>        |
| rho                             | (t, x, y, z) | The air density.                                                                                                                                                                   | kg m<sup>-3</sup>       |
| sfc\_flux\_rad\_lw              | (t)          | The surface upwelling longwave radiative flux.                                                                                                                                     | W m<sup>-2</sup>        |
| sfc\_flux\_rad\_sw              | (t)          | The surface upwelling shortwave radiative flux.                                                                                                                                    | W m<sup>-2</sup>        |
| sfc\_heat\_flux\_latent         | (t)          | The surface latent heat flux.                                                                                                                                                      | W m<sup>-2</sup>        |
| sfc\_heat\_flux\_sensible       | (t)          | The surface sensible heat flux.                                                                                                                                                    | W m<sup>-2</sup>        |
| theta\_li                       | (t, x, y, z) | The liquid-ice potential temperature.                                                                                                                                              | K            |
| theta\_li\_diffusive\_flux\_z   | (t, x, y, z) | The vertical diffusive flux of liquid-ice potential temperature.                                                                                                                   | K kg m<sup>-2</sup> s<sup>-1</sup> |
| theta\_li\_les\_tendency        | (t, x, y, z) | The tendency of liquid-ice potential temperature due to LES dynamics (convection + diffusion).                                                                                     | K s<sup>-1</sup>        |
| theta\_li\_microphysics\_source | (t, x, y, z) | The tendency of liquid-ice potential temperature due to microphysical sources (autoconversion, accretion, evaporation).                                                            | K s<sup>-1</sup>        |
| u                               | (t, x, y, z) | The zonal velocity.                                                                                                                                                                | m s<sup>-1</sup>        |
| v                               | (t, x, y, z) | The meridional velocity.                                                                                                                                                           | m s<sup>-1</sup>        |
| w                               | (t, x, y, z) | The vertical velocity.                                                                                                                                                             | m s<sup>-1</sup>        |

