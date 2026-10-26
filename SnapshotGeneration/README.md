# Snapshot generation (OpenMC)

This folder is **optional**. The main notebooks use precomputed flux snapshots in `Data/` at the repo root. You only need this environment if you want to regenerate Monte Carlo data with OpenMC.

## Setup

Create and activate a dedicated conda environment (OpenMC is best installed via conda-forge):

```bash
conda env create -f environment.yml
conda activate sdn4neutronics-openmc
```

## Cross-section data

OpenMC requires nuclear cross-section libraries. After activating the environment, download data and set `OPENMC_CROSS_SECTIONS`, for example:

```bash
openmc-data-download -d openmc_data -i nds
export OPENMC_CROSS_SECTIONS=$PWD/openmc_data/cross_sections.xml
```

See the [OpenMC documentation](https://docs.openmc.org/en/stable/usersguide/data.html) for other data sources and install options.

## Running the script

From this directory, with the conda environment active and cross sections configured:

```bash
python openmc_data_generation.py
```

The core simulation logic lives in `triga_fct_total()`. **Note:** the driver loop at the bottom of the script currently renames existing OpenMC output files (`statepoint.*.h5`, etc.) and does **not** call `triga_fct_total()`. To actually run new simulations, restore or uncomment the call to `triga_fct_total()` inside that loop with your desired rod positions and batch settings.

## Output

Simulations write OpenMC statepoint and tally files in this directory. Converting those outputs into the `.npz` / `.vtk` files used by the notebooks is a separate post-processing step (not included in this repo).
