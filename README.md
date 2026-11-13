# Toward Neuroprotective Cooling of the Eye: A Scleral-Contact Thermoelectric Device

PhD dissertation of Luigi Mecacci, Department of Biomedical Engineering, University of Illinois Chicago (2026),
with the analysis code and the figures it uses.

| Folder | Contents |
|---|---|
| `thesis/` | The dissertation (PDF) |
| `figures/` | The figures of the dissertation made by the author (figures reproduced from other publications are not included) |
| `code/ch2-exvivo-analysis/` | Chapter 2: thermistor processing, quality control, endpoints, uncertainty and statistics of the ex-vivo cooling experiments on human and porcine eyes (`run_analysis.py`; tests in `tests/`) |
| `code/ch2-ct-segmentation/` | Chapter 2: segmentation of the micro-CT scans of the eyes and of the pig MRI, globe outline and axis, and the position of each thermistor in the eye (`run.py`; scans listed in `config.json`) |
| `code/ch2-bath-vs-perfusion/` | Chapter 2: conduction estimate of the heated oil bath used in place of perfusion |
| `code/ch3-model-analysis/` | Chapter 3: analysis of the COMSOL model exports (`run_analysis.py`; tests in `tests/`) |
| `code/ch3-heat-replica/` | Chapter 3: finite-element heat-transfer program that re-solves the COMSOL models from the model files (`replica/`, `run_all.py`, `run_waterblock.py`, and `dev/new_runs.py` for the corrected and extended cases) |
| `code/ch4-live-pigs/` | Chapter 4: conversion of the ring thermistor recordings, ERG extraction and quality control, and the cohort ERG statistics (`erg_cohort.py`) |
| `code/figures/` | Scripts that draw the figures in `figures/` |

## Requirements

Python 3.11 or newer (the thesis numbers were produced with Python 3.13). Each analysis folder lists its
packages in `requirements.txt` where versions matter; install them with

    pip install -r code/ch2-exvivo-analysis/requirements.txt

The finite-element program also needs `pypardiso` (and `mmgpy` for mesh adaptation, `gmsh` for meshes made
from STEP files).

## Data

This repository holds the code, the dissertation and its figures, and the instructions to run them. **It holds no
data.** The data are available from the author on request: please open an issue on this repository, saying what you
need and what for.

Two packages can be provided:

- **Processed data** (about 2.6 GB, most of it the micro-CT segmentation outputs). The inputs each script reads below the raw-data level: the per-eye micro-CT
  measurements (`code/ch2-exvivo-analysis/config/vitreous_depth_ct.csv`, `vitreous_position_ct.csv`), the reference
  results the tests compare against (`code/*/tests/reference/`), the stored inputs of every figure and table
  (`figures/<chapter>/sources/`), and the results of the ex-vivo analysis. The package unpacks over this repository,
  so that every figure, table and number of the dissertation can be regenerated with the code here.
- **Raw data**. The ex-vivo thermistor recordings, the micro-CT scans, the in-vivo session folders (temperature and
  ERG recordings, photographs) and the COMSOL model files. Paths to them are set at the top of each script or through
  environment variables named there.

Without the processed data the tests in `code/*/tests/` and the figure scripts cannot run: they stop at the first
input file they cannot find.

## Licence

The code in `code/` is released under the MIT licence (`LICENSE`). The dissertation and the figures are
copyright Luigi Mecacci, 2026; please cite the dissertation when using them.
