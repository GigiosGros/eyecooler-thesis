# Toward Neuroprotective Cooling of the Eye: A Scleral-Contact Thermoelectric Device

PhD dissertation of Luigi Mecacci, Department of Biomedical Engineering, University of Illinois Chicago (2026),
with the analysis code and the figures it uses.

| Folder | Contents |
|---|---|
| `thesis/` | The dissertation (PDF) |
| `figures/` | The figures of the dissertation made by the author (figures reproduced from other publications are not included) |
| `code/ch2-exvivo-analysis/` | Chapter 2: thermistor processing, quality control, endpoints, uncertainty and statistics of the ex-vivo cooling experiments on human and porcine eyes (`run_analysis.py`; tests in `tests/`) |
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

The raw recordings (ex-vivo thermistor files, in-vivo session folders) and the COMSOL model files are not in
this repository; they are available from the author on request. Paths to them are set at the top of each
script or through environment variables named there.
