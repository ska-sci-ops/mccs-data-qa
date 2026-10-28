# mccs-data-qa

* mccs_data_qa/ (`mccs-spider`): Runs on MCCS server, extracts HDF5 file metadata and creates database.
* mccs_data_qa/analyse.py (`mccs-analyse`): Builds the analysis workbook (sweep QA, coverage histograms, calibration sets) from `db/latest.csv`.
* mccs_data_qa/convert_set.py (`mccs-convert-set`): Converts every sweep in a calibration set to UVX, with a `calibration_set.yaml`.
* mccs_data_qa/reports/ (`mccs-report-ui`): Web UI and scripts for QA reports (quicklook, stefcal, delay fit, Sun holography, sensitivity) on calibration sets.
* app.py: Runs on scs server, simple Panel app to view the MCCS observation database.

### Usage

To update the database, run `mccs-spider` on the MCCS server. 

To do so: 
* login to the Juptyerhub at https://k8s.mccs.low.internal.skao.int/jupyterhub/
* open a new terminal session
* cd `/home/jovyan/shared/Danny/mccs-spider-search`
* run `mccs-spider` (after `uv pip install .`; add `--no-skip-existing` to re-spider everything)

This will spider the files and create the database (a CSV file). This file is uploaded to 
acacia via rclone:

```
rclone copy db/latest.csv SKAO:/aa05/mccs-spider-search/db/
```

To build the analysis workbook from the database:

```
mccs-analyse [db/latest.csv] [out.xlsx] [calibration.xlsx]
```

Sweeps are QA'd on file count: 385 (64-448), or the daytime ranges 321 (128-448), 298 (128-425) and 362 (64-425).


### Reports

The reports need the `reports` extra (python 3.11+, pulls ska-ost-low-uv from GitLab):

```
uv pip install '.[reports]'
```

Convert a calibration set, then browse to its parent folder in the report UI:

```
mccs-convert-set 614 --out-dir sweeps
mccs-report-ui --root sweeps    # then open http://127.0.0.1:8080
```

Each script also runs on its own, e.g. `python -m mccs_data_qa.reports.run_stefcal --help`.
