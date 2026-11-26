# mccs-data-qa

* mccs_data_qa/ (`mccs-spider`): Runs on MCCS server, extracts HDF5 file metadata and creates database.
* mccs_data_qa/analyse.py (`mccs-analyse`): Builds the analysis workbook (sweep QA, coverage histograms, calibration sets) from `db/latest.csv`.
* mccs_data_qa/convert_set.py (`mccs-convert-set`): Converts every sweep in a calibration set to UVX, with a `calibration_set.yaml`. `mccs-convert-eb` converts a single EB. Both take `--lossy [STEP]` for quantised (lossy) crosses.
* mccs_data_qa/reports/ (`mccs-report-ui`): Web UI and scripts for QA reports (quicklook, stefcal, delay fit, Sun holography, sensitivity, diagnose) on calibration sets.
* app.py: Runs on scs server, simple Panel app to view the MCCS observation database.

### Usage

To update the database, run `mccs-spider` on the MCCS server. 

To do so: 
* login to the Juptyerhub at https://k8s.mccs.low.internal.skao.int/jupyterhub/
* open a new terminal session
* cd `/home/jovyan/shared/Danny/mccs-spider-search`
* run `mccs-spider --analyse` (after `uv pip install .`; add `--no-skip-existing` to re-spider everything; `--analyse` also rebuilds the workbooks below)

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

The reports need the `reports` extra (python 3.12+, pulls ska-ost-low-uv from GitLab):

```
uv pip install '.[reports]'
```

Convert a calibration set, then browse to its parent folder in the report UI:

```
mccs-convert-set 614 --out-dir sweeps
mccs-report-ui --root sweeps    # then open http://127.0.0.1:8080
```

Or convert from the UI: start it with the database folder (and the DAQ folder, if not `/home/jovyan/daq-data`),

```
mccs-report-ui --root sweeps --db db [--daq-dir /home/jovyan/daq-data]
```

then open **Calibration sets** in the header. It lists every set in `db/calibration_sets.xlsx`, with
per-column filters, a search box, CSV export, the set QA marked in the report UI, and a Convert button per
row. Conversions go into `--root` with the Convert options. **Update database** runs `mccs-spider --analyse`
(spider, then `mccs-analyse`, then the acacia upload, which a switch can turn off) and reloads the table.
These jobs run one at a time, alongside the reports.

A Set ID is the set's UTC date and LST, `yyyymmdd` then LST in 0.1 hr (2026-09-15 at LST 14.4 is
`20260915144`, converted into `set_2026-09-15_144/`), so deleting or adding data never renumbers other
sets. Folders converted under the old sequential IDs (`set12_...`) are renamed once with `python -m mccs_data_qa.migrate_set_ids --sweeps-dir sweeps --dry-run`
(drop `--dry-run` to apply), then rerun `mccs-analyse`.

QA marks record the reviewer's initials, set from the
header chip and kept per browser.

On the MCCS JupyterHub, start the UI from a hub terminal and open it through jupyter-server-proxy,
at `https://k8s.mccs.low.internal.skao.int/jupyterhub/user/<name>/proxy/8080/` (`mccs-report-ui` prints the path). If `pip show jupyter-server-proxy` finds nothing, `pip install --user
jupyter-server-proxy` and restart your server from the Hub Control Panel. The UI stops when the
terminal closes or the server is culled; `nohup mccs-report-ui --root sweeps > report_ui.log 2>&1 &`
keeps it past the terminal.

Each script also runs on its own, e.g. `python -m mccs_data_qa.reports.run_stefcal --help`.

The Diagnose task checks every antenna of a set and reports the ones that look faulty. In daytime data it
phases the station up on the Sun and finds antennas missing from the aperture; with a calibration run it
also finds antennas whose phase differs from their neighbours', the sign of an antenna out of position or
a cable on the wrong input. It compares each baseline's correlation with that of baselines of similar
length to find swapped polarisations and dead inputs, compares bandpasses across the station, follows
power and correlation across integrations, and, with a calibration run, looks for poor or outlying delay
fits. The report lists every failure with its likely cause, shows a pass, warn or fail matrix of all
antennas, and writes the failures to `issues.csv`. `--freq-idx` sets the channel used for the holography,
UV distance and time checks.

The Calibration task can also run the MCCS station calibration solver with the SFT's GSM, SUN and
SUN_FAST settings (`--cal-model mccs-gsm`, `mccs-sun`, `mccs-sunfast`; `--leakage solve` is the
polarised solve). GSM and SUN need the MCCS EEPs, set as for the SFT notebooks with `$EEP_PATH`,
`$EEP_FILE_NAME` and `$EEP_SUFFIX` (default `.npz`) before starting `mccs-report-ui`. To use the
compressed (spherical-harmonic `.h5`) EEPs instead, set `$POSTX_EEP_DIR` to their folder, e.g.
`export POSTX_EEP_DIR=/path/to/shc-eeps` holding `ska_low_harp.h5` and `ska_low_feko.h5`, and pick
`--mccs-eep compressed-harp` or `compressed-feko` (the "mccs-eep" select in the UI; `regular` is the npz
EEPs). `$POSTX_EEP_DIR` also adds the `eep-harp` and `eep-feko` `--beam` choices. The solver's
ska-sdp-* dependencies come from the SKAO index, which `uv` picks up from `pyproject.toml`; with pip,
add `--extra-index-url https://artefact.skao.int/repository/pypi-internal/simple`.
