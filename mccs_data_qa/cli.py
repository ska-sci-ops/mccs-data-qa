""" spider.py -- create a CSV of single-station observation metadata. """

import argparse
import os
import sys
import re
import glob
import pandas as pd
import tqdm
import numpy as np
from astropy.time import Time
from astropy.coordinates import EarthLocation
import h5py
import yaml
from dataclasses import dataclass
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger

# Map of all station IDs
from .station_ids import sid_map

# Setup logger
logger.remove(0)
logger.add(sys.stderr, format="<level>{level}</level> | {message}", level="INFO")

# Set SKA-Low Earth Location
eloc = EarthLocation(x=-2561290.83467119, y=5085918.51537833, z=-2864050.87177975, unit='m')

# Define a dataclass to store metadata in
@dataclass
class ObservationMetadata:
    """ Simple dataclass to store observation metadata. """
    obs_id: str
    pb_id: str=''
    station: str=''
    mode: str=''
    sub_mode: str=''
    intent: str=''
    notes: str=''
    observer: str=''
    reference: str=''
    utc_start: str=''
    obs_duration: str=''
    loop_duration: str=''
    loop_rest_interval: str=''
    lst_start: float=''
    start_channel: int=''
    n_channel: int=''
    start_frequency: float=''
    obs_bandwidth: float=''
    samples_per_frame: int=''
    time_resolution: float=''
    n_timesteps: int=''
    tracking: str=''
    source_name: str=''
    right_ascension: str=''
    declination: str=''
    altitude: float=''
    azimuth: float=''
    obs_date: str=''
    n_files: int=0
    size_mb: float=0.0
    qa: str=''
    error: str=''

# Convert HDF5 name to a mode and submode
obs_types = {
    'stationbeam_integ': ['beamformer', 'power'],
    'channel_burst': ['channel-voltages', 'sweep'],
    'channel_integ': ['antenna-bandpass', ' '],
    'channel_cont': ['channel-voltages', 'fixed'],
    'correlation_burst': ['correlator', 'sweep'], # NOTE: Could be fixed submode too
    'raw_burst': ['adc-samples', 'synchronous'],  # NOTE: Could be asynchronous submode too
    'beamformed_burst': ['beamformer', 'voltages'],
}

###################
## Metadata readers
###################

def get_md_corr(filelist: list, obs_md: ObservationMetadata) -> ObservationMetadata:
    """ Get correlator metadata from filelist. """

    dirname = os.path.dirname(filelist[0])

    # Map (channel, sequence) -> actual path. Filenames are opened from this map,
    # never rebuilt: the date/seconds fields can differ between files (e.g. a
    # sweep crossing midnight), so a reconstructed name may not exist.
    files = {}
    for f in sorted(filelist):
        parts = os.path.basename(f).replace('.hdf5', '').split('_')
        if len(parts) != 6 or '_'.join(parts[:2]) != 'correlation_burst':
            logger.warning(f"{obs_md.obs_id}/{obs_md.pb_id} - Mixed HDF5 file types in {dirname}: skipping {os.path.basename(f)}")
            continue
        files.setdefault((int(parts[2]), int(parts[5])), f)
    if not files:
        raise RuntimeError("No parseable correlation_burst files")

    # Get unique channels and sequence IDs (timesteps)
    chans = sorted({c for c, _ in files})
    tsteps = sorted({t for _, t in files})

    n_chan = len(chans)
    n_step = len(tsteps)

    obs_md.n_channel     = n_chan
    obs_md.n_timesteps   = n_step
    obs_md.start_channel = np.min(chans)
    obs_md.start_frequency = 0.78125 * obs_md.start_channel
    obs_md.obs_bandwidth   = 0.78125 * obs_md.n_channel

    # Figure out if sweep mode (n_step == 1) or fixed mode (n_chan == 1)
    try:
        assert n_chan == 1 or n_step == 1
    except AssertionError:
        raise RuntimeError(f"Cannot determine mode! n_chan {n_chan} n_step {n_step}")

    obs_md.sub_mode = 'fixed' if n_chan == 1 else 'sweep'

    # Find first file, extract metadata
    first_file = files[(chans[0], tsteps[0])]
    with h5py.File(first_file, mode='r') as h:
        t0 = Time(h['sample_timestamps']['data'][0, 0], format='unix')
        obs_md.time_resolution = np.round( h['root'].attrs['tsamp'], 11)
        obs_md.utc_start = t0.iso
        obs_md.n_timesteps *= h['sample_timestamps']['data'].shape[0]

    # Get last timestamp: in sweep mode, use last channel
    if obs_md.sub_mode == 'sweep':
        with h5py.File(files[(chans[-1], tsteps[0])], mode='r') as h:
            t1 = Time(h['sample_timestamps']['data'][0, 0] + obs_md.time_resolution, format='unix')

    # Get last timestamp: in fixed mode, this will be last file in sequence
    elif n_step > 1:
        with h5py.File(files[(chans[0], tsteps[-1])], mode='r') as h:
            t1 = Time(h['sample_timestamps']['data'][-1, -1] + obs_md.time_resolution, format='unix')
    else:
        with h5py.File(first_file, mode='r') as h:
            rr = h['root'].attrs
            t1 = Time(rr['ts_start'] + rr['tsamp'], format='unix')

    obs_md.obs_duration = np.round((t1 - t0).sec, 11)

    return obs_md


def get_md_adc(filelist: list, obs_md: ObservationMetadata) -> ObservationMetadata:
    """ Get ADC sample metadata from filelist. """
    with h5py.File(filelist[0], mode='r') as h:

        # ADC dataset is always 32768 in size. Here we check if data > 4096 are zeros,
        # which indicates synchronous mode was used
        dcount = np.sum(h['raw_']['data'][:][4096:])
        obs_md.sub_mode = 'synchronous' if dcount == 0 else 'asynchronous'
        obs_md.time_resolution = np.round(1 / 800e6, 11)
        n_samp = 4096 if dcount == 0 else 32768
        obs_md.obs_duration = n_samp * obs_md.time_resolution
        obs_md.n_timesteps  = n_samp

    return obs_md

def get_md_power_beam(filelist: list, obs_md: ObservationMetadata) -> ObservationMetadata:
    """ Get station beamformer (power) metadata from filelist. """
    # This will sort files in time order
    # Files appear to follow stationbeam_integ_0_20240918_11242_0.hdf5 (X and SEQ zero)
    filelist = sorted(filelist)


    with h5py.File(filelist[0], mode='r') as h:
        obs_md.time_resolution = h['root'].attrs['tsamp']
        obs_md.n_timesteps  = h['root'].attrs['n_blocks']
        obs_md.n_channel    = h['root'].attrs['n_chans']

        t0 = h['root'].attrs['ts_start']
        t1 = h['root'].attrs['ts_end']

    if len(filelist) > 1:
        with h5py.File(filelist[-1], mode='r') as h:
            t1 = h['root'].attrs['ts_end']
    obs_md.obs_duration = np.round(t1 - t0, 11)
    obs_md.n_timesteps *= len(filelist)

    return obs_md


######################
## Per-observation work
######################

def _read_header(fh, obs_md: ObservationMetadata) -> None:
    """ Read start time and station ID from an open HDF5 file. """
    t = Time(fh['root'].attrs['ts_start'], format='unix', location=eloc)
    obs_md.utc_start = t.iso
    obs_md.lst_start = np.round(t.sidereal_time('apparent').value, 3)

    if fh['root'].attrs['station_id'] != 0:
        station_id = str(fh['root'].attrs['station_id'])

        # As of Jan 2025 some stations have integer IDs. This fixes 'em
        # station_id in
        # https://gitlab.com/ska-telescope/ska-telmodel-data/-/blob/main/tmdata/instrument/ska1_low/layout/data.json
        obs_md.station = sid_map.get(station_id, station_id)
    else:
        # Station ID is sometimes stored in description field
        # e.g. "s10-3, sun SFT" or "s9-2 CAL TEST"
        description = _description(fh).split(', ')
        description = description[0].split(' ')
        if description[0].lower().startswith('s'):
            obs_md.station = description[0].upper()
        if len(description) > 1:
            obs_md.intent = description[1]


def _description(fh) -> str:
    """ observation_info description attribute, decoded if stored as bytes. """
    description = fh['observation_info'].attrs['description']
    return description.decode() if isinstance(description, bytes) else str(description)


def _read_description(fh, obs_md: ObservationMetadata) -> None:
    """ Sometimes the source RA/DEC or name is in the description field. """
    description = _description(fh)
    paren_match = re.search(r"\(([^)]+)\)", description)
    if not paren_match:
        return
    paren_value = paren_match.group(1)
    if paren_value == "NAMED":
        target_match = re.search(r"pointing at ([A-Z0-9_-]+) \(NAMED\)", description)
        obs_md.source_name = target_match.group(1) if target_match else ''
        return
    radec_match = re.search(r"\[([^\]]+)\]", description)
    if not radec_match:
        return
    ra, dec = [float(x.strip()) for x in radec_match.group(1).split(",")]
    if paren_value == "ALT_AZ":
        obs_md.altitude = ra
        obs_md.azimuth = dec
    else:
        obs_md.right_ascension = ra
        obs_md.declination = dec
    obs_md.source_name = f"{paren_value} [{ra:.3f}, {dec:.3f}]"


def _read_yaml(yaml_path: str, obs_md: ObservationMetadata) -> None:
    """ Override metadata with values from obs_metadata.yaml (missing keys keep HDF5 values). """
    with open(yaml_path, 'r') as fh:
        log = yaml.safe_load(fh) or {}
    if not isinstance(log, dict):
        raise ValueError(f"{yaml_path} is not a mapping")
    keymap = {'station': 'station', 'intent': 'intent', 'observer': 'observer',
              'reference': 'reference', 'notes': 'notes', 'qa': 'qa', 'tracking': 'tracking',
              'source_name': 'source_name', 'ra': 'right_ascension', 'dec': 'declination',
              'alt': 'altitude', 'az': 'azimuth'}
    for key, attr in keymap.items():
        setattr(obs_md, attr, log.get(key, getattr(obs_md, attr)))


def _fill_md(h5list: list, dirs: list, obs_md: ObservationMetadata) -> None:
    """ Fill obs_md from the HDF5 files (and YAML) of one processing block.

    Optional metadata (header, description, YAML) is read in its own try block so a
    bad attribute is logged without losing the rest. Errors in the mode-specific
    readers propagate to the caller, which records them in obs_md.error.
    """
    tag = f"{obs_md.obs_id}/{obs_md.pb_id}"
    with h5py.File(h5list[0], 'r') as fh:
        for reader in (_read_header, _read_description):
            try:
                reader(fh, obs_md)
            except Exception as exc:
                logger.warning(f"{tag} - {reader.__name__}: {type(exc).__name__}: {exc} ({h5list[0]})")

    for d in dirs:
        yaml_path = os.path.join(d, "obs_metadata.yaml")
        if os.path.exists(yaml_path):
            try:
                _read_yaml(yaml_path, obs_md)
            except Exception as exc:
                logger.warning(f"{tag} - bad YAML: {type(exc).__name__}: {exc} ({yaml_path})")
            break

    # Get mode and submode from HDF5 filename
    obstype = "_".join(os.path.basename(h5list[0]).split('_')[:2])
    if obstype not in obs_types:
        obs_md.mode = 'unknown'
        obs_md.error = f"unknown file type {obstype}"
        logger.warning(f"{tag} - {obs_md.error} ({h5list[0]})")
        return

    obs_md.mode, obs_md.sub_mode = obs_types[obstype]
    if obs_md.mode == 'correlator':
        get_md_corr(h5list, obs_md)
    elif obs_md.mode == 'adc-samples':
        get_md_adc(h5list, obs_md)
    elif obs_md.mode == 'beamformer' and obs_md.sub_mode == 'power':
        get_md_power_beam(h5list, obs_md)


def _process_obs(obs: str) -> list:
    """ Build ObservationMetadata record(s) for a single eb-* folder.

    Pulled out of the main loop so it can be dispatched to a thread pool --
    the work here is almost entirely I/O (HDF5 reads, YAML reads, stat
    calls), so running several observations concurrently overlaps that I/O
    instead of waiting on it one folder at a time.

    An execution block can hold several processing blocks
    (ska-low-mccs/<pb-id>/...), with HDF5 files either directly in the PB
    directory or nested deeper (e.g. <pb-id>/OSODriftScan/SUN/). One row is
    returned per directory of HDF5 files. A row is always kept, even if its
    metadata can't be read: the reason goes in the 'error' column.
    """
    obs_id = os.path.basename(obs)
    mccs_dir = os.path.join(obs, "ska-low-mccs")
    try:
        pb_ids = sorted(os.listdir(mccs_dir))
    except OSError as exc:
        logger.warning(f"{obs_id} - cannot list {mccs_dir}: {exc}")
        return []

    results = []
    for pb_id in pb_ids:
        pb_dir = os.path.join(mccs_dir, pb_id)
        if not os.path.isdir(pb_dir):
            continue
        for dirpath, _, filenames in os.walk(pb_dir, onerror=lambda e: logger.warning(f"{obs_id}/{pb_id} - {e}")):
            names = sorted(f for f in filenames if f.endswith('.hdf5'))
            if not names:
                continue
            obs_md = ObservationMetadata(obs_id, pb_id=pb_id)
            h5list = [os.path.join(dirpath, f) for f in names]
            try:
                obs_md.n_files = len(h5list)
                obs_md.size_mb = np.round(sum(os.path.getsize(f) for f in h5list) / 1e6, 2)
                # YAML may sit beside the HDF5 files or at the top of the PB directory
                _fill_md(h5list, [dirpath, pb_dir], obs_md)
            except Exception as exc:
                obs_md.error = f"{type(exc).__name__}: {exc}"
                logger.warning(f"{obs_id}/{pb_id} - {obs_md.error} ({dirpath})")
            results.append(obs_md)

    return results


##############
## Main loop
##############

def run_spider(datapath: str, outdir: str='db', skip_existing: bool=True, max_workers: int=8):
    """ Run the spider script to create a CSV file.

    Loops through execution block directories in datapath.

    Args:
        datapath: Path to the directory containing eb-* observation folders.
        outdir: Output directory for CSV files.
        skip_existing: If True, read latest.csv and skip any folders whose
                       obs_id is already present, appending new rows to the
                       existing records. Defaults to True.
        max_workers: Number of observation folders to process concurrently.
                     The per-observation work is I/O-bound (HDF5/YAML reads,
                     stat calls), so a thread pool overlaps that I/O instead
                     of processing one folder at a time. Defaults to 8.
    """
    logger.info(f"Spidering {datapath}...")
    # Match both the old eb-t<date>-<seq> folders and the newer
    # short-alphanumeric eb-<id> folders (e.g. eb-18jjem39q2h9).
    obslist = sorted(glob.glob(f"{datapath}/eb-*"))

    # Display-friendly names. Defined here (not just later, near the rename
    # of the newly-spidered df) so it can also be used to normalize whatever
    # column names happen to already be in latest.csv when we load it below.
    col_names = {
        'lst_start': 'LST start (hr)',
        'obs_duration': 'Duration (s)',
        'obs_id': 'Observation ID',
        'mode': 'Mode',
        'station': 'Station ID',
        'sub_mode': 'Sub-mode',
        'utc_start': 'UTC Start',
        'qa': 'QA',
        'bandwidth': 'Bandwidth (MHz)',
        'observer': 'Observer',
        'pb_id': 'pb-id',
    }

    # Load existing records and build a set of already-spidered obs_ids
    existing_df = None
    existing_ids: set = set()
    if skip_existing:
        latest_csv = os.path.join(outdir, "latest.csv")
        if os.path.exists(latest_csv):
            try:
                existing_df = pd.read_csv(latest_csv)
                # latest.csv may have been written by an older version of this
                # script (raw dataclass field names) or a newer one (display
                # names). Normalize to display names either way -- rename()
                # silently leaves any column not in col_names untouched, so
                # this is a no-op for a file that's already in the new format.
                existing_df = existing_df.rename(columns=col_names)
                id_col = 'Observation ID' if 'Observation ID' in existing_df.columns else 'obs_id'
                existing_ids = set(existing_df[id_col].dropna().astype(str))
                logger.info(f"Loaded {len(existing_ids)} existing records from {latest_csv}; skipping those folders.")
            except Exception as exc:
                logger.warning(f"Could not read {latest_csv}: {exc}. Spidering all folders.")

    # Filter out already-spidered folders up front (this is what actually
    # keeps a daily incremental run fast -- everything below only ever runs
    # against the handful of new folders, not the whole archive).
    todo = [obs for obs in obslist if os.path.basename(obs) not in existing_ids]
    n_skipped = len(obslist) - len(todo)
    if n_skipped:
        logger.debug(f"Skipping {n_skipped} already-spidered folders")

    obs_table = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_process_obs, obs): obs for obs in todo}
        for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
            obs = futures[future]
            try:
                obs_mds = future.result()
            except Exception as exc:
                logger.warning(f"Unhandled error processing {os.path.basename(obs)}: {exc}")
                continue
            obs_table.extend(obs_mds)

    df = pd.DataFrame(obs_table)

    # Convert floats into string (helps searching). Guard against an empty
    # DataFrame (e.g. every folder was already in latest.csv, or none had
    # readable metadata) -- in that case pd.DataFrame([]) has no columns at
    # all, so df['lst_start'] would raise a KeyError.
    if not df.empty:
        df['lst_start']    = pd.to_numeric(df['lst_start'], errors='coerce').round(1).astype('str')
        df['obs_duration'] = pd.to_numeric(df['obs_duration'], errors='coerce').round(3).astype('str')

    # Rename to display-friendly names (col_names is defined earlier, where
    # it's also used to normalize existing_df loaded from latest.csv).
    df = df.rename(columns=col_names)

    # Merge newly spidered rows with any existing records from latest.csv
    if skip_existing and existing_df is not None and not df.empty:
        df = pd.concat([existing_df, df], ignore_index=True)
    elif skip_existing and existing_df is not None and df.empty:
        logger.info("No new observations found; latest.csv is already up to date.")
        df = existing_df

    if df is None or df.empty:
        logger.warning("No observations found (new or existing); nothing to save.")
        return

    filename = datetime.now().strftime(f"{outdir}/%Y-%m-%d.csv")
    logger.info(f"Saving to {filename}")
    df.to_csv(filename, index=False)
    # Save to latest.csv also
    df.to_csv(f"{outdir}/latest.csv", index=False)

def main(argv=None):
    parser = argparse.ArgumentParser(description="Create a CSV of single-station observation metadata.")
    parser.add_argument("datapath", nargs="?", default="/home/jovyan/daq-data", help="directory containing eb-* folders")
    parser.add_argument("-o", "--outdir", default="db", help="output directory (default: db)")
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false",
                        help="re-spider every folder instead of skipping those already in latest.csv")
    parser.add_argument("-j", "--max-workers", type=int, default=8, help="concurrent folders (default: 8)")
    parser.add_argument("--no-upload", dest="upload", action="store_false", help="skip the rclone upload")
    args = parser.parse_args(argv)

    now_str = datetime.now().strftime("%Y-%m-%d")
    os.makedirs(args.outdir, exist_ok=True)
    logger.add(f"{args.outdir}/loguru_{now_str}.log", format="<level>{level}</level> | {message}", level="INFO")
    run_spider(args.datapath, outdir=args.outdir, skip_existing=args.skip_existing, max_workers=args.max_workers)

    if args.upload:
        print("Updating database on acacia")
        os.system(f"/home/jovyan/shared/Danny/rclone/rclone copy {args.outdir}/ SKAO:/aa05/mccs-spider-search/db")


if __name__ == "__main__":
    main()
