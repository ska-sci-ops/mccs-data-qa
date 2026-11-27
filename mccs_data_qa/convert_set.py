""" convert_set.py -- convert every sweep in a calibration set (by Set ID) to UVX. """

import argparse
import glob
import json
import os
import sys
import pandas as pd
import yaml


def parse_n_expected(n_expected):
    """ '321,385' (or an int, or None) -> list of allowed file counts; empty means don't check. """
    if n_expected in (None, ''):
        return []
    return [int(x) for x in str(n_expected).split(',') if x.strip()]


def get_eb(daq_dir, eb_id, n_expected):
    """ Returns the path to the directory for a given eb_id; n_expected is a count or comma-separated counts. """
    dirs = glob.glob(f"{daq_dir}/{eb_id}/ska-low-mccs/*")
    if not dirs:
        raise FileNotFoundError(f"EB not found: {daq_dir}/{eb_id}/ska-low-mccs/*")
    dirpath = dirs[0]
    # some EBs keep the files in a correlator_data subdirectory
    if os.path.isdir(f"{dirpath}/correlator_data"):
        dirpath = f"{dirpath}/correlator_data"
    n_files = len(glob.glob(f"{dirpath}/correlation*.hdf5"))
    allowed = parse_n_expected(n_expected)
    if allowed and n_files not in allowed:
        raise RuntimeError(f"Wrong number of files: {n_files} / {','.join(map(str, allowed))}")
    return dirpath


def convert(daq_dir, eb_id, out_dir, n_expected, quantise=None):
    # ponytail: lazy import, this only exists on the DAQ / jupyter host
    # hdf5_sweep_to_uvx writes zstd by default; ska_ost_low_uv.io imports hdf5plugin itself
    from ska_ost_low_uv.io import hdf5_sweep_to_uvx, get_hdf5_metadata

    dirpath = get_eb(daq_dir, eb_id, n_expected)
    # station ID from the first readable file: hdf5_sweep_to_uvx skips unreadable channels, so must we
    for fn in sorted(glob.glob(f"{dirpath}/correlation*.hdf5")):
        try:
            station_id = get_hdf5_metadata(fn)['station_id'].lower()
            break
        except (AttributeError, IndexError, KeyError, OSError):
            continue
    else:
        raise RuntimeError(f"No readable correlation files in {dirpath}")
    fn_out = f"{out_dir}/{eb_id}_{station_id}_sweep.uvx"
    # pattern: EB dirs can hold other DAQ HDF5 files (no 'root' group) alongside the correlation files
    hdf5_sweep_to_uvx(dirpath, fn_out, telescope_name=station_id, pattern='correlation*',
                      quantise=quantise, return_uvx=False)
    return station_id, fn_out


def write_set_yaml(out_dir, set_id, s_all, filelist, args, argv):
    """ Write calibration_set.yaml (set rows, file list, args) into out_dir. """
    info = dict(set_id=set_id,
                set_details=json.loads(s_all.to_json(orient='records', date_format='iso')),
                filelist=filelist, args=args, argv=argv)
    with open(f"{out_dir}/calibration_set.yaml", 'w') as f:
        yaml.safe_dump(info, f, sort_keys=False)


def add_quantise_arg(p):
    # quantise=None is lossless zstd; a step (in thermal-noise sigma) stores crosses as int16 (lossy)
    p.add_argument('--lossy', '--quantise', dest='quantise', type=float, nargs='?', const=0.5, default=None,
                   metavar='STEP', help="lossy: quantise crosses to STEP x thermal noise (default step 0.5, ~4x smaller)")


def main_eb():
    """ mccs-convert-eb: convert a single EB (an EB directory, or an EB ID under --daq-dir) to UVX. """
    p = argparse.ArgumentParser(description=main_eb.__doc__)
    p.add_argument('eb', help="EB directory (e.g. daq-data/eb-...) or EB ID")
    p.add_argument('--daq-dir', default='/home/jovyan/daq-data', help="where to look up a bare EB ID")
    p.add_argument('--out-dir', default='.')
    p.add_argument('--n-expected', default=None, help="fail unless the EB has this many correlation files (comma-separated list allowed)")
    add_quantise_arg(p)
    a = p.parse_args()

    eb = a.eb.rstrip('/')
    daq_dir, eb_id = os.path.split(os.path.abspath(eb)) if os.path.isdir(eb) else (a.daq_dir, eb)
    os.makedirs(a.out_dir, exist_ok=True)
    station_id, fn_out = convert(daq_dir, eb_id, a.out_dir, a.n_expected, a.quantise)
    print(f"OK   {eb_id} {station_id} -> {fn_out}")


def set_folder(out_dir, set_id, date):
    """ Folder a set is converted to: <out_dir>/set_<UTC date, yyyy-mm-dd>_<LST, 0.1 hr>, e.g. set_2026-10-01_205.

    The Set ID is yyyymmdd * 1000 + LST in 0.1 hr (analyse.set_id), so the folder holds the same information, readably.
    """
    return f"{out_dir}/set_{pd.to_datetime(date):%Y-%m-%d}_{set_id % 1000:03d}"


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('set_id', type=int, help="Set ID from calibration_sets.xlsx")
    p.add_argument('--xlsx', default='db/calibration_sets.xlsx')
    p.add_argument('--daq-dir', default='/home/jovyan/daq-data')
    p.add_argument('--out-dir', default='sweeps')
    p.add_argument('--n-expected', default='321,385', help="expected correlation files per EB (comma-separated list allowed)")
    p.add_argument('--include-failed', action='store_true', help="also convert observations flagged FAILED")
    p.add_argument('--list', action='store_true', help="show the set's observations and exit")
    add_quantise_arg(p)
    return p


def main():
    a = build_parser().parse_args()

    det = pd.read_excel(a.xlsx, sheet_name='Set Details')
    s = det[det['Set ID'] == a.set_id]
    if s.empty:
        sys.exit(f"Set ID {a.set_id} not found. Valid: {det['Set ID'].min()}-{det['Set ID'].max()}")

    cols = [c for c in ('Station ID', 'Observation ID', 'UTC Start', 'n_files', 'Status') if c in s]
    print(s[cols].to_string(index=False))
    if a.list:
        return

    out_dir = set_folder(a.out_dir, a.set_id, s['Date (UTC)'].iloc[0])

    s_all = s
    if not a.include_failed:
        skipped = s[s['Status'] == 'FAILED']
        if len(skipped):
            print(f"Skipping {len(skipped)} FAILED (use --include-failed): {', '.join(skipped['Observation ID'])}")
        s = s[s['Status'] != 'FAILED']

    os.makedirs(out_dir, exist_ok=True)
    bad, filelist = [], []
    for eb_id in s['Observation ID']:
        try:
            station_id, fn_out = convert(a.daq_dir, eb_id, out_dir, a.n_expected, a.quantise)
            print(f"OK   {eb_id} {station_id} -> {fn_out}")
            filelist.append(dict(observation_id=eb_id, station_id=station_id, file=fn_out, status='ok'))
        except Exception as e:
            print(f"FAIL {eb_id}: {e}")
            bad.append(eb_id)
            filelist.append(dict(observation_id=eb_id, status='failed', error=str(e)))
    write_set_yaml(out_dir, a.set_id, s_all, filelist, vars(a), sys.argv)
    print(f"Set {a.set_id}: {len(s) - len(bad)} converted, {len(bad)} failed")
    sys.exit(bool(bad))


if __name__ == '__main__':
    main()
