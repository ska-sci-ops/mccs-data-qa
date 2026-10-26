""" convert_set.py -- convert every sweep in a calibration set (by Set ID) to UVX. """

import argparse
import glob
import json
import os
import sys
import pandas as pd
import yaml


def get_eb(daq_dir, eb_id, n_expected):
    """ Returns the path to the directory for a given eb_id. """
    dirpath = glob.glob(f"{daq_dir}/{eb_id}/ska-low-mccs/*")[0]
    # some EBs keep the files in a correlator_data subdirectory
    if os.path.isdir(f"{dirpath}/correlator_data"):
        dirpath = f"{dirpath}/correlator_data"
    n_files = len(glob.glob(f"{dirpath}/correlation*.hdf5"))
    if n_files != n_expected:
        raise RuntimeError(f"Wrong number of files: {n_files} / {n_expected}")
    return dirpath


def convert(daq_dir, eb_id, out_dir, n_expected):
    # ponytail: lazy import, these only exist on the DAQ / jupyter host
    import hdf5plugin  # noqa: F401
    from ska_ost_low_uv.io import hdf5_sweep_to_uvx, get_hdf5_metadata

    dirpath = get_eb(daq_dir, eb_id, n_expected)
    md = get_hdf5_metadata(glob.glob(f"{dirpath}/correlation*.hdf5")[0])
    station_id = md['station_id'].lower()
    fn_out = f"{out_dir}/{eb_id}_{station_id}_sweep.uvx"
    hdf5_sweep_to_uvx(dirpath, fn_out, telescope_name=station_id, return_uvx=False)
    return station_id, fn_out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('set_id', type=int, help="Set ID from calibration_sets.xlsx")
    p.add_argument('--xlsx', default='db/calibration_sets.xlsx')
    p.add_argument('--daq-dir', default='/home/jovyan/daq-data')
    p.add_argument('--out-dir', default='sweeps')
    p.add_argument('--n-expected', type=int, default=385, help="expected correlation files per EB")
    p.add_argument('--include-failed', action='store_true', help="also convert observations flagged FAILED")
    p.add_argument('--list', action='store_true', help="show the set's observations and exit")
    a = p.parse_args()

    det = pd.read_excel(a.xlsx, sheet_name='Set Details')
    s = det[det['Set ID'] == a.set_id]
    if s.empty:
        sys.exit(f"Set ID {a.set_id} not found. Valid: {det['Set ID'].min()}-{det['Set ID'].max()}")

    cols = [c for c in ('Station ID', 'Observation ID', 'UTC Start', 'n_files', 'Status') if c in s]
    print(s[cols].to_string(index=False))
    if a.list:
        return

    # per-set subdirectory: <out-dir>/set<id>_<UTC date, yyyy-mm-dd>
    out_dir = f"{a.out_dir}/set{a.set_id}_{pd.to_datetime(s['Date (UTC)'].iloc[0]):%Y-%m-%d}"

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
            station_id, fn_out = convert(a.daq_dir, eb_id, out_dir, a.n_expected)
            print(f"OK   {eb_id} {station_id} -> {fn_out}")
            filelist.append(dict(observation_id=eb_id, station_id=station_id, file=fn_out, status='ok'))
        except Exception as e:
            print(f"FAIL {eb_id}: {e}")
            bad.append(eb_id)
            filelist.append(dict(observation_id=eb_id, status='failed', error=str(e)))
    info = dict(set_id=a.set_id,
                set_details=json.loads(s_all.to_json(orient='records', date_format='iso')),
                filelist=filelist, args=vars(a), argv=sys.argv)
    with open(f"{out_dir}/calibration_set.yaml", 'w') as f:
        yaml.safe_dump(info, f, sort_keys=False)
    print(f"Set {a.set_id}: {len(s) - len(bad)} converted, {len(bad)} failed")
    sys.exit(bool(bad))


if __name__ == '__main__':
    main()
