""" organize_sweeps.py -- move loose eb-* sweep files into set<id>_<UTC date>/ folders, adding calibration_set.yaml where missing. """

import argparse
import glob
import os
import shutil
import pandas as pd
from mccs_data_qa.convert_set import write_set_yaml


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sweeps-dir', default='sweeps')
    p.add_argument('--xlsx', default='db/calibration_sets.xlsx')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()

    det = pd.read_excel(a.xlsx, sheet_name='Set Details')
    by_eb = det.drop_duplicates('Observation ID').set_index('Observation ID')

    for name in sorted(os.listdir(a.sweeps_dir)):
        src = f"{a.sweeps_dir}/{name}"
        eb_id = name.split('_s')[0]
        if not name.startswith('eb-') or eb_id not in by_eb.index:
            if name.startswith('eb-'):
                print(f"SKIP {name}: not in sheet")
            continue
        r = by_eb.loc[eb_id]
        dst_dir = f"{a.sweeps_dir}/set{r['Set ID']}_{pd.to_datetime(r['Date (UTC)']):%Y-%m-%d}"
        print(f"MOVE {name} -> {os.path.basename(dst_dir)}/")
        if a.dry_run:
            continue
        os.makedirs(dst_dir, exist_ok=True)
        if os.path.exists(f"{dst_dir}/{name}"):
            print(f"  exists, skipping")
            continue
        shutil.move(src, dst_dir)

    for d in sorted(glob.glob(f"{a.sweeps_dir}/set*_*")):
        if os.path.exists(f"{d}/calibration_set.yaml"):
            continue
        set_id = int(os.path.basename(d)[3:].split('_')[0])
        rows = det[det['Set ID'] == set_id]
        files = {f.split('_s')[0]: f for f in os.listdir(d) if f.endswith('.uvx')}
        filelist = [dict(observation_id=e, station_id=r['Station ID'].lower(), file=f"{d}/{files[e]}", status='ok')
                    for e, r in rows.set_index('Observation ID').iterrows() if e in files]
        print(f"YAML {d} ({len(filelist)}/{len(rows)} files)")
        if not a.dry_run:
            write_set_yaml(d, set_id, rows, filelist, dict(vars(a), generated='retroactively by organize_sweeps'), None)


if __name__ == '__main__':
    main()
