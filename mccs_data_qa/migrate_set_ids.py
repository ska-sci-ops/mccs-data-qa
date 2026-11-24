""" migrate_set_ids.py -- one-off: rename set<old id>_<date>/ folders to the stable Set IDs (see analyse.set_id).

Reads each folder's calibration_set.yaml (date + LST are recorded there, so the current, renumbered
workbook is not needed), renames the folder, and rewrites the folder name in its yaml / json files
(set_id, file list, run.json argv) and in every other set's (calibration runs read from another set).

    python -m mccs_data_qa.migrate_set_ids --sweeps-dir sweeps --dry-run
"""

import argparse
import os
import re
from pathlib import Path

import yaml

from mccs_data_qa.analyse import set_id
from mccs_data_qa.convert_set import set_folder


def plan(sweeps: Path) -> dict[Path, tuple[int, Path]]:
    """ {old folder: (new id, new folder)} for every set folder whose name is not already its stable ID. """
    moves = {}
    for d in sorted(sweeps.glob('set*_*')):
        y = d / 'calibration_set.yaml'
        if not y.exists():
            print(f"SKIP {d.name}: no calibration_set.yaml, rename by hand")
            continue
        info = yaml.safe_load(y.read_text())
        first = (info.get('set_details') or [{}])[0]
        if 'Date (UTC)' not in first or 'LST (hr)' not in first:
            print(f"SKIP {d.name}: yaml has no Date (UTC) / LST (hr)")
            continue
        new_id = set_id(first['Date (UTC)'], first['LST (hr)'])
        new = Path(set_folder(sweeps, new_id, first['Date (UTC)']))
        if new != d:
            moves[d] = (new_id, new)
    return moves


def rename_refs(text: str, names: dict[str, str]) -> str:
    """ Replace whole folder names only: 'set12_2026-01-01' must not match 'set112_...' or 'cal_from_set12_...'. """
    pat = re.compile(r'(?<![\w-])(' + '|'.join(map(re.escape, names)) + r')(?![\w-])')
    return pat.sub(lambda m: names[m.group(1)], text)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--sweeps-dir', default='sweeps')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    sweeps = Path(a.sweeps_dir)

    moves = plan(sweeps)
    clash = [new for new in {n for _, n in moves.values()} if new.exists() and new not in moves]
    targets = [n for _, n in moves.values()]
    if clash or len(set(targets)) != len(targets):
        raise SystemExit(f"Target folders exist or collide, nothing changed: {clash or targets}")
    for old, (new_id, new) in moves.items():
        print(f"MOVE {old.name} -> {new.name}")
    if a.dry_run or not moves:
        print(f"{len(moves)} folders to rename" + (' (dry run)' if a.dry_run else ''))
        return

    # Rewrite references first (in every set folder, not just moved ones), then rename
    names = {old.name: new.name for old, (_, new) in moves.items()}
    for f in [*sweeps.glob('set*_*/calibration_set.yaml'), *sweeps.glob('set*_*/reports/**/*.json')]:
        text = f.read_text()
        out = rename_refs(text, names)
        if f.name == 'calibration_set.yaml' and f.parent in moves:
            out = re.sub(r'^set_id: .*$', f'set_id: {moves[f.parent][0]}', out, count=1, flags=re.M)
        if out != text:
            f.write_text(out)
    for old, (_, new) in moves.items():
        os.rename(old, new)
    print(f"Renamed {len(moves)} folders. Now rerun mccs-analyse so calibration_sets.xlsx has the new IDs.")


if __name__ == '__main__':
    main()
