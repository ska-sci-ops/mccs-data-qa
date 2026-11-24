""" migrate_set_ids.py -- one-off: rename set<old id>_<date>/ folders to set_<date>_<LST>/, the stable Set IDs (see analyse.set_id).

Each folder's set is worked out from its UVX files (<EB id>_<station>_sweep.uvx) looked up in the workbook,
not from the folder name or calibration_set.yaml, which the renumbering may have made wrong. A folder whose
files are all from one set is renamed, and the folder name is rewritten in its yaml / json files (set_id,
file list, run.json argv) and in every other set's (calibration runs read from another set).

Folders it can't place safely are listed and left alone, to sort out by hand:
  - mixed: UVX files from more than one set (a set converted into another set's folder)
  - duplicate: two folders hold the same set
  - the yaml describes a different set to the files, or no file is in the workbook

Safe to rerun: renamed folders are already at their stable name.

    python -m mccs_data_qa.migrate_set_ids --sweeps-dir sweeps --dry-run
"""

import argparse
import os
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
import yaml

from mccs_data_qa.analyse import set_id
from mccs_data_qa.convert_set import set_folder


def eb_of(f: Path) -> str:
    """ EB id of a converted file, <EB id>_<station>_sweep.uvx (same split as organize_sweeps). """
    return f.name.split('_s')[0]


def describe(d: Path) -> str:
    """ Folder name, UVX count and the reports run in it, so you can tell which copy to keep. """
    reports = sorted(p.name for p in (d / 'reports').glob('*') if p.is_dir() and p.name != 'convert')
    return f"{d.name} ({len(list(d.glob('*.uvx')))} UVX, reports: {', '.join(reports) or 'none'})"


def yaml_set(d: Path) -> int | None:
    """ Stable ID of the set calibration_set.yaml describes, or None without a usable yaml. """
    y = d / 'calibration_set.yaml'
    first = ((yaml.safe_load(y.read_text()) or {}).get('set_details') or [{}])[0] if y.exists() else {}
    return set_id(first['Date (UTC)'], first['LST (hr)']) if {'Date (UTC)', 'LST (hr)'} <= set(first) else None


def plan(sweeps: Path, obs: dict[str, tuple[int, object]]) -> tuple[dict[Path, tuple[int, Path]], list[str]]:
    """ ({old folder: (new id, new folder)}, [why each other folder is left alone]).

    Args:
        obs (dict): {observation id: (stable set id, UTC date)} from the workbook.
    """
    want, skip = {}, []
    for d in sorted(sweeps.glob('set*_*')):
        sets = defaultdict(list)
        for f in sorted(d.glob('*.uvx')):
            sets[obs.get(eb_of(f))].append(f)
        unknown = sets.pop(None, [])
        note = f" ({len(unknown)} UVX not in the workbook, moved with it)" if unknown else ''
        if not sets:
            skip.append(f"{describe(d)}: no UVX file is in the workbook")
        elif len(sets) > 1:
            parts = ', '.join(f"set{i}: {len(fs)} UVX" for (i, _), fs in sorted(sets.items()))
            skip.append(f"{describe(d)}: mixed, files from {len(sets)} sets ({parts})")
        elif (y := yaml_set(d)) not in (None, next(iter(sets))[0]):
            skip.append(f"{describe(d)}: yaml describes set{y} but the files are set{next(iter(sets))[0]}")
        else:
            want[d] = next(iter(sets))
            if note:
                print(f"NOTE {d.name}{note}")

    by_target = defaultdict(list)
    for d, key in want.items():
        by_target[key].append(d)
    moves = {}
    for (i, date), ds in by_target.items():
        new = Path(set_folder(sweeps, i, date))
        if len(ds) > 1:
            skip.append(f"{new.name}: duplicate, same set in {'; '.join(map(describe, ds))}")
        elif new.exists() and new != ds[0]:
            skip.append(f"{describe(ds[0])}: {new.name} already exists")
        elif new != ds[0]:
            moves[ds[0]] = (i, new)
    return moves, skip


def rename_refs(text: str, names: dict[str, str]) -> str:
    """ Replace whole folder names only: 'set12_2026-01-01' must not match 'set112_...' or 'cal_from_set12_...'. """
    pat = re.compile(r'(?<![\w-])(' + '|'.join(map(re.escape, names)) + r')(?![\w-])')
    return pat.sub(lambda m: names[m.group(1)], text)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--sweeps-dir', default='sweeps')
    p.add_argument('--xlsx', default='db/calibration_sets.xlsx', help="workbook whose Set Details place each EB")
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    sweeps = Path(a.sweeps_dir)

    det = pd.read_excel(a.xlsx, sheet_name='Set Details')
    obs = {o: (set_id(dt, lst), dt) for o, dt, lst in zip(det['Observation ID'], det['Date (UTC)'], det['LST (hr)'])}
    moves, skip = plan(sweeps, obs)
    for s in skip:
        print(f"SKIP {s}")
    for old, (_, new) in moves.items():
        print(f"MOVE {old.name} -> {new.name}")
    summary = f"{len(moves)} to rename, {len(skip)} left for you to sort out"
    if a.dry_run or not moves:
        print(summary + (' (dry run)' if a.dry_run else ''))
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
    print(summary.replace('to rename', 'renamed') + ". Then rerun mccs-analyse so calibration_sets.xlsx has the new IDs.")


if __name__ == '__main__':
    main()
