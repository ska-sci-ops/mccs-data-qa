import sys

import pandas as pd
import yaml

from mccs_data_qa import migrate_set_ids
from mccs_data_qa.analyse import add_derived, calibration_sets, set_id


def sweeps(sets):
    """ Two stations sweeping at each (UTC start, LST). """
    rows = [dict(Mode='correlator', **{'Sub-mode': 'sweep', 'UTC Start': t, 'LST start (hr)': lst,
                                         'Station ID': st, 'Observation ID': f'eb-{t}-{st}', 'n_files': 385})
            for t, lst in sets for st in ('s8-1', 's8-6')]
    return add_derived(pd.DataFrame(rows))


def test_set_id():
    assert set_id('2026-09-15', 14.4) == 20260915144
    assert set_id(pd.Timestamp('2026-09-15T00:00:00'), 0.04) == 20260915000


def test_ids_survive_deleting_other_sets():
    sets = [('2026-09-10 01:00', 3.1), ('2026-09-15 02:00', 14.4), ('2026-09-15 05:00', 17.4)]
    before = calibration_sets(sweeps(sets))[0].set_index('Set ID')['LST (hr)']
    after = calibration_sets(sweeps(sets[1:]))[0].set_index('Set ID')['LST (hr)']
    assert list(before.index) == [20260910031, 20260915144, 20260915174]
    assert after.to_dict() == before.drop(20260910031).to_dict()


def test_migrate(tmp_path, monkeypatch):
    def make(name, lst, argv):
        d = tmp_path / name
        (d / 'reports' / 'calibration' / 'run1').mkdir(parents=True)
        info = dict(set_id=int(name[3:].split('_')[0]),
                    set_details=[{'Date (UTC)': '2026-09-15T00:00:00.000', 'LST (hr)': lst}],
                    filelist=[dict(file=f'{d}/eb-x.uvx')])
        (d / 'calibration_set.yaml').write_text(yaml.safe_dump(info, sort_keys=False))
        (d / 'reports' / 'calibration' / 'run1' / 'run.json').write_text(argv)
        return d

    make('set1_2026-09-15', 14.4, f'["--datadir={tmp_path}/set1_2026-09-15"]')
    # set11 reads a calibration run from set1; its own run name merely contains 'set1_...' and must not change
    make('set11_2026-09-15', 17.4, f'["--caldir", "{tmp_path}/set1_2026-09-15/reports/calibration/cal_from_set1_2026-09-15"]')
    monkeypatch.setattr(sys, 'argv', ['x', '--sweeps-dir', str(tmp_path)])
    migrate_set_ids.main()

    assert sorted(p.name for p in tmp_path.iterdir()) == ['set20260915144_2026-09-15', 'set20260915174_2026-09-15']
    d = tmp_path / 'set20260915144_2026-09-15'
    info = yaml.safe_load((d / 'calibration_set.yaml').read_text())
    assert info['set_id'] == 20260915144 and info['filelist'][0]['file'] == f'{d}/eb-x.uvx'
    other = (tmp_path / 'set20260915174_2026-09-15' / 'reports' / 'calibration' / 'run1' / 'run.json').read_text()
    assert other == f'["--caldir", "{d}/reports/calibration/cal_from_set1_2026-09-15"]'
