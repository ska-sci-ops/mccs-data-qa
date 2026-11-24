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
    sweeps, xlsx = tmp_path / 'sweeps', tmp_path / 'sets.xlsx'
    # Workbook: EB -> set (date, LST); eb-b, eb-c, eb-d are other sets the same day
    lst = {'eb-a': 14.4, 'eb-b': 17.4, 'eb-c': 15.0, 'eb-d': 16.0}
    pd.DataFrame({'Observation ID': list(lst), 'Date (UTC)': pd.Timestamp('2026-09-15'), 'LST (hr)': list(lst.values())}
                 ).to_excel(xlsx, sheet_name='Set Details', index=False)

    def make(name, ebs, lst_yaml, argv='[]'):
        d = sweeps / name
        (d / 'reports' / 'calibration' / 'run1').mkdir(parents=True)
        for eb in ebs:
            (d / f'{eb}_s8-1_sweep.uvx').touch()
        info = dict(set_id=int(name[3:].split('_')[0]),
                    set_details=[{'Date (UTC)': '2026-09-15T00:00:00.000', 'LST (hr)': lst_yaml}],
                    filelist=[dict(file=f'{d}/{eb}_s8-1_sweep.uvx') for eb in ebs])
        (d / 'calibration_set.yaml').write_text(yaml.safe_dump(info, sort_keys=False))
        (d / 'reports' / 'calibration' / 'run1' / 'run.json').write_text(argv)
        return d

    make('set1_2026-09-15', ['eb-a'], 14.4, f'["--datadir={sweeps}/set1_2026-09-15"]')
    # set11 reads a calibration run from set1; its own run name merely contains 'set1_...' and must not change
    make('set11_2026-09-15', ['eb-b'], 17.4, f'["--caldir", "{sweeps}/set1_2026-09-15/reports/calibration/cal_from_set1_2026-09-15"]')
    make('set2_2026-09-15', ['eb-c', 'eb-b'], 17.4)  # mixed: eb-b converted into set2's folder after a renumbering
    make('set3_2026-09-15', ['eb-d'], 16.0)          # duplicate: set3 and set4 hold the same set
    make('set4_2026-09-15', ['eb-d'], 16.0)
    monkeypatch.setattr(sys, 'argv', ['x', '--sweeps-dir', str(sweeps), '--xlsx', str(xlsx)])
    migrate_set_ids.main()

    assert sorted(p.name for p in sweeps.iterdir()) == [
        'set2_2026-09-15', 'set3_2026-09-15', 'set4_2026-09-15', 'set_2026-09-15_144', 'set_2026-09-15_174']
    d = sweeps / 'set_2026-09-15_144'
    info = yaml.safe_load((d / 'calibration_set.yaml').read_text())
    assert info['set_id'] == 20260915144 and info['filelist'][0]['file'] == f'{d}/eb-a_s8-1_sweep.uvx'
    other = (sweeps / 'set_2026-09-15_174' / 'reports' / 'calibration' / 'run1' / 'run.json').read_text()
    assert other == f'["--caldir", "{d}/reports/calibration/cal_from_set1_2026-09-15"]'

    migrate_set_ids.main()  # A rerun changes nothing
    assert len(list(sweeps.iterdir())) == 5
