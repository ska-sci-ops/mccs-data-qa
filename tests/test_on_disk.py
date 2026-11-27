import pandas as pd

from mccs_data_qa import analyse, cli
from mccs_data_qa.reports import report_ui


def test_on_disk(tmp_path):
    """ --check-exists marks rows whose files are gone; the set counts them; Convert is off when none are left. """
    daq, db = tmp_path / 'daq', tmp_path / 'db'
    (daq / 'eb-a' / 'ska-low-mccs' / 'pb-1').mkdir(parents=True)  # eb-b has been deleted
    db.mkdir()
    rows = [
        {'Observation ID': eb, 'pb-id': 'pb-1', 'Station ID': st, 'Mode': 'correlator', 'Sub-mode': 'sweep',
         'UTC Start': '2026-11-17 03:00:00', 'LST start (hr)': 11.0, 'n_files': 385, 'size_mb': 1.0}
        for eb, st in (('eb-a', 's8-1'), ('eb-b', 's8-6'))
    ]
    pd.DataFrame(rows).to_csv(db / 'latest.csv', index=False)

    cli.run_spider(str(daq), outdir=str(db), check_only=True, max_workers=2)
    df = pd.read_csv(db / 'latest.csv')
    assert dict(zip(df['Observation ID'], df['on_disk'])) == {'eb-a': True, 'eb-b': False}

    summary, detail = analyse.calibration_sets(analyse.add_derived(df))
    assert summary['On disk'].tolist() == [1]
    assert detail['on_disk'].tolist() == [True, False]

    row = {'folder': str(tmp_path / 'set')}
    assert report_ui.set_row_state(row | {'on_disk': 0}, set())['action'] == 'missing'
    assert report_ui.set_row_state(row | {'on_disk': 1}, set())['action'] == 'convert'
    assert report_ui.set_row_state(row | {'on_disk': None}, set())['action'] == 'convert'
