"""test_convert_set: calibration_sets.xlsx, as written by mccs-analyse, read by the report UI's sets page."""

import pandas as pd

from mccs_data_qa.analyse import add_derived, write_calibration_report
from mccs_data_qa.convert_set import build_parser, set_folder
from mccs_data_qa.reports.argparse_form import form_fields
from mccs_data_qa.reports.report_ui import CONVERT_SET_ARGS, SET_YAML, convert_argv, load_sets, set_columns


def _sweeps() -> pd.DataFrame:
    """Two stations sweeping at the same time: one calibration set."""
    rows = [
        {'Observation ID': f'eb-{i}', 'Station ID': st, 'Mode': 'correlator', 'Sub-mode': 'sweep',
         'UTC Start': '2026-07-29 02:46:00', 'LST start (hr)': 7.0, 'n_files': 385, 'size_mb': 1.0,
         'pb-id': 'pb-1', 'n_channel': 1, 'Duration (s)': 760.0, 'error': ''}
        for i, st in enumerate(('s8-1', 's9-2'))
    ]
    return add_derived(pd.DataFrame(rows))


def test_sets_page_reads_analyse_workbook(tmp_path):
    xlsx = tmp_path / 'calibration_sets.xlsx'
    write_calibration_report(_sweeps(), str(xlsx))
    rows, details = load_sets(xlsx, tmp_path)

    assert len(rows) == 1
    row = rows[0]
    # Every grid column is filled from the workbook or the row state
    assert {c['field'] for c in set_columns()} <= set(row)
    assert row['folder'] == set_folder(tmp_path, 20260729070, '2026-07-29') == f'{tmp_path}/set20260729070_2026-07-29'
    assert (row['action'], row['qa'], row['status']) == ('convert', '', 'OK')
    assert [o['Station ID'] for o in details[20260729070]] == ['s8-1', 's9-2']

    # Once converted, the row offers a re-convert and shows the set as unreviewed
    (tmp_path / 'set20260729070_2026-07-29').mkdir()
    (tmp_path / 'set20260729070_2026-07-29' / SET_YAML).write_text('set_id: 20260729070\n')
    row = load_sets(xlsx, tmp_path)[0][0]
    from mccs_data_qa.reports.report_ui import set_row_state

    assert set_row_state(row, {'set20260729070_2026-07-29'})['action'] == 'reconvert'
    assert set_row_state(row, {'set20260729070_2026-07-29'})['qa'] == 'unreviewed'


def test_convert_form_is_lossless_unless_switched():
    fields = [f for f in form_fields(build_parser()) if f.dest not in CONVERT_SET_ARGS]
    assert {f.dest for f in fields} == {'n_expected', 'include_failed'}
    values = {'n_expected': '321,385', 'include_failed': False, 'lossy': False, 'step': 0}
    assert convert_argv(fields, values) == []  # A stray step never turns on lossy
    assert convert_argv(fields, values | {'lossy': True}) == ['--lossy=0.5']
    argv = convert_argv(fields, values | {'lossy': True, 'step': 0.25})
    assert build_parser().parse_args(['1', *argv]).quantise == 0.25


def test_n_expected_accepts_comma_separated_list():
    from mccs_data_qa.convert_set import parse_n_expected

    assert parse_n_expected('321,385') == [321, 385]
    assert parse_n_expected(385) == [385]
    assert parse_n_expected(None) == parse_n_expected('') == []
    assert build_parser().parse_args(['1']).n_expected == '321,385'
