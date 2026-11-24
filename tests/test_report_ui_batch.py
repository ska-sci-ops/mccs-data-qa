"""test_report_ui_batch: which sets a batch of one report can be queued on, in mccs_data_qa/reports/report_ui.py."""

import pytest
import yaml


@pytest.fixture
def ui(monkeypatch):
    pytest.importorskip('nicegui')
    pytest.importorskip('mccs_data_qa.reports.report')
    from mccs_data_qa.reports import report_ui

    monkeypatch.setattr(report_ui, 'QUEUE', [])
    monkeypatch.setattr(report_ui, 'RUNS', {})
    return report_ui


def make_set(ui, root, name, cal_runs=(), day='Day'):
    s = root / name
    s.mkdir()
    (s / ui.SET_YAML).write_text(yaml.safe_dump({'set_details': [{'Day/Night (AWST)': day}]}))
    for run in cal_runs:
        d = ui.report_dir(s, 'calibration') / run
        d.mkdir(parents=True)
        (d / 's8-1.cal').touch()
    return s


def test_cal_paths(ui, tmp_path):
    assert ui.cal_paths(tmp_path, ['run1', 'delay_fit/run1']) == [
        tmp_path / 'reports' / 'calibration' / 'run1',
        tmp_path / 'reports' / 'delay_fit' / 'run1',
    ]


def test_batch_state(ui, tmp_path):
    a = make_set(ui, tmp_path, 'a', ['run1'])
    b = make_set(ui, tmp_path, 'b', ['run2'])
    c = make_set(ui, tmp_path, 'c')
    night = make_set(ui, tmp_path, 'night', day='Night')

    assert ui.batch_state(a, 'delay_fit', {}, ['run1']) == ('', False)
    assert ui.batch_state(a, 'delay_fit', {}, []) == ('No calibration run picked.', False)
    assert ui.batch_state(b, 'delay_fit', {}, ['run1']) == ('No calibration run run1.', False)
    assert ui.batch_state(c, 'delay_fit', {}, ['run1'])[0].startswith('Needs a calibration run')
    assert ui.batch_state(night, 'holo_suncal', {}, [])[0].startswith('Night-time data')

    # An existing report is overwritten
    (ui.report_dir(a, 'delay_fit') / 'index.html').parent.mkdir(parents=True)
    (ui.report_dir(a, 'delay_fit') / 'index.html').touch()
    assert ui.batch_state(a, 'delay_fit', {}, ['run1']) == ('', True)

    # A calibration run queued earlier in a batch counts; the run itself can't be queued twice
    job = ui.Job(b, 'calibration', [], 0, ui.report_dir(b, 'calibration') / 'run1')
    ui.QUEUE.append(job)
    assert ui.batch_state(b, 'delay_fit', {}, ['run1']) == ('', False)
    ui.QUEUE.append(ui.Job(a, 'delay_fit', [], 0, ui.run_dir(a, 'delay_fit')))
    assert ui.batch_state(a, 'delay_fit', {}, ['run1']) == ('Already queued or running.', True)
