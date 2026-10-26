"""test_argparse_form: argparse parsers as form fields, for mccs_data_qa/reports/report_ui.py."""

import argparse
import importlib
from pathlib import Path

import pytest
from ska_ost_low_uv.postx.aeps.eep_models import EEP_DIR_ENV

from mccs_data_qa.reports.argparse_form import build_argv, default_values, form_fields, missing_required

SCRIPT_NAMES = ('run_stefcal', 'quicklook_aa1', 'station_sensitivity', 'delay_fit_report', 'holo_suncal')


def _toy_parser():
    p = argparse.ArgumentParser()
    p.add_argument('cal_files', nargs='+', help='Cal files, default %(default)s')
    p.add_argument('--n', type=int, default=4)
    p.add_argument('--x', type=float, default=None)
    p.add_argument('--name', default='a')
    p.add_argument('--sky', choices=['gsm16', 'lfsm'], default='gsm16')
    p.add_argument('--steps', nargs='+', choices=['cal', 'plot'], default=['cal', 'plot'])
    p.add_argument('--uvx', nargs='*', default=[])
    p.add_argument('--no-sun', action='store_true')
    return p


def test_form_fields_kinds():
    """Each argparse argument type maps to its form field kind."""
    fields = {f.dest: f for f in form_fields(_toy_parser())}
    kinds = {k: f.kind for k, f in fields.items()}
    assert kinds == {
        'cal_files': 'list',
        'n': 'number',
        'x': 'number',
        'name': 'text',
        'sky': 'select',
        'steps': 'multiselect',
        'uvx': 'list',
        'no_sun': 'switch',
    }
    assert fields['cal_files'].flag is None and fields['cal_files'].required
    assert fields['n'].flag == '--n' and not fields['n'].required
    assert 'default None' in fields['cal_files'].help


def test_build_argv_defaults_and_changes():
    """Defaults add no options; changed values are emitted and parse back, with quoting."""
    p = _toy_parser()
    fields = form_fields(p)
    values = default_values(fields)
    assert missing_required(fields, values) == ['cal_files']

    values['cal_files'] = 'a.cal "b c.cal"'
    assert build_argv(fields, values) == ['a.cal', 'b c.cal']

    values |= {
        'n': 5.0,
        'x': -1.5,
        'name': '-odd',
        'sky': 'lfsm',
        'steps': ['cal'],
        'uvx': 'u1 u2',
        'no_sun': True,
    }
    argv = build_argv(fields, values)
    assert argv[:2] == ['a.cal', 'b c.cal']
    args = p.parse_args(argv)
    assert (args.n, args.x, args.name, args.sky) == (5, -1.5, '-odd', 'lfsm')
    assert (args.steps, args.uvx, args.no_sun) == (['cal'], ['u1', 'u2'], True)

    values |= {'uvx': ''}
    assert '--uvx' not in build_argv(fields, values)


@pytest.mark.parametrize('name', SCRIPT_NAMES)
def test_real_parsers_roundtrip(name):
    """Default values give the parser's own defaults, and each field set alone parses back to what was set."""
    parser = pytest.importorskip(f'mccs_data_qa.reports.{name}').build_parser()
    fields = form_fields(parser)
    base = default_values(fields)
    required = {f.dest: 'x.cal' for f in fields if f.required}
    base |= required
    expected = vars(parser.parse_args(build_argv(fields, base)))
    for f in fields:
        if f.dest not in required:
            assert expected[f.dest] == f.default, f.dest

    for f in fields:
        values = dict(base)
        if f.kind == 'switch':
            values[f.dest] = not f.default
            want = not f.default
        elif f.kind == 'select':
            want = next(c for c in f.choices if c != f.default)
            values[f.dest] = want
        elif f.kind == 'multiselect':
            want = f.choices[:1]
            values[f.dest] = want
        elif f.kind == 'number':
            want = 7 if f.type is int else 7.5
            values[f.dest] = want
        elif f.kind == 'list':
            want = ['p', 'q']
            values[f.dest] = 'p q'
        else:
            want = 'foo'
            values[f.dest] = want
        args = vars(parser.parse_args(build_argv(fields, values)))
        assert args[f.dest] == want, f.dest
        assert {k: v for k, v in args.items() if k != f.dest} == {
            k: v for k, v in expected.items() if k != f.dest
        }


def test_report_ui_set_argv(tmp_path):
    """Each report parses the set-derived arguments plus its default form, with outputs under reports/."""
    pytest.importorskip('nicegui')
    from mccs_data_qa.reports import report
    from mccs_data_qa.reports import report_ui

    caldir = report_ui.report_dir(tmp_path, 'calibration') / 'run1'
    caldir.mkdir(parents=True)
    (caldir / 's8-1.cal').touch()
    assert {report_ui.report_dir(tmp_path, n).name for n in report_ui.SCRIPTS} == set(report.REPORTS)
    for name, script in report_ui.SCRIPTS.items():
        parser = pytest.importorskip(f'mccs_data_qa.reports.{script}').build_parser()
        fields = [f for f in form_fields(parser) if f.dest not in report_ui.SET_ARGS]
        values = default_values(fields)
        assert not missing_required(fields, values), name
        d = report_ui.run_dir(tmp_path, name, values)
        job = report_ui.Job(tmp_path, name, [], 0, d, [caldir] if name in report_ui.NEEDS_CAL else [])
        args = parser.parse_args(report_ui.set_argv(job) + build_argv(fields, values))
        assert Path(args.outdir) == report_ui.report_dir(tmp_path, name)
        # The run folders the calibration scripts write into
        if name == 'calibration':
            assert d == Path(args.outdir) / 'sun-gsm16_leakage_aep-harp'
        if name == 'holo_suncal':
            assert d == Path(args.outdir) / report.HOLO_RUN
        if name == 'delay_fit':
            assert args.caldir == [str(caldir)]
        if name == 'sensitivity':
            assert Path(args.caldir) == caldir
            # A delay fit run is read with the calibration run of the same name
            fits = report_ui.report_dir(tmp_path, 'delay_fit') / 'run1'
            job = report_ui.Job(tmp_path, name, [], 0, d, [fits])
            args = parser.parse_args(report_ui.set_argv(job) + build_argv(fields, values))
            assert (Path(args.caldir), Path(args.delay_caldir)) == (caldir, fits)


def test_report_ui_queue(tmp_path, monkeypatch):
    """One run at a time; quicklooks jump the queue; reports that read calibration runs wait for them."""
    pytest.importorskip('nicegui')
    from mccs_data_qa.reports import report_ui

    started = []

    def fake_start(job):
        r = report_ui.Run(job.dir, report_ui.SCRIPTS[job.name], job.argv, 0, live=True)
        report_ui.RUNS[r.dir] = r
        started.append((job.set.name, job.name))
        return r

    def finish():
        report_ui.running().live = False
        report_ui.start_queued()

    monkeypatch.setattr(report_ui, 'RUNS', {})
    monkeypatch.setattr(report_ui, 'QUEUE', [])
    monkeypatch.setattr(report_ui, 'start_run', fake_start)
    a, b = tmp_path / 'a', tmp_path / 'b'
    run1, run2 = (report_ui.report_dir(a, 'calibration') / f'run{i}' for i in (1, 2))

    assert report_ui.enqueue(a, 'calibration', [], run1) is not None
    assert report_ui.enqueue(a, 'delay_fit', [], caldirs=[run1]) is None
    assert report_ui.enqueue(b, 'calibration', []) is None
    assert report_ui.enqueue(a, 'quicklook', []) is None
    assert report_ui.enqueue(b, 'quicklook', []) is None
    order = [(j.set.name, j.name) for j in report_ui.QUEUE]
    assert order == [('a', 'quicklook'), ('b', 'quicklook'), ('a', 'delay_fit'), ('b', 'calibration')]
    assert report_ui.queued(report_ui.report_dir(a, 'delay_fit')) == 3

    finish()  # calibration run1 on a, without writing a cal
    finish()  # quicklook on a
    finish()  # quicklook on b; delay_fit on a still has no cal, so calibration on b goes first
    assert started[1:] == [('a', 'quicklook'), ('b', 'quicklook'), ('b', 'calibration')]

    run1.mkdir(parents=True)
    (run1 / 's8-1.cal').touch()
    report_ui.move_job(report_ui.QUEUE[0], 1)
    assert started[-1] == ('b', 'calibration')  # Still running, so nothing else starts
    finish()
    assert started[-1] == ('a', 'delay_fit')
    assert report_ui.QUEUE == []

    # Each waits for the calibration runs it reads, including a Sun holography run
    finish()
    holo = report_ui.run_dir(a, 'holo_suncal')
    assert report_ui.enqueue(a, 'calibration', [], run2) is not None
    report_ui.enqueue(a, 'sensitivity', [], caldirs=[holo])
    report_ui.enqueue(a, 'delay_fit', [], caldirs=[run1, run2])
    report_ui.enqueue(a, 'holo_suncal', [])
    assert report_ui.cal_runs_coming(a) == [run2, holo]
    run2.mkdir()
    (run2 / 's8-1.cal').touch()
    finish()  # calibration run2
    assert started[-1] == ('a', 'delay_fit')
    finish()
    assert started[-1] == ('a', 'holo_suncal')
    holo.mkdir()
    (holo / 's8-1.cal').touch()
    finish()
    assert started[-1] == ('a', 'sensitivity')
    assert set(report_ui.cal_runs(a)) == {run1, run2, holo}


def test_report_ui_delete_cal(tmp_path, monkeypatch):
    """A calibration run in use can't be deleted; deleting one removes its delay fits and rebuilds both pages."""
    pytest.importorskip('nicegui')
    from mccs_data_qa.reports import report_ui

    monkeypatch.setattr(report_ui, 'RUNS', {})
    monkeypatch.setattr(report_ui, 'QUEUE', [])
    cal, fits = report_ui.report_dir(tmp_path, 'calibration'), report_ui.report_dir(tmp_path, 'delay_fit')
    for run in ('run1', 'run2'):
        (cal / run).mkdir(parents=True)
        (cal / run / 's8-1.cal').touch()
        (fits / run).mkdir(parents=True)
        (fits / run / 's8-1_delay.cal').touch()
    sens = report_ui.Job(
        tmp_path, 'sensitivity', [], 0, report_ui.report_dir(tmp_path, 'sensitivity'), [cal / 'run2']
    )
    report_ui.QUEUE.append(sens)
    assert report_ui.cal_in_use(cal / 'run2')
    assert not report_ui.cal_in_use(cal / 'run1')
    report_ui.QUEUE.clear()
    fitting = report_ui.Run(fits, 'delay_fit_report', ['--caldir', str(cal / 'run2')], 0, live=True)
    report_ui.RUNS[fits] = fitting
    assert report_ui.cal_in_use(cal / 'run2')
    fitting.live = False

    report_ui.delete_cal_run(cal / 'run2')
    assert not (cal / 'run2').exists() and not (fits / 'run2').exists()
    assert 'url=run1/index.html' in (cal / 'index.html').read_text()
    assert 'url=run1/index.html' in (fits / 'index.html').read_text()
    report_ui.delete_cal_run(cal / 'run1')
    assert not (cal / 'index.html').exists() and not (fits / 'index.html').exists()


def test_report_ui_unavailable(tmp_path, monkeypatch):
    """Sun holography needs daytime data; delay fit and sensitivity need a calibration run."""
    pytest.importorskip('nicegui')
    from mccs_data_qa.reports import report_ui

    monkeypatch.setattr(report_ui, 'RUNS', {})
    monkeypatch.setattr(report_ui, 'QUEUE', [])
    night, day, unknown = tmp_path / 'night', tmp_path / 'day', tmp_path / 'unknown'
    for d, dn in ((night, 'Night'), (day, 'Day'), (unknown, None)):
        d.mkdir()
        details = [{'Station ID': 'S8-1'} | ({'Day/Night (AWST)': dn} if dn else {})]
        (d / report_ui.SET_YAML).write_text(report_ui.yaml.safe_dump({'set_details': details}))

    assert report_ui.unavailable(night, 'holo_suncal')
    assert not report_ui.unavailable(day, 'holo_suncal')
    assert not report_ui.unavailable(unknown, 'holo_suncal')
    assert report_ui.unavailable(day, 'delay_fit')
    report_ui.QUEUE.append(
        report_ui.Job(day, 'calibration', [], 0, report_ui.report_dir(day, 'calibration') / 'run1')
    )
    assert not report_ui.unavailable(day, 'delay_fit')
    assert not report_ui.unavailable(night, 'quicklook')

    stefcal = {'cal_model': 'sun+sky', 'leakage': 'solve', 'beam': 'aep-harp', 'sky_model': 'gsm16'}
    assert not report_ui.unavailable(night, 'calibration', stefcal)
    assert report_ui.unavailable(day, 'calibration', stefcal | {'leakage': 'none'})
    assert report_ui.unavailable(night, 'calibration', stefcal | {'cal_model': 'sun'})
    assert not report_ui.unavailable(day, 'calibration', stefcal | {'cal_model': 'sun', 'leakage': 'none'})
    assert report_ui.unavailable(night, 'quicklook', stefcal | {'cal_model': 'sun'})
    assert 'Sun is up' in report_ui.cal_warning(day, stefcal | {'cal_model': 'sky'})
    assert not report_ui.cal_warning(night, stefcal | {'cal_model': 'sky'})
    assert not report_ui.cal_warning(day, stefcal)

    import h5py

    for d, n_t in ((night, 1), (day, 2)):
        with h5py.File(d / 'x_sweep.uvx', 'w') as h:
            h['visibilities/coords/time/unix'] = range(n_t)
    report_ui.n_integrations.cache_clear()
    assert 'integration' in report_ui.unavailable(night, 'sensitivity')
    assert not report_ui.unavailable(day, 'sensitivity')


def test_report_ui_qa(tmp_path, monkeypatch):
    """Station QA marks set the run's worst mark and go stale on a re-run; the set has its own mark."""
    pytest.importorskip('nicegui')
    from mccs_data_qa.reports import report_ui

    monkeypatch.setattr(report_ui, 'RUNS', {})
    monkeypatch.setitem(report_ui.STATE, 'initials', 'ab')
    quick = report_ui.report_dir(tmp_path, 'quicklook')
    cal = report_ui.report_dir(tmp_path, 'calibration')
    assert report_ui.qa_choices(quick) == report_ui.STATION_QA
    assert report_ui.qa_choices(cal / 'run1') == report_ui.CAL_QA
    assert report_ui.qa_choices(cal) == report_ui.qa_choices(report_ui.report_dir(tmp_path, 'delay_fit')) == ()

    quick.mkdir(parents=True)
    (quick / 'index.html').touch()
    report_ui.Run(quick, 'x', [], 100.0, returncode=0).save()
    assert report_ui.run_qa(quick) == 'unreviewed'
    report_ui.save_station_qa(quick, 's8-1', 'healthy', '')
    assert report_ui.run_qa(quick) == 'healthy'
    report_ui.save_station_qa(quick, 's8-2', 'fail', 'dead LNA')
    report_ui.save_station_qa(quick, 's8-3', 'degraded', '')
    assert report_ui.run_qa(quick) == 'fail'
    report_ui.save_station_qa(quick, 's8-2', None, 'dead LNA')  # Cleared
    assert report_ui.run_qa(quick) == 'degraded'
    assert (
        report_ui.station_marks(quick)['s8-3'] | {'status': 'degraded', 'by': 'ab'}
        == (report_ui.station_marks(quick)['s8-3'])
    )

    report_ui.Run(quick, 'x', [], 200.0, returncode=0).save()  # Re-run
    assert report_ui.run_qa(quick) == 'stale'
    assert report_ui.station_marks(quick)['s8-2'] == {'note': 'dead LNA'}
    report_ui.save_station_qa(quick, 's8-1', 'healthy', '')
    assert report_ui.run_qa(quick) == 'healthy'  # Earlier marks dropped, notes kept
    assert report_ui.station_marks(quick)['s8-3'] == {'note': ''}

    # Auto marks from flagged antennas, keeping marks made on this run
    assert [report_ui.flag_qa(n) for n in (9, 10, 20, 21)] == [
        'healthy',
        'mostly_healthy',
        'mostly_healthy',
        'degraded',
    ]
    report_ui.auto_station_qa(quick)  # No flags.json: unchanged
    assert report_ui.run_qa(quick) == 'healthy'
    (quick / 'flags.json').write_text('{"s8-1": 30, "s8-2": 5, "s8-3": 15}')
    report_ui.auto_station_qa(quick)
    marks = report_ui.station_marks(quick)
    assert marks['s8-1']['by'] == 'ab' and marks['s8-1']['status'] == 'healthy'
    assert marks['s8-2'] | {'status': 'healthy', 'note': 'Flagged antennas: 5', 'by': 'auto'} == marks['s8-2']
    assert marks['s8-3']['status'] == 'mostly_healthy'
    assert report_ui.run_qa(quick) == 'mostly_healthy'
    report_ui.save_station_qa(quick, 's8-3', 'fail', 'override')
    assert report_ui.run_qa(quick) == 'fail'

    # A calibration run is degraded while over half its stations pass
    cal = report_ui.report_dir(tmp_path, 'calibration') / 'run1'
    cal.mkdir(parents=True)
    report_ui.Run(cal, 'x', [], 100.0, returncode=0).save()
    for st, status in (('s8-1', 'pass'), ('s8-2', 'pass'), ('s8-3', 'fail')):
        report_ui.save_station_qa(cal, st, status, '')
    assert report_ui.run_qa(cal) == 'degraded'
    assert report_ui.qa_label(cal, 'degraded') == 'QA degraded (1/3)'
    report_ui.save_station_qa(cal, 's8-4', 'fail', '')
    assert report_ui.run_qa(cal) == 'fail'  # Half pass
    report_ui.save_station_qa(cal, 's8-3', 'pass', '')
    report_ui.save_station_qa(cal, 's8-4', 'pass', '')
    assert report_ui.run_qa(cal) == 'pass'
    assert report_ui.qa_label(quick, 'mostly_healthy') == 'QA mostly healthy'

    assert report_ui.set_qa(tmp_path) == 'unreviewed'
    report_ui.save_set_qa(tmp_path, 'mostly_healthy', 's8-2 down')
    assert report_ui.set_qa(tmp_path) == 'mostly_healthy'
    assert report_ui.read_json(report_ui.set_qa_path(tmp_path))['by'] == 'ab'

    stations = [f's8-{i}' for i in range(1, 11)]
    filelist = [{'station_id': st} for st in ['s10-1', *stations]]
    (tmp_path / report_ui.SET_YAML).write_text(report_ui.yaml.safe_dump({'filelist': filelist}))
    assert report_ui.set_stations(tmp_path) == [*stations, 's10-1']  # s8 before s10


def test_report_ui_set_runs(tmp_path, monkeypatch):
    """The run picker lists the calibration comparison page and runs, then reports with a page or a log."""
    pytest.importorskip('nicegui')
    from mccs_data_qa.reports import report_ui

    monkeypatch.setattr(report_ui, 'RUNS', {})
    stefcal = report_ui.report_dir(tmp_path, 'calibration')
    (stefcal / 'run1').mkdir(parents=True)
    (stefcal / 'index.html').touch()
    (stefcal / 'run1' / 'index.html').touch()
    report_ui.report_dir(tmp_path, 'quicklook').mkdir()
    (report_ui.report_dir(tmp_path, 'quicklook') / 'run.json').touch()
    report_ui.report_dir(tmp_path, 'delay_fit').mkdir()

    assert report_ui.set_runs(tmp_path) == [
        stefcal,
        stefcal / 'run1',
        report_ui.report_dir(tmp_path, 'quicklook'),
    ]
    assert report_ui.set_of(stefcal / 'run1') == tmp_path
    assert report_ui.run_label(stefcal / 'run1') == f'{tmp_path.name} / calibration/run1'


def test_report_beams(tmp_path, monkeypatch):
    """--beam offers each packaged AEP and each EEP file in $POSTX_EEP_DIR, named by model."""
    from mccs_data_qa.reports import report

    eep_dir = tmp_path / 'eeps'
    eep_dir.mkdir()
    for name in ('ska_low_feko', 'ska_low_harp'):
        (eep_dir / f'{name}.h5').touch()
    try:
        monkeypatch.setenv(EEP_DIR_ENV, str(eep_dir))
        importlib.reload(report)  # BEAMS is read from $POSTX_EEP_DIR on import
        assert list(report.BEAMS) == ['aep-feko', 'aep-harp', 'eep-feko', 'eep-harp']
        assert report.BEAMS['aep-harp'] == {'aep': 'ska_low_harp'}
        assert report.BEAMS['eep-feko'] == {'eep': 'ska_low_feko'}
        assert not report.cal_args_problem('sky', 'solve', 'aep-feko')
        assert not report.cal_args_problem('sky', 'solve', 'eep-feko')
        monkeypatch.setenv(EEP_DIR_ENV, str(tmp_path))
        assert 'ska_low_feko' in report.cal_args_problem('sky', 'solve', 'eep-feko')

        monkeypatch.delenv(EEP_DIR_ENV)
        importlib.reload(report)
        assert list(report.BEAMS) == ['aep-feko', 'aep-harp']
    finally:
        monkeypatch.undo()
        importlib.reload(report)


def test_report_ui_queue_guards(tmp_path, monkeypatch):
    """Stale clicks on finished runs and departed jobs are no-ops; a bad run.json loads as no run."""
    pytest.importorskip('nicegui')
    import asyncio

    from mccs_data_qa.reports import report_ui

    monkeypatch.setattr(report_ui, 'RUNS', {})
    monkeypatch.setattr(report_ui, 'QUEUE', [])
    done = report_ui.Run(tmp_path, 'quicklook_aa1', [], 0, returncode=0)
    asyncio.run(report_ui.cancel_run(done))
    assert done.status == 'done'

    job = report_ui.Job(tmp_path, 'quicklook', [], 0, report_ui.report_dir(tmp_path, 'quicklook'))
    report_ui.move_job(job, 1)
    report_ui.remove_job(job)

    (tmp_path / 'run.json').write_text('')
    assert report_ui.Run.load(tmp_path) is None


def test_report_ui_cal_ready(tmp_path, monkeypatch):
    """Sensitivity on a delay fit run waits for a queued re-run of its calibration run or delay fit."""
    pytest.importorskip('nicegui')
    from mccs_data_qa.reports import report_ui

    monkeypatch.setattr(report_ui, 'RUNS', {})
    monkeypatch.setattr(report_ui, 'QUEUE', [])
    cal = report_ui.report_dir(tmp_path, 'calibration') / 'run1'
    fits = report_ui.report_dir(tmp_path, 'delay_fit') / 'run1'
    for d, fn in ((cal, 's8-1.cal'), (fits, 's8-1_delay.cal')):
        d.mkdir(parents=True)
        (d / fn).touch()
    sens = report_ui.Job(tmp_path, 'sensitivity', [], 0, report_ui.report_dir(tmp_path, 'sensitivity'), [fits])
    assert report_ui.cal_ready(sens)
    report_ui.QUEUE.append(report_ui.Job(tmp_path, 'calibration', [], 0, cal))
    assert not report_ui.cal_ready(sens)
    report_ui.QUEUE[:] = [
        report_ui.Job(tmp_path, 'delay_fit', [], 0, report_ui.report_dir(tmp_path, 'delay_fit'), [cal])
    ]
    assert not report_ui.cal_ready(sens)
