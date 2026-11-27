"""report_ui.py: Web UI for running the report scripts on calibration sets.

Browse to a folder of calibration sets (subfolders holding `calibration_set.yaml` and one UVX sweep per
station), pick a set to see its yaml, and run reports on it. Each report is written to
`<set>/reports/<name>/`, along with its log. Each calibration run (a stefcal option combination, or
Sun holography) goes in its own folder `<set>/reports/calibration/<run>/`, with the `{station}.cal` files
that the delay fit and sensitivity reports read; `reports/calibration/index.html` compares the runs. Delay
fit and sensitivity read the runs picked on their cards. Runs one report at a time; others wait in a queue
shared across sets, managed from the Queue button in the header. Quicklooks go to the front of the queue.

The Calibration browser page (`sets/`) lists every set in `<db>/calibration_sets.xlsx`, converts a set into
`--root` with mccs-convert-set, and updates the database (mccs-spider, then mccs-analyse). These run in their own
queue, one at a time, alongside the reports.

Usage:
    mccs-report-ui [--root DIR] [--db db] [--daq-dir DIR] [--port 8080] # then open http://127.0.0.1:8080

In JupyterHub (needs jupyter-server-proxy), open <hub>/user/<name>/proxy/8080/ instead.
"""

import argparse
import asyncio
import codecs
import contextlib
import functools
import importlib
import json
import os
import re
import shlex
import shutil
import signal
import sys
import time
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from urllib.parse import quote

import h5py
import openpyxl  # noqa: F401  # import eagerly: pandas' lazy import from concurrent io_bound threads races ("partially initialized module")
import pandas  # noqa: F401  # same reason
import psutil
import yaml
from mccs_data_qa.reports.argparse_form import Field, build_argv, default_values, form_fields, missing_required
from fastapi import HTTPException
from fastapi.responses import FileResponse
from loguru import logger
from nicegui import app, background_tasks, run, ui

# The report scripts are imported here to rebuild comparison pages, which only save plots
os.environ.setdefault('MPLBACKEND', 'Agg')
HERE = Path(__file__).parent
DEFAULT_ROOT = Path.cwd()
SET_YAML = 'calibration_set.yaml'
# Report name: script, in card order
SCRIPTS = {
    'quicklook': 'quicklook_aa1',
    'calibration': 'run_stefcal',
    'delay_fit': 'delay_fit_report',
    'holo_suncal': 'holo_suncal',
    'sensitivity': 'station_sensitivity',
    'diagnose': 'diagnose',
}
CALIBRATION = ('calibration', 'holo_suncal')  # Write a calibration run folder in reports/calibration/
NEEDS_CAL = ('delay_fit', 'sensitivity')  # Read the {station}.cal files of calibration runs
OPTIONAL_CAL = ('diagnose',)  # Can read a calibration run, if one is picked
MULTI_CAL = ('delay_fit',)  # Reads several calibration runs, and compares them
DELAY_CAL = ('sensitivity',)  # Can also read a delay fit run, with its calibration run's amplitudes
INDENTED = ('delay_fit',)  # Card shown under the Calibration card
PRIORITY = ('quicklook',)  # Queued ahead of the other reports
DAY_ONLY = ('holo_suncal',)  # Need the Sun up
TWO_INTEGRATIONS = ('sensitivity',)  # Need two integrations per channel, for difference images
# Report name: (icon, title, one-line description) for its card
INFO = {
    'quicklook': ('visibility', 'Quicklook', 'Calibrates one channel; all-sky images and station comparison.'),
    'calibration': ('tune', 'Calibration', 'Stefcal on every channel; writes the {station}.cal files.'),
    'delay_fit': ('timeline', 'Delay fit', 'Fits delays to the gain phases of the calibration runs.'),
    'holo_suncal': (
        'wb_sunny',
        'Sun holography',
        'Self-holography on the Sun; a run in the calibration summary.',
    ),
    'sensitivity': ('speed', 'Sensitivity', 'SEFD and A/T from difference images; needs two integrations.'),
    'diagnose': (
        'troubleshoot',
        'Diagnose',
        'Checks every antenna for faults and lists the suspects; a calibration run adds delay checks.',
    ),
}
# Status: badge colour
STATUS_COLOR = {'done': 'positive', 'running': 'primary', 'queued': 'orange', 'cancelled': 'grey'}
SET_ARGS = ('datadir', 'eb', 'outdir', 'caldir', 'delay_caldir')  # Filled in from the selected set
HIDDEN_ARGS = {'run_stefcal': ('reuse_cal',)}  # Not in a script's form: a re-run replaces its cal files
# Form fields shown only for these --cal-model values
CAL_MODEL_ONLY = {
    'sky_model': ('sun+sky', 'sky'),
    'sun_fwhm': ('sun+sky',),
    'n_flux_iter': ('sun+sky',),
    'min_gsm_snr': ('sun+sky',),
    'min_sun_alt': ('sun+sky',),
    # The MCCS solver has its own antenna flagging, beam (EEPs) and sky model; only mccs-gsm has an NSIDE
    'sigma_thr': ('sun+sky', 'sky', 'sun'),
    'beam': ('sun+sky', 'sky', 'sun'),
    'nside': ('sun+sky', 'sky', 'sun', 'mccs-gsm'),
    'mccs_eep': ('mccs-gsm', 'mccs-sun'),  # mccs-sunfast has no EEPs
}
# Form select labels, by field and choice
CHOICE_LABELS = {
    'cal_model': {
        'sun+sky': 'sun (extended) + sky',
        'sun': 'sun (point source)',
        'mccs-gsm': 'MCCS GSM (GSM + point Sun, EEPs)',
        'mccs-sun': 'MCCS SUN (point Sun, EEPs)',
        'mccs-sunfast': 'MCCS SUN_FAST (point Sun, no EEPs)',
    },
    'mccs_eep': {'regular': 'regular (npz, $EEP_PATH)'},
}
ANSI = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')
LOG_TAIL = 3000  # Lines of a finished run's log shown
N_RECENT = 20  # Finished runs listed in the Queue menu
SET_QA = ('healthy', 'mostly_healthy', 'degraded', 'fail')
STATION_QA = ('healthy', 'mostly_healthy', 'degraded', 'fail')  # Per station, in a quicklook
CAL_QA = ('pass', 'fail')  # Per station, in a calibration run
# QA mark: colour, Pantone 346 C, 374 C, 393 C, 178 C and 428 C, in order of severity. Stale means the mark
# is for an earlier run.
QA_HEX = {
    'pass': '#71CC98',
    'healthy': '#71CC98',
    'mostly_healthy': '#C5E86C',
    'degraded': '#F0EC74',
    'fail': '#FF585D',
    'unreviewed': '#C1C6C8',
    'stale': '#C1C6C8',
}
QA_ICON = {
    'pass': 'check_circle',
    'healthy': 'check_circle',
    'mostly_healthy': 'check_circle_outline',
    'degraded': 'warning',
    'fail': 'cancel',
}
# SKAO brand: Blueshift navy and Redshift magenta, in the shades of the SKAO slide template, with the
# Noto Sans family (Light for headings, Regular for small text, Mono for technical text, Verdana fallback).
SKAO_COLORS = {'primary': '#0A1D45', 'secondary': '#AF0B62', 'accent': '#E20E7F', 'dark': '#0A1D45'}
SKAO_HEAD = """<script>
// Report URLs are page-relative ('files/...'), so the page must end in '/': opened at <hub>/proxy/8080
// they would resolve to <hub>/proxy/files/... and 404. Add the slash before anything loads.
if (!location.pathname.endsWith('/')) location.replace(location.pathname + '/' + location.search + location.hash);
</script>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Noto+Sans:ital,wght@0,300;0,400;0,600;0,700;1,400&family=Noto+Sans+Mono&display=swap">
<style>
body, .q-field, .q-btn, .text-h6, .text-subtitle1, .text-caption {
  font-family: 'Noto Sans', Verdana, sans-serif; }
.font-mono, .nicegui-log { font-family: 'Noto Sans Mono', monospace; }
body { background: #F6F7FA; color: #0A1D45; }
.text-h6 { font-weight: 300; letter-spacing: 0; }
.text-grey-7 { color: #5B6478 !important; }
.eyebrow { text-transform: uppercase; letter-spacing: .12em; font-size: .72rem; font-weight: 600;
  color: #AF0B62; }
.q-header { background: linear-gradient(100deg, #0A1D45 45%, #761A63 80%, #E20E7F); }
.q-card--bordered, .q-tab-panels { border: 1px solid #E3E6EE; border-radius: 8px; }
.skao-selected { background: #EEF0F6; box-shadow: inset 3px 0 #AF0B62; }
.skao-logo svg { display: block; width: 100%; height: auto; }
.q-table th { font-weight: 700; }
.skao-needs-cal { border-left: 3px solid #C9CEDB; }
</style>"""


@dataclass
class Run:
    """A script run, live or loaded from its report folder."""

    dir: Path
    script: str
    argv: list[str]
    started: float
    live: bool = False
    lines: list[str] = field(default_factory=list)
    progress: str = ''
    returncode: int | None = None
    cancelled: bool = False
    finished: float | None = None
    proc: asyncio.subprocess.Process | None = None
    rss: int = 0  # Memory of the script and the processes it started, bytes, while running
    peak_rss: int = 0

    @property
    def status(self) -> str:
        """One of running, done, failed (code), cancelled, interrupted."""
        if self.returncode is None:
            return 'running' if self.live else 'interrupted'
        if self.cancelled:
            return 'cancelled'
        return 'done' if self.returncode == 0 else f'failed ({self.returncode})'

    @property
    def elapsed(self) -> str:
        """Run time so far, or in total once finished, as h:mm:ss; '' if unknown."""
        if not (self.live or self.finished):
            return ''
        return str(timedelta(seconds=int((self.finished or time.time()) - self.started)))

    @property
    def module(self) -> str:
        """Module run with `python -m`: a report script's name, or a dotted module (convert, spider)."""
        return self.script if '.' in self.script else f'mccs_data_qa.reports.{self.script}'

    @property
    def command(self) -> str:
        """The command line, for display."""
        return shlex.join(['python', '-m', self.module, *self.argv])

    def report(self) -> Path | None:
        """The report page, once the script has written it."""
        path = self.dir / 'index.html'
        return path if path.exists() else None

    def save(self):
        """Write run.json."""
        meta = {
            'script': self.script,
            'argv': self.argv,
            'started': self.started,
            'returncode': self.returncode,
            'cancelled': self.cancelled,
            'finished': self.finished,
            'peak_rss': self.peak_rss,
        }
        (self.dir / 'run.json').write_text(json.dumps(meta))

    @classmethod
    def load(cls, path: Path) -> 'Run | None':
        """Load a run from its report folder, or None if the UI never ran it or its run.json is unreadable."""
        if not (path / 'run.json').exists():
            return None
        try:
            meta = json.loads((path / 'run.json').read_text())
            log = path / 'run.log'
            lines = log.read_text(errors='replace').splitlines()[-LOG_TAIL:] if log.exists() else []
            return cls(
                path,
                meta['script'],
                meta['argv'],
                meta['started'],
                lines=lines,
                returncode=meta['returncode'],
                cancelled=meta.get('cancelled', False),
                finished=meta.get('finished'),
                peak_rss=meta.get('peak_rss', 0),
            )
        except (OSError, ValueError, KeyError) as err:
            logger.warning(f'Cannot read {path / "run.json"}: {err!r}')
            return None


@dataclass
class Job:
    """A report waiting in the queue, with its options as they were when queued."""

    set: Path
    name: str
    argv: list[str]
    queued: float
    dir: Path  # Run folder
    caldirs: list[Path] = field(default_factory=list)  # Calibration run folders read


RUNS: dict[Path, Run] = {}  # Runs started since the UI started, by report folder
QUEUE: list[Job] = []  # In memory, so lost if the UI restarts
# Convert and database-update jobs: their own queue, one at a time, alongside the reports
TOOL_RUNS: dict[Path, Run] = {}
TOOL_QUEUE: list[Run] = []  # Not started yet
FORM_VALUES: dict[str, dict] = {}
PARSERS: dict = {}
# Browsed folder and selected set, kept across page reloads; db and DAQ folders for the sets page
STATE = {'dir': DEFAULT_ROOT, 'set': None, 'root': DEFAULT_ROOT, 'db': Path('db'), 'daq_dir': Path('/home/jovyan/daq-data')}


def reviewer() -> str:
    """Reviewer initials of this browser, recorded with QA marks."""
    return app.storage.user.get('initials', '')


def by_line(mark: dict) -> str:
    """'by DP, 2 h ago' for a QA mark, or ''."""
    if not mark.get('updated'):
        return ''
    return f'by {mark.get("by") or "?"}, {ago(mark["updated"])}'


def url(path: Path) -> str:
    """URL of a file in a reports folder, versioned by its modification time so a rewrite is never cached.

    Relative to the page (served only at '/'), so it resolves under any proxy prefix, e.g. JupyterHub's
    <hub>/user/<name>/proxy/<port>/, without needing the proxy to send X-Forwarded-Prefix.
    """
    return f'files{quote(path.resolve().as_posix())}?v={path.stat().st_mtime_ns}'


@app.api_route('/files/{path:path}', methods=['GET', 'HEAD'])
def serve_file(path: str):
    """Serve files inside a set's `reports` folder, and nothing else on disk."""
    p = Path('/', path).resolve()
    reports = next((q for q in p.parents if q.name == 'reports'), None)
    if reports is None or not (reports.parent / SET_YAML).exists() or not p.is_file():
        raise HTTPException(404)
    # Reports are rewritten in place on a re-run, so the browser must revalidate
    return FileResponse(p, headers={'Cache-Control': 'no-cache'})


def report_dir(setdir: Path, name: str) -> Path:
    """Output folder of a report on a set: `reports/calibration` for the calibrations, else `reports/<name>`."""
    return setdir / 'reports' / ('calibration' if name in CALIBRATION else name)


def run_dir(setdir: Path, name: str, values: dict | None = None) -> Path:
    """Folder of a run of a report: `reports/calibration/<run>` for a calibration, else `reports/<name>`.

    A stefcal run is named from its form values; without them, this is the calibration folder.
    """
    from mccs_data_qa.reports.report import HOLO_RUN, cal_variant  # Imported with the scripts, after startup

    if name == 'holo_suncal':
        return report_dir(setdir, name) / HOLO_RUN
    if name == 'calibration' and values:
        run = cal_variant(
            values['cal_model'],
            values['leakage'],
            values['beam'],
            values['sky_model'],
            values['nside'],
            values.get('mccs_eep', 'regular'),
        )
        return report_dir(setdir, name) / run
    return report_dir(setdir, name)


def set_of(d: Path) -> Path:
    """Set folder of a run folder."""
    return next(p for p in d.parents if p.name == 'reports').parent


def run_name(d: Path) -> str:
    """'<report>[/<run>]' for a run folder."""
    return d.relative_to(set_of(d) / 'reports').as_posix()


def run_label(d: Path) -> str:
    """'<set> / <report>[/<run>]' for a run folder."""
    return f'{set_of(d).name} / {run_name(d)}'


def cal_runs(setdir: Path) -> list[Path]:
    """Calibration run folders of a set that hold cal files, newest first."""
    runs = [d for d in report_dir(setdir, 'calibration').glob('*/') if any(d.glob('*.cal'))]
    return sorted(runs, key=lambda d: d.stat().st_mtime, reverse=True)


def set_runs(setdir: Path) -> list[Path]:
    """Run folders of a set the UI can show: the calibration comparison page and runs, then the other reports."""
    cal = report_dir(setdir, 'calibration')
    dirs = [cal, *sorted(cal.glob('*/'))] + [report_dir(setdir, n) for n in SCRIPTS if n not in CALIBRATION]
    return [d for d in dirs if (d / 'index.html').exists() or (d / 'run.json').exists() or d in RUNS]


def read_json(path: Path) -> dict:
    """A JSON file's contents, or {} if it doesn't exist."""
    return json.loads(path.read_text()) if path.exists() else {}


def started_at(d: Path) -> float:
    """Start time of the run in a folder, or its report's time if written outside the UI."""
    r = RUNS.get(d)
    if r:
        return r.started
    meta = read_json(d / 'run.json')
    return meta['started'] if 'started' in meta else (d / 'index.html').stat().st_mtime


def qa_color(qa: str) -> str:
    """Colour name of a QA mark, as registered with ui.colors()."""
    return f'qa-{qa.replace("_", "-")}'


def run_qa(d: Path) -> str:
    """QA mark of the run in a folder: its worst station mark, 'stale' if for an earlier run, or 'unreviewed'."""
    qa = read_json(d / 'qa.json')
    if not qa.get('status'):
        return 'unreviewed'
    return qa['status'] if qa.get('run_started') == started_at(d) else 'stale'


def qa_choices(d: Path) -> tuple:
    """Station QA marks of the run in a folder: STATION_QA for a quicklook, CAL_QA for a calibration run, else ()."""
    s = set_of(d)
    if d == report_dir(s, 'quicklook'):
        return STATION_QA
    return CAL_QA if d.parent == report_dir(s, 'calibration') else ()


def qa_label(d: Path, qa: str) -> str:
    """Chip text of a run's QA mark, with the failed stations of a degraded calibration run, e.g. (2/14)."""
    text = f'QA {qa.replace("_", " ")}'
    if qa == 'degraded' and qa_choices(d) == CAL_QA:
        status = [m.get('status') for m in station_marks(d).values() if m.get('status')]
        text += f' ({status.count("fail")}/{len(status)})'
    return text


def station_marks(d: Path) -> dict:
    """Station QA marks of the run in a folder, {station_id: {status, note, by, updated}}.

    Marks made on an earlier run keep only their note.
    """
    marks = read_json(d / 'qa.json').get('stations', {})
    if run_qa(d) == 'stale':
        return {st: {'note': m.get('note', '')} for st, m in marks.items()}
    return marks


def write_station_marks(d: Path, marks: dict, by: str):
    """Write the station QA marks of the run in a folder, and the run's status from them.

    A calibration run passes if every marked station passes, is degraded if over half pass, and else fails.
    A quicklook takes its worst station mark.
    """
    status = [m['status'] for m in marks.values() if m.get('status')]
    worst = max(status, key=list(QA_HEX).index, default=None)
    if worst == 'fail' and qa_choices(d) == CAL_QA and status.count('pass') > len(status) / 2:
        worst = 'degraded'
    qa = {'status': worst, 'by': by, 'updated': time.time(), 'run_started': started_at(d)}
    (d / 'qa.json').write_text(json.dumps(qa | {'stations': marks}, indent=1))


def save_station_qa(d: Path, station: str, status: str | None, note: str):
    """Write the QA mark and note of a station in the run in a folder."""
    marks = station_marks(d)
    marks[station] = {'status': status, 'note': note, 'by': reviewer(), 'updated': time.time()}
    write_station_marks(d, marks, reviewer())


def flag_qa(n_flagged: int) -> str:
    """Station QA mark from its number of flagged antennas: healthy under 10, degraded over 20."""
    if n_flagged < 10:
        return 'healthy'
    return 'mostly_healthy' if n_flagged <= 20 else 'degraded'


def auto_station_qa(d: Path):
    """Mark the stations of a quicklook from their flagged antennas in its flags.json.

    Stations already marked on this run keep their mark.
    """
    flags = read_json(d / 'flags.json')
    if not flags:
        return
    marks = station_marks(d)
    for st, n in flags.items():
        if not marks.get(st, {}).get('status'):
            marks[st] = {
                'status': flag_qa(n),
                'note': f'Flagged antennas: {n}',
                'by': 'auto',
                'updated': time.time(),
            }
    write_station_marks(d, marks, 'auto')


def set_stations(setdir: Path) -> list[str]:
    """Station IDs in a set's file list, s8 before s9 before s10."""
    from mccs_data_qa.reports.report import station_key

    files = read_set(setdir).get('filelist') or []
    return sorted({str(f['station_id']) for f in files if f.get('station_id')}, key=station_key)


def set_qa_path(setdir: Path) -> Path:
    """File of a set's QA mark, {status, note, by, updated}."""
    return setdir / 'reports' / 'set_qa.json'


def set_qa(setdir: Path) -> str:
    """QA mark of a set: one of SET_QA, or 'unreviewed'."""
    return read_json(set_qa_path(setdir)).get('status') or 'unreviewed'


def save_set_qa(setdir: Path, status: str | None, note: str):
    """Write the QA mark and note of a set."""
    path = set_qa_path(setdir)
    path.parent.mkdir(parents=True, exist_ok=True)
    mark = {'status': status, 'note': note, 'by': reviewer(), 'updated': time.time()}
    path.write_text(json.dumps(mark, indent=1))


def set_argv(job: Job) -> list[str]:
    """Arguments that point a script at the set's files, its report folder, and the calibration runs it reads."""
    args = {'datadir': job.set, 'eb': '*', 'outdir': report_dir(job.set, job.name)}
    argv = [f'--{k}={v}' for k, v in args.items()]
    if not job.caldirs:
        return argv
    # A delay fit run is read with the calibration run of the same name, for its amplitudes
    delay = [d for d in job.caldirs if d.parent.name == 'delay_fit']
    cals = [report_dir(job.set, 'calibration') / d.name if d in delay else d for d in job.caldirs]
    return argv + ['--caldir', *map(str, cals)] + (['--delay-caldir', *map(str, delay)] if delay else [])


@functools.lru_cache
def n_integrations(setdir: Path) -> int | None:
    """Integrations per channel in the set's first UVX file, or None if it can't be read."""
    try:
        with h5py.File(sorted(setdir.glob('*.uvx'))[0], 'r') as h:
            return len(h['visibilities/coords/time/unix'])
    except (IndexError, OSError, KeyError):
        return None


@functools.lru_cache
def is_daytime(setdir: Path) -> bool | None:
    """Whether the set was observed in the day, from its yaml, or None if the yaml doesn't say."""
    days = {d.get('Day/Night (AWST)') for d in read_set(setdir).get('set_details') or []} - {None}
    return 'Day' in days if days else None


def unavailable(setdir: Path, name: str, values: dict | None = None) -> str:
    """Why a report can't be run or queued on a set with these form values, or '' if it can."""
    n = n_integrations(setdir) if name in TWO_INTEGRATIONS else None
    if n is not None and n < 2:
        return 'One integration per channel: needs two for difference images.'
    if name in NEEDS_CAL and not cal_coming(setdir):
        return 'Needs a calibration run: run Calibration or Sun holography first.'
    if name in DAY_ONLY and is_daytime(setdir) is False:
        return 'Night-time data: needs the Sun up.'
    if values and 'cal_model' in values:
        from mccs_data_qa.reports.report import cal_args_problem

        if values['cal_model'] in ('sun', 'mccs-sun', 'mccs-sunfast') and is_daytime(setdir) is False:
            return 'Night-time data: a Sun-only cal needs the Sun up.'
        return cal_args_problem(
            values['cal_model'], values['leakage'], values['beam'], values.get('mccs_eep', 'regular')
        )
    return ''


def cal_warning(setdir: Path, values: dict) -> str:
    """A warning about form values that can run on a set but probably shouldn't, or ''."""
    if values.get('cal_model') == 'sky' and is_daytime(setdir):
        return 'Daytime data: the Sun is up but not in the sky model. sun (extended) + sky includes it.'
    return ''


def ago(t: float) -> str:
    """Time since t, e.g. '5 min ago'."""
    m = (time.time() - t) / 60
    if m < 1:
        return 'just now'
    if m < 60:
        return f'{m:.0f} min ago'
    return f'{m / 60:.0f} h ago' if m < 48 * 60 else f'{m / 1440:.0f} d ago'


def read_set(setdir: Path) -> dict:
    """The set's calibration_set.yaml."""
    return yaml.safe_load((setdir / SET_YAML).read_text()) or {}


def load_parser(script: str):
    """Import a script and cache its argument parser. Takes a few seconds, so call from a thread."""
    PARSERS[script] = importlib.import_module(f'mccs_data_qa.reports.{script}').build_parser()


async def preload_parsers():
    """Import every script at startup, so the forms appear without waiting. A failed import is logged."""
    for script in SCRIPTS.values():
        try:
            await run.io_bound(load_parser, script)
        except Exception:
            logger.exception(f'Cannot load the options of {script}')


def form_state(script: str) -> tuple[list[Field], dict]:
    """Form fields of a script, without the set-derived ones, and their values."""
    hidden = SET_ARGS + HIDDEN_ARGS.get(script, ())
    fields = [f for f in form_fields(PARSERS[script]) if f.dest not in hidden]
    return fields, FORM_VALUES.setdefault(script, default_values(fields))


async def _pump(r: Run, stream: asyncio.StreamReader, log):
    """Copy a subprocess's output into the run's lines and log file, splitting tqdm redraws from lines."""
    buf = ''
    # Incremental, so a multibyte character split across reads (e.g. a tqdm bar) decodes whole
    decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
    while chunk := await stream.read(4096):
        buf += decoder.decode(chunk)
        *done, buf = buf.split('\n')
        for line in done:
            line = ANSI.sub('', line).rsplit('\r', 1)[-1]
            r.lines.append(line)
            log.write(line + '\n')
        log.flush()
        r.progress = ANSI.sub('', buf).rsplit('\r', 1)[-1] if '\r' in buf else ''
    if buf := buf + decoder.decode(b'', final=True):
        line = ANSI.sub('', buf).rsplit('\r', 1)[-1]
        r.lines.append(line)
        log.write(line + '\n')
    r.progress = ''


def gib(n: int) -> str:
    """Bytes as GiB, e.g. '12.3 GiB'."""
    return f'{n / 2**30:.1f} GiB'


def tree_rss(pid: int) -> int:
    """Resident memory of a process and all its descendants (e.g. loky workers), bytes; 0 if it has gone.

    Pages shared between processes count once per process, so this overstates a little.
    """
    try:
        root = psutil.Process(pid)
        procs = [root, *root.children(recursive=True)]
    except psutil.Error:
        return 0
    total = 0
    for proc in procs:
        with contextlib.suppress(psutil.Error):  # Exited since listed
            total += proc.memory_info().rss
    return total


def memory_use() -> tuple[int, int]:
    """(used, limit) memory in bytes: of this container if it has a cgroup v2 limit (as in JupyterHub, where
    the machine's figures are the host's), else of the machine."""
    # ponytail: cgroup v2 only; v1 (memory/memory.limit_in_bytes) falls back to the machine's figures
    cg = Path('/sys/fs/cgroup')
    try:
        limit = (cg / 'memory.max').read_text().strip()
        if limit != 'max':
            stat = dict(line.split() for line in (cg / 'memory.stat').read_text().splitlines())
            # The working set, as kubectl top reports: page cache that can be dropped doesn't count
            return int((cg / 'memory.current').read_text()) - int(stat.get('inactive_file', 0)), int(limit)
    except (OSError, ValueError):
        pass
    vm = psutil.virtual_memory()
    return vm.total - vm.available, vm.total


async def _sample_memory(r: Run, every: float = 2):
    """Track a run's memory, and its peak, until cancelled."""
    while True:
        r.rss = tree_rss(r.proc.pid)
        r.peak_rss = max(r.peak_rss, r.rss)
        await asyncio.sleep(every)


async def _execute(r: Run, then=None):
    """Run a script, streaming its output into the run, then start the next queued job (`then`, default
    `start_queued`)."""
    env = {**os.environ, 'MPLBACKEND': 'Agg', 'PYTHONUNBUFFERED': '1'}
    cmd = [sys.executable, '-m', r.module, *r.argv]
    shutdown = False
    sampler = None
    try:
        with open(r.dir / 'run.log', 'w') as log:
            # Own session, so cancel also reaches the dask/loky workers the script starts
            r.proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=r.dir,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
            )
            sampler = asyncio.create_task(_sample_memory(r))
            await _pump(r, r.proc.stdout, log)
            r.returncode = await r.proc.wait()
    except asyncio.CancelledError:
        # The UI is shutting down: start nothing else
        shutdown = True
        raise
    finally:
        if sampler:
            sampler.cancel()
        if r.returncode is None:
            _kill_group(r, signal.SIGKILL)
        r.rss = 0
        r.finished = time.time()
        r.live = False
        r.save()
        if not shutdown:
            if r.returncode == 0 and r.script == SCRIPTS['quicklook']:
                try:
                    auto_station_qa(r.dir)
                except Exception:
                    logger.exception(f'Cannot set station QA marks for {r.dir}')
            (then or start_queued)()


def start_run(job: Job) -> Run:
    """Start a queued report script, in its run folder."""
    r = Run(job.dir, SCRIPTS[job.name], set_argv(job) + job.argv, time.time(), live=True)
    r.dir.mkdir(parents=True, exist_ok=True)
    r.save()
    RUNS[r.dir] = r
    background_tasks.create(_execute(r))
    return r


def run_for(d: Path, name: str) -> Run | None:
    """The run in a folder started since the UI started, or the last one on disk.

    A report written outside the UI (e.g. from the command line), or the calibration comparison page,
    has no run.json, so it is returned as a finished run with no log.
    """
    r = RUNS.get(d) or Run.load(d)
    if r is None and (d / 'index.html').exists():
        r = Run(d, SCRIPTS[name], [], (d / 'index.html').stat().st_mtime, returncode=0)
    return r


def start_tool():
    """Start the first queued convert / database-update job, if none is running."""
    if TOOL_QUEUE and not any(r.live for r in TOOL_RUNS.values()):
        run_tool_now(TOOL_QUEUE[0])


def run_tool_now(r: Run):
    """Start a queued convert / database-update job, alongside any running; the queue waits for all to finish."""
    if r not in TOOL_QUEUE:
        return
    TOOL_QUEUE.remove(r)
    r.started, r.live = time.time(), True
    r.dir.mkdir(parents=True, exist_ok=True)
    r.save()
    TOOL_RUNS[r.dir] = r
    background_tasks.create(_execute(r, start_tool))


def enqueue_tool(d: Path, module: str, argv: list[str]) -> Run:
    """Queue a convert / database-update job (`python -m module argv`, logging to d/run.log)."""
    r = Run(d, module, argv, time.time())
    TOOL_QUEUE.append(r)
    start_tool()
    return r


def running() -> Run | None:
    """The run in progress, if any."""
    return next((r for r in RUNS.values() if r.live), None)


def _writes(job: Job) -> set[Path]:
    """Run folders a queued job will write: its own, and for a delay fit, `delay_fit/<run>` per run read."""
    if job.name == 'delay_fit':
        return {job.dir} | {report_dir(job.set, 'delay_fit') / d.name for d in job.caldirs}
    return {job.dir}


def _reads(job: Job) -> set[Path]:
    """Calibration run folders a job reads."""
    # A delay fit run is read with the calibration run of the same name, see set_argv()
    return set(job.caldirs) | {
        report_dir(job.set, 'calibration') / d.name for d in job.caldirs if d.parent.name == 'delay_fit'
    }


def cal_ready(job: Job) -> bool:
    """Whether the calibration runs a job reads are done, and no queued job will rewrite them."""
    if job.name not in NEEDS_CAL and not job.caldirs:
        return True
    pending = set().union(*map(_writes, QUEUE))
    return bool(job.caldirs) and all(any(d.glob('*.cal')) for d in job.caldirs) and not _reads(job) & pending


def ready(job: Job) -> bool:
    """Whether a queued job can start now."""
    return running() is None and cal_ready(job)


def run_now_blocked(job: Job) -> str:
    """Why a queued job can't start alongside the running ones, or '' if it can."""
    if not cal_ready(job):
        return 'Waits for calibration'
    live = {r.dir for r in RUNS.values() if r.live}
    # A folder clashes if a live run writes it or a folder above it (e.g. delay_fit/ over delay_fit/<run>)
    if any(p in live or live & set(p.parents) for p in _writes(job) | _reads(job)):
        return 'A running report uses its folders'
    return ''


def run_now(job: Job):
    """Start a queued job straight away, alongside any running; the queue waits for all of them to finish."""
    if job in QUEUE and not run_now_blocked(job):
        QUEUE.remove(job)
        start_run(job)


def start_queued():
    """Start the first queued job that can start now."""
    job = next((j for j in QUEUE if ready(j)), None)
    if job:
        QUEUE.remove(job)
        start_run(job)


def enqueue(
    setdir: Path, name: str, argv: list[str], d: Path | None = None, caldirs: list[Path] | None = None
) -> Run | None:
    """Queue a report, then start whatever can start. Returns its run if it started straight away.

    Quicklooks go ahead of the other reports, behind any quicklooks already queued.

    Args:
        setdir (Path): Calibration set.
        name (str): Report name.
        argv (list): Form arguments.
        d (Path): Run folder. (default: `run_dir()` without form values)
        caldirs (list): Calibration run folders, for delay fit and sensitivity.
    """
    job = Job(setdir, name, argv, time.time(), d or run_dir(setdir, name), caldirs or [])
    if name in PRIORITY:
        QUEUE.insert(next((i for i, j in enumerate(QUEUE) if j.name not in PRIORITY), len(QUEUE)), job)
    else:
        QUEUE.append(job)
    start_queued()
    r = RUNS.get(job.dir)
    return r if r and r.live else None


def queued(d: Path) -> int | None:
    """1-based queue position of the run in a folder, or None if it is not queued."""
    return next((i + 1 for i, j in enumerate(QUEUE) if j.dir == d), None)


def cal_runs_coming(setdir: Path) -> list[Path]:
    """Folders of the set's calibration runs going or queued."""
    live = [r.dir for r in RUNS.values() if r.live and r.dir.parent == report_dir(setdir, 'calibration')]
    return live + [j.dir for j in QUEUE if j.set == setdir and j.name in CALIBRATION]


def picked_cals(setdir: Path, sel) -> list[Path]:
    """Calibration (or delay fit, 'delay_fit/<run>') run folders picked in a card's dropdown, or [] for none."""
    if sel is None or not sel.value:
        return []
    return cal_paths(setdir, sel.value if sel.multiple else [sel.value])


def cal_paths(setdir: Path, names: list[str]) -> list[Path]:
    """Calibration (or delay fit, 'delay_fit/<run>') run folders of these names in a set."""
    return [setdir / 'reports' / n if '/' in n else report_dir(setdir, 'calibration') / n for n in names]


def batch_state(setdir: Path, name: str, values: dict, cal_names: list[str]) -> tuple[str, bool]:
    """(why the report can't be queued on a set in a batch, or ''; whether it overwrites a report there).

    A batch runs the same options, and reads calibration runs of the same names, on every set.
    """
    d = run_dir(setdir, name, values)
    have = {x.name for x in cal_runs(setdir) + cal_runs_coming(setdir)}
    have |= {f'delay_fit/{x.name}' for x in delay_fit_runs(setdir)} if name in DELAY_CAL else set()
    missing = [n for n in cal_names if n not in have]
    reason = (
        unavailable(setdir, name, values)
        or ('No calibration run picked.' if name in NEEDS_CAL and not cal_names else '')
        or (f'No calibration run {", ".join(missing)}.' if missing else '')
        or ('Already queued or running.' if queued(d) or (d in RUNS and RUNS[d].live) else '')
    )
    return reason, (d / 'index.html').exists()


def delay_fit_runs(setdir: Path) -> list[Path]:
    """Delay fit run folders of a set that hold delay cal files."""
    return sorted(d for d in report_dir(setdir, 'delay_fit').glob('*/') if any(d.glob('*_delay.cal')))


def cal_in_use(d: Path) -> str:
    """Why a calibration run can't be deleted now, or '' if it can."""
    fits = report_dir(set_of(d), 'delay_fit') / d.name  # Deleted with it
    if any(j.dir == d or {d, fits} & set(j.caldirs) for j in QUEUE):
        return 'It is queued, or a queued report reads it.'
    if any(r.live and (r.dir == d or str(d) in r.argv) for r in RUNS.values()):
        return 'It is running, or a running report reads it.'
    return ''


def delete_cal_run(d: Path):
    """Delete a calibration run and its delay fits, and rebuild the comparison pages of both."""
    from mccs_data_qa.reports.delay_fit_report import delay_fit_summary
    from mccs_data_qa.reports.run_stefcal import write_summary

    s = set_of(d)
    shutil.rmtree(d)
    write_summary(report_dir(s, 'calibration'))
    fits = report_dir(s, 'delay_fit') / d.name
    if fits.is_dir():
        shutil.rmtree(fits)
        delay_fit_summary(report_dir(s, 'delay_fit'))


def import_cal_run(src: Path, dest: Path):
    """Copy a calibration run of another set into a set, and rebuild the set's comparison page.

    The copy keeps the run's report, log and QA mark; its run.json still names the source set's files.
    """
    from mccs_data_qa.reports.run_stefcal import write_summary

    shutil.copytree(src, dest)
    write_summary(dest.parent)


def importable_sets(setdir: Path) -> dict[str, list[str]]:
    """Sets next to a set with calibration runs, {set folder: [run names]}, in natural order."""
    from mccs_data_qa.reports.report import station_key

    sets = sorted(
        (p for p in setdir.parent.iterdir() if (p / SET_YAML).exists()), key=lambda p: station_key(p.name)
    )
    return {str(p): names for p in sets if p != setdir and (names := [d.name for d in cal_runs(p)])}


def cal_coming(setdir: Path) -> bool:
    """Whether the set has cal files, or a calibration run going or queued that will write them."""
    return bool(cal_runs(setdir) or cal_runs_coming(setdir))


def move_job(job: Job, step: int):
    """Move a queued job up (-1) or down (+1); nothing if it has left the queue."""
    if job not in QUEUE:
        return
    i = QUEUE.index(job)
    j = min(max(i + step, 0), len(QUEUE) - 1)
    QUEUE.insert(j, QUEUE.pop(i))
    start_queued()


def remove_job(job: Job):
    """Take a job out of the queue, if it is still there."""
    if job in QUEUE:
        QUEUE.remove(job)
    start_queued()


def latest_run() -> Run | None:
    """The most recently started run, preferring one still running."""
    return max(RUNS.values(), key=lambda r: (r.live, r.started), default=None)


def _kill_group(r: Run, sig: int):
    """Signal a run's process group, if it started."""
    if r.proc is None:
        return
    try:
        os.killpg(r.proc.pid, sig)
    except (ProcessLookupError, PermissionError):
        pass


async def cancel_run(r: Run):
    """Stop a run and everything it started; nothing if it has already finished."""
    if not r.live or r.proc is None:
        return
    r.cancelled = True
    _kill_group(r, signal.SIGTERM)
    try:
        await asyncio.wait_for(r.proc.wait(), 5)
    except asyncio.TimeoutError:
        pass
    _kill_group(r, signal.SIGKILL)


def set_text(label, text: str):
    """Set a label's text, sending an update only if it changed (tick() runs every 0.3 s)."""
    if label.text != text:
        label.set_text(text)


def _field_widget(f: Field, values: dict):
    """One form widget bound to values[f.dest], with its help text."""
    label = f'{f.flag or f.dest}{" *" if f.required else ""}'
    with ui.column().classes('w-full gap-0') as col:
        if f.kind == 'switch':
            w = ui.switch(label)
        elif f.kind == 'select':
            labels = CHOICE_LABELS.get(f.dest, {})
            w = ui.select({c: labels.get(c, c) for c in f.choices}, label=label).classes('w-full')
        elif f.kind == 'multiselect':
            w = ui.select(f.choices, label=label, multiple=True).props('use-chips').classes('w-full')
        elif f.kind == 'number':
            w = ui.number(label=label, precision=0 if f.type is int else None).classes('w-full')
        else:
            w = ui.input(label=label, placeholder='space-separated' if f.kind == 'list' else None).classes(
                'w-full'
            )
        w.bind_value(values, f.dest)
        if f.help:
            ui.label(f.help).classes('text-xs text-grey-7')
    if f.dest in CAL_MODEL_ONLY and 'cal_model' in values:
        col.bind_visibility_from(values, 'cal_model', backward=lambda m: m in CAL_MODEL_ONLY[f.dest])


def _options_form(script: str):
    """The script's options, grouped as in its parser."""
    fields, values = form_state(script)
    groups = {}
    for f in fields:
        groups.setdefault(f.group, []).append(f)
    for title, fs in groups.items():
        ui.label(title).classes('eyebrow mt-3')
        for f in fs:
            _field_widget(f, values)


async def ask_reviewer() -> bool:
    """Ask for the reviewer's initials if this browser has none; False if the user cancels."""
    if reviewer():
        return True
    with ui.dialog() as dialog, ui.card().classes('w-80'):
        ui.label('Who is reviewing?').classes('text-h6')
        ui.label('Your initials are recorded with your QA marks (kept in this browser).').classes(
            'text-xs text-grey-7'
        )
        box = ui.input('Initials').props('autofocus').classes('w-full')
        box.on('keydown.enter', lambda: dialog.submit(box.value.strip()))
        with ui.row().classes('w-full justify-end'):
            ui.button('Cancel', on_click=lambda: dialog.submit('')).props('flat no-caps')
            ui.button('Save', on_click=lambda: dialog.submit(box.value.strip())).props('unelevated no-caps')
    initials = await dialog
    dialog.delete()
    if initials:
        app.storage.user['initials'] = initials
    return bool(initials)


def reviewer_chip():
    """Header button showing the reviewer's initials, highlighted when unset, with a menu to change them."""
    store = app.storage.user
    store.setdefault('initials', '')
    btn = ui.button(icon='person').props('unelevated rounded dense no-caps padding="2px 12px"')

    def show():
        i = store['initials']
        btn.set_text(f'Reviewer: {i}' if i else 'Set reviewer')
        btn.props(f'color={"white" if i else "accent"} text-color={"primary" if i else "white"}')

    with btn, ui.menu().classes('p-3 w-64'):
        ui.label('Reviewer initials').classes('eyebrow')
        ui.label('Recorded with your QA marks, in this browser.').classes('text-xs text-grey-7')
        ui.input(placeholder='e.g. AB', on_change=show).bind_value(store, 'initials').props('dense autofocus')
    show()


def brand(title: str):
    """SKAO colours, fonts and logo, and the page title, at the start of a page's header."""
    ui.colors(**SKAO_COLORS, **{f'qa_{k}': v for k, v in QA_HEX.items()})
    ui.add_head_html(SKAO_HEAD)
    ui.html((HERE / 'skao_logo.svg').read_text(), sanitize=False).classes('skao-logo w-20 mx-2')
    ui.label(title).classes('text-h6')


@ui.page('/')
def index():
    """The single page: folder browser in a drawer, report cards, and set info, log and report tabs."""
    view = {'run': latest_run(), 'n': 0, 'state': None, 'loading': False, 'queue': None, 'open': set()}
    cards = {}  # Report name: dict of the card's live widgets
    runs = {}  # Run folder: last run on disk, read when first needed by a card

    def browse(d: Path):
        STATE['dir'] = d
        if (d / SET_YAML).exists():
            # Stay in the parent folder, so the other sets are still listed
            STATE['dir'], STATE['set'] = d.parent, d
            set_info.refresh()
            reports.refresh()
            follow_set(d)
            tabs.set_value(t_set)
            pick_options()
        folder.refresh()

    def open_report(setdir: Path, name: str):
        browse(setdir)
        show(run_for(run_dir(setdir, name), name), t_report)

    def card_dir(name: str) -> Path:
        """Run folder of a card's report with its current options."""
        script = SCRIPTS[name]
        return run_dir(STATE['set'], name, form_state(script)[1] if script in PARSERS else None)

    def mark_row(label: str, choices: tuple, mark: dict, save):
        """QA mark buttons and a note, passed to save(status, note) when either changes."""
        status = [mark.get('status')]
        with ui.row().classes('w-full items-center no-wrap gap-2'):
            ui.label(label).classes('text-xs font-mono w-16')
            btns = {
                c: ui.button(
                    c.replace('_', ' ').capitalize(), icon=QA_ICON[c], on_click=lambda c=c: pick(c)
                ).props('dense no-caps unelevated padding="2px 8px"')
                for c in choices
            }
            note = ui.input(placeholder='Note', value=mark.get('note', '')).props('dense').classes('grow')
            by = ui.label(by_line(mark)).classes('text-xs text-grey-7 whitespace-nowrap')
        last = [(status[0], note.value)]

        async def pick(c: str):
            # A second click on the selected mark clears it
            status[0] = None if status[0] == c else c
            await on_change()

        def paint():
            for c, b in btns.items():
                on = c == status[0]
                color = qa_color(c) if on else 'grey-2'
                b.props(f'color={color} text-color={"primary" if on else "grey-8"}')

        async def on_change():
            if (status[0], note.value) != last[0]:
                if not await ask_reviewer():
                    status[0], _ = last[0]  # Not saved without initials
                    note.value = last[0][1]
                    paint()
                    return
                last[0] = (status[0], note.value)
                save(status[0], note.value)
                by.set_text(f'by {reviewer()}, just now')
                folder.refresh()
            paint()

        paint()
        note.on('blur', on_change)

    def qa_menu(d: Path, choices: tuple):
        """Button showing the run's QA mark, with a popup of station marks, in the Report tab."""

        def save(st: str, status: str | None, note: str):
            save_station_qa(d, st, status, note)
            qa = run_qa(d)
            btn.set_text(qa_label(d, qa))
            btn.props(f'color={qa_color(qa)}')

        qa = run_qa(d)
        marks = station_marks(d)
        with ui.button(qa_label(d, qa), icon='fact_check').props(
            f'unelevated dense no-caps color={qa_color(qa)} text-color=primary'
        ) as btn:
            with ui.menu().classes('p-3 w-[40rem] max-w-full'):
                ui.label('Station QA').classes('eyebrow mb-1')
                for st in set_stations(set_of(d)):
                    mark_row(st, choices, marks.get(st, {}), lambda status, note, st=st: save(st, status, note))

    @ui.refreshable
    def folder():
        d = STATE['dir']
        with ui.row().classes('w-full items-center no-wrap'):
            ui.button(icon='arrow_upward', on_click=lambda: browse(d.parent)).props('flat dense').tooltip('Up')
            ui.input('Folder', value=str(d)).classes('grow').on(
                'keydown.enter', lambda e: browse(Path(e.sender.value).expanduser())
            )
        if not d.is_dir():
            ui.label('Not a folder').classes('text-negative')
            return
        from mccs_data_qa.reports.report import station_key  # Natural sort, so set614 comes before set1172

        subdirs = sorted(
            (p for p in d.iterdir() if p.is_dir() and not p.name.startswith('.')),
            key=lambda p: station_key(p.name),
        )
        with ui.list().props('dense separator').classes('w-full'):
            for p in subdirs:
                is_set = (p / SET_YAML).exists()
                with ui.item(on_click=lambda p=p: browse(p)).classes(
                    'skao-selected' if p == STATE['set'] else ''
                ):
                    # Quasar's 56px avatar column leaves too little room for the name beside 5 report icons
                    with ui.item_section().props('avatar').style('min-width: 36px'):
                        if is_set:
                            qa = set_qa(p)
                            ui.icon('dataset', color=qa_color(qa)).tooltip(f'Set QA: {qa.replace("_", " ")}')
                        else:
                            ui.icon('folder', color='grey')
                    with ui.item_section():
                        # Truncate rather than run under the report icons
                        ui.label(p.name).classes('ellipsis').tooltip(p.name)
                    done = [n for n in SCRIPTS if (run_dir(p, n) / 'index.html').exists()] if is_set else []
                    if done:
                        with ui.item_section().props('side'), ui.row().classes('gap-1 no-wrap'):
                            for n in done:
                                ui.icon(INFO[n][0], size='xs', color='grey-7').classes(
                                    'cursor-pointer hover:text-primary'
                                ).tooltip(f'Open the {INFO[n][1]} report').on(
                                    'click.stop', lambda p=p, n=n: open_report(p, n)
                                )

    @ui.refreshable
    def set_info():
        s = STATE['set']
        set_title.set_text(s.name if s else '')
        if s is None:
            ui.label('Select a calibration set from the folder panel.')
            return
        from mccs_data_qa.reports.report import station_key

        mark_row('Set QA', SET_QA, read_json(set_qa_path(s)), lambda status, note: save_set_qa(s, status, note))
        meta = read_set(s)
        ui.label(f'Calibration set {meta.get("set_id", "?")}').classes('eyebrow')
        ui.label(s.name).classes('text-h6')
        ui.label(str(s)).classes('text-xs text-grey-7 font-mono')
        details = sorted(meta.get('set_details') or [], key=lambda d: station_key(str(d.get('Station ID', ''))))
        if details:
            keys = [k for k in details[0] if k not in ('Set ID', 'pb-id', 'n_files')]
            keys.insert(keys.index('n_channel') + 1 if 'n_channel' in keys else len(keys), 'n_timestep')
            n_time = n_integrations(s)
            fmt = {
                'Date (UTC)': lambda v: v[:10],
                'UTC Start': lambda v: v[:19],
                'Perth Local Time (AWST)': lambda v: v[11:16],
            }
            cols = [{'name': k, 'label': k, 'field': k, 'align': 'left'} for k in keys]
            rows = [
                {
                    k: str(n_time or '')
                    if k == 'n_timestep'
                    else ''
                    if d.get(k) is None
                    else fmt.get(k, str)(str(d[k]))
                    for k in keys
                }
                for d in details
            ]
            ui.table(columns=cols, rows=rows).props('dense flat').classes('w-full')
        files = sorted(meta.get('filelist') or [], key=lambda f: station_key(str(f.get('station_id', ''))))
        if files:
            ui.label('Files').classes('eyebrow mt-4')
            keys = ['station_id', 'observation_id', 'status', 'file']
            cols = [{'name': k, 'label': k, 'field': k, 'align': 'left'} for k in keys]
            rows = [{k: str(f.get(k, '')) for k in keys} for f in files]
            ui.table(columns=cols, rows=rows).props('dense flat').classes('w-full')

    def toggle_options(name: str, col):
        view['open'] ^= {name}
        col.set_visibility(name in view['open'])

    @ui.refreshable
    def reports():
        cards.clear()
        runs.clear()
        s = STATE['set']
        view['loading'] = False
        if s is None:
            ui.label('Select a calibration set to run reports on it.').classes('text-grey-7')
            return
        for name, script in SCRIPTS.items():
            icon, title, desc = INFO[name]
            indent = 'ml-6 skao-needs-cal' if name in INDENTED else ''
            with ui.card().classes(f'w-full {indent}').props('flat bordered'):
                with ui.row().classes('w-full items-center no-wrap gap-3'):
                    ui.icon(icon, size='md', color='primary')
                    with ui.column().classes('grow gap-0'):
                        with ui.row().classes('items-center gap-2'):
                            ui.label(title).classes('text-subtitle1')
                            badge = ui.badge().props('rounded')
                            qa_badge = ui.badge().props('rounded text-color=primary')
                        ui.label(desc).classes('text-xs text-grey-7')
                cal = None
                if name in NEEDS_CAL + OPTIONAL_CAL:
                    multi = name in MULTI_CAL
                    label = 'Calibration runs' if multi else 'Calibration run'
                    cal = ui.select([], label=label + (' (optional)' if name in OPTIONAL_CAL else ''), multiple=multi)
                    cal.classes('w-full').props('use-chips' if multi else 'clearable' if name in OPTIONAL_CAL else '')
                run_text = report_btn = log_btn = opts_btn = summary_btn = None
                if name == 'calibration':
                    # New runs are set up in a dialog; the card lists the runs
                    with ui.dialog() as dlg, ui.card().classes('w-[34rem] max-w-full'):
                        ui.label('Add a calibration run').classes('text-h6')
                        run_text = ui.label().classes('text-xs font-mono')
                        opts = ui.column().classes('w-full gap-1')
                        reason = ui.label().classes('text-xs text-orange-9')
                        with ui.row().classes('w-full justify-end'):
                            ui.button('Cancel', on_click=dlg.close).props('flat no-caps')
                            batch_btn = ui.button(
                                'Batch',
                                icon='library_add',
                                on_click=lambda dlg=dlg, name=name, script=script: batch_cal(dlg, name, script),
                            ).props('flat no-caps').tooltip('Add this run to several sets')
                            run_btn = ui.button(
                                on_click=lambda dlg=dlg, name=name, script=script: add_cal(dlg, name, script)
                            ).props('unelevated no-caps')
                    cal_list()
                    with ui.row().classes('w-full items-center gap-1'):
                        summary_btn = (
                            ui.button(
                                'Compare',
                                icon='compare',
                                on_click=lambda: show(
                                    run_for(report_dir(s, 'calibration'), 'calibration'), t_report
                                ),
                            )
                            .props('flat dense no-caps')
                            .tooltip('Compare the calibration runs')
                        )
                        ui.space()
                        ui.button('Import', icon='download', on_click=lambda: import_dialog(s)).props(
                            'flat dense no-caps'
                        ).tooltip('Copy a calibration run from another set')
                        ui.button('Add cal', icon='add', on_click=dlg.open).props('unelevated dense no-caps')
                else:
                    reason = ui.label().classes('text-xs text-orange-9')
                    with ui.row().classes('w-full items-center gap-1'):
                        # A report reading several calibration runs opens on its comparison page
                        multi = name in MULTI_CAL
                        report_btn = ui.button(
                            'Compare' if multi else 'Report',
                            icon='compare' if multi else 'article',
                            on_click=lambda name=name: show(run_for(card_dir(name), name), t_report),
                        ).props('flat dense no-caps')
                        if multi:
                            report_btn.tooltip('Compare the delay fits of the calibration runs')
                        log_btn = ui.button(
                            'Log',
                            icon='terminal',
                            on_click=lambda name=name: show(run_for(card_dir(name), name), t_log),
                        ).props('flat dense no-caps')
                    with ui.row().classes('w-full items-center gap-1'):
                        opts_btn = ui.button('Options', icon='settings').props(
                            'flat dense no-caps color=grey-8'
                        )
                        ui.space()
                        batch_btn = ui.button(
                            'Batch',
                            icon='library_add',
                            on_click=lambda name=name, script=script: batch_dialog(name, script),
                        ).props('flat dense no-caps').tooltip('Queue with these options on several sets')
                        run_btn = ui.button(
                            on_click=lambda name=name, script=script: on_run(name, script)
                        ).props('unelevated dense no-caps')
                    opts = ui.column().classes('w-full gap-1')
                    opts.set_visibility(name in view['open'])
                    opts_btn.on_click(lambda name=name, opts=opts: toggle_options(name, opts))
                with opts:
                    if script in PARSERS:
                        _options_form(script)
                    else:
                        ui.label('Loading options...').classes('text-xs')
                        view['loading'] = True
                cards[name] = {
                    'badge': badge,
                    'qa': qa_badge,
                    'run_label': run_text,
                    'cal': cal,
                    'summary': summary_btn,
                    'reason': reason,
                    'run': run_btn,
                    'batch': batch_btn,
                    'report': report_btn,
                    'log': log_btn,
                    'opts': opts_btn,
                }

    def status_of(d: Path) -> tuple:
        """(run, status text, badge colour, busy) of the run in a folder."""
        if d not in runs:
            runs[d] = Run.load(d)
        r = RUNS.get(d) or runs[d]
        pos = queued(d)
        busy = bool(pos) or (r is not None and r.live)
        if pos:
            return r, f'queued #{pos}', STATUS_COLOR['queued'], busy
        if r is not None:
            return r, f'{r.status}, {ago(r.started)}', STATUS_COLOR.get(r.status, 'negative'), busy
        if (d / 'index.html').exists():
            return r, 'done', STATUS_COLOR['done'], busy
        return r, 'not run', 'grey-5', busy

    def cal_list_state(s: Path) -> tuple:
        """Per calibration run: folder, status, colour, QA mark, QA chip text, and why it can't be deleted."""
        out = []
        for d in dict.fromkeys(cal_runs(s) + cal_runs_coming(s)):
            _, status, color, busy = status_of(d)
            qa = run_qa(d) if (d / 'index.html').exists() and not busy else 'unreviewed'
            out.append((d, status, color, qa, qa_label(d, qa), cal_in_use(d)))
        return tuple(out)

    @ui.refreshable
    def cal_list():
        s = STATE['set']
        rows = cal_list_state(s) if s else ()
        view['cal_list'] = rows
        if not rows:
            ui.label('No calibration runs yet: add one.').classes('text-xs text-grey-7')
        for d, status, color, qa, label, in_use in rows:
            name = 'holo_suncal' if d == run_dir(s, 'holo_suncal') else 'calibration'
            with ui.row().classes('w-full items-center no-wrap gap-1'):
                ui.label(d.name).classes('text-xs font-mono grow break-all')
                ui.badge(status, color=color).props('rounded')
                if qa != 'unreviewed':
                    ui.badge(label).props(f'rounded color={qa_color(qa)} text-color=primary')
                ui.button(
                    icon='article', on_click=lambda d=d, name=name: show(run_for(d, name), t_report)
                ).props('flat dense size=sm').tooltip('Report').set_enabled((d / 'index.html').exists())
                ui.button(icon='terminal', on_click=lambda d=d, name=name: show(run_for(d, name), t_log)).props(
                    'flat dense size=sm'
                ).tooltip('Log').set_enabled(status != 'not run' and not status.startswith('queued'))
                ui.button(icon='delete', color='negative', on_click=lambda d=d: confirm_delete(d)).props(
                    'flat dense size=sm'
                ).tooltip(in_use or 'Delete this run and its delay fits').set_enabled(not in_use)

    async def add_cal(dlg, name: str, script: str):
        dlg.close()
        await on_run(name, script)

    async def batch_cal(dlg, name: str, script: str):
        dlg.close()
        await batch_dialog(name, script)

    async def batch_dialog(name: str, script: str):
        """Queue a report with the card's options, and calibration runs of the same names, on several sets
        of the selected set's folder."""
        from mccs_data_qa.reports.report import station_key

        s = STATE['set']
        fields, values = form_state(script)
        values = dict(values)  # The options as they are now
        argv = script_argv(fields, values)
        sel = cards[name]['cal']
        cal_names = [] if sel is None or not sel.value else sel.value if sel.multiple else [sel.value]
        sets = sorted(
            (p for p in s.parent.iterdir() if (p / SET_YAML).exists()), key=lambda p: station_key(p.name)
        )
        # Reads each set's yaml and first UVX file
        states = await run.io_bound(lambda: [batch_state(p, name, values, cal_names) for p in sets])
        boxes = {}
        with ui.dialog() as dialog, ui.card().classes('w-[40rem] max-w-full'):
            ui.label(f'Batch {INFO[name][1]}').classes('text-h6')
            if name == 'calibration':
                ui.label(f'Run: {run_dir(s, name, values).name}').classes('text-xs font-mono')
            ui.label(f'Options on every set: {shlex.join(argv) or "defaults"}').classes(
                'text-xs font-mono break-all'
            )
            if cal_names:
                ui.label(f'Calibration runs read: {" ".join(cal_names)}').classes('text-xs font-mono')
            with ui.row().classes('w-full items-center gap-1'):
                ui.label(f'Sets in {s.parent.name}/').classes('eyebrow grow')
                ui.button('All', on_click=lambda: tick_all(True)).props('flat dense no-caps')
                ui.button('None', on_click=lambda: tick_all(False)).props('flat dense no-caps')
            # Tall list with tight rows, so many sets fit on screen at once
            with ui.scroll_area().classes('w-full h-[65vh] border rounded'), ui.column().classes('w-full gap-0'):
                for p, (reason, overwrites) in zip(sets, states, strict=True):
                    with ui.row().classes('w-full items-center no-wrap gap-2'):
                        box = ui.checkbox(p.name, value=not reason and not overwrites, on_change=lambda: count())
                        box.props('dense').set_enabled(not reason)
                        if not reason:
                            boxes[p] = box
                        note = reason or ('re-run (overwrites)' if overwrites else '')
                        ui.label(note).classes(f'text-xs {"text-orange-9" if reason else "text-grey-7"}')
            with ui.row().classes('w-full justify-end'):
                ui.button('Cancel', on_click=lambda: dialog.submit(False)).props('flat no-caps')
                go = ui.button(icon='playlist_add', on_click=lambda: dialog.submit(True)).props(
                    'unelevated no-caps'
                )

        def tick_all(on: bool):
            for b in boxes.values():
                b.value = on

        def count():
            n = sum(b.value for b in boxes.values())
            go.set_text(f'Queue {n} run{"s" * (n != 1)}')
            go.set_enabled(n > 0)

        count()
        ok = await dialog
        dialog.delete()
        if not ok:
            return
        n = 0
        for p, b in boxes.items():
            # Re-checked, as the queue may have changed while the dialog was open
            if b.value and not batch_state(p, name, values, cal_names)[0]:
                enqueue(p, name, argv, run_dir(p, name, values), cal_paths(p, cal_names))
                n += 1
        ui.notify(f'Queued {n} {INFO[name][1]} run{"s" * (n != 1)}')

    async def import_dialog(s: Path):
        """Ask for a calibration run of another set, then copy it into this set."""
        sets = await run.io_bound(importable_sets, s)
        if not sets:
            ui.notify(f'No other set in {s.parent} has calibration runs')
            return
        with ui.dialog() as dialog, ui.card().classes('w-[34rem] max-w-full'):
            ui.label('Import a calibration run').classes('text-h6')
            ui.label(f'Copies the run folder into {s.name}/reports/calibration/.').classes('text-xs text-grey-7')
            src_set = ui.select({k: Path(k).name for k in sets}, label='From set', value=next(iter(sets)))
            src_run = ui.select([], label='Calibration run')
            name = ui.input('Name in this set')
            for w in (src_set, src_run, name):
                w.classes('w-full')

            def on_set():
                src_run.set_options(sets[src_set.value], value=sets[src_set.value][0])

            def on_pick():
                name.set_value(f'{src_run.value}_from_{Path(src_set.value).name}' if src_run.value else '')

            src_set.on_value_change(on_set)
            src_run.on_value_change(on_pick)
            on_set()
            with ui.row().classes('w-full justify-end'):
                ui.button('Cancel', on_click=lambda: dialog.submit(None)).props('flat no-caps')
                ui.button(
                    'Import',
                    icon='download',
                    on_click=lambda: dialog.submit(
                        (report_dir(Path(src_set.value), 'calibration') / src_run.value, name.value.strip())
                    ),
                ).props('unelevated no-caps')
        picked = await dialog
        dialog.delete()
        if not picked:
            return
        src, new = picked
        dest = report_dir(s, 'calibration') / new
        # One folder name, inside this set's calibration folder
        if not new or Path(new).name != new or new.startswith('.'):
            ui.notify(f'Not imported: {new!r} is not a folder name')
        elif dest.exists():
            ui.notify(f'Not imported: {run_label(dest)} already exists')
        elif why := cal_in_use(src):
            ui.notify(f'Not imported: {why}')
        else:
            await run.io_bound(import_cal_run, src, dest)
            ui.notify(f'Imported {run_label(src)} as {run_label(dest)}')
            folder.refresh()

    async def confirm_delete(d: Path):
        """Ask, then delete a calibration run and its delay fits."""
        s = set_of(d)
        fits = report_dir(s, 'delay_fit') / d.name
        with ui.dialog() as dialog, ui.card().classes('max-w-md'):
            ui.label(f'Delete calibration run {d.name}?').classes('text-h6')
            ui.label(f'This deletes {d.relative_to(s)}/, with its cal files, report and QA mark.')
            if fits.is_dir():
                ui.label(f'It also deletes its delay fits in {fits.relative_to(s)}/.').classes('text-orange-9')
            with ui.row().classes('w-full justify-end'):
                ui.button('Cancel', on_click=lambda: dialog.submit(False)).props('flat no-caps')
                ui.button(
                    'Delete', icon='delete', color='negative', on_click=lambda: dialog.submit(True)
                ).props('unelevated no-caps')
        ok = await dialog
        dialog.delete()
        if not ok:
            return
        if why := cal_in_use(d):
            ui.notify(f'Not deleted: {why}')
            return
        await run.io_bound(delete_cal_run, d)
        for x in (d, fits):
            RUNS.pop(x, None)
            runs.pop(x, None)
        if view['run'] and view['run'].dir in (d, fits):
            set_run(None)
        ui.notify(f'Deleted {run_label(d)}')
        folder.refresh()

    @ui.refreshable
    def report():
        r = view['run']
        path = r.report() if r else None
        if path is None:
            ui.label('No report (index.html) in this run yet.')
            return
        choices = () if r.live else qa_choices(r.dir)
        if choices and run_qa(r.dir) == 'stale':
            mark = read_json(r.dir / 'qa.json')
            ui.label(
                f'Marked {mark["status"]} on an earlier run ({ago(mark["updated"])}); not reviewed since the re-run.'
            ).classes('text-xs text-orange-9')
        with ui.row().classes('w-full items-center no-wrap'):
            if choices:
                qa_menu(r.dir, choices)
            ui.space()
            ui.button(icon='open_in_new', on_click=lambda: ui.navigate.to(url(path), new_tab=True)).props(
                'flat dense'
            ).tooltip('Open in a new tab')
        ui.element('iframe').props(f'src="{url(path)}"').classes('w-full').style('height: 78vh; border: 0')

    @ui.refreshable
    def files():
        r = view['run']
        for p in sorted(r.dir.rglob('*')) if r else []:
            if p.is_file():
                ui.link(p.relative_to(r.dir).as_posix(), url(p), new_tab=True)

    def on_set_tab() -> bool:
        return tabs.value in (t_set, 'Set')

    def pick_options():
        """List the set's summary (the Set tab) and its runs in the run picker."""
        r = view['run']
        s = set_of(r.dir) if r and not on_set_tab() else STATE['set']
        run_pick.set_visibility(s is not None)
        if s is None:
            return
        run_set.set_text(f'{s.name} /')
        runs = [*set_runs(s), *([r.dir] if r and set_of(r.dir) == s else [])]
        options = {str(s): 'summary'} | {str(d): run_name(d) for d in dict.fromkeys(runs)}
        value = str(s) if on_set_tab() else (str(r.dir) if r else None)
        # Setting options re-renders the select, which closes its open menu
        if (options, value) != (run_pick.options, run_pick.value):
            run_pick.set_options(options, value=value)

    def pick_run(value: str | None):
        if not value:
            return
        d = Path(value)
        if (d / SET_YAML).exists():
            if not on_set_tab():
                browse(d)
        elif on_set_tab() or view['run'] is None or d != view['run'].dir:
            show(run_for(d, run_name(d).split('/')[0]), None if on_set_tab() else tabs.value)

    def show(r: Run | None, tab=None):
        if r is None:
            ui.notify('Not run from the UI yet')
            return
        set_run(r)
        tabs.set_value(tab or (t_report if r.status == 'done' else t_log))

    def set_run(r: Run | None):
        """Show a run, or none, in the Log, Report and Files tabs."""
        view.update(run=r, n=0, state=None)
        log.clear()
        if r is None:  # tick() redraws the tabs only while a run is shown
            pick_options()
            report.refresh()
            files.refresh()

    def follow_set(d: Path):
        """Show the same report of a newly selected set, or none if it has no such report."""
        r = view['run']
        if r is None or set_of(r.dir) == d:
            return
        rel = run_name(r.dir)
        name = rel.split('/')[0]
        set_run(run_for(d / 'reports' / rel, name) or run_for(report_dir(d, name), name))

    async def confirm_rerun(s: Path, name: str, d: Path) -> bool:
        """Ask before a run overwrites an existing report."""
        with ui.dialog() as dialog, ui.card().classes('max-w-md'):
            ui.label(f'Re-run {INFO[name][1]} on {s.name}?').classes('text-h6')
            ui.label(f'This overwrites the existing report in {d.relative_to(s)}/.')
            if name in CALIBRATION:
                ui.label(
                    'It also replaces the cal files of this run that delay fit and sensitivity read.'
                ).classes('text-orange-9')
            with ui.row().classes('w-full justify-end'):
                ui.button('Cancel', on_click=lambda: dialog.submit(False)).props('flat no-caps')
                ui.button('Re-run', icon='replay', on_click=lambda: dialog.submit(True)).props(
                    'unelevated no-caps'
                )
        ok = await dialog
        dialog.delete()
        return bool(ok)

    async def on_run(name: str, script: str):
        s = STATE['set']
        d = card_dir(name)
        if (d / 'index.html').exists() and not await confirm_rerun(s, name, d):
            return
        fields, values = form_state(script)
        r = enqueue(s, name, script_argv(fields, values), d, picked_cals(s, cards[name]['cal']))
        if r:
            show(r)
        else:
            ui.notify(f'Queued {run_label(d)} at position {queued(d)}')

    def show_from_queue(r: Run):
        view['queue_menu'].close()  # Else the menu covers the log it opened
        show(r)

    @ui.refreshable
    def queue_view():
        live = [r for r in RUNS.values() if r.live]
        view['live_rows'] = []  # (run, label) of each running report, updated by tick()
        with ui.row().classes('w-full items-center no-wrap'):
            ui.label('Running').classes('eyebrow grow')
            view['mem'] = ui.label().classes('text-xs text-grey-7').tooltip(
                'Memory in use, of the container limit if it has one, else of the machine'
            )
        if not live:
            ui.label('Nothing running.').classes('text-xs text-grey-7')
        for r in live:
            with ui.row().classes('w-full items-center no-wrap'):
                ui.label(run_label(r.dir)).classes('grow')
                view['live_rows'].append((r, ui.label().classes('text-xs text-grey-7')))
                ui.button(icon='visibility', on_click=lambda r=r: show_from_queue(r)).props('flat dense').tooltip(
                    'Show log'
                )
                ui.button(icon='stop', color='negative', on_click=lambda r=r: cancel_run(r)).props(
                    'flat dense'
                ).tooltip('Cancel')
        with ui.row().classes('w-full items-center mt-4'):
            ui.label(f'Queued ({len(QUEUE)})').classes('eyebrow grow')
            ui.button('Clear', icon='clear_all', on_click=lambda: QUEUE.clear()).props(
                'flat dense'
            ).set_enabled(bool(QUEUE))
        if not QUEUE:
            ui.label('The queue is empty. Reports started while another runs are queued here.').classes(
                'text-xs text-grey-7'
            )
        for i, job in enumerate(QUEUE):
            with ui.row().classes('w-full items-center no-wrap'):
                ui.label(f'{i + 1}.').classes('w-6')
                with ui.column().classes('grow gap-0'):
                    ui.label(run_label(job.dir))
                    extra = shlex.join(job.argv) or 'default options'
                    if job.caldirs:
                        extra += f', cal {" ".join(d.name for d in job.caldirs)}'
                    wait = ', waits for calibration' if not cal_ready(job) else ''
                    ui.label(f'{extra}{wait}').classes('text-xs text-grey-7 font-mono break-all')
                blocked = run_now_blocked(job)
                ui.button(icon='play_arrow', color='positive', on_click=lambda job=job: run_now(job)).props(
                    'flat dense'
                ).tooltip(blocked or 'Run now, alongside the running report').set_enabled(not blocked)
                ui.button(
                    icon='vertical_align_top', on_click=lambda job=job: move_job(job, -len(QUEUE))
                ).props('flat dense').tooltip('Move to top')
                ui.button(icon='arrow_upward', on_click=lambda job=job: move_job(job, -1)).props('flat dense')
                ui.button(icon='arrow_downward', on_click=lambda job=job: move_job(job, 1)).props('flat dense')
                ui.button(icon='delete', color='negative', on_click=lambda job=job: remove_job(job)).props(
                    'flat dense'
                ).tooltip('Remove from the queue')
        done = sorted(
            (r for r in RUNS.values() if not r.live), key=lambda r: r.finished or r.started, reverse=True
        )
        ui.label('Recently completed').classes('eyebrow mt-4')
        if not done:
            ui.label('No reports have finished since the UI started.').classes('text-xs text-grey-7')
        for r in done[:N_RECENT]:
            with ui.row().classes('w-full items-center no-wrap'):
                ui.label(run_label(r.dir)).classes('grow')
                ui.badge(r.status, color=STATUS_COLOR.get(r.status, 'negative')).props('rounded')
                peak = f', peak {gib(r.peak_rss)}' if r.peak_rss else ''
                ui.label(f'{r.elapsed}{peak}, {ago(r.finished or r.started)}').classes('text-xs text-grey-7')
                ui.button(icon='visibility', on_click=lambda r=r: show_from_queue(r)).props('flat dense').tooltip(
                    'Show'
                )

    def update_card(s: Path, name: str, c: dict):
        d = card_dir(name)
        r, status, color, busy = status_of(d)
        has_report = (d / 'index.html').exists()
        n_runs = len(cal_runs(s)) if name == 'calibration' else 0
        if name == 'calibration':
            status, color = f'{n_runs} run{"s" * (n_runs != 1)}', 'positive' if n_runs else 'grey-5'
        script = SCRIPTS[name]
        fields, values = form_state(script) if script in PARSERS else ([], {})
        reason = unavailable(s, name, values)
        note = reason or cal_warning(s, values)
        if c['cal']:
            sel = c['cal']
            # Finished calibration runs, newest first, then those going or queued
            options = list(dict.fromkeys(x.name for x in cal_runs(s) + cal_runs_coming(s)))
            if name in DELAY_CAL:
                options += [f'delay_fit/{x.name}' for x in delay_fit_runs(s)]
            if options != sel.options:
                if sel.multiple:
                    kept = [v for v in sel.value or [] if v in options]
                    sel.set_options(options, value=kept or options)
                else:
                    sel.set_options(options, value=sel.value if sel.value in options else None)
            if not sel.multiple and sel.value is None and options and name not in OPTIONAL_CAL:
                sel.set_value(options[0])
        caldirs = picked_cals(s, c['cal'])
        start_now = ready(Job(s, name, [], 0, d, caldirs))
        # A batch checks each set itself, so it needs only the options and calibration runs
        batch_ok = (
            script in PARSERS and not missing_required(fields, values) and (c['cal'] is None or bool(caldirs) or name in OPTIONAL_CAL)
        )
        enabled = batch_ok and not busy and not reason
        changed = sum(values[f.dest] != v for f, v in zip(fields, default_values(fields).values(), strict=True))
        qa = run_qa(d) if has_report and not busy and name != 'calibration' else 'unreviewed'
        label = qa_label(d, qa)
        state = (
            *(status, color, note, start_now, enabled, batch_ok),
            *(has_report, r is not None, changed, d, n_runs, label),
        )
        if state == c.get('state'):
            return
        c['state'] = state
        c['qa'].set_text(label)
        c['qa'].props(f'color={qa_color(qa)}')
        c['qa'].set_visibility(qa != 'unreviewed')
        if c['run_label']:
            c['run_label'].set_text(f'Run: {d.name}')
        if c['summary']:
            c['summary'].set_enabled(n_runs > 0)
        c['badge'].set_text(status)
        c['badge'].props(f'color={color}')
        c['reason'].set_text(note)
        c['reason'].set_visibility(bool(note))
        if start_now:
            c['run'].set_text('Re-run' if has_report else 'Run')
            c['run'].props(f'icon={"replay" if has_report else "play_arrow"}')
        else:
            c['run'].set_text('Queue re-run' if has_report else 'Queue')
            c['run'].props('icon=playlist_add')
        c['run'].set_enabled(enabled)
        c['batch'].set_enabled(batch_ok)
        if c['opts']:
            c['report'].set_enabled(has_report)
            c['log'].set_enabled(r is not None)
            c['opts'].set_text(f'Options ({changed} changed)' if changed else 'Options')

    def tick():
        s = STATE['set']
        if view['loading'] and all(script in PARSERS for script in SCRIPTS.values()):
            reports.refresh()
        for name, c in cards.items():
            update_card(s, name, c)
        if 'calibration' in cards and cal_list_state(s) != view.get('cal_list'):
            cal_list.refresh()
        q = (tuple(map(id, QUEUE)), tuple(r.dir for r in RUNS.values() if r.live))
        if q != view['queue']:
            view['queue'] = q
            queue_view.refresh()
            n = len(QUEUE) + len(q[1])
            view['queue_badge'].set_text(str(n))
            view['queue_badge'].set_visibility(n > 0)
        used, limit = memory_use()
        set_text(view['mem'], f'Memory {gib(used)} of {gib(limit)}')
        for lr, lbl in view['live_rows']:
            set_text(lbl, f'{lr.elapsed}, {gib(lr.rss)} (peak {gib(lr.peak_rss)})')
        r = view['run']
        if r is not None and RUNS.get(r.dir) not in (None, r):
            # A queued re-run of the shown folder has started
            set_run(RUNS[r.dir])
            r = view['run']
        cancel_btn.set_visibility(bool(r and r.live) and not on_set_tab())
        status_label.set_visibility(not on_set_tab())
        if r is None:
            return
        for line in r.lines[view['n'] :]:
            log.push(line)
        view['n'] = len(r.lines)
        took = f', {"running for" if r.live else "took"} {r.elapsed}' if r.elapsed else ''
        mem = f', {gib(r.rss)} memory' if r.live else ''
        mem += f' (peak {gib(r.peak_rss)})' if r.peak_rss else ''
        set_text(status_label, f'{r.status}{took}{mem}')
        cmd.set_text(r.command)
        progress.set_text(r.progress)
        # Also watch the report file, as runs started by another process or before a restart
        # never change status here
        path = r.report()
        state = (r.status, path, path.stat().st_mtime if path else None)
        if state != view['state']:
            view['state'] = state
            pick_options()
            report.refresh()
            files.refresh()
            folder.refresh()

    def toggle_tasks():
        if tasks_col.visible:
            view['tasks_width'] = splitter.value
        tasks_col.set_visibility(not tasks_col.visible)
        splitter.set_value(view.get('tasks_width', 34) if tasks_col.visible else 0)

    with ui.header().classes('items-center gap-2'):
        ui.button(icon='menu', on_click=lambda: drawer.toggle()).props('flat round color=white').tooltip(
            'Show or hide the folder panel'
        )
        ui.button(icon='view_sidebar', on_click=lambda: toggle_tasks()).props('flat round color=white').tooltip(
            'Show or hide the report tasks'
        )
        brand('SKA-Low UV reports')
        set_title = ui.label().classes('text-subtitle1 opacity-80')
        ui.space()
        with ui.button('Queue', icon='queue').props('flat no-caps color=white'):
            view['queue_badge'] = ui.badge(color='accent').props('floating rounded')
            ui.tooltip('Running, queued and recently completed reports')
            with ui.menu().classes('p-4').style('width: 640px; max-width: 95vw') as view['queue_menu']:
                queue_view()
        ui.button('Calibration browser', icon='table_view', on_click=lambda: ui.navigate.to('sets/')).props(
            'flat no-caps color=white'
        ).tooltip('Browse, convert and update the calibration sets')
        reviewer_chip()
    with ui.left_drawer(value=True, bordered=True).props('width=360') as drawer:
        folder()
    with ui.splitter(value=34).classes('w-full') as splitter:
        with splitter.before, ui.column().classes('w-full pr-4 gap-2') as tasks_col:
            with ui.row().classes('w-full items-center no-wrap'):
                ui.label('Reports').classes('eyebrow grow')
                ui.button(icon='chevron_left', on_click=lambda: toggle_tasks()).props('flat dense').tooltip(
                    'Hide the report tasks (show again from the header)'
                )
            reports()
        with splitter.after, ui.column().classes('w-full pl-4'):
            with ui.row().classes('w-full items-center no-wrap gap-1'):
                run_set = ui.label().classes('text-subtitle1')
                run_pick = (
                    ui.select({}, on_change=lambda e: pick_run(e.value))
                    .props('dense borderless options-dense')
                    .classes('text-subtitle1')
                    .on('popup-show', lambda: pick_options())
                )
                status_label = ui.label('No run selected').classes('text-subtitle1 text-grey-7 grow ml-2')
                cancel_btn = ui.button(
                    'Cancel', icon='stop', color='negative', on_click=lambda: cancel_run(view['run'])
                ).props('outline dense')
            with ui.tabs(on_change=lambda: pick_options()).props(
                'active-color=primary indicator-color=secondary'
            ) as tabs:
                t_set = ui.tab('Set')
                t_log = ui.tab('Log')
                t_report = ui.tab('Report')
                t_files = ui.tab('Files')
            with ui.tab_panels(tabs, value=t_set).classes('w-full'):
                with ui.tab_panel(t_set):
                    set_info()
                with ui.tab_panel(t_log):
                    with ui.row().classes('w-full items-center no-wrap'):
                        cmd = ui.label().classes('font-mono text-xs break-all grow')
                        ui.button(icon='content_copy', on_click=lambda: ui.clipboard.write(cmd.text)).props(
                            'flat dense'
                        ).tooltip('Copy command')
                    progress = ui.label().classes('font-mono text-xs')
                    log = ui.log(max_lines=10000).classes('w-full h-[70vh]')
                with ui.tab_panel(t_report):
                    report()
                with ui.tab_panel(t_files):
                    files()
    pick_options()
    ui.timer(0.3, tick)
    if view['run']:
        show(view['run'])


# --- Calibration browser page: browse calibration_sets.xlsx, convert sets, update the database ----------------

# calibration_sets.xlsx 'Calibration Sets' column: grid field
SET_FIELDS = {
    'Set ID': 'set_id',
    'Date (UTC)': 'date',
    'LST (hr)': 'lst',
    'UTC Time': 'utc',
    'Perth Local Time (AWST)': 'awst',
    'Day/Night (AWST)': 'day',
    'Observations': 'n_obs',
    'Stations': 'stations',
    'Failed': 'n_failed',
    'Failed stations': 'failed_stations',
    'Status': 'status',
}
DETAIL_COLS = ('Station ID', 'Observation ID', 'UTC Start', 'n_files', 'Duration (s)', 'Status', 'error')
# Set folder options of mccs-convert-set, filled in by the page
# quantise has an optional value (--lossy [STEP]), so the page shows it as a switch and a step instead
CONVERT_SET_ARGS = ('set_id', 'xlsx', 'daq_dir', 'out_dir', 'list', 'quantise')
LOSSY_STEP = 0.5  # mccs-convert-set's default step
CONVERT, SPIDER = 'mccs_data_qa.convert_set', 'mccs_data_qa.cli'
SETS_CSS = """<style>
.sets-grid { --ag-font-family: 'Noto Sans', Verdana, sans-serif; --ag-font-size: 13px;
  --ag-header-background-color: #EEF0F6; --ag-header-text-color: #0A1D45; --ag-accent-color: #AF0B62;
  --ag-selected-row-background-color: #F7E6EF; --ag-row-hover-color: #F3F4F8; --ag-border-radius: 8px; }
.sets-grid .ag-row.sets-failed { background-color: #FFF4F4; }
.sets-grid .ag-row.sets-failed.ag-row-selected { background-color: #F7E6EF; }
.sets-act { display: inline-flex; align-items: center; gap: 6px; height: 26px; padding: 0 12px; border-radius: 13px;
  font: 600 12px 'Noto Sans', sans-serif; border: 1px solid transparent; cursor: pointer; line-height: 1; }
.sets-act.primary { background: #0A1D45; color: #fff; }
.sets-act.primary:hover { background: #AF0B62; }
.sets-act.ghost { background: transparent; color: #0A1D45; border-color: #C9CEDB; }
.sets-act.ghost:hover { border-color: #AF0B62; color: #AF0B62; }
.sets-act.warn { background: #FFE3E4; color: #9B1C22; }
.sets-act.busy { background: #EEF0F6; color: #5B6478; cursor: default; }
.sets-spin { width: 10px; height: 10px; border: 2px solid #C9CEDB; border-top-color: #AF0B62; border-radius: 50%;
  animation: sets-spin .8s linear infinite; }
@keyframes sets-spin { to { transform: rotate(360deg); } }
.sets-chip { display: inline-flex; align-items: center; gap: 4px; padding: 1px 8px; border-radius: 10px;
  font-size: 12px; line-height: 18px; color: #0A1D45; }
.sets-chip .material-icons, .sets-icons .material-icons { font-size: 15px; }
.sets-icons { display: inline-flex; gap: 4px; color: #5B6478; }
.sets-link { cursor: pointer; }
.sets-link:hover .sets-chip, .sets-link:hover.sets-icons { outline: 1px solid #AF0B62; }
</style>"""


def _chip(color: str, text: str, icon: str = '') -> str:
    """JS template literal for a chip; color, text and icon are JS expressions."""
    i = f'<span class="material-icons">${{{icon}}}</span>' if icon else ''
    return f'`<span class="sets-chip" style="background:${{{color}}}">{i}${{{text}}}</span>`'


ACTION_JS = """p => {
  const v = p.value || '';
  if (v === 'running') return '<span class="sets-act busy"><span class="sets-spin"></span>Converting</span>';
  if (v.startsWith('queued')) return `<span class="sets-act busy">Queued #${v.split(' ')[1]}</span>`;
  const [label, cls] = {convert: ['Convert', 'primary'], reconvert: ['Re-convert', 'ghost'],
                        failed: ['Retry', 'warn']}[v] || [v, 'ghost'];
  return `<button class="sets-act ${cls}">${label}</button>`;
}"""
QA_JS = f"""p => {{
  if (!p.value) return '<span style="color:#C1C6C8">—</span>';
  const hex = {json.dumps(QA_HEX)}, icon = {json.dumps(QA_ICON)};
  return `<span class="sets-link">` + {_chip("hex[p.value] || '#C1C6C8'", "p.value.replace('_', ' ')", "icon[p.value] || 'help_outline'")} + '</span>';
}}"""
STATUS_JS = f"""p => {_chip("p.value === 'FAILED' ? '#FFE3E4' : '#E3F5EA'", 'p.value')}"""
DAY_JS = """p => p.value ? `<span class="sets-icons"><span class="material-icons">${p.value === 'Day' ? 'wb_sunny' : 'bedtime'}</span>${p.value}</span>` : ''"""
REPORTS_JS = f"""p => {{
  const info = {json.dumps({n: [INFO[n][0], INFO[n][1]] for n in INFO})};
  const v = p.value || [];
  return v.length ? '<span class="sets-icons sets-link">' + v.map(n =>
    `<span class="material-icons" title="${{info[n][1]}}">${{info[n][0]}}</span>`).join('') + '</span>' : '';
}}"""
TIME_JS = "p => p.value ? p.value.slice(11, 16) : ''"
# Date filter on ISO strings: compare their date part with the picked local midnight
DATE_FILTER = """{comparator: (f, v) => { if (!v) return -1; const d = new Date(v.slice(0, 10) + 'T00:00:00');
  return d < f ? -1 : d > f ? 1 : 0; }, buttons: ['clear'], inRangeInclusive: true}"""


def set_columns() -> list[dict]:
    """AG Grid column definitions of the sets grid. Fields hold raw values; renderers draw chips and buttons."""
    num = {'filter': 'agNumberColumnFilter'}
    return [
        {
            'field': 'action',
            'headerName': '',
            'pinned': 'left',
            'width': 140,
            'sortable': False,
            'headerTooltip': 'Convert the set to UVX in the --root folder (options: Convert options)',
            'filter': False,
            ':cellRenderer': ACTION_JS,
        },
        {'field': 'set_id', 'headerName': 'Set', 'width': 95, 'sort': 'desc', **num},
        {
            'field': 'date',
            'headerName': 'Date (UTC)',
            'width': 150,
            'filter': 'agDateColumnFilter',
            ':filterParams': DATE_FILTER,
            ':valueFormatter': "p => p.value ? p.value.slice(0, 10) : ''",
        },
        {'field': 'qa', 'headerName': 'Set QA', 'width': 160, 'tooltipField': 'qa_tip', ':cellRenderer': QA_JS},
        {'field': 'reports', 'headerName': 'Reports', 'width': 130, ':cellRenderer': REPORTS_JS,
         'headerTooltip': 'Reports run on the set; click to open the set in the report UI'},
        {'field': 'status', 'headerName': 'Status', 'width': 105, ':cellRenderer': STATUS_JS},
        {'field': 'lst', 'headerName': 'LST (h)', 'width': 105, ':valueFormatter': 'p => p.value?.toFixed(1)', **num},
        {'field': 'utc', 'headerName': 'UTC', 'width': 95, ':valueFormatter': TIME_JS},
        {'field': 'awst', 'headerName': 'AWST', 'width': 95, ':valueFormatter': TIME_JS},
        {'field': 'day', 'headerName': 'Day/Night', 'width': 115, ':cellRenderer': DAY_JS},
        {'field': 'n_obs', 'headerName': 'Obs', 'width': 85, **num},
        {'field': 'stations', 'headerName': 'Stations', 'flex': 2, 'minWidth': 180, 'tooltipField': 'stations'},
        {'field': 'n_failed', 'headerName': 'Failed', 'width': 95, **num},
        {'field': 'failed_stations', 'headerName': 'Failed stations', 'flex': 1, 'minWidth': 140,
         'tooltipField': 'failed_stations'},
    ]


@functools.lru_cache(maxsize=2)
def _read_sets(xlsx: Path, mtime: float) -> tuple[list[dict], dict[int, list[dict]]]:
    """The set rows and their observations from calibration_sets.xlsx (cached per file version)."""
    import pandas as pd

    sheets = pd.read_excel(xlsx, sheet_name=['Calibration Sets', 'Set Details'])
    summary = sheets['Calibration Sets'].rename(columns=SET_FIELDS)
    rows = json.loads(summary[[c for c in SET_FIELDS.values() if c in summary]].to_json(orient='records', date_format='iso'))
    det = sheets['Set Details']
    det = det[[c for c in ('Set ID', *DETAIL_COLS) if c in det]]
    details = {
        int(k): json.loads(g.drop(columns='Set ID').to_json(orient='records', date_format='iso'))
        for k, g in det.groupby('Set ID')
    }
    return rows, details


def load_sets(xlsx: Path, root: Path) -> tuple[list[dict], dict[int, list[dict]]]:
    """Set rows for the grid, each with the folder it converts to, and the observations of each set.

    Returns ([], {}) if the workbook doesn't exist yet. Rows are copies, so a page can update them.
    """
    from mccs_data_qa.convert_set import set_folder

    if not xlsx.exists():
        return [], {}
    rows, details = _read_sets(xlsx, xlsx.stat().st_mtime)
    rows = [r | {'folder': set_folder(root, r['set_id'], r['date'])} | set_row_state(r, set()) for r in rows]
    return rows, details


def convert_dir(setdir: Path) -> Path:
    """Folder of a set's convert run (its run.log)."""
    return setdir / 'reports' / 'convert'


def tool_queued(d: Path) -> int | None:
    """1-based position of a run folder in the convert / database-update queue, or None."""
    return next((i + 1 for i, r in enumerate(TOOL_QUEUE) if r.dir == d), None)


def set_row_state(row: dict, existing: set[str]) -> dict:
    """The parts of a set's grid row that change: action, set QA (+ tooltip) and reports.

    Args:
        row (dict): Grid row, with 'folder'.
        existing (set): Names of the folders in the convert folder, so sets never converted cost no disk reads.
    """
    d = Path(row['folder']) if 'folder' in row else None
    st = {'action': 'convert', 'qa': '', 'qa_tip': 'Not converted', 'reports': []}
    if d is None:
        return st
    conv = convert_dir(d)
    r = TOOL_RUNS.get(conv)
    if pos := tool_queued(conv):
        st['action'] = f'queued {pos}'
    elif r is not None and r.live:
        st['action'] = 'running'
    if d.name not in existing:
        return st
    # mccs-convert-set exits 1 if any EB failed, but still writes the yaml: offer a retry either way
    rc = r.returncode if r is not None else read_json(conv / 'run.json').get('returncode')
    if st['action'] == 'convert' and rc not in (0, None):
        st['action'] = 'failed'
    if (d / SET_YAML).exists():
        if st['action'] == 'convert':
            st['action'] = 'reconvert'
        mark = read_json(set_qa_path(d))
        st['qa'] = mark.get('status') or 'unreviewed'
        tip = ', '.join(x for x in (by_line(mark), mark.get('note')) if x)
        st['qa_tip'] = tip or 'Not reviewed yet: click to open the set and mark it'
        st['reports'] = [n for n in SCRIPTS if (run_dir(d, n) / 'index.html').exists()]
    return st


def script_argv(fields: list[Field], values: dict) -> list[str]:
    """`build_argv()`, leaving out the options hidden for the --cal-model (e.g. --beam for mccs-*)."""
    model = values.get('cal_model')
    hidden = {d for d, models in CAL_MODEL_ONLY.items() if model and model not in models}
    return build_argv(fields, {f.dest: f.default if f.dest in hidden else values[f.dest] for f in fields})


def convert_argv(fields: list[Field], values: dict) -> list[str]:
    """mccs-convert-set options from the Convert options form: lossless unless the lossy switch is on."""
    step = values.get('step') or LOSSY_STEP
    return build_argv(fields, values) + ([f'--lossy={float(step)!r}'] if values.get('lossy') else [])


def tool_label(r: Run) -> str:
    """'Convert set614_2025-03-01' or 'Update database'."""
    return 'Update database' if r.script == SPIDER else f'Convert {r.dir.parent.parent.name}'


def tool_status(r: Run) -> tuple[str, str]:
    """(status text, badge colour) of a convert / database-update job."""
    if pos := tool_queued(r.dir):
        return f'queued #{pos}', STATUS_COLOR['queued']
    return r.status, STATUS_COLOR.get(r.status, 'negative')


@ui.page('/sets/')
async def sets_page():
    """Calibration browser: a grid of calibration_sets.xlsx with a Convert button per set, and the database update."""
    root, db = STATE['root'], STATE['db']
    xlsx = db / 'calibration_sets.xlsx'
    # Filled by reload_if_new() once the page is sent: reading the workbook can outlast the page's 3 s response_timeout
    rows, details, by_id = [], {}, {}
    view = {'set': None, 'run': None, 'n': 0, 'mtime': None, 'jobs': None}
    if PARSERS.get(CONVERT) is None:
        from mccs_data_qa.convert_set import build_parser

        PARSERS[CONVERT] = build_parser()
    fields = [f for f in form_fields(PARSERS[CONVERT]) if f.dest not in CONVERT_SET_ARGS]
    values = FORM_VALUES.setdefault(CONVERT, default_values(fields) | {'lossy': False, 'step': LOSSY_STEP})
    spider = FORM_VALUES.setdefault(SPIDER, {'all': False, 'upload': True})

    def existing() -> set[str]:
        return {p.name for p in root.iterdir()} if root.is_dir() else set()

    def refresh_rows():
        """Push the rows whose action, QA or reports changed, keeping the grid's filters, sort and scroll."""
        names = existing()
        changed = []
        for r in rows:
            st = set_row_state(r, names)
            if any(r[k] != v for k, v in st.items()):
                r.update(st)
                changed.append(r)
        if changed:
            grid.run_grid_method('applyTransaction', {'update': changed})
        if view['set'] is not None and view['set'] in {r['set_id'] for r in changed}:
            selected.refresh()

    async def reload_if_new():
        """Reload the grid when the database update rewrites calibration_sets.xlsx."""
        mtime = xlsx.stat().st_mtime if xlsx.exists() else None
        if mtime == view['mtime']:
            return
        first = view['mtime'] is None
        view['mtime'] = mtime
        try:
            new, det = await run.io_bound(load_sets, xlsx, root)
        finally:
            grid.run_grid_method('setGridOption', 'loading', False)  # Don't leave the overlay up on a bad workbook
        rows[:] = new
        details.clear()
        details.update(det)
        by_id.clear()
        by_id.update({r['set_id']: r for r in rows})
        # rows is the list in grid.options, refilled in place; assigning grid.options would rebuild the grid
        # and drop its filters and sort
        grid.run_grid_method('setGridOption', 'rowData', rows)
        updated.set_text(workbook_text())
        empty.set_visibility(not rows)
        refresh_rows()
        if not first:
            ui.notify(f'Reloaded {len(rows)} calibration sets')

    def workbook_text() -> str:
        return f'{xlsx.name}, updated {ago(view["mtime"])}' if view['mtime'] else f'No {xlsx}'

    def open_in_reports(d: Path):
        STATE['dir'], STATE['set'] = d.parent, d
        ui.navigate.to('../')

    async def convert(row: dict):
        d = Path(row['folder'])
        conv = convert_dir(d)
        if row['action'] == 'running' or row['action'].startswith('queued'):
            show_job(TOOL_RUNS.get(conv) or next(r for r in TOOL_QUEUE if r.dir == conv))
            return
        if (d / SET_YAML).exists() and not await confirm_reconvert(row, d):
            return
        argv = [
            str(row['set_id']),
            f'--xlsx={xlsx}',
            f'--daq-dir={STATE["daq_dir"]}',
            f'--out-dir={root}',
            *convert_argv(fields, values),
        ]
        r = enqueue_tool(conv, CONVERT, argv)
        show_job(TOOL_RUNS.get(conv) or r)
        refresh_rows()

    async def confirm_reconvert(row: dict, d: Path) -> bool:
        with ui.dialog() as dialog, ui.card().classes('max-w-md'):
            ui.label(f'Re-convert set {row["set_id"]}?').classes('text-h6')
            ui.label(f'This rewrites the UVX files and {SET_YAML} in {d}. Its reports are kept.')
            with ui.row().classes('w-full justify-end'):
                ui.button('Cancel', on_click=lambda: dialog.submit(False)).props('flat no-caps')
                ui.button('Re-convert', icon='replay', on_click=lambda: dialog.submit(True)).props(
                    'unelevated no-caps'
                )
        ok = await dialog
        dialog.delete()
        return bool(ok)

    def update_db():
        d = root / '.update_db'  # Hidden from the folder browser, and outside db/ so it isn't uploaded
        if TOOL_RUNS.get(d, Run(d, '', [], 0)).live or tool_queued(d):
            ui.notify('A database update is already running or queued')
            return
        argv = [str(STATE['daq_dir']), f'--outdir={db}', '--analyse']
        argv += ['--no-skip-existing'] * spider['all'] + ['--no-upload'] * (not spider['upload'])
        show_job(TOOL_RUNS.get(d) or enqueue_tool(d, SPIDER, argv))

    async def on_cell(e):
        a = e.args
        row = by_id.get((a.get('data') or {}).get('set_id'))
        if row is None:
            return
        view['set'] = row['set_id']
        selected.refresh()
        col = a.get('colId')
        if col == 'action':
            await convert(row)
        elif col in ('qa', 'reports') and (Path(row['folder']) / SET_YAML).exists():
            open_in_reports(Path(row['folder']))

    def show_job(r: Run):
        view.update(run=r, n=0)
        log.clear()
        jobs.refresh()

    @ui.refreshable
    def selected():
        row = by_id.get(view['set'])
        if row is None:
            ui.label('Click a set to see its observations. Convert starts from the button on each row.').classes(
                'text-xs text-grey-7'
            )
            return
        d = Path(row['folder'])
        converted = (d / SET_YAML).exists()
        ui.label(f'Calibration set {row["set_id"]}').classes('eyebrow')
        with ui.row().classes('w-full items-center gap-2'):
            lst = f'LST {row["lst"]:.1f} h' if row.get('lst') is not None else ''
            title = ' · '.join(x for x in ((row.get('date') or '')[:10], lst, row.get('day')) if x)
            ui.label(title).classes('text-h6')
            ui.space()
            if converted:
                ui.button('Open in reports', icon='open_in_new', on_click=lambda: open_in_reports(d)).props(
                    'flat dense no-caps'
                )
            if (convert_dir(d) / 'run.json').exists() or convert_dir(d) in TOOL_RUNS:
                ui.button(
                    'Convert log',
                    icon='terminal',
                    on_click=lambda: show_job(TOOL_RUNS.get(convert_dir(d)) or Run.load(convert_dir(d))),
                ).props('flat dense no-caps')
            label = {'convert': 'Convert', 'reconvert': 'Re-convert', 'failed': 'Retry'}.get(row['action'])
            ui.button(label or row['action'].capitalize(), icon='sync', on_click=lambda: convert(row)).props(
                'unelevated dense no-caps'
            ).set_enabled(bool(label))
        ui.label(str(d) + ('' if converted else '  (not converted)')).classes('text-xs text-grey-7 font-mono')
        obs = details.get(row['set_id'], [])
        if obs:
            keys = list(obs[0])
            cols = [{'name': k, 'label': k, 'field': k, 'align': 'left', 'sortable': True} for k in keys]
            fmt = lambda k, v: '' if v is None else str(v)[:19].replace('T', ' ') if k == 'UTC Start' else str(v)  # noqa: E731
            data = [{k: fmt(k, o[k]) for k in keys} for o in obs]
            ui.table(columns=cols, rows=data).props('dense flat').classes('w-full')

    @ui.refreshable
    def jobs():
        live = [r for r in TOOL_RUNS.values() if r.live]
        done = sorted((r for r in TOOL_RUNS.values() if not r.live), key=lambda r: r.finished or 0, reverse=True)
        items = live + TOOL_QUEUE + done[:N_RECENT]
        if not items:
            ui.label('No conversions or database updates since the UI started.').classes('text-xs text-grey-7')
        for r in items:
            status, color = tool_status(r)
            sel = 'skao-selected' if view['run'] is r else ''
            with ui.row().classes(f'w-full items-center no-wrap gap-1 px-1 rounded {sel}'):
                ui.label(tool_label(r)).classes('grow text-sm cursor-pointer').on('click', lambda r=r: show_job(r))
                ui.badge(status, color=color).props('rounded')
                if r.live or r.finished:
                    ui.label(r.elapsed).classes('text-xs text-grey-7')
                if r.live:
                    ui.button(icon='stop', color='negative', on_click=lambda r=r: cancel_run(r)).props(
                        'flat dense size=sm'
                    ).tooltip('Cancel')
                elif r in TOOL_QUEUE:
                    ui.button(
                        icon='play_arrow', color='positive', on_click=lambda r=r: (run_tool_now(r), refresh_rows())
                    ).props('flat dense size=sm').tooltip('Run now, alongside the running job')
                    ui.button(
                        icon='vertical_align_top',
                        on_click=lambda r=r: r in TOOL_QUEUE
                        and (TOOL_QUEUE.remove(r), TOOL_QUEUE.insert(0, r), refresh_rows()),
                    ).props('flat dense size=sm').tooltip('Move to top')
                    ui.button(
                        icon='delete', color='negative', on_click=lambda r=r: (TOOL_QUEUE.remove(r), refresh_rows())
                    ).props('flat dense size=sm').tooltip('Remove from the queue')

    def tick():
        refresh_rows()
        state = (tuple(map(id, TOOL_QUEUE)), tuple((id(r), r.status) for r in TOOL_RUNS.values()))
        if state != view['jobs']:
            view['jobs'] = state
            jobs.refresh()
        r = view['run']
        if r is None:
            return
        for line in r.lines[view['n'] :]:
            log.push(line)
        view['n'] = len(r.lines)
        status, _ = tool_status(r)
        took = f', {"running for" if r.live else "took"} {r.elapsed}' if r.elapsed else ''
        job_title.set_text(f'{tool_label(r)}: {status}{took}')
        cmd.set_text(r.command)
        progress.set_text(r.progress)

    ui.add_head_html(SETS_CSS)
    with ui.header().classes('items-center gap-2'):
        ui.button(icon='arrow_back', on_click=lambda: ui.navigate.to('../')).props('flat round color=white').tooltip(
            'Back to the reports'
        )
        brand('SKA-Low UV reports')
        ui.label('Calibration browser').classes('text-subtitle1 opacity-80')
        ui.space()
        reviewer_chip()

    with ui.column().classes('w-full gap-3'):
        with ui.row().classes('w-full items-center gap-2'):
            search = (
                ui.input(placeholder='Search all columns: a station, date, Night, FAILED...')
                .props('dense outlined clearable')
                .classes('w-96')
            )
            search.on_value_change(lambda e: grid.run_grid_method('setGridOption', 'quickFilterText', e.value or ''))
            ui.button(
                'Clear filters',
                icon='filter_alt_off',
                on_click=lambda: (search.set_value(''), grid.run_grid_method('setFilterModel', None)),
            ).props('flat dense no-caps')
            ui.button(
                'CSV',
                icon='download',
                on_click=lambda: grid.run_grid_method('exportDataAsCsv', {'fileName': 'calibration_sets.csv'}),
            ).props('flat dense no-caps').tooltip('Download the filtered rows')
            ui.space()
            updated = ui.label(workbook_text()).classes('text-xs text-grey-7')
            with ui.button('Convert options', icon='tune').props('flat dense no-caps'):
                with ui.menu().classes('p-3 w-96'), ui.column().classes('w-full gap-1'):
                    ui.label('mccs-convert-set options').classes('eyebrow')
                    ui.label("Used by every set's Convert button.").classes('text-xs text-grey-7')
                    for f in fields:
                        _field_widget(f, values)
                    ui.switch('--lossy').bind_value(values, 'lossy')
                    with ui.column().classes('w-full gap-0') as lossy:
                        ui.number('STEP', min=0.01, step=0.1).bind_value(values, 'step').classes('w-full')
                    lossy.bind_visibility_from(values, 'lossy')
                    ui.label(
                        'Off: lossless (zstd). On: store crosses as int16, quantised to STEP x thermal noise '
                        f'(default {LOSSY_STEP}, ~4x smaller).'
                    ).classes('text-xs text-grey-7')
            with ui.button('Update database', icon='cloud_sync').props('unelevated dense no-caps'):
                with ui.menu().classes('p-3 w-96') as db_menu, ui.column().classes('w-full gap-1'):
                    ui.label('Update database').classes('eyebrow')
                    ui.label(
                        f'Runs mccs-spider on {STATE["daq_dir"]}, then mccs-analyse, rewriting {db}/latest.csv '
                        'and calibration_sets.xlsx. The table reloads when it finishes.'
                    ).classes('text-xs text-grey-7')
                    ui.switch('Re-spider every folder (slow)').bind_value(spider, 'all')
                    ui.switch('Upload to acacia (rclone)').bind_value(spider, 'upload')
                    ui.button(
                        'Update now', icon='play_arrow', on_click=lambda: (db_menu.close(), update_db())
                    ).props('unelevated no-caps')
        empty = ui.label(f'No {xlsx} yet: run Update database to build it.').classes('text-orange-9')
        empty.set_visibility(not xlsx.exists())
        grid = ui.aggrid(
            {
                'columnDefs': set_columns(),
                'rowData': rows,
                # Loading overlay until reload_if_new() fills the rows; with no workbook there is nothing to load
                'loading': xlsx.exists(),
                'defaultColDef': {
                    'sortable': True,
                    'resizable': True,
                    'filter': 'agTextColumnFilter',
                    'floatingFilter': True,
                    'suppressHeaderMenuButton': True,
                },
                ':getRowId': 'p => String(p.data.set_id)',
                ':rowClassRules': "{'sets-failed': p => p.data.status === 'FAILED'}",
                'rowSelection': {'mode': 'singleRow', 'checkboxes': False, 'enableClickSelection': True},
                'pagination': True,
                'paginationPageSize': 50,
                'paginationPageSizeSelector': [25, 50, 100, 500],
                'tooltipShowDelay': 300,
                'animateRows': False,
            },
            auto_size_columns=False,
        ).classes('sets-grid w-full').style('height: 62vh')
        grid.on('cellClicked', on_cell)

        with ui.row().classes('w-full no-wrap items-start gap-3'):
            with ui.card().props('flat bordered').classes('w-1/2'):
                selected()
            with ui.card().props('flat bordered').classes('w-1/2'):
                ui.label('Jobs').classes('eyebrow')
                ui.label('Conversions and database updates run one at a time, alongside the reports.').classes(
                    'text-xs text-grey-7'
                )
                jobs()
                job_title = ui.label('No job selected').classes('text-subtitle1 mt-2')
                with ui.row().classes('w-full items-center no-wrap'):
                    cmd = ui.label().classes('font-mono text-xs break-all grow')
                    ui.button(icon='content_copy', on_click=lambda: ui.clipboard.write(cmd.text)).props(
                        'flat dense'
                    ).tooltip('Copy command')
                progress = ui.label().classes('font-mono text-xs')
                log = ui.log(max_lines=10000).classes('w-full h-[40vh]')
    ui.timer(1.0, tick)
    ui.timer(5.0, reload_if_new)
    await ui.context.client.connected()
    await reload_if_new()


app.on_startup(preload_parsers)


@app.on_shutdown
def _stop_live():
    for r in [*RUNS.values(), *TOOL_RUNS.values()]:
        if r.live:
            _kill_group(r, signal.SIGKILL)


def main():
    cli = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_argument(
        '--root', type=Path, default=DEFAULT_ROOT, help='Folder of calibration sets to start in, and convert sets to'
    )
    cli.add_argument('--port', type=int, default=8080)
    cli.add_argument(
        '--db', type=Path, default=Path('db'), help='Database folder: latest.csv and calibration_sets.xlsx'
    )
    cli.add_argument('--daq-dir', type=Path, default=STATE['daq_dir'], help='Folder of the eb-* DAQ folders')
    opts = cli.parse_known_args()[0]
    STATE['dir'] = STATE['root'] = opts.root.expanduser().resolve()
    STATE['db'], STATE['daq_dir'] = opts.db.expanduser().resolve(), opts.daq_dir.expanduser()
    if hub := os.environ.get('JUPYTERHUB_SERVICE_PREFIX'):
        print(f'In JupyterHub, open {hub}proxy/{opts.port}/ (needs jupyter-server-proxy)', flush=True)
    # 127.0.0.1, as the default 0.0.0.0 would expose a no-login launcher of arbitrary report jobs
    # The secret only signs the cookie that keeps each browser's reviewer initials; it is not a login
    secret = os.environ.get('MCCS_REPORT_UI_SECRET', 'mccs-report-ui')
    ui.run(
        host='127.0.0.1', port=opts.port, reload=False, show=False, title='SKA-Low UV reports', storage_secret=secret
    )


if __name__ in {'__main__', '__mp_main__'}:
    main()
