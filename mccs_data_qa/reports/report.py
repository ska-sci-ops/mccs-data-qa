"""Shared command-line arguments and HTML report page (report_template.html.j2) for the report scripts."""

import argparse
import glob
import html
import inspect
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import astropy.units as u
import matplotlib as mpl
import numpy as np
from jinja2 import Environment, FileSystemLoader
from loguru import logger

from ska_ost_low_uv.io import read_uvx
from ska_ost_low_uv.postx.aeps.eep_models import EEP_DIR_ENV, _eep_path
from ska_ost_low_uv.postx.calibration.flagging import (
    apply_channel_flags,
    flag_antennas_multichannel,
    flag_channels,
    flag_channels_from_cal,
)
from ska_ost_low_uv.postx.calibration.simple_cal import simple_stefcal, simple_stefcal_4pol
from ska_ost_low_uv.postx.calibration.suncal import suncal
from ska_ost_low_uv.postx.sky_model import RadioSource
from ska_ost_low_uv.utils import get_resource_path

# Report folders under a calibration set's reports/ folder, in the order of the nav bar between them.
# calibration/ holds a folder per calibration run: one per stefcal option combination, and HOLO_RUN.
REPORTS = ('quicklook', 'calibration', 'delay_fit', 'sensitivity')
HOLO_RUN = 'sun-holography'
# Report plots open full size in a modal, so are saved above matplotlib's 100 dpi
mpl.rcParams['savefig.dpi'] = 150
_ENV = Environment(loader=FileSystemLoader(Path(__file__).parent), trim_blocks=True, lstrip_blocks=True)
_SUNCAL_ARGS = ('sun_fwhm', 'n_flux_iter', 'min_gsm_snr', 'min_sun_alt')
_SUNCAL_DEFAULTS = {k: inspect.signature(suncal).parameters[k].default for k in _SUNCAL_ARGS}
_SUNCAL_HELP = {
    'sun_fwhm': (float, 'FWHM of the Gaussian Sun model (deg)'),
    'n_flux_iter': (int, 'Maximum suncal Sun flux fits'),
    'min_gsm_snr': (float, 'GSM detection S/N needed for suncal to determine the Sun flux'),
    'min_sun_alt': (float, 'Sun elevation (deg) below which suncal calibrates against the sky model only'),
}
# The sky turns 0.25 deg in 60 s, small against the ~3 deg station beam
CHANNEL_TIME_TOL_S = 60
CAL_MODELS = ('sun+sky', 'sky', 'sun')
LEAKAGE = ('solve', 'none')
DEFAULT_SIGMA_THR = 5
# Fixed 3x6 AA1 station layout, a row per cluster (s8, s9, s10): subplot (rows, cols, index) per station
AA1_GRID_ROWS, AA1_GRID_COLS = 3, 6
AA1_GRID = {f's{c}-{i}': (3, 6, r * 6 + i) for r, c in enumerate((8, 9, 10)) for i in range(1, 7)}


def _beams() -> dict:
    """--beam choices, e.g. 'aep-harp': element pattern kwargs per packaged AEP and per $POSTX_EEP_DIR EEP.

    EEP files are not packaged, so there are no EEP choices if $POSTX_EEP_DIR is unset.
    """
    aeps = [p.stem for p in Path(get_resource_path('postx/data/aeps')).glob('*.h5') if '_jones' not in p.stem]
    eep_dir = os.environ.get(EEP_DIR_ENV)
    eeps = [p.stem for p in Path(eep_dir).glob('*.h5')] if eep_dir else []
    beams = {f'aep-{n.removeprefix("ska_low_")}': {'aep': n} for n in aeps}
    beams |= {f'eep-{n.removeprefix("ska_low_")}': {'eep': n} for n in eeps}
    return dict(sorted(beams.items()))


BEAMS = _beams()  # Element pattern keyword arguments


def input_parser(description: str, outdir: str, static_flags: bool = True) -> argparse.ArgumentParser:
    """Argument parser with the input and output options shared by the report scripts.

    Args:
        description (str): Parser description, usually the script's module docstring.
        outdir (str): Default output directory.
        static_flags (bool): Add --ignore-static-flags, for scripts that use the UVX antenna flags.

    Returns:
        parser (argparse.ArgumentParser): Parser with --datadir, --eb and --outdir.
    """
    p = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    io = p.add_argument_group('input and output')
    io.add_argument('--datadir', required=True, help='Folder of UVX station files')
    io.add_argument('--eb', default='eb-t0001-20260505-0000*', help='Glob pattern for station files')
    io.add_argument('--outdir', default=outdir, help='Output directory for the report')
    if static_flags:
        p.add_argument(
            '--ignore-static-flags',
            action='store_true',
            help='Ignore the static antenna flags stored in the UVX files',
        )
    return p


def base_parser(description: str, outdir: str, sigma_thr: float | None = None) -> argparse.ArgumentParser:
    """Argument parser with the input, output and calibration options shared by the report scripts.

    Args:
        description (str): Parser description, usually the script's module docstring.
        outdir (str): Default output directory.
        sigma_thr (float): Default stefcal/suncal antenna outlier threshold. None leaves out the
            calibration options other than --sky-model, for scripts that read an existing cal.

    Returns:
        parser (argparse.ArgumentParser): Parser with input, sky model and Sun calibration options.
    """
    p = input_parser(description, outdir)

    cal = p.add_argument_group('calibration')
    if sigma_thr is None:
        cal.add_argument('--sky-model', default='gsm16', help='Diffuse sky model, also the flux reference')
        return p
    p.add_argument(
        '--sigma-thr',
        type=float,
        default=sigma_thr,
        help='Flag antennas with gain outliers beyond this many sigma',
    )
    cal.add_argument(
        '--cal-model',
        choices=CAL_MODELS,
        default='sun+sky',
        help='sun+sky: the diffuse sky plus the Sun as an extended source when it is up (suncal); sky: the '
        'diffuse sky only; sun: the Sun only, as a point source (no flux scale)',
    )
    cal.add_argument('--sky-model', default='gsm16', help='Diffuse sky model, also the flux reference')
    cal.add_argument(
        '--leakage',
        choices=LEAKAGE,
        default='solve',
        help='solve: full 2x2 Jones gains, with XY/YX leakage; none: diagonal X and Y gains only',
    )
    cal.add_argument(
        '--beam',
        choices=list(BEAMS),
        default='aep-harp',
        help='Element pattern for the sky model and its EM model: aep: average, packaged; eep: per-antenna '
        f'embedded, from ${EEP_DIR_ENV} ({os.environ.get(EEP_DIR_ENV, "unset")})',
    )
    add_suncal_args(cal)
    return p


def add_suncal_args(group, names: tuple = _SUNCAL_ARGS):
    """Add suncal() options, with suncal()'s defaults, to a parser or argument group.

    Args:
        group: argparse parser or argument group.
        names (tuple): suncal() parameter names, a subset of 'sun_fwhm', 'n_flux_iter', 'min_gsm_snr' and
            'min_sun_alt'.
    """
    for k in names:
        typ, help_ = _SUNCAL_HELP[k]
        group.add_argument(f'--{k.replace("_", "-")}', type=typ, default=_SUNCAL_DEFAULTS[k], help=help_)


def read_station(fn: str, ignore_static_flags: bool = False):
    """Read a UVX station file, optionally clearing its static antenna flags."""
    uvx = read_uvx(fn)
    if ignore_static_flags:
        uvx.antennas.attrs['flags'][:] = False
    return uvx


def station_key(name: str) -> list:
    """Natural sort key, so s8-1 < s9-1 < s10-1. Works on station names and file names."""
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', name)]


def drop_repeated_stations(files: list[str], stations=None) -> list[str]:
    """Keep one file per station, in station order (s8-1, ..., s10-6), warning about repeats.

    A repeated station keeps the file whose start time is closest to the mean start time of the
    non-repeated stations (or of all files, if every station repeats).

    Args:
        files (list): UVX files.
        stations (Iterable): Allowed station names, e.g. `AA1_GRID`. None allows any.

    Raises:
        ValueError: If a station appears in more than two files, or is not in `stations`.
    """
    by_name = {}
    for fn in files:
        uvx = read_uvx(fn)
        by_name.setdefault(uvx.name, []).append((fn, uvx.timestamps[0].unix))
    if stations is not None and (unknown := sorted(set(by_name) - set(stations), key=station_key)):
        raise ValueError(f'Stations not in the AA1 layout: {unknown}')
    too_many = {n: [os.path.basename(fn) for fn, _ in v] for n, v in by_name.items() if len(v) > 2}
    if too_many:
        raise ValueError(f'Stations with more than two files, narrow --eb to select one scan: {too_many}')

    singles = [v[0][1] for v in by_name.values() if len(v) == 1]
    t_ref = np.mean(singles or [t for v in by_name.values() for _, t in v])
    keep = []
    for name, v in sorted(by_name.items(), key=lambda kv: station_key(kv[0])):
        best = min(v, key=lambda x: abs(x[1] - t_ref))[0]
        if len(v) > 1:
            dropped = [os.path.basename(fn) for fn, _ in v if fn != best]
            logger.warning(f'{name} has {len(v)} files: using {os.path.basename(best)}, ignoring {dropped}')
        keep.append(best)
    return keep


def find_station_files(parser: argparse.ArgumentParser, args: argparse.Namespace, stations=None) -> list[str]:
    """UVX files matching --datadir and --eb, one per station, or exit with a parser error.

    Args:
        parser (argparse.ArgumentParser): Parser to report errors with.
        args (argparse.Namespace): Parsed arguments with `datadir` and `eb`.
        stations (Iterable): Allowed station names, see `drop_repeated_stations()`.

    Returns:
        files (list): One file per station, in station order.
    """
    files = sorted(glob.glob(os.path.join(glob.escape(args.datadir), f'{args.eb}.uvx')))
    if not files:
        parser.error(f'No files match {os.path.join(args.datadir, args.eb)}.uvx')
    try:
        return drop_repeated_stations(files, stations)
    except ValueError as err:
        parser.error(str(err))


def parse_freq_ranges(s: str) -> list:
    """Parse comma-separated MHz ranges, e.g. '87.5-108,137-138', into (start, stop) Quantity pairs.

    Raises:
        ValueError: If a range is not two numbers separated by '-', or starts above its end.
    """
    ranges = [tuple(float(x) * u.MHz for x in r.split('-')) for r in s.split(',') if r.strip()]
    if any(len(r) != 2 or r[0] > r[1] for r in ranges):
        raise ValueError(f'Cannot parse frequency ranges {s!r}')
    return ranges


def parse_flag_freqs(parser: argparse.ArgumentParser, s: str) -> list:
    """`parse_freq_ranges()` of --flag-freqs, or exit with a parser error."""
    try:
        return parse_freq_ranges(s)
    except ValueError:
        parser.error(f'Cannot parse --flag-freqs {s!r}, expected increasing ranges, e.g. 87.5-108,137-138')


def cal_args_problem(cal_model: str, leakage: str, beam: str) -> str:
    """Why a combination of calibration options can't be run, or '' if it can."""
    if cal_model == 'sun+sky' and leakage == 'none':
        return 'sun+sky (suncal) solves full-Jones gains only: use --leakage solve, or another --cal-model.'
    if 'eep' in BEAMS.get(beam, {}):
        try:
            _eep_path(BEAMS[beam]['eep'])
        except FileNotFoundError as err:
            return str(err)
    return ''


def check_cal_args(parser: argparse.ArgumentParser, args: argparse.Namespace):
    """Exit with a parser error if the calibration options can't be run together."""
    if problem := cal_args_problem(args.cal_model, args.leakage, args.beam):
        parser.error(problem)


def cal_variant(cal_model: str, leakage: str, beam: str, sky_model: str = 'gsm16') -> str:
    """Folder name of a stefcal run with these options, e.g. 'sun-gsm16_leakage_aep-harp'."""
    model = {'sun+sky': f'sun-{sky_model}', 'sky': sky_model, 'sun': 'sun'}[cal_model]
    return f'{model}_{"leakage" if leakage == "solve" else "noleakage"}_{beam}'


def station_cal_model(aa, cal_model: str) -> str:
    """The model to calibrate a station with: sun+sky becomes sky when the Sun is down.

    Raises:
        ValueError: For a Sun-only cal with the Sun below the horizon.
    """
    if aa.coords.get_alt_az(aa.coords.get_sun()).alt.deg > 0:
        return cal_model
    if cal_model == 'sun':
        raise ValueError(f'{aa.name}: the Sun is below the horizon, so it cannot be calibrated on the Sun only')
    return 'sky'


def calibrate_channel(
    aa,
    cal_model: str,
    leakage: str,
    beam: str,
    sky_model: str,
    sigma_thr: float,
    antenna_flags=None,
    suncal_kwargs: dict | None = None,
):
    """Calibrate the selected channel of a station.

    Args:
        aa (ApertureArray): Station to calibrate, at its selected channel.
        cal_model (str): 'sun+sky' (suncal), 'sky' or 'sun', see `station_cal_model()`.
        leakage (str): 'solve' for full-Jones gains, 'none' for diagonal gains.
        beam (str): Element pattern, a key of `BEAMS`.
        sky_model (str): Diffuse sky model.
        sigma_thr (float): Antenna gain outlier threshold.
        antenna_flags (np.ndarray): Antenna flags.
        suncal_kwargs (dict): suncal() options.

    Returns:
        cal (UVXAntennaCal): The calibration.
        dg (float): Final StefCal convergence: dG for full-Jones gains, the larger of the two pols' z
            for diagonal gains.
    """
    suncal_kwargs = suncal_kwargs or {}
    common = {'sigma_thr': sigma_thr, 'antenna_flags': antenna_flags, **BEAMS[beam]}
    if cal_model == 'sun+sky':
        cal = suncal(aa, diffuse_model=sky_model, **common, **suncal_kwargs)
    else:
        if cal_model == 'sun':
            model = {'sky_model': {'Sun': RadioSource(aa.coords.get_sun(), mag=1.0)}}  # Point source
        else:
            model = {'diffuse_model': sky_model}
        stefcal = simple_stefcal_4pol if leakage == 'solve' else simple_stefcal
        cal = stefcal(aa, diffuse_model_units='Jysr', **model, **common)
        if cal_model == 'sun':
            # The Sun model is 1 Jy, so the gains have no flux scale
            cal.provenance.pop('unit', None)
    if 'stefcal_4pol' in cal.provenance:
        return cal, cal.provenance['stefcal_4pol']['dG']
    return cal, max(p['z'] for p in cal.provenance['stefcal'].values())


def calibrate_sweep(aa, calibrate, flag_freqs: list | None = None):
    """Flag antennas and RFI channels of a station sweep, calibrate it, then flag channels on the cal.

    Flagged channels are those with a bad timestamp (see `bad_channel_times()`), in `flag_freqs`, already
    flagged in the cal returned by `calibrate`, or with RFI in the autocorrelation or gain spectra (see
    `flag_channels()` and `flag_channels_from_cal()`).

    Args:
        aa (ApertureArray): Station sweep.
        calibrate (callable): calibrate(aa, antenna_flags) returning the UVXAntennaCal of every channel.
            Per-channel StefCal convergence in its `provenance['sweep']['dG']`, if any, is used in the
            gain-spectrum channel flagging.
        flag_freqs (list): (start, stop) Quantity frequency ranges to flag, e.g. known RFI bands.

    Returns:
        cal (UVXAntennaCal): The cal, with flagged channels zeroed.
    """
    ant_flags = flag_antennas_multichannel(aa)
    rfi = flag_channels(aa, antenna_flags=ant_flags, initial_flags=flag_freqs)
    cal = calibrate(aa, ant_flags)
    bad = bad_channel_times(aa.uvx) | rfi | cal.flags.channel.values
    dg = cal.provenance.get('sweep', {}).get('dG')
    bad |= flag_channels_from_cal(cal, dG=dg, initial_flags=bad)
    return apply_channel_flags(cal, bad)


def suncal_kwargs_from_args(args: argparse.Namespace) -> dict:
    """suncal() keyword arguments from parsed `base_parser()` arguments."""
    return {k: getattr(args, k) for k in _SUNCAL_ARGS}


def table(header: list, rows: list) -> str:
    """HTML table with one header row."""
    th = ''.join(f'<th>{h}</th>' for h in header)
    trs = ''.join('<tr>' + ''.join(f'<td>{c}</td>' for c in row) + '</tr>' for row in rows)
    return f'<table><tr>{th}</tr>{trs}</table>'


def args_table(args: dict) -> str:
    """Settings table of command-line arguments, from {dest: value}."""
    return table(
        ['Argument', 'Value'], [(f'--{k.replace("_", "-")}', html.escape(str(v))) for k, v in args.items()]
    )


def figs(names: list, prefix: str = 'plots/', captions: list | None = None) -> str:
    """Grid of linked PNG figures `{prefix}{name}.png`, captioned by name unless captions are given."""
    captions = captions or names
    items = ''.join(
        f'<figure><a href="{prefix}{n}.png"><img src="{prefix}{n}.png" alt="{n}" loading="lazy"></a>'
        f'<figcaption>{cap}</figcaption></figure>'
        for n, cap in zip(names, captions, strict=True)
    )
    return f'<div class="fig-grid">{items}</div>'


def subtabs(tabs: list) -> str:
    """Tab row nested inside a report tab, from (id, label, html) per tab; the first is shown."""
    btns = ''.join(
        f'<button class="tab-btn{" active" if i == 0 else ""}" data-tab="{tid}">{label}</button>'
        for i, (tid, label, _) in enumerate(tabs)
    )
    panels = ''.join(
        f'<div class="tab-panel{" active" if i == 0 else ""}" id="{tid}">{body}</div>'
        for i, (tid, _, body) in enumerate(tabs)
    )
    return f'<div class="tabs">{btns}</div>{panels}'


def cluster_tabs(tab_id: str, items: dict) -> str:
    """Sub-tabs per station cluster (S8, S9, ...) from {station: [(stem, caption), ...]}."""
    groups = {}
    for name, stems in items.items():
        groups.setdefault(name.split('-')[0].upper(), []).extend(stems)
    return subtabs(
        [(f'{tab_id}-{g}', g, figs([s for s, _ in v], captions=[c for _, c in v])) for g, v in groups.items()]
    )


def obs_start(uvx) -> tuple[str, str]:
    """UTC and apparent LST of the first integration, as strings."""
    t0 = uvx.timestamps[0]
    lst = t0.sidereal_time('apparent', longitude=uvx.origin.lon)
    return t0.strftime('%Y-%m-%d %H:%M:%S'), lst.to_string(unit='hourangle', sep=':', precision=0, pad=True)


def bad_channel_times(uvx, tol_s: float = CHANNEL_TIME_TOL_S) -> np.ndarray:
    """Channels of a frequency sweep whose timestamp is off the sweep's regular time step.

    Sweeps step through channels at a fixed interval. A channel more than `tol_s` from that line
    has a bad timestamp (e.g. the DAQ session start time), so its sky model is for the wrong time.
    Logs a warning listing any bad channels.

    Args:
        uvx (UVX): Station data. Without per-channel times (not a sweep), no channels are bad.
        tol_s (float): Allowed offset from the regular time step, in seconds.

    Returns:
        bad (np.ndarray): (n_freq,) boolean, True for channels with a bad timestamp.
    """
    n = uvx.data.sizes['frequency']
    if 'channel_unix' not in uvx.data.coords or n < 3:
        return np.zeros(n, dtype=bool)
    t = uvx.data.coords['channel_unix'].values.astype('float64')
    step = np.median(np.diff(t))
    idx = np.arange(n)
    t_fit = np.median(t - idx * step) + idx * step
    bad = np.abs(t - t_fit) > tol_s
    if bad.any():
        logger.warning(
            f'{uvx.name}: {bad.sum()} channel(s) with bad timestamps, off the sweep time step by '
            f'{np.round(t[bad] - t_fit[bad]).astype(int).tolist()} s: channels {np.flatnonzero(bad).tolist()}'
        )
    return bad


def uvx_metadata(fname: str, uvx) -> dict:
    """Header metadata of a UVX file as {field: value}, for side-by-side comparison of stations."""
    d = uvx.data
    f_mhz = d.frequency.values / 1e6
    fa, ta = d.frequency.attrs, d.time.attrs
    utc0, lst0 = obs_start(uvx)
    loc = uvx.origin.to_geodetic()
    pc = uvx.phase_center
    meta = {'File': html.escape(os.path.basename(fname)), 'Name': uvx.name}
    meta |= {
        f'Context: {k}': html.escape(str(v))
        for k, v in uvx.context.items()
        if k != 'description' and not isinstance(v, dict)
    }
    meta |= {
        'UTC start': utc0,
        'UTC end': uvx.timestamps[-1].strftime('%Y-%m-%d %H:%M:%S'),
        'LST start': lst0,
        'Integrations': f'{d.sizes["time"]} × {ta.get("resolution", float("nan")):.3g} s',
        'Channels': f'{d.sizes["frequency"]}: {f_mhz.min():.2f}&ndash;{f_mhz.max():.2f} MHz',
        'Channel spacing / bandwidth [kHz]': (
            f'{fa.get("channel_spacing", float("nan")) / 1e3:.1f} / {fa.get("channel_bandwidth", float("nan")) / 1e3:.1f}'
        ),
        'Polarizations': ', '.join(d.polarization.values),
        'Antennas (flagged in file)': (
            f'{uvx.antennas.sizes["antenna"]} ({int(uvx.antennas.attrs["flags"].values.sum())})'
        ),
        'Data unit': d.attrs.get('unit', '?'),
        'Phase tracking applied': bool(d.attrs.get('phase_center_tracking_applied', False)),
        'Phase centre (RA, Dec) [deg]': f'{pc.ra.deg.ravel()[0]:.3f}, {pc.dec.deg.ravel()[0]:.3f}',
        'Origin (lat, lon) [deg], height [m]': f'{loc.lat.deg:.5f}, {loc.lon.deg:.5f}, {loc.height.value:.1f}',
        'Input files': len([k for k in uvx.provenance.get('input_files', {}) if k != 'description']),
        'Bad channel timestamps': ', '.join(map(str, np.flatnonzero(bad_channel_times(uvx)))) or 'none',
    }
    versions = uvx.provenance.get('ska_ost_low_uv_config', {})
    meta |= {f'Written with {k}': v for k, v in versions.items() if k != 'description'}
    return meta


def metadata_table(stations: list, files: list, uvxs: list) -> str:
    """Table of UVX header metadata, one column per station."""
    metas = [uvx_metadata(fn, u) for fn, u in zip(files, uvxs, strict=True)]
    fields = list(dict.fromkeys(k for m in metas for k in m))
    return table(['Field', *stations], [(k, *(m.get(k, '') for m in metas)) for k in fields])


def write_redirect(path: str | Path, target: str):
    """Write an HTML page at `path` that redirects to the relative URL `target`."""
    Path(path).write_text(
        f'<!DOCTYPE html><meta http-equiv="refresh" content="0; url={target}"><a href="{target}">{target}</a>'
    )


def write_report(path: str | Path, title: str, info: list, tabs: list, version: str, card_extra: str = ''):
    """Render the tabbed report page to `path`.

    A report written to `<set>/reports/<name>/`, or a run of it in `<set>/reports/<name>/<run>/`, gets a
    nav bar linking the other reports of the set.

    Args:
        path (str | Path): Output HTML file.
        title (str): Page title and heading.
        info (list): (label, value) pairs for the summary card.
        tabs (list): (id, label, html) per tab; the first is shown on load.
        version (str): Software version string for the subtitle.
        card_extra (str): HTML placed below the summary card's info grid.
    """
    path = Path(path).resolve()
    # Levels from the page's folder up to reports/
    up = next((i for i in (1, 2) if i < len(path.parents) and path.parents[i].name == 'reports'), None)
    html = _ENV.get_template('report_template.html.j2').render(
        nav=REPORTS if up else (),
        nav_prefix='../' * (up or 0),
        current=path.parents[up - 1].name if up else '',
        set_name=path.parents[up + 1].name if up else '',
        title=title,
        generated_at=datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC'),
        version=version,
        info=info,
        tabs=tabs,
        card_extra=card_extra,
    )
    path.write_text(html)
