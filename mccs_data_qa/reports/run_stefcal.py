"""run_stefcal.py: Calibrate AA1 station sweeps across frequency and generate a calibration report webpage.

This is the step that writes the `{station}.cal` files used by delay_fit_report.py and
station_sensitivity.py. Calibrates every channel of each station against the model chosen by
--cal-model, --leakage and --beam, writes `{station}.cal`, and plots the gain magnitude and phase
against frequency. Images at a single channel are in quicklook_aa1.py.

Each combination of options is a run, written to its own folder `<outdir>/<run>/`, e.g.
`calibration/sun-gsm16_leakage_aep-harp/`. `<outdir>/index.html` compares the runs, including a Sun
holography run from holo_suncal.py, or redirects to the only one.

Usage:
    python -m mccs_data_qa.reports.run_stefcal --datadir DIR --eb 'eb-t0001-20260505-0000*' --outdir calibration
"""

import argparse
import os
import warnings
from pathlib import Path

import numpy as np
import pylab as plt
from loguru import logger
from mccs_data_qa.reports.report import (
    AA1_GRID,
    AA1_GRID_COLS,
    AA1_GRID_ROWS,
    DEFAULT_SIGMA_THR,
    args_table,
    base_parser,
    cal_variant,
    calibrate_channel,
    calibrate_sweep,
    check_cal_args,
    figs,
    find_station_files,
    metadata_table,
    obs_start,
    parse_flag_freqs,
    read_station,
    station_cal_model,
    station_key,
    suncal_kwargs_from_args,
    table,
    write_redirect,
    write_report,
)

from ska_ost_low_uv import show_version
from ska_ost_low_uv.io import read_cal, write_cal
from ska_ost_low_uv.parallelize import run_in_parallel, task
from ska_ost_low_uv.postx import ApertureArray
from ska_ost_low_uv.postx.calibration.suncal import merge_suncal_provenance
from ska_ost_low_uv.postx.channel_utils import iterate_channels_cal

ANT_FLAG_FRACTION = 0.5  # An antenna is flagged if zeroed in more than this fraction of unflagged channels


def _calibrate_sweep(
    fn: str,
    outdir: str,
    sky_model: str = 'gsm16',
    sigma_thr: float = DEFAULT_SIGMA_THR,
    cal_model: str = 'sun+sky',
    leakage: str = 'solve',
    beam: str = 'aep-harp',
    suncal_kwargs: dict | None = None,
    reuse: bool = False,
    ignore_static_flags: bool = False,
    flag_freqs: list | None = None,
) -> str:
    """Calibrate every channel of one station file and write `{outdir}/{station}.cal`.

    See `report.calibrate_channel()` for the options; sun+sky (suncal) falls back to the diffuse sky
    on channels where the Sun is down. The final StefCal convergence of each channel is kept in `provenance['sweep']['dG']`,
    and suncal results as per-channel arrays in `provenance['suncal']`. Channels with a bad timestamp
    (see `report.bad_channel_times()`), where suncal found no flux scale, in `flag_freqs`, or with
    RFI in the autocorrelation or gain spectra (see `flag_channels()`) are flagged.

    Args:
        fn (str): UVX station file.
        outdir (str): Output directory.
        sky_model (str): Diffuse sky model to calibrate against.
        sigma_thr (float): Antenna gain outlier threshold.
        cal_model (str): 'sun+sky', 'sky' or 'sun'.
        leakage (str): 'solve' for full-Jones gains, 'none' for diagonal gains.
        beam (str): Element pattern, a key of `report.BEAMS`.
        suncal_kwargs (dict): Extra suncal() options.
        reuse (bool): Keep an existing cal file instead of recalibrating.
        ignore_static_flags (bool): Ignore the static antenna flags stored in the UVX file.
        flag_freqs (list): (start, stop) Quantity frequency ranges to flag, e.g. known RFI bands.

    Returns:
        cal_path (str): The cal file.
    """
    aa = ApertureArray(read_station(fn, ignore_static_flags))
    aa.set_task_verbosity('silent')
    cal_path = os.path.join(outdir, f'{aa.name}.cal')
    if reuse and os.path.exists(cal_path):
        logger.info(f'{aa.name}: using existing {cal_path}')
        return cal_path

    if cal_model == 'sun':
        # Raises if the Sun is down at either end of the sweep
        for i in (0, len(aa.f) - 1):
            with aa.select(f=i):
                station_cal_model(aa, cal_model)

    def calibrate(aa, flags):
        dgs, suns = [], []

        def cal_channel(aa):
            cal, dg = calibrate_channel(
                aa, cal_model, leakage, beam, sky_model, sigma_thr, flags, suncal_kwargs
            )
            if 'suncal' in cal.provenance:
                suns.append(cal.provenance['suncal'])
            dgs.append(dg)
            return cal

        logger.info(f'{aa.name}: {cal_model} model, leakage {leakage}, {beam} on {len(aa.f)} channels')
        cal = iterate_channels_cal(aa, 'all', cal_channel)
        cal.provenance['sweep'] = {'dG': np.array(dgs)}
        if suns:
            cal.provenance['suncal'] = merge_suncal_provenance(suns)
            # Channels where suncal kept the Sun-only gains (1 Jy Sun) are not in Jy, so are flagged
            no_scale = ~cal.provenance['suncal']['flux_scale']
            if no_scale.any():
                logger.warning(
                    f'{aa.name}: no flux scale (Sun-only gains) in {no_scale.sum()} channels, flagged: '
                    f'{np.round(aa.f.to("MHz").value[no_scale], 1).tolist()} MHz'
                )
            cal.flags.channel.values[no_scale] = True
            cal.provenance['unit'] = 'Jy'
        return cal

    cal = calibrate_sweep(aa, calibrate, flag_freqs)
    write_cal(cal, cal_path)
    logger.info(f'{aa.name}: wrote {cal_path}, median dG {np.median(cal.provenance["sweep"]["dG"]):.2e}')
    return cal_path


def _plot_gains_vs_freq(cals: dict, outdir: Path, plot_type: str = 'mag') -> str:
    """Grid of per-station gain magnitude or phase against frequency, XX and YY, one line per antenna."""
    plt.figure(figsize=(AA1_GRID_COLS * 3, AA1_GRID_ROWS * 2.5))
    for name, cal in cals.items():
        plt.subplot(*AA1_GRID[name])
        f_mhz = cal.gains.frequency.values / 1e6
        # Diagonal terms: XX, YY for a full-Jones cal, X, Y for a diagonal one; flagged set to NaN
        bad = (
            cal.flags.antenna.values[None, :, None]
            | cal.flags.channel.values[:, None, None]
            | (cal.gains.values == 0)
        )
        g = np.where(bad[..., [0, -1]], np.nan, cal.gains.values[..., [0, -1]])
        if plot_type == 'phs':
            # Each channel's overall phase is arbitrary, so reference to an antenna unflagged at every good channel
            ok = np.flatnonzero(np.isfinite(g[~cal.flags.channel.values]).all(axis=(0, 2)))
            y = np.full(g.shape, np.nan)
            if len(ok):
                y = np.rad2deg(np.angle(g * np.conj(g[:, ok[0] : ok[0] + 1])))
            plt.ylim(-185, 185)
            ref_name = cal.gains.antenna.values[ok[0]] if len(ok) else 'none'
            plt.ylabel(f'Phase rel. {ref_name} [deg]')
        else:
            y = np.abs(g)
            if np.isfinite(y).any():
                plt.ylim(0, 1.2 * np.nanpercentile(y, 99))
            plt.ylabel('Gain magnitude')
        for p_idx, (pol, color) in enumerate((('XX', 'C0'), ('YY', 'C1'))):
            plt.plot(f_mhz, y[..., p_idx], color=color, alpha=0.1, lw=0.5)
            plt.plot([], [], color=color, label=pol)
        plt.title(name)
        plt.xlabel('Frequency [MHz]')
        plt.legend(loc='upper right', fontsize='small')
    plt.tight_layout()
    fn = f'cal_{plot_type}_vs_freq.png'
    plt.savefig(outdir / fn)
    plt.close()
    return fn


def _flagged_antennas(cal) -> np.ndarray:
    """Bool mask of antennas flagged, or zeroed (gain outliers) in most unflagged channels of the sweep cal."""
    zeroed = (cal.gains.values == 0).any(axis=-1)[~cal.flags.channel.values]  # (N_good_freq, N_ant)
    if not len(zeroed):
        return cal.flags.antenna.values.copy()
    return cal.flags.antenna.values | (zeroed.mean(axis=0) > ANT_FLAG_FRACTION)


def _summary_row(name: str, cal) -> tuple:
    """Per-station sweep summary: flags, median dG, and the median suncal Sun flux if the Sun was used."""
    n_ch = cal.flags.channel.sizes['frequency']
    good = ~cal.flags.channel.values
    sun = cal.provenance.get('suncal')
    sun_flux = ''
    if sun is not None:
        s = np.asarray(sun['sun_flux_jy'])[good]
        sun_flux = f'{np.nanmedian(s):.4g}' if np.isfinite(s).any() else 'not determined'
    return (
        name,
        'suncal' if sun is not None else cal.method,
        f'{int(_flagged_antennas(cal).sum())} / {cal.flags.antenna.sizes["antenna"]}',
        f'{int((~good).sum())} / {n_ch}',
        f'{np.median(cal.provenance["sweep"]["dG"][good]):.2e}' if good.any() else '&ndash;',
        sun_flux,
    )


def run_stefcal(
    files: list[str],
    outdir: str = 'calibration',
    sigma_thr: float = DEFAULT_SIGMA_THR,
    sky_model: str = 'gsm16',
    cal_model: str = 'sun+sky',
    leakage: str = 'solve',
    beam: str = 'aep-harp',
    suncal_kwargs: dict | None = None,
    n_jobs: int = 4,
    reuse_cal: bool = False,
    ignore_static_flags: bool = False,
    flag_freqs: list | None = None,
    cli_args: dict | None = None,
) -> Path:
    """Calibrate every channel of each station and generate the calibration report webpage.

    The cal files and report go in `outdir/<run>/`, named by `report.cal_variant()`. Then
    `outdir/index.html` is rewritten to compare every run in `outdir`.

    Args:
        files (list): UVX files, one per station.
        outdir (str): Parent folder of the run folders.
        sigma_thr (float): Antenna gain outlier threshold.
        sky_model (str): Diffuse sky model to calibrate against.
        cal_model (str): 'sun+sky' (suncal when the Sun is up), 'sky' or 'sun'.
        leakage (str): 'solve' for full-Jones gains, 'none' for diagonal gains.
        beam (str): Element pattern, a key of `report.BEAMS`.
        suncal_kwargs (dict): Extra suncal() options, e.g. from `report.suncal_kwargs_from_args()`.
        n_jobs (int): Stations calibrated in parallel.
        reuse_cal (bool): Keep existing `{station}.cal` files in the run folder instead of recalibrating.
        ignore_static_flags (bool): Ignore the static antenna flags stored in the UVX files.
        flag_freqs (list): (start, stop) Quantity frequency ranges to flag, e.g. known RFI bands.
        cli_args (dict): Command-line arguments for the Settings tab (default: the arguments above).

    Returns:
        index_fn (Path): The run's index.html.
    """
    outdir_path = Path(outdir) / cal_variant(cal_model, leakage, beam, sky_model)
    outdir_path.mkdir(parents=True, exist_ok=True)

    # Each station is independent; a worker holds one station in memory
    tasks = [
        task(_calibrate_sweep)(
            fn,
            str(outdir_path),
            sky_model=sky_model,
            sigma_thr=sigma_thr,
            cal_model=cal_model,
            leakage=leakage,
            beam=beam,
            suncal_kwargs=suncal_kwargs,
            reuse=reuse_cal,
            ignore_static_flags=ignore_static_flags,
            flag_freqs=flag_freqs,
        )
        for fn in files
    ]
    cal_paths = run_in_parallel(tasks, n_workers=n_jobs, verbose=True)
    uvxs = [read_station(fn, ignore_static_flags) for fn in files]
    cals = {uvx.name: read_cal(p) for uvx, p in zip(uvxs, cal_paths, strict=True)}
    fn_cal_mag = _plot_gains_vs_freq(cals, outdir_path, 'mag')
    fn_cal_phs = _plot_gains_vs_freq(cals, outdir_path, 'phs')

    stations = list(cals)
    f_mhz = next(iter(cals.values())).gains.frequency.values / 1e6
    utc0, lst0 = obs_start(uvxs[0])
    info = [
        ('Stations', f'{len(stations)}: {", ".join(stations)}'),
        ('Frequency', f'{f_mhz.min():.1f}&ndash;{f_mhz.max():.1f} MHz ({len(f_mhz)} channels)'),
        ('UTC start', utc0),
        ('LST start', lst0),
        ('Run', outdir_path.name),
    ]
    if cli_args is None:
        cli_args = {
            'sigma_thr': sigma_thr,
            'sky_model': sky_model,
            'cal_model': cal_model,
            'leakage': leakage,
            'beam': beam,
            **(suncal_kwargs or {}),
            'n_jobs': n_jobs,
            'reuse_cal': reuse_cal,
            'ignore_static_flags': ignore_static_flags,
            'flag_freqs': flag_freqs,
            'outdir': outdir,
        }
    settings_table = args_table(cli_args)
    summary = table(
        ['Station', 'Method', 'Flagged antennas', 'Flagged channels', 'Median dG', 'Median Sun flux [Jy]'],
        [_summary_row(name, cal) for name, cal in cals.items()],
    )
    meta_table = metadata_table(stations, files, uvxs)

    def fig(fn, caption):
        return figs([fn.removesuffix('.png')], prefix='', captions=[caption])

    cal_links = ''.join(
        f'<li><a href="{os.path.basename(p)}">{os.path.basename(p)}</a></li>'
        for p in sorted(cal_paths, key=station_key)
    )
    tabs = [
        (
            'calfreq',
            'Calibration vs frequency',
            f'<div class="card"><h2>Calibration summary</h2>{summary}'
            f'<h2>Calibration files</h2><ul>{cal_links}</ul></div>'
            + fig(fn_cal_mag, 'Gain magnitude (correction applied to the data), XX and YY')
            + fig(fn_cal_phs, 'Gain phase relative to a reference antenna, XX and YY'),
        ),
        ('metadata', 'Metadata', f'<div class="card"><h2>UVX file metadata</h2>{meta_table}</div>'),
        ('settings', 'Settings', f'<div class="card"><h2>Runtime arguments</h2>{settings_table}</div>'),
    ]
    index_fn = outdir_path / 'index.html'
    write_report(index_fn, f'Stefcal Calibration Report: {outdir_path.name}', info, tabs, show_version())
    write_summary(outdir)
    return index_fn


def _station_curve(cal, kind: str) -> np.ndarray | None:
    """Per-channel StefCal convergence ('dg'), or median unflagged XX/YY gain magnitude ('mag'); NaN if flagged.

    Returns None for 'dg' on a cal with no StefCal convergence, e.g. from holography.
    """
    if kind == 'dg':
        if 'sweep' not in cal.provenance:
            return None
        y = np.asarray(cal.provenance['sweep']['dG'], dtype=float)
    else:
        g = np.abs(cal.gains.values[..., [0, -1]])
        g = np.where(cal.flags.antenna.values[None, :, None] | (g == 0), np.nan, g)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'All-NaN slice')
            y = np.nanmedian(g, axis=(1, 2))
    return np.where(cal.flags.channel.values, np.nan, y)


def _plot_runs_vs_freq(runs: dict, outdir: Path, kind: str) -> str:
    """Grid of per-station curves from `_station_curve()` against frequency, one line per run."""
    plt.figure(figsize=(AA1_GRID_COLS * 3, AA1_GRID_ROWS * 2.5))
    stations = sorted({s for cals in runs.values() for s in cals}, key=list(AA1_GRID).index)
    for name in stations:
        plt.subplot(*AA1_GRID[name])
        for i, (run, cals) in enumerate(runs.items()):
            y = _station_curve(cals[name], kind) if name in cals else None
            if y is not None:
                f_mhz = cals[name].gains.frequency.values / 1e6
                plt.semilogy(f_mhz, y, color=f'C{i}', lw=0.8, label=run)
        plt.title(name)
        plt.xlabel('Frequency [MHz]')
        plt.ylabel('StefCal dG' if kind == 'dg' else 'Median gain magnitude')
        plt.legend(loc='upper right', fontsize='xx-small')
    plt.tight_layout()
    fn = f'runs_{kind}_vs_freq.png'
    plt.savefig(outdir / fn)
    plt.close()
    return fn


def write_summary(outdir: str | Path) -> Path | None:
    """Write `outdir/index.html` comparing the runs in outdir's subfolders, or redirecting to the only one.

    With no runs left, the page is removed.

    Args:
        outdir (str | Path): Parent folder of the run folders written by `run_stefcal()` and `holo_suncal()`.

    Returns:
        index_fn (Path): The page, or None if there are no runs.
    """
    outdir = Path(outdir)
    index_fn = outdir / 'index.html'
    run_dirs = sorted(d for d in outdir.iterdir() if d.is_dir() and any(d.glob('*.cal')))
    if len(run_dirs) <= 1:
        if run_dirs:
            write_redirect(index_fn, f'{run_dirs[0].name}/index.html')
        else:
            index_fn.unlink(missing_ok=True)
        return index_fn if run_dirs else None

    runs = {
        d.name: {p.stem: read_cal(str(p)) for p in sorted(d.glob('*.cal'), key=lambda p: station_key(p.name))}
        for d in run_dirs
    }
    stations = sorted({s for cals in runs.values() for s in cals}, key=station_key)

    def run_row(run, cals):
        good = {s: ~c.flags.channel.values for s, c in cals.items()}
        dg = [
            np.asarray(c.provenance['sweep']['dG'])[good[s]] for s, c in cals.items() if 'sweep' in c.provenance
        ]
        dg = np.concatenate(dg) if dg else np.array([])
        sun = [
            np.asarray(c.provenance['suncal']['sun_flux_jy'])[good[s]]
            for s, c in cals.items()
            if 'suncal' in c.provenance
        ]
        sun = np.concatenate(sun) if sun else np.array([])
        return (
            f'<a href="{run}/index.html">{run}</a>',
            len(cals),
            sum(int(_flagged_antennas(c).sum()) for c in cals.values()),
            sum(int((~g).sum()) for g in good.values()),
            f'{np.median(dg):.2e}' if dg.size else '&ndash;',
            f'{np.nanmedian(sun):.4g}' if np.isfinite(sun).any() else '&ndash;',
        )

    runs_table = table(
        ['Run', 'Stations', 'Flagged antennas', 'Flagged channels', 'Median dG', 'Median Sun flux [Jy]'],
        [run_row(run, cals) for run, cals in runs.items()],
    )
    station_table = table(
        ['Station', *runs],
        [
            (s, *(int(_flagged_antennas(c[s]).sum()) if s in c else '&ndash;' for c in runs.values()))
            for s in stations
        ],
    )
    fn_dg = _plot_runs_vs_freq(runs, outdir, 'dg')
    fn_mag = _plot_runs_vs_freq(runs, outdir, 'mag')
    body = (
        f'<div class="card"><h2>Runs</h2>{runs_table}<h2>Flagged antennas per station</h2>{station_table}</div>'
        + figs(
            [fn_dg.removesuffix('.png'), fn_mag.removesuffix('.png')],
            prefix='',
            captions=[
                'StefCal convergence (dG) per channel, flagged channels blank; holography runs have none',
                'Median XX/YY gain magnitude over unflagged antennas; Sun-only runs have no flux scale',
            ],
        )
    )
    info = [('Runs', len(runs)), ('Stations', f'{len(stations)}: {", ".join(stations)}')]
    write_report(
        index_fn, 'Calibration Run Comparison', info, [('compare', 'Comparison', body)], show_version()
    )
    return index_fn


def build_parser() -> argparse.ArgumentParser:
    """Command-line argument parser."""
    parser = base_parser(__doc__, outdir='calibration', sigma_thr=DEFAULT_SIGMA_THR)
    parser.add_argument(
        '--n-jobs', type=int, default=4, help='Stations calibrated in parallel (~3 GB RAM each)'
    )
    parser.add_argument(
        '--reuse-cal',
        action='store_true',
        help='Keep existing {station}.cal files in the run folder instead of recalibrating',
    )
    parser.add_argument(
        '--flag-freqs',
        type=str,
        default='',
        help="Comma-separated MHz ranges to flag as RFI, e.g. '87.5-108,137-138'",
    )
    return parser


def main():
    """CLI entry point: parse args and generate the report."""
    parser = build_parser()
    args = parser.parse_args()
    check_cal_args(parser, args)
    flag_freqs = parse_flag_freqs(parser, args.flag_freqs)
    files = find_station_files(parser, args, AA1_GRID)
    index_fn = run_stefcal(
        files,
        outdir=args.outdir,
        sigma_thr=args.sigma_thr,
        sky_model=args.sky_model,
        cal_model=args.cal_model,
        leakage=args.leakage,
        beam=args.beam,
        suncal_kwargs=suncal_kwargs_from_args(args),
        n_jobs=args.n_jobs,
        reuse_cal=args.reuse_cal,
        ignore_static_flags=args.ignore_static_flags,
        flag_freqs=flag_freqs,
        cli_args=vars(args),
    )
    print(f'Wrote {index_fn}')


if __name__ == '__main__':
    main()
