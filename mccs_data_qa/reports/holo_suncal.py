"""holo_suncal.py: Calibrate station frequency sweeps on the Sun by self-holography, then fit delays.

Runs JishnuCal self-holography (Thekkeppattu et al. 2024) on every channel of each UVX sweep, with the
Sun as the calibrator. This is a calibration run like those of run_stefcal.py: `{station}.cal` goes in
`<outdir>/sun-holography/`, and `<outdir>/index.html` compares it with the other runs there. Delays are
then fitted to the gain phases (see delay_fit_report.py), written to the run's `delay/` folder, and the
report adds aperture illumination images and far-field beam cuts at --freq-idx, before calibration,
with the holography cal, and with the delay-model cal, grouped by station cluster (S8, S9, S10).

Usage:
    python -m mccs_data_qa.reports.holo_suncal --datadir DIR --eb 'eb-t0001-20260505-0000*' --outdir calibration
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pylab as plt
from mccs_data_qa.reports.delay_fit_report import add_fit_args, antenna_positions, delay_fit_report, parse_ref_ant, run_delay_fits
from loguru import logger
from mccs_data_qa.reports.report import (
    AA1_GRID,
    HOLO_RUN,
    calibrate_sweep,
    cluster_tabs,
    find_station_files,
    input_parser,
    parse_flag_freqs,
    read_station,
)
from mccs_data_qa.reports.run_stefcal import write_summary

from ska_ost_low_uv.io import read_cal, write_cal
from ska_ost_low_uv.parallelize import run_in_parallel, task
from ska_ost_low_uv.postx import ApertureArray

DEFAULT_FREQ_IDX = 192
MIN_SUN_ALT = 20  # deg; below this the Sun no longer dominates the diffuse sky in every channel


def _holo_sweep(
    fn: str, outdir: str, reuse: bool = False, ignore_static_flags: bool = False, flag_freqs: list | None = None
) -> str:
    """Run JishnuCal on every channel of one station sweep and write `{outdir}/{station}.cal`.

    Antennas flagged by `flag_antennas_multichannel()` are flagged in the cal. Channels with a bad
    timestamp, in `flag_freqs`, or with RFI in the autocorrelation or gain spectra are flagged.

    Args:
        fn (str): UVX station sweep file.
        outdir (str): Output directory.
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

    sun_alt = []
    for i in (0, len(aa.f) - 1):
        with aa.select(f=i):
            sun_alt.append(float(aa.coords.get_alt_az(aa.coords.get_sun()).alt.deg))
    if min(sun_alt) < MIN_SUN_ALT:
        logger.warning(f'{aa.name}: Sun altitude {sun_alt[0]:.1f} to {sun_alt[1]:.1f} deg across the sweep')

    def calibrate(aa, ant_flags):
        # The Sun's ICRS position barely moves during a sweep; each channel is phased at its own time
        aa.calibration.holography.set_cal_src(aa.coords.get_sun())
        logger.info(f'{aa.name}: JishnuCal on {len(aa.f)} channels')
        cal = aa.calibration.holography.run_jishnucal(channel_range='all')
        cal.flags.antenna.values[ant_flags] = True
        return cal

    cal = calibrate_sweep(aa, calibrate, flag_freqs)
    write_cal(cal, cal_path)
    logger.info(f'{aa.name}: wrote {cal_path}')
    return cal_path


def _plot_apertures(fn: str, cal_paths: dict, freq_idx: int, plotdir: Path, ignore_static_flags: bool) -> tuple:
    """Aperture illumination and far-field beam before calibration and with each cal in `cal_paths`.

    Uses the channel nearest freq_idx that is unflagged in every cal.

    Args:
        fn (str): UVX station sweep file.
        cal_paths (dict): {label: cal file path}.
        freq_idx (int): Channel index to plot.
        plotdir (Path): Output directory for the PNGs.
        ignore_static_flags (bool): Ignore the static antenna flags stored in the UVX file.

    Returns:
        name (str): Station name.
        apertures (list): (stem, caption) per aperture illumination plot.
        farfields (list): (stem, caption) per far-field beam plot.
    """
    cals = {label: read_cal(path) for label, path in cal_paths.items()}
    ok = np.flatnonzero(~np.any([c.flags.channel.values for c in cals.values()], axis=0))
    if ok.size and freq_idx not in ok:
        new_idx = int(ok[np.argmin(np.abs(ok - freq_idx))])
        logger.info(f'Channel {freq_idx} is flagged, using channel {new_idx} for the aperture images')
        freq_idx = new_idx
    aa = ApertureArray(read_station(fn, ignore_static_flags))
    aa.set_task_verbosity('silent')
    aa.set_idx(f=freq_idx)
    holo = aa.calibration.holography
    holo.set_cal_src(aa.coords.get_sun())
    f_mhz = aa.f[freq_idx].to('MHz').value

    apertures, farfields = [], []
    for label, cal in {'uncalibrated': None, **cals}.items():
        aa.set_cal(cal)
        holo.run_selfholo(vis='data' if cal is None else 'cal')
        holo.plot_aperture_xy(show_flags=True)
        plt.suptitle(f'{aa.name} aperture illumination, {f_mhz:.1f} MHz, {label}')
        stem = f'{aa.name}_aperture_{label.replace(" ", "_")}'
        plt.savefig(plotdir / f'{stem}.png', bbox_inches='tight')
        plt.close('all')
        apertures.append((stem, f'{aa.name}: {label}, flagged antennas circled in red'))

        holo.plot_farfield_beam_pattern()
        plt.title(f'{aa.name} far-field beam, {f_mhz:.1f} MHz, {label}')
        stem = f'{aa.name}_farfield_{label.replace(" ", "_")}'
        plt.savefig(plotdir / f'{stem}.png', bbox_inches='tight')
        plt.close('all')
        farfields.append((stem, f'{aa.name}: {label}, simulated array factor grey dashed'))
    return aa.name, apertures, farfields


def holo_suncal(
    files: list[str],
    outdir: str = 'calibration',
    freq_idx: int = DEFAULT_FREQ_IDX,
    n_jobs: int = 4,
    reuse_cal: bool = False,
    ignore_static_flags: bool = False,
    flag_freqs: list | None = None,
    ref_ant: int | str | None = 'auto',
    min_coherence: float = 0.5,
    sigma_thr: float = 5,
    cli_args: dict | None = None,
) -> Path:
    """Holography cal of each station sweep, delay fits, and the report webpage, in `outdir/sun-holography/`.

    Then `outdir/index.html` is rewritten to compare every calibration run in `outdir`.

    Args:
        files (list): UVX sweep files, one per station.
        outdir (str): Parent folder of the calibration run folders.
        freq_idx (int): Channel index for the aperture illumination images.
        n_jobs (int): Stations calibrated in parallel.
        reuse_cal (bool): Keep existing `{station}.cal` files in the run folder instead of recalibrating.
        ignore_static_flags (bool): Ignore the static antenna flags stored in the UVX files.
        flag_freqs (list): (start, stop) Quantity frequency ranges to flag, e.g. known RFI bands.
        ref_ant (int, str, or None): Reference antenna for the delay fits.
        min_coherence (float): Flag antennas with delay fit coherence below this.
        sigma_thr (float): Delay fit antenna and channel outlier threshold.
        cli_args (dict): Command-line arguments for the Settings tab. (default: None)

    Returns:
        index_fn (Path): The run's index.html.
    """
    rundir = Path(outdir) / HOLO_RUN
    plotdir = rundir / 'plots'
    plotdir.mkdir(parents=True, exist_ok=True)
    # Kept out of the run folder, whose *.cal files are read as the station cals
    (rundir / 'delay').mkdir(exist_ok=True)

    tasks = [task(_holo_sweep)(fn, str(rundir), reuse_cal, ignore_static_flags, flag_freqs) for fn in files]
    cal_paths = run_in_parallel(tasks, n_workers=n_jobs, verbose=True)
    results = run_delay_fits(cal_paths, rundir / 'delay', flag_freqs, ref_ant, min_coherence, sigma_thr)

    apertures, farfields = {}, {}
    for fn, res in zip(files, results.values(), strict=True):
        cals = {'holography cal': res['file'], 'delay-model cal': res['delay_cal']}
        name, ap, ff = _plot_apertures(fn, cals, freq_idx, plotdir, ignore_static_flags)
        apertures[name], farfields[name] = ap, ff
    extra_tabs = [
        ('aperture', 'Aperture', cluster_tabs('aperture', apertures)),
        ('farfield', 'Beam pattern', cluster_tabs('farfield', farfields)),
    ]

    index_fn = delay_fit_report(
        results,
        rundir,
        positions=antenna_positions(files),
        extra_tabs=extra_tabs,
        title='Sun Holography Calibration Report',
        cli_args=cli_args,
    )
    write_summary(outdir)
    return index_fn


def build_parser() -> argparse.ArgumentParser:
    """Command-line argument parser."""
    p = input_parser(__doc__, outdir='calibration')
    io = p.add_argument_group('input and output')
    io.add_argument('--n-jobs', type=int, default=4, help='Stations calibrated in parallel')
    io.add_argument(
        '--reuse-cal',
        action='store_true',
        help='Keep existing {station}.cal files in the run folder instead of recalibrating',
    )
    io.add_argument(
        '--freq-idx', type=int, default=DEFAULT_FREQ_IDX, help='Channel index for the aperture images'
    )
    add_fit_args(p)
    return p


def main():
    """CLI entry point: parse args, calibrate and generate the report."""
    p = build_parser()
    args = p.parse_args()
    flag_freqs = parse_flag_freqs(p, args.flag_freqs)
    files = find_station_files(p, args, AA1_GRID)

    index_fn = holo_suncal(
        files,
        outdir=args.outdir,
        freq_idx=args.freq_idx,
        n_jobs=args.n_jobs,
        reuse_cal=args.reuse_cal,
        ignore_static_flags=args.ignore_static_flags,
        flag_freqs=flag_freqs,
        ref_ant=parse_ref_ant(args.ref_ant),
        min_coherence=args.min_coherence,
        sigma_thr=args.sigma_thr,
        cli_args=vars(args),
    )
    print(f'Wrote {index_fn}')


if __name__ == '__main__':
    main()
