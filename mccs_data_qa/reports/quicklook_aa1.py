"""quicklook_aa1.py: Generate an AA1 quick-look QA report webpage from a set of UVX station files.

Calibrates each station at one channel, then renders antenna-flagging, correlation-matrix, all-sky-image,
fractional-difference and station-averaged healpix plots into an HTML report, with a healpix FITS file of
the combined image. All output is written into one directory.

Usage:
    python -m mccs_data_qa.reports.quicklook_aa1 --datadir DIR --eb 'eb-t0001-20260505-0000*' --outdir qa_report
"""

import argparse
import html
import json
import warnings
from pathlib import Path

import healpy as hp
import numpy as np
import pylab as plt
from loguru import logger
from mccs_data_qa.reports.report import (
    AA1_GRID,
    DEFAULT_SIGMA_THR,
    args_table,
    bad_channel_times,
    base_parser,
    calibrate_channel,
    check_cal_args,
    figs,
    find_station_files,
    guarded_tab,
    metadata_table,
    obs_start,
    read_station,
    station_cal_model,
    station_grid,
    suncal_kwargs_from_args,
    table,
    write_report,
)

from ska_ost_low_uv import show_version
from ska_ost_low_uv.postx import ApertureArray
from ska_ost_low_uv.postx.coords.coord_utils import sky2hpix
from ska_ost_low_uv.postx.imaging.pol_image import PolHpxImage
from ska_ost_low_uv.postx.simulation.beam_sim import sim_station_beam_properties

DEFAULT_FREQ_IDX = 77
DEFAULT_DG_THRESHOLD = 0.005
SUN_MASK_FWHM = 3  # Radius, in station beam FWHMs, excluded around the Sun when setting colour scales


def _load_and_calibrate(
    files: list[str],
    freq_idx: int,
    sigma_thr: float,
    dg_threshold: float,
    sky_model: str = 'gsm16',
    cal_model: str = 'sun+sky',
    leakage: str = 'solve',
    beam: str = 'aep-harp',
    suncal_kwargs: dict | None = None,
    ignore_static_flags: bool = False,
) -> tuple[dict, dict, dict, list[str], dict, dict]:
    """Read each UVX file, run flagging + stefcal at freq_idx, and flag bad stations by dG.

    The calibration options are those of `report.calibrate_channel()`. With sun+sky and the Sun up,
    suncal (options in `suncal_kwargs`) also fits the Sun's flux density.

    Returns:
        aas (dict): ApertureArray per station name.
        dgs (dict): Final stefcal dG per station name.
        suns (dict): suncal provenance per station name, or None where the Sun wasn't used.
        bad_stations (list): Stations with dG above dg_threshold (or NaN), a bad timestamp at freq_idx,
            or no suncal flux scale.
        failed (dict): Error per station (file name, if it could not be read) that failed to load or
            calibrate; these are left out of the other outputs.
        srcs (dict): UVX file per station in `aas`.
    """
    aas, failed, srcs = {}, {}, {}
    for fn in files:
        try:
            aa = ApertureArray(read_station(fn, ignore_static_flags))
        except Exception as err:
            logger.exception(f'Cannot read {fn}')
            failed[Path(fn).name] = repr(err)
            continue
        aa.set_task_verbosity('silent')
        # All calibration, plots and images use this channel
        aa.set_idx(f=freq_idx)
        aas[aa.name], srcs[aa.name] = aa, fn

    dgs, suns, bad_stations = {}, {}, []
    for name, aa in list(aas.items()):
        try:
            flags = aa.calibration.flagging.run_flagging()
            model = station_cal_model(aa, cal_model)
            cal, dg = calibrate_channel(aa, model, leakage, beam, sky_model, sigma_thr, flags, suncal_kwargs)
            aa.set_cal(cal)
        except Exception as err:
            # One station's failure shouldn't lose the others' report
            logger.exception(f'Cannot calibrate {name}')
            failed[name] = repr(err)
            del aas[name], srcs[name]
            continue
        dgs[name] = dg
        suns[name] = cal.provenance.get('suncal')
        # With a bad timestamp its sky model (and so its cal and images) is for the wrong time, and
        # without a suncal flux scale its gains are not in Jy. A NaN dG means the cal diverged.
        bad_time = bad_channel_times(aa.uvx)[freq_idx]
        no_scale = suns[name] is not None and not suns[name]['flux_scale']
        if not dgs[name] <= dg_threshold or bad_time or no_scale:
            bad_stations.append(name)
    if not aas:
        raise RuntimeError(f'Every station failed to load or calibrate: {failed}')
    return aas, dgs, suns, bad_stations, failed, srcs


def _nanmean(arrays: list) -> np.ndarray:
    """Mean over stations, ignoring each station's NaN (or masked) pixels; NaN where none has data."""
    stack = np.array([np.ma.filled(np.ma.asarray(a), np.nan) for a in arrays])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)  # Mean of empty slice
        return np.nanmean(stack, axis=0)


def _plot_antenna_flags(aas: dict, bad_stations: list[str], outdir: Path) -> str:
    grid, rows, cols = station_grid(aas)
    plt.figure(figsize=(cols * 3, rows * 3))
    for name, aa in aas.items():
        title = name if name not in bad_stations else f'{name} (cal fail)'
        aa.plotting.plot_antennas(flags=aa.cal, subplot_id=grid[name], reuse_fig=True, title=title)
    plt.tight_layout()
    fn = 'antenna_flags.png'
    plt.savefig(outdir / fn)
    plt.close()
    return fn


def _plot_corr_matrices(aas: dict, bad_stations: list[str], outdir: Path) -> str:
    grid, rows, cols = station_grid(aas)
    plt.figure(figsize=(cols * 3, rows * 2.3))
    for name, aa in aas.items():
        title = name if name not in bad_stations else f'{name} (cal fail)'
        aa.plotting.plot_corr_matrix(subplot_id=grid[name], reuse_fig=True, title=title)
    plt.tight_layout()
    fn = 'corr_matrix.png'
    plt.savefig(outdir / fn)
    plt.close()
    return fn


def _plot_uvdist(aas: dict, bad_stations: list[str], outdir: Path) -> list[str]:
    """One 2x2 (per-pol) calibrated amplitude vs UV distance figure per station."""
    fns = []
    for name, aa in aas.items():
        title = name if name not in bad_stations else f'{name} (cal fail)'
        aa.plotting.plot_uvdist_amp_4pol(title=title)
        fn = f'uvdist_{name}.png'
        plt.savefig(outdir / fn)
        plt.close()
        fns.append(fn)
    return fns


def _plot_allsky_images(aas: dict, bad_stations: list[str], outdir: Path, vis: str = 'cal') -> tuple[dict, str]:
    """Grid of per-station all-sky images from the 'cal' or uncalibrated 'data' visibilities."""
    grid, rows, cols = station_grid(aas)
    plt.figure(figsize=(cols * 3, rows * 3))
    imgs = {}
    for name, aa in aas.items():
        img = aa.imaging.make_image(vis=vis)
        title = name if name not in bad_stations else f'{name} (cal fail)'
        aa.viewer.orthview(
            img,
            mode='stokes',
            reuse_fig=True,
            subplot_id=grid[name],
            title=title,
            colorbar=True,
        )
        imgs[name] = img
    plt.tight_layout()
    fn = 'allsky_images.png' if vis == 'cal' else f'allsky_images_{vis}.png'
    plt.savefig(outdir / fn)
    plt.close()
    return imgs, fn


def _plot_frac_diff(
    aas: dict,
    imgs: dict,
    bad_stations: list[str],
    img_avg: np.ndarray,
    pol_idx: int,
    label: str,
    outdir: Path,
) -> str:
    grid, rows, cols = station_grid(aas)
    plt.figure(figsize=(cols * 3, rows * 3))
    for name, aa in aas.items():
        img_norm = (imgs[name] - img_avg) / img_avg
        title = name if name not in bad_stations else f'{name} (excluded)'
        aa.viewer.orthview(
            img_norm,
            title=title,
            colorbar=True,
            cmap='RdBu_r',
            clim=(-0.5, 0.5),
            overlay_alpha=0.8,
            reuse_fig=True,
            subplot_id=grid[name],
            pol_idx=pol_idx,
            overlay_coords='altaz',
        )
    plt.suptitle(f'Fractional difference from all-station average, {label}')
    plt.tight_layout()
    fn = f'frac_diff_{label.lower().replace("-", "")}.png'
    plt.savefig(outdir / fn)
    plt.close()
    return fn


def _combined_healpix(aas: dict, bad_stations: list[str], freq_idx: int, outdir: Path) -> tuple[str, str]:
    """Average the healpix image across good stations, plot it, and write it to FITS."""
    good = {name: aa for name, aa in aas.items() if name not in bad_stations}
    hpxs = {name: aa.imaging.make_healpix(vis='cal', include_autos=True) for name, aa in good.items()}
    aa = next(iter(good.values()))
    # One station's NaN pixels would otherwise blank those pixels in the average
    hpx_avg = PolHpxImage(
        np.ma.masked_invalid(_nanmean(list(hpxs.values()))),
        basis='linear',
        unit=aa.vis_unit('cal'),
        fill_value=np.nan,
    )

    # Colour scale from the sky outside the Sun, so the diffuse emission is visible.
    # SUN_MASK_FWHM beam widths covers the Sun's main lobe and first sidelobes.
    img_i = np.ma.filled(hpx_avg.stokes[..., 0], np.nan)
    if not np.isfinite(img_i).any():
        raise ValueError('The station-averaged image has no finite pixels')
    n_side = hp.npix2nside(len(img_i))
    fwhm = sim_station_beam_properties(aa)['fwhm']
    sun_pix = hp.query_disc(
        n_side, hp.pix2vec(n_side, sky2hpix(n_side, aa.coords.get_sun())), radius=SUN_MASK_FWHM * fwhm
    )
    sky = img_i.copy()
    sky[sun_pix] = np.nan
    # If the Sun mask leaves no finite sky, scale to the whole image (an all-NaN max breaks the colorbar)
    vmax = np.nanmax(sky) if np.isfinite(sky).any() else np.nanmax(img_i)

    f_mhz = aa.f[freq_idx].to_value('MHz')
    plt.figure(figsize=(6, 6))
    aa.viewer.mollview(
        hpx_avg,
        overlay_srcs=True,
        mode='stokes',
        overlay_alpha=0.3,
        title=f'Station-averaged, {f_mhz:.2f} MHz',
        pol_idx=0,
        colorbar=True,
        max=vmax,
        reuse_fig=True,
    )
    fn_png = 'combined_healpix.png'
    plt.savefig(outdir / fn_png, bbox_inches='tight')
    plt.close()

    fn_fits = 'combined_healpix.fits'
    aa.viewer.write_fits(hpx_avg, str(outdir / fn_fits), overwrite=True)
    return fn_png, fn_fits


def quicklook_aa1(
    files: list[str],
    outdir: str = 'aa1_qa_report',
    freq_idx: int = DEFAULT_FREQ_IDX,
    dg_threshold: float = DEFAULT_DG_THRESHOLD,
    sigma_thr: float = DEFAULT_SIGMA_THR,
    sky_model: str = 'gsm16',
    cal_model: str = 'sun+sky',
    leakage: str = 'solve',
    beam: str = 'aep-harp',
    suncal_kwargs: dict | None = None,
    ignore_static_flags: bool = False,
    cli_args: dict | None = None,
) -> Path:
    """Generate the AA1 quick-look report webpage (and combined healpix FITS) from a list of UVX files.

    Args:
        files (list): UVX files, one per station.
        outdir (str): Output directory for the report.
        freq_idx (int): Frequency channel index to calibrate and image at.
        dg_threshold (float): Stations with a final stefcal dG above this are excluded.
        sigma_thr (float): Antenna gain outlier threshold.
        sky_model (str): Diffuse sky model to calibrate against.
        cal_model (str): 'sun+sky' (suncal when the Sun is up), 'sky' or 'sun'.
        leakage (str): 'solve' for full-Jones gains, 'none' for diagonal gains.
        beam (str): Element pattern, a key of `report.BEAMS`.
        suncal_kwargs (dict): Extra suncal() options, e.g. from `report.suncal_kwargs_from_args()`.
        ignore_static_flags (bool): Ignore the static antenna flags stored in the UVX files.
        cli_args (dict): Command-line arguments for the Settings tab (default: the arguments above).

    Returns:
        index_fn (Path): The generated index.html.
    """
    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)

    aas, dgs, suns, bad_stations, failed, srcs = _load_and_calibrate(
        files,
        freq_idx,
        sigma_thr,
        dg_threshold,
        sky_model,
        cal_model,
        leakage,
        beam,
        suncal_kwargs,
        ignore_static_flags,
    )

    def fig(fn, caption):
        return figs([fn.removesuffix('.png')], prefix='', captions=[caption])

    # Each plot tab is guarded: a plot that fails (e.g. on NaN data) shows its error, the rest still render
    imgs = {}

    def allsky():
        cal_imgs, fn_cal = _plot_allsky_images(aas, bad_stations, outdir_path)
        imgs.update(cal_imgs)  # For the station average below
        _, fn_uncal = _plot_allsky_images(aas, bad_stations, outdir_path, vis='data')
        return fig(fn_cal, 'Calibrated') + fig(fn_uncal, 'Before calibration')

    plot_tabs = [
        guarded_tab('allsky', 'All-sky images', allsky),  # First tab is the one open by default
        guarded_tab(
            'flags',
            'Flagged antennas',
            lambda: fig(_plot_antenna_flags(aas, bad_stations, outdir_path), 'Flagged antennas'),
        ),
        guarded_tab(
            'corr',
            'Correlation matrix',
            lambda: fig(_plot_corr_matrices(aas, bad_stations, outdir_path), 'Correlation matrix'),
        ),
        guarded_tab(
            'uvdist',
            'UV distance',
            lambda: figs(
                [fn.removesuffix('.png') for fn in _plot_uvdist(aas, bad_stations, outdir_path)],
                prefix='',
                captions=list(aas),
            ),
        ),
    ]

    # The station average, fractional differences and combined image need at least one good station
    good_imgs = [imgs[name] for name in aas if name not in bad_stations and name in imgs]
    if good_imgs:
        img_avg = _nanmean(good_imgs)

        def frac_diff():
            fn_x = _plot_frac_diff(aas, imgs, bad_stations, img_avg, 0, 'X-pol', outdir_path)
            fn_y = _plot_frac_diff(aas, imgs, bad_stations, img_avg, 3, 'Y-pol', outdir_path)
            return fig(fn_x, 'X-pol') + fig(fn_y, 'Y-pol')

        def combined():
            fn_png, fn_fits = _combined_healpix(aas, bad_stations, freq_idx, outdir_path)
            return fig(fn_png, f'Station-averaged healpix image, <a href="{fn_fits}">download FITS</a>')

        plot_tabs += [
            guarded_tab('fracdiff', 'Fractional differences', frac_diff),
            guarded_tab('combined', 'Combined image', combined),
        ]

    stations = list(aas)
    any_aa = next(iter(aas.values()))
    utc0, lst0 = obs_start(any_aa.uvx)
    info = [
        ('Stations', f'{len(stations)}: {", ".join(stations)}'),
        ('Frequency', f'{any_aa.f[freq_idx].to("MHz").value:.2f} MHz (channel {freq_idx})'),
        ('UTC start', utc0),
        ('LST start', lst0),
    ]
    badges = ''.join(f'<span class="badge">{name} excluded</span>' for name in bad_stations)
    badges += ''.join(f'<span class="badge">{html.escape(name)} failed</span>' for name in failed)
    if cli_args is None:
        cli_args = {
            'freq_idx': freq_idx,
            'dg_threshold': dg_threshold,
            'sigma_thr': sigma_thr,
            'sky_model': sky_model,
            'cal_model': cal_model,
            'leakage': leakage,
            'beam': beam,
            'ignore_static_flags': ignore_static_flags,
            **(suncal_kwargs or {}),
            'outdir': outdir,
        }
    settings_table = args_table(cli_args)

    def sun_cols(p):
        if p is None:
            return ('', '', '')
        if not np.isfinite(p['sun_flux_jy']):
            return (f'{p["sun_alt_deg"]:.1f}', 'not determined', '')
        return (
            f'{p["sun_alt_deg"]:.1f}',
            f'{p["sun_flux_jy"]:.4g} &plusmn; {p["sun_flux_err_jy"]:.2g}',
            f'{p["activity_factor"]:.2f}',
        )

    n_flagged = {s: int(aa.cal.flags.antenna.values.sum()) for s, aa in aas.items()}
    # Read by report_ui to mark each station's QA
    (outdir_path / 'flags.json').write_text(json.dumps(n_flagged))
    cal_table = table(
        ['Station', 'Flagged antennas', 'dG', 'Excluded', 'Sun alt [deg]', 'Sun flux [Jy]', 'S / quiet Sun'],
        [
            (
                s,
                n_flagged[s],
                f'{dgs[s]:.2e}',
                'yes' if s in bad_stations else '',
                *sun_cols(suns[s]),
            )
            for s in aas
        ]
        + [(html.escape(s), 'failed: ' + html.escape(err), '', '', '', '', '') for s, err in failed.items()],
    )
    meta_table = metadata_table(stations, list(srcs.values()), [aa.uvx for aa in aas.values()])

    suncal_note = ' (suncal when the Sun is up)' if cal_model == 'sun+sky' else ''
    tabs = [
        *plot_tabs,
        ('metadata', 'Metadata', f'<div class="card"><h2>UVX file metadata</h2>{meta_table}</div>'),
        (
            'settings',
            'Settings',
            f'<div class="card"><h2>Runtime arguments</h2>{settings_table}'
            f'<h2>Calibration at channel {freq_idx}{suncal_note}</h2>{cal_table}</div>',
        ),
    ]
    index_fn = outdir_path / 'index.html'
    write_report(index_fn, 'AA1 QA Report', info, tabs, show_version(), card_extra=badges)
    return index_fn


def build_parser() -> argparse.ArgumentParser:
    """Command-line argument parser."""
    parser = base_parser(__doc__, outdir='aa1_qa_report', sigma_thr=DEFAULT_SIGMA_THR)
    parser.add_argument(
        '--freq-idx',
        type=int,
        default=DEFAULT_FREQ_IDX,
        help='Frequency channel index to calibrate at',
    )
    parser.add_argument(
        '--dg-threshold',
        type=float,
        default=DEFAULT_DG_THRESHOLD,
        help='dG threshold above which a station is excluded',
    )
    return parser


def main():
    """CLI entry point: parse args and generate the report."""
    parser = build_parser()
    args = parser.parse_args()
    check_cal_args(parser, args)
    files = find_station_files(parser, args, AA1_GRID)
    index_fn = quicklook_aa1(
        files,
        outdir=args.outdir,
        freq_idx=args.freq_idx,
        dg_threshold=args.dg_threshold,
        sigma_thr=args.sigma_thr,
        sky_model=args.sky_model,
        cal_model=args.cal_model,
        leakage=args.leakage,
        beam=args.beam,
        suncal_kwargs=suncal_kwargs_from_args(args),
        ignore_static_flags=args.ignore_static_flags,
        cli_args=vars(args),
    )
    print(f'Wrote {index_fn}')


if __name__ == '__main__':
    main()
