"""delay_fit_report.py: Fit delays to the gain phases of calibration sweeps and generate a report webpage.

Fits a delay and phase offset to each antenna and pol of every `{station}.cal` in each --caldir (a
calibration run folder written by run_stefcal.py or holo_suncal.py), writes the delay-model calibration
as `{station}_delay.cal`, and plots the phase fits, the delay spectra and the delays at the antenna positions of the UVX files in
--datadir. Each caldir gets its own report in `<outdir>/<caldir name>/`; `<outdir>/index.html`
compares every run folder in outdir by station, or redirects to the only one.

Usage:
    python -m mccs_data_qa.reports.delay_fit_report --caldir calibration/RUN1 calibration/RUN2 --datadir DIR --eb 'eb-*'
"""

import argparse
import glob
import inspect
import os
import warnings
from pathlib import Path

import astropy.units as u
import numpy as np
import pylab as plt
from loguru import logger
from mccs_data_qa.reports.report import (
    args_table,
    cluster_tabs,
    figs,
    input_parser,
    parse_flag_freqs,
    read_station,
    station_key,
    subtabs,
    table,
    write_redirect,
    write_report,
)

from ska_ost_low_uv import show_version
from ska_ost_low_uv.io import read_cal, write_cal
from ska_ost_low_uv.postx.calibration.delay_fit import LIGHT_SPEED, delay_cal, fit_delays
from ska_ost_low_uv.postx.calibration.flagging import _resolve_channel_flags

POLS = ('X', 'Y')
_FIT_PARAMS = inspect.signature(fit_delays).parameters
DEFAULT_MIN_COHERENCE = _FIT_PARAMS['min_coherence'].default
DEFAULT_SIGMA_THR = _FIT_PARAMS['sigma_thr'].default
# Columns of `_summary_row()`
SUMMARY_COLS = (
    'Method',
    'Channels fitted',
    'Ref. antenna',
    'Flagged X / Y',
    'Median coherence X / Y',
    'Median resid rms [deg] X / Y',
    'Delay std [ns] X / Y',
    'Median delay err [ps] X / Y',
)


def run_delay_fits(
    cal_files: list[str],
    outdir: str | Path,
    flag_freqs: list | None = None,
    ref_ant: int | str | None = 'auto',
    min_coherence: float = DEFAULT_MIN_COHERENCE,
    sigma_thr: float = DEFAULT_SIGMA_THR,
) -> dict:
    """Fit delays to each calibration file and write the model as `{outdir}/{name}_delay.cal`.

    Args:
        cal_files (list): Multi-channel calibration files.
        outdir (str | Path): Output directory.
        flag_freqs (list): (start, stop) Quantity frequency ranges to leave out of the fits.
        ref_ant (int, str, or None): Reference antenna, passed to `fit_delays()`.
        min_coherence (float): Flag antennas with fit coherence below this.
        sigma_thr (float): Antenna and channel outlier threshold.

    Returns:
        results (dict): Per file stem, a dict with the input 'file', its 'cal' and its 'method', the
            `fit_delays()` output 'fit', and the 'delay_cal' path.
    """
    results = {}
    for fn in cal_files:
        name = Path(fn).stem
        cal = read_cal(fn)
        flags = _resolve_channel_flags(cal.gains.frequency.values * u.Hz, flag_freqs)
        fit = fit_delays(cal, flags, ref_ant=ref_ant, min_coherence=min_coherence, sigma_thr=sigma_thr)
        path = Path(outdir) / f'{name}_delay.cal'
        write_cal(delay_cal(cal, fit), str(path))
        results[name] = {'file': fn, 'cal': cal, 'method': cal.method, 'fit': fit, 'delay_cal': str(path)}
    return results


def antenna_positions(uvx_files: list[str]) -> dict:
    """ENU antenna positions per station name, as {name: {antenna identifier: (e, n, u)}}."""
    pos = {}
    for fn in uvx_files:
        uvx = read_station(fn)
        ids = uvx.antennas.attrs['identifier'].values.astype(str)
        pos[uvx.name] = dict(zip(ids, uvx.antennas.enu.values, strict=True))
    return pos


def _wrap_deg(phs: np.ndarray) -> np.ndarray:
    """Phase in rad wrapped to [-180, 180) deg."""
    return np.rad2deg(np.angle(np.exp(1j * phs)))


def _model_phase(f: np.ndarray, fit: dict) -> np.ndarray:
    """Model phase (N_freq, N_ant, 2) in rad."""
    return 2 * np.pi * f[:, None, None] * fit['delay'][None] / LIGHT_SPEED + fit['phase0'][None]


def _plot_phase_fits(name: str, res: dict, p_idx: int, plotdir: Path) -> str:
    """Grid of measured phase against frequency per antenna, with the delay model. Flagged antennas shaded."""
    fit, cal = res['fit'], res['cal']
    f = cal.gains.frequency.values
    model = _model_phase(f, fit)[..., p_idx]
    data = _wrap_deg(model + fit['resid'][..., p_idx])
    model = _wrap_deg(model)
    # Break the model line where it wraps
    model[1:][np.abs(np.diff(model, axis=0)) > 180] = np.nan
    ants = cal.gains.antenna.values
    n = int(np.ceil(np.sqrt(len(ants))))

    fig, axes = plt.subplots(n, n, figsize=(n * 1.25, n), sharex=True, sharey=True)
    for i, ax in enumerate(np.ravel(axes)):
        ax.set_xticks([])
        ax.set_yticks([])
        if i >= len(ants):
            ax.axis('off')
            continue
        ax.plot(f / 1e6, data[:, i], '.', ms=1, color='#333333')
        if fit['flags'][i, p_idx]:
            ax.set_facecolor('#f6d5d5')
        else:
            ax.plot(f / 1e6, model[:, i], color='#cc0000', lw=0.6)
        ax.set_title(f'{ants[i]} {fit["delay_ns"][i, p_idx]:.1f} ns', fontsize=6, pad=2)
    ax.set_ylim(-180, 180)
    ref = f', relative to {fit["ref_ant"]}' if fit['ref_ant'] else ''
    fig.suptitle(
        f'{name} {POLS[p_idx]}: phase vs frequency ({f.min() / 1e6:.0f}-{f.max() / 1e6:.0f} MHz){ref}', y=0.995
    )
    fig.tight_layout(pad=0.3, h_pad=0.3, w_pad=0.1)
    stem = f'{name}_phase_{POLS[p_idx]}'
    fig.savefig(plotdir / f'{stem}.png')
    plt.close(fig)
    return stem


def _plot_delay_spectra(name: str, res: dict, plotdir: Path) -> str:
    """Normalized delay spectrum |P(tau)| per antenna, X and Y."""
    fit = res['fit']
    tau_ns = fit['taus'] / LIGHT_SPEED * 1e9
    good = fit['delay_ns'][~fit['flags']]
    lim = min(np.abs(tau_ns).max(), 1.5 * np.abs(good).max() + 20) if good.size else np.abs(tau_ns).max()
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    for p_idx, ax in enumerate(axes):
        im = ax.imshow(
            fit['power'][:, :, p_idx].T,
            aspect='auto',
            interpolation='nearest',
            extent=(tau_ns[0], tau_ns[-1], fit['power'].shape[1] - 0.5, -0.5),
            vmin=0,
            vmax=1,
        )
        ax.set_xlim(-lim, lim)
        ax.set_xlabel('Delay [ns]')
        ax.set_title(f'{name} {POLS[p_idx]}')
    axes[0].set_ylabel('Antenna index')
    fig.colorbar(im, ax=axes, label='Coherence |P(τ)| / N_chan')
    stem = f'{name}_delay_spectra'
    fig.savefig(plotdir / f'{stem}.png')
    plt.close(fig)
    return stem


def _plot_residuals(name: str, res: dict, plotdir: Path) -> str:
    """Residual phase (data - fit) per antenna and frequency, with its median over unflagged antennas."""
    fit = res['fit']
    f_mhz = res['cal'].gains.frequency.values / 1e6
    resid = np.rad2deg(np.where(fit['flags'][None], np.nan, fit['resid']))
    rms = np.rad2deg(fit['resid_rms'][~fit['flags']])
    lim = max(10, 3 * np.median(rms)) if rms.size else 45
    df = np.median(np.diff(f_mhz)) if len(f_mhz) > 1 else 1

    # The colorbar gets its own gridspec column, so the image and line panels share x extents
    fig = plt.figure(figsize=(12.5, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=(3, 1), width_ratios=(1, 1, 0.04), wspace=0.15)
    axes = np.empty((2, 2), dtype=object)
    axes[0, 0] = fig.add_subplot(gs[0, 0])
    for r, c in ((0, 1), (1, 0), (1, 1)):
        axes[r, c] = fig.add_subplot(gs[r, c], sharex=axes[0, 0], sharey=axes[r, 0] if c else None)
    for p_idx in range(2):
        ax_img, ax_med = axes[:, p_idx]
        im = ax_img.imshow(
            resid[..., p_idx].T,
            aspect='auto',
            interpolation='nearest',
            cmap='RdBu_r',
            vmin=-lim,
            vmax=lim,
            extent=(f_mhz[0] - df / 2, f_mhz[-1] + df / 2, resid.shape[1] - 0.5, -0.5),
        )
        ax_img.set_title(f'{name} {POLS[p_idx]}')
        with np.errstate(invalid='ignore'), warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'All-NaN slice')
            lo, med, hi = np.nanpercentile(resid[..., p_idx], (16, 50, 84), axis=1)
        ax_med.fill_between(f_mhz, lo, hi, color='#999999', alpha=0.4, lw=0, label='16-84%')
        ax_med.plot(f_mhz, med, color='#333333', lw=0.8, label='median')
        ax_med.axhline(0, color='#cc0000', lw=0.5)
        ax_med.set_ylim(-lim, lim)
        ax_med.set_xlabel('Frequency [MHz]')
    axes[0, 0].set_ylabel('Antenna index')
    axes[1, 0].set_ylabel('Residual [deg]')
    axes[1, 1].legend(loc='upper right', fontsize='small')
    for ax in axes.flat:
        ax.label_outer()
    fig.colorbar(im, cax=fig.add_subplot(gs[0, 2]), label='Residual phase [deg]')
    stem = f'{name}_residuals'
    fig.savefig(plotdir / f'{stem}.png')
    plt.close(fig)
    return stem


def _plot_delays(name: str, res: dict, positions: dict | None, plotdir: Path) -> str:
    """Delays at the antenna positions if known, otherwise against antenna index."""
    fit, cal = res['fit'], res['cal']
    ants = cal.gains.antenna.values.astype(str)
    enu = None
    if positions and cal.telescope in positions and all(a in positions[cal.telescope] for a in ants):
        enu = np.array([positions[cal.telescope][a] for a in ants])
    elif positions:
        logger.warning(f'{name}: no UVX antenna positions for station {cal.telescope}, plotting against index')

    good = ~fit['flags']
    vals = fit['delay_ns'][good]
    vlim = np.percentile(np.abs(vals), 98) if vals.size else 1
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharex=True, sharey=True)
    for p_idx, ax in enumerate(axes):
        ok = good[:, p_idx]
        if enu is not None:
            ax.scatter(*enu[~ok, :2].T, marker='x', color='#999999', s=15, label='flagged')
            sc = ax.scatter(
                *enu[ok, :2].T, c=fit['delay_ns'][ok, p_idx], cmap='RdBu_r', vmin=-vlim, vmax=vlim, s=30
            )
            ax.set_aspect('equal')
            ax.set_xlabel('E [m]')
            axes[0].set_ylabel('N [m]')
        else:
            idx = np.arange(len(ants))
            sc = ax.scatter(
                idx[ok],
                fit['delay_ns'][ok, p_idx],
                c=fit['delay_ns'][ok, p_idx],
                cmap='RdBu_r',
                vmin=-vlim,
                vmax=vlim,
                s=10,
            )
            ax.scatter(
                idx[~ok], fit['delay_ns'][~ok, p_idx], marker='x', color='#999999', s=15, label='flagged'
            )
            ax.set_xlabel('Antenna index')
            axes[0].set_ylabel('Delay [ns]')
        ax.set_title(f'{name} {POLS[p_idx]}')
        ax.legend(loc='upper right', fontsize='small')
    fig.colorbar(sc, ax=axes, label='Delay [ns]')
    stem = f'{name}_delays'
    fig.savefig(plotdir / f'{stem}.png')
    plt.close(fig)
    return stem


def _summary_row(name: str, res: dict) -> tuple:
    fit = res['fit']
    good = ~fit['flags']
    n_ch = len(fit['channel_flags'])

    def per_pol(fn, fmt):
        return ' / '.join(fmt.format(fn(p)) if good[:, p].any() else '&ndash;' for p in range(2))

    return (
        name,
        res['method'],
        f'{n_ch - fit["channel_flags"].sum()} / {n_ch}',
        fit['ref_ant'] or '&ndash;',
        f'{fit["flags"][:, 0].sum()} / {fit["flags"][:, 1].sum()}',
        per_pol(lambda p: np.median(fit['coherence'][good[:, p], p]), '{:.3f}'),
        per_pol(lambda p: np.rad2deg(np.median(fit['resid_rms'][good[:, p], p])), '{:.1f}'),
        per_pol(lambda p: np.std(fit['delay_ns'][good[:, p], p]), '{:.2f}'),
        per_pol(lambda p: np.median(fit['delay_err'][good[:, p], p]) / LIGHT_SPEED * 1e12, '{:.0f}'),
    )


def _antenna_table(res: dict) -> str:
    fit = res['fit']
    err_ns = fit['delay_err'] / LIGHT_SPEED * 1e9
    rows = []
    for i, ant in enumerate(res['cal'].gains.antenna.values):
        cols = []
        for p in range(2):
            cols += [
                f'{fit["delay_ns"][i, p]:.3f} &plusmn; {err_ns[i, p]:.3f}',
                f'{np.rad2deg(fit["phase0"][i, p]):.1f}',
                f'{fit["coherence"][i, p]:.3f}',
                f'{np.rad2deg(fit["resid_rms"][i, p]):.1f}',
                'flagged' if fit['flags'][i, p] else '',
            ]
        rows.append((ant, *cols))
    header = ['Antenna'] + [
        f'{p} {h}' for p in POLS for h in ('delay [ns]', 'φ0 [deg]', 'coherence', 'resid rms [deg]', 'flag')
    ]
    return table(header, rows)


def delay_fit_report(
    results: dict,
    outdir: str | Path,
    positions: dict | None = None,
    extra_tabs: list | None = None,
    title: str = 'Delay Fit Report',
    cli_args: dict | None = None,
) -> Path:
    """Plot the delay fits from `run_delay_fits()` and render the report page.

    Args:
        results (dict): Output of `run_delay_fits()`.
        outdir (str | Path): Output directory; plots go in `plots/`.
        positions (dict): Antenna positions from `antenna_positions()`, keyed by station name. (default: None)
        extra_tabs (list): (id, label, html) tabs inserted after the delay tabs. (default: None)
        title (str): Page title.
        cli_args (dict): Command-line arguments for the Settings tab. (default: None)

    Returns:
        index_fn (Path): The generated index.html.
    """
    outdir = Path(outdir)
    plotdir = outdir / 'plots'
    plotdir.mkdir(parents=True, exist_ok=True)

    phase_figs, phase_caps, resid_figs, spec_figs, delay_figs = [], [], [], [], []
    for name, res in results.items():
        for p_idx in range(2):
            phase_figs.append(_plot_phase_fits(name, res, p_idx, plotdir))
            phase_caps.append(f'{name} {POLS[p_idx]}: measured phase (black) and delay model (red)')
        resid_figs.append(_plot_residuals(name, res, plotdir))
        spec_figs.append(_plot_delay_spectra(name, res, plotdir))
        delay_figs.append(_plot_delays(name, res, positions, plotdir))

    summary = table(['Calibration', *SUMMARY_COLS], [_summary_row(name, res) for name, res in results.items()])

    def rel(path: str) -> str:
        return Path(path).resolve().relative_to(outdir.resolve()).as_posix()

    cal_links = ''.join(
        f'<li><a href="{rel(r["delay_cal"])}">{rel(r["delay_cal"])}</a> (from {r["file"]})</li>'
        for r in results.values()
    )
    ant_tables = ''.join(
        f'<details><summary>{name}</summary>{_antenna_table(res)}</details>' for name, res in results.items()
    )

    f_all = np.concatenate([r['cal'].gains.frequency.values for r in results.values()]) / 1e6
    fits = [r['fit'] for r in results.values()]
    info = [
        ('Calibrations', f'{len(results)}: {", ".join(results)}'),
        ('Frequency', f'{f_all.min():.1f}&ndash;{f_all.max():.1f} MHz'),
        (
            'Antennas flagged X / Y',
            ' / '.join(str(sum(ft['flags'][:, p].sum() for ft in fits)) for p in range(2)),
        ),
        ('Model', 'g(f) = exp(i (2π f τ / c + φ0))'),
    ]
    tabs = [
        (
            'summary',
            'Summary',
            f'<div class="card"><h2>Delay fits</h2>{summary}'
            f'<h2>Delay-model calibration files</h2><ul>{cal_links}</ul></div>',
        ),
        ('phase', 'Phase fits', figs(phase_figs, captions=phase_caps)),
        (
            'residuals',
            'Residuals',
            figs(
                resid_figs,
                captions=[
                    f'{n}: residual phase (data - fit), flagged antennas and channels blank. Structure common '
                    f'to every antenna includes the reference antenna ({results[n]["fit"]["ref_ant"]}), negated.'
                    if results[n]['fit']['ref_ant']
                    else f'{n}: residual phase (data - fit), flagged antennas and channels blank.'
                    for n in results
                ],
            ),
        ),
        (
            'spectra',
            'Delay spectra',
            figs(spec_figs, captions=[f'{n}: delay spectrum per antenna' for n in results]),
        ),
        (
            'delays',
            'Delays',
            figs(delay_figs, captions=[f'{n}: fitted delays' for n in results])
            + f'<div class="card"><h2>Per-antenna fits</h2>{ant_tables}</div>',
        ),
        *(extra_tabs or []),
    ]
    if cli_args:
        settings = args_table(cli_args)
        tabs.append(('settings', 'Settings', f'<div class="card"><h2>Runtime arguments</h2>{settings}</div>'))

    index_fn = outdir / 'index.html'
    write_report(index_fn, title, info, tabs, show_version())
    return index_fn


def _plot_delay_comparison(
    ref: str, run: str, ref_results: dict, run_results: dict, plotdir: Path
) -> str | None:
    """Grid of one run's fitted delays against the reference run's, per station, X and Y.

    Returns None, with no plot, if the runs have no station in common.
    """
    names = [n for n in ref_results if n in run_results]
    if not names:
        return None
    n = int(np.ceil(np.sqrt(len(names))))
    fig, axes = plt.subplots(n, n, figsize=(n * 3, n * 3), squeeze=False)
    for ax in axes.flat[len(names) :]:
        ax.axis('off')
    for ax, name in zip(axes.flat[: len(names)], names, strict=True):
        a, b = ref_results[name]['fit'], run_results[name]['fit']
        for p_idx in range(2):
            ok = ~(a['flags'][:, p_idx] | b['flags'][:, p_idx])
            x, y = a['delay_ns'][ok, p_idx], b['delay_ns'][ok, p_idx]
            ax.plot(x, y, '.', ms=3, color=f'C{p_idx}', label=f'{POLS[p_idx]}: rms diff {np.std(y - x):.2f} ns')
        ax.axline((0, 0), slope=1, color='#999999', lw=0.5)
        ax.set_title(name, fontsize='small')
        ax.set_xlabel(f'{ref} [ns]', fontsize='x-small')
        ax.set_ylabel(f'{run} [ns]', fontsize='x-small')
        ax.legend(loc='upper left', fontsize='xx-small')
    fig.tight_layout()
    stem = f'delays_{run}_vs_{ref}'
    fig.savefig(plotdir / f'{stem}.png')
    plt.close(fig)
    return stem


def load_delay_fits(rundir: Path) -> dict:
    """The delay fits of a run folder, from its `{station}_delay.cal` files.

    Returns:
        results (dict): Per station, a dict with the fit 'method' and the `fit_delays()` output 'fit'.
    """
    results = {}
    for path in sorted(rundir.glob('*_delay.cal'), key=lambda p: station_key(p.name)):
        dcal = read_cal(str(path))
        fit = dict(dcal.provenance['delay_fit'])
        if 'flags' not in fit:  # Written before the per-pol flags were stored
            fit['flags'] = np.repeat(dcal.flags.antenna.values[:, None], 2, axis=1)
        fit['ref_ant'] = fit.get('ref_ant') or None
        results[path.stem.removesuffix('_delay')] = {'method': fit.get('cal_method', ''), 'fit': fit}
    return results


def _plot_station_runs(name: str, fits: dict, plotdir: Path) -> str:
    """Delay, residual rms and coherence of each antenna of one station, one line per calibration run.

    Delays are relative to each run's reference antenna, so each run's median delay is removed.
    """
    fig, axes = plt.subplots(3, 2, figsize=(11, 8), sharex=True, squeeze=False)
    for i, (run, fit) in enumerate(fits.items()):
        for p_idx in range(2):
            y = {
                'delay': fit['delay_ns'][:, p_idx],
                'resid': np.rad2deg(fit['resid_rms'][:, p_idx]),
                'coherence': fit['coherence'][:, p_idx],
            }
            # The reference antenna is zero residual and unit coherence by construction
            bad = fit['flags'][:, p_idx] | (fit['resid_rms'][:, p_idx] == 0)
            if (~bad).any():
                y['delay'] = y['delay'] - np.median(y['delay'][~bad])
            for ax, v in zip(axes[:, p_idx], y.values(), strict=True):
                ax.plot(np.where(bad, np.nan, v), '.-', ms=3, lw=0.6, color=f'C{i}', label=run)
    labels = ('Delay - median [ns]', 'Residual rms [deg]', 'Coherence')
    for p_idx in range(2):
        axes[0, p_idx].set_title(f'{name} {POLS[p_idx]}')
        axes[-1, p_idx].set_xlabel('Antenna index')
        for ax, label in zip(axes[:, p_idx], labels, strict=True):
            ax.set_ylabel(label)
            ax.grid(alpha=0.3)
    axes[0, 0].legend(loc='upper right', fontsize='x-small')
    fig.tight_layout()
    stem = f'{name}_runs'
    fig.savefig(plotdir / f'{stem}.png')
    plt.close(fig)
    return stem


def delay_fit_summary(outdir: str | Path) -> Path | None:
    """Write `outdir/index.html` comparing the delay fits in outdir's run folders, or redirecting to the only one.

    Args:
        outdir (str | Path): Output directory holding a folder of `{station}_delay.cal` per calibration run.

    Returns:
        index_fn (Path): The page, or None (and no page) if there are no runs.
    """
    outdir = Path(outdir)
    index_fn = outdir / 'index.html'
    run_dirs = [d for d in sorted(outdir.glob('*/')) if any(d.glob('*_delay.cal'))]
    if len(run_dirs) <= 1:
        if run_dirs:
            write_redirect(index_fn, f'{run_dirs[0].name}/index.html')
        else:
            index_fn.unlink(missing_ok=True)
        return index_fn if run_dirs else None
    runs = {d.name: load_delay_fits(d) for d in run_dirs}

    plotdir = outdir / 'plots'
    plotdir.mkdir(parents=True, exist_ok=True)
    stations = sorted({s for results in runs.values() for s in results}, key=station_key)
    rows = [
        (s, f'<a href="{run}/index.html">{run}</a>', *_summary_row(s, results[s])[1:])
        for s in stations
        for run, results in runs.items()
        if s in results
    ]
    summary = table(['Station', 'Calibration run', *SUMMARY_COLS], rows)
    by_station = {
        s: [
            (
                _plot_station_runs(s, {run: r[s]['fit'] for run, r in runs.items() if s in r}, plotdir),
                f"{s}: fitted delay (each run's median removed), residual rms and coherence per antenna; "
                'flagged and reference antennas left out',
            )
        ]
        for s in stations
    }
    # A sub-tab per reference run, with every other run against it
    compare = []
    for i, ref in enumerate(runs):
        stems = {
            run: _plot_delay_comparison(ref, run, runs[ref], runs[run], plotdir) for run in runs if run != ref
        }
        stems = {run: stem for run, stem in stems.items() if stem}
        captions = [
            f'{run} against {ref}: fitted delays of antennas unflagged in both. A constant offset comes from '
            'a different reference antenna.'
            for run in stems
        ]
        compare.append((f'compare-{i}', ref, figs(list(stems.values()), captions=captions)))
    info = [
        ('Calibration runs', f'{len(runs)}: {", ".join(runs)}'),
        ('Stations', len(stations)),
    ]
    tabs = [
        ('summary', 'Summary', f'<div class="card"><h2>Delay fits per station and run</h2>{summary}</div>'),
        ('stations', 'By station', cluster_tabs('stations', by_station)),
        ('compare', 'Delay comparison', subtabs(compare)),
    ]
    write_report(index_fn, 'Delay Fit Comparison', info, tabs, show_version())
    return index_fn


def parse_ref_ant(s: str) -> int | str | None:
    """'none' -> None, an integer -> antenna index, anything else -> antenna name or 'auto'."""
    if s.lower() == 'none':
        return None
    return int(s) if s.lstrip('-').isdigit() else s


def add_fit_args(p: argparse.ArgumentParser):
    """Add the delay fit options to a parser."""
    g = p.add_argument_group('delay fit')
    g.add_argument(
        '--ref-ant',
        default='auto',
        help="Reference antenna index or name, 'auto' (antenna valid in most channels) or 'none'",
    )
    g.add_argument(
        '--min-coherence',
        type=float,
        default=DEFAULT_MIN_COHERENCE,
        help='Flag antennas with fit coherence below this (1 = perfectly linear phase)',
    )
    g.add_argument(
        '--sigma-thr', type=float, default=DEFAULT_SIGMA_THR, help='Antenna and channel outlier threshold'
    )
    g.add_argument(
        '--flag-freqs',
        type=str,
        default='',
        help="Comma-separated MHz ranges to leave out, e.g. '87.5-108,137-138'",
    )


def build_parser() -> argparse.ArgumentParser:
    """Command-line argument parser."""
    p = input_parser(__doc__, outdir='delay_fit', static_flags=False)
    io = p.add_argument_group('input and output')
    io.add_argument('--caldir', required=True, nargs='+', help='Calibration run folders of {station}.cal files')
    add_fit_args(p)
    return p


def main():
    """CLI entry point: fit delays to the calibration files and generate the report."""
    p = build_parser()
    args = p.parse_args()
    flag_freqs = parse_flag_freqs(p, args.flag_freqs)
    if len({Path(d).name for d in args.caldir}) < len(args.caldir):
        p.error(f'Each --caldir needs a different folder name: {args.caldir}')
    cal_files = {
        Path(d).name: sorted(glob.glob(os.path.join(glob.escape(d), '*.cal')), key=station_key)
        for d in args.caldir
    }
    empty = [d for d, files in zip(args.caldir, cal_files.values(), strict=True) if not files]
    if empty:
        p.error(f'No .cal files in {empty}; run run_stefcal.py first')
    positions = antenna_positions(sorted(glob.glob(os.path.join(glob.escape(args.datadir), f'{args.eb}.uvx'))))

    for run, files in cal_files.items():
        rundir = Path(args.outdir) / run
        rundir.mkdir(parents=True, exist_ok=True)
        results = run_delay_fits(
            files, rundir, flag_freqs, parse_ref_ant(args.ref_ant), args.min_coherence, args.sigma_thr
        )
        index_fn = delay_fit_report(
            results, rundir, positions=positions, title=f'Delay Fit Report: {run}', cli_args=vars(args)
        )
        print(f'Wrote {index_fn}')
    # Compares every run fitted into outdir, including those from earlier runs of this script
    print(f'Wrote {delay_fit_summary(args.outdir)}')


if __name__ == '__main__':
    main()
