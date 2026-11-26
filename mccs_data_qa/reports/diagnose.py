"""diagnose.py: Check every antenna of a calibration set for faults and report the suspects.

Five checks run on each station, each only where the data allow it:

* Holography, when the Sun is up at --freq-idx. The station is phased up on the Sun and each antenna's
  correlation with it measured. An antenna far below the station median is missing. With a calibration
  run, an antenna whose calibrated phase differs from its neighbours' is reported too: the aperture image
  draws every antenna at its surveyed position, so a misplaced or mis-wired antenna shows a wrong phase
  rather than a wrong position.
* UV distance, at --freq-idx, on raw visibilities so that a calibration cannot hide a swap. Each
  baseline's normalised correlation is compared with that of baselines of about the same length, which
  removes the structure of the sky (sinc-like when the Sun is up). Swapped polarisations and dead or
  noisy inputs stand out.
* Bandpass, for frequency sweeps: the level and shape of each autocorrelation spectrum against the
  station median.
* Time stability, when there are several integrations: power and correlation across them.
* Calibration, with --caldir: flagged antennas, gain outliers, poor delay fits, outlying delays and
  outlying X-Y delay differences.

An antenna fails a check when it is a robust outlier (median and MAD) within its station and, where the
check has one, also beyond an absolute limit. The report lists the failures, gives a pass, warn or fail
matrix of every antenna, and writes the failures to `issues.csv`.

Usage:
    python -m mccs_data_qa.reports.diagnose --datadir DIR --eb '*' --outdir diagnose [--caldir calibration/RUN]
"""

import argparse
import copy
import csv
import html
import warnings
from functools import partial
from pathlib import Path

import astropy.units as u
import numpy as np
import pylab as plt
from loguru import logger
from mccs_data_qa.reports.delay_fit_report import _plot_delays, add_fit_args, parse_ref_ant, pin_phase0
from mccs_data_qa.reports.holo_suncal import DEFAULT_FREQ_IDX, MIN_SUN_ALT, aperture_phase_range
from mccs_data_qa.reports.report import (
    AA1_GRID,
    args_table,
    bad_channel_times,
    cluster_tabs,
    find_station_files,
    guarded,
    input_parser,
    parse_flag_freqs,
    read_station,
    station_key,
    subtabs,
    table,
    uvx_metadata,
    write_report,
)

from ska_ost_low_uv import show_version
from ska_ost_low_uv.io import read_cal
from ska_ost_low_uv.postx import ApertureArray
from ska_ost_low_uv.postx.calibration.delay_fit import fit_delays
from ska_ost_low_uv.postx.calibration.flagging import _resolve_channel_flags, antenna_correlation, flag_channels

# Absolute limits an antenna must also pass, beyond --sigma-thr robust sigma, to fail. Real stations vary
# more than a Gaussian does, so a tight MAD alone would fail harmless differences, such as 1 deg phases.
LIMITS = {
    'missing_db': -10.0,  # Holography amplitude below the station median, dB (no sigma test)
    'phase_deg': 30.0,  # Aperture phase away from the neighbours, deg
    'bandpass_dex': 0.2,  # Bandpass level or ripple away from the station median, dex (0.2 dex = 2 dB)
    'time_dex': 0.05,  # Power variation between integrations, dex
    'xy_delay_ns': 2.0,  # X-Y delay difference away from the station median, ns (no sigma test)
}
MIN_SCORE_DROP = 0.2  # Correlation score drop between integrations, as a fraction of the antenna's median
SWAP_RATIO = 10  # Relative cross / parallel-pol correlation for a swap: about 1 when healthy, 100 when swapped
N_NEIGHBOURS = 6  # Antennas whose aperture phase an antenna is compared with
N_UV_BINS = 40  # UV distance bins for the reference correlation; narrow enough to follow the Sun's lobes
POLS = ('X', 'Y')
ISSUE_COLS = ('station', 'antenna', 'pol', 'check', 'finding', 'value', 'limit', 'likely_cause')
CHECKS = {
    'holography': 'Holography',
    'uvdist': 'UV distance',
    'bandpass': 'Bandpass',
    'time': 'Time stability',
    'cal': 'Calibration',
}
# Finding: (text, likely cause)
FINDINGS = {
    'missing': ('Missing from the aperture', 'Dead or disconnected signal chain, or data from another input'),
    'phase': ('Phase differs from its neighbours', 'Antenna out of position, or cable on the wrong input'),
    'swap': ('X and Y swapped', 'Polarisations swapped between the antenna and the TPM'),
    'low_corr': ('Low correlation', 'Dead, noisy or mis-wired input'),
    'high_corr': ('Excess correlation', 'RFI or crosstalk in the signal chain'),
    'no_signal': ('No signal', 'Signal chain off or disconnected'),
    'level': ('Bandpass level differs from the station', 'Weak or noisy signal chain'),
    'shape': ('Bandpass shape differs from the station', 'Reflection, filter fault or narrowband RFI'),
    'power_var': ('Power varies between integrations', 'Intermittent fault, such as a loose connector'),
    'corr_drop': ('Correlation drops between integrations', 'Intermittent fault, such as a loose connector'),
    'cal_flag': ('Flagged by the calibration', 'See the other checks of this antenna'),
    'gain': ('Gain amplitude differs from the station', 'Weak or noisy signal chain'),
    'poor_fit': ('Poor delay fit', 'Broken or intermittent signal chain'),
    'delay': ('Delay differs from the station', 'Cable of the wrong length'),
    'delay_holo': ('Delay differs from the station', 'Cable of the wrong length, or antenna out of position'),
    'xy_delay': ('X and Y delays differ', "Delay error in one polarisation's signal chain"),
}
# Matrix cell colours, as the QA marks of report_ui: pass, warn, fail, not run, flagged in the station config
GRADE_HEX = {0: '#71CC98', 1: '#F0EC74', 2: '#FF585D', None: '#E3E6EE', -1: '#8A8F99'}
GRADE_TEXT = {0: 'passed', 1: 'passed, but more than halfway to failing', 2: 'failed', None: 'not run', -1: 'flagged'}


class NotRun(Exception):
    """A check that the data do not allow, e.g. holography at night."""


def robust_z(x: np.ndarray, mask: np.ndarray | None = None, axis: int = 0) -> np.ndarray:
    """(x - median) / (1.4826 MAD) along `axis`, with the median and MAD of the entries not in `mask`.

    NaN entries stay NaN. A zero MAD gives +-inf for entries away from the median and 0 on it.
    """
    x = np.asarray(x, dtype='float64')
    ref = x if mask is None else np.where(np.expand_dims(mask, tuple(range(1, x.ndim))), np.nan, x)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)  # All-NaN slices
        med = np.nanmedian(ref, axis=axis, keepdims=True)
        mad = 1.4826 * np.nanmedian(np.abs(ref - med), axis=axis, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        z = (x - med) / mad
    return np.where((x == med) & np.isfinite(x), 0.0, z)


def _nanmedian(x, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmedian(x, **kwargs)


class Findings:
    """The failures of one check on one station, and every antenna's grade in it.

    Grades are 0 (passed), 1 (passed, but more than halfway to failing) and 2 (failed), per antenna and
    pol. A finding for both pols ('XY') grades both.
    """

    def __init__(self, station: str, ants: np.ndarray, check: str):
        self.station, self.ants, self.check = station, ants, check
        self.rows = []
        self.grade = np.zeros((len(ants), 2), dtype=int)

    def add(self, finding: str, fail, warn, values, pol: str, limit: str, fmt: str = '.3g', unit: str = ''):
        """Record the antennas in `fail` as failures, and grade every antenna on `fail` and `warn`."""
        fail, warn = np.asarray(fail, bool), np.asarray(warn, bool)
        cols = {'X': [0], 'Y': [1]}.get(pol, [0, 1])
        self.grade[:, cols] = np.maximum(self.grade[:, cols], np.where(fail, 2, np.where(warn, 1, 0))[:, None])
        text, cause = FINDINGS[finding]
        self.rows += [
            dict(
                zip(
                    ISSUE_COLS,
                    (self.station, self.ants[i], pol, self.check, text, f'{values[i]:{fmt}}{unit}', limit, cause),
                    strict=True,
                )
            )
            for i in np.flatnonzero(fail)
        ]


#################
# Statistics, testable without data
#################


def uvdist_stats(rho: np.ndarray, bl_len: np.ndarray, ant1: np.ndarray, ant2: np.ndarray, n_ant: int) -> dict:
    """Each antenna's correlation relative to baselines of about the same length.

    Correlation is not flat with baseline length: the diffuse sky raises it on short baselines, and with
    the Sun up it follows the Sun's sinc-like visibility. Dividing each baseline by the median of its UV
    distance bin removes that structure; each antenna then takes the median over its baselines, which also
    averages over the baselines' orientations.

    Args:
        rho (np.ndarray): (N_bl, 4) normalised correlation |V_ij^pq| / sqrt(|V_ii^pp| |V_jj^qq|), pols
            XX, XY, YX, YY. NaN for flagged antennas.
        bl_len (np.ndarray): (N_bl,) baseline length, m.
        ant1, ant2 (np.ndarray): (N_bl,) antenna indexes, ant1 != ant2.
        n_ant (int): Number of antennas.

    Returns:
        stats (dict): (N_ant,) 'xx' and 'yy', the relative parallel-pol correlation (1 is typical), and
            'cross', (XY + YX) / (XX + YY) of the relative correlations (1 is typical, about 100 for swapped
            pols); also 'bin_edges' and 'bin_median' (N_bins, 4), for plots.
    """
    edges = np.unique(np.quantile(bl_len, np.linspace(0, 1, N_UV_BINS + 1)))
    b = np.clip(np.searchsorted(edges, bl_len, side='right') - 1, 0, len(edges) - 2)
    bin_med = np.array([_nanmedian(rho[b == i], axis=0) for i in range(len(edges) - 1)])
    with np.errstate(divide='ignore', invalid='ignore'):
        norm = rho / bin_med[b]
        cross = (norm[:, 1] + norm[:, 2]) / (norm[:, 0] + norm[:, 3])
    xx, yy, cr = (np.full(n_ant, np.nan) for _ in range(3))
    # ponytail: loop over antennas, O(N_ant * N_bl); fine for 256 antennas
    for a in range(n_ant):
        sel = (ant1 == a) | (ant2 == a)
        xx[a], yy[a], cr[a] = _nanmedian(norm[sel, 0]), _nanmedian(norm[sel, 3]), _nanmedian(cross[sel])
    return {'xx': xx, 'yy': yy, 'cross': cr, 'bin_edges': edges, 'bin_median': bin_med}


def bandpass_stats(autos: np.ndarray, chan_flags: np.ndarray, ant_flags: np.ndarray) -> dict:
    """Each antenna's autocorrelation level and ripple relative to the station median spectrum.

    Args:
        autos (np.ndarray): (N_freq, N_ant, 2) autocorrelation power, X and Y.
        chan_flags (np.ndarray): (N_freq,) channels to leave out.
        ant_flags (np.ndarray): (N_ant,) antennas left out of the station median.

    Returns:
        stats (dict): 'ratio' (N_freq, N_ant, 2), log10 of the autos over the station median spectrum (NaN
            in flagged channels and for zero autos), and (N_ant, 2) 'level', the median ratio, and 'ripple',
            the robust standard deviation of the ratio over channels, both in dex.
    """
    a = np.where(autos > 0, autos, np.nan).astype('float64')
    a[chan_flags] = np.nan
    ref = _nanmedian(a[:, ~ant_flags], axis=1)  # (N_freq, 2)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.log10(a / ref[:, None])
    level = _nanmedian(ratio, axis=0)
    ripple = 1.4826 * _nanmedian(np.abs(ratio - level), axis=0)
    return {'ratio': ratio, 'level': level, 'ripple': ripple}


def time_stats(power: np.ndarray, score: np.ndarray, ant_flags: np.ndarray) -> dict:
    """Each antenna's stability of autocorrelation power and correlation score across integrations.

    Args:
        power (np.ndarray): (N_time, N_ant, 2) autocorrelation power, X and Y.
        score (np.ndarray): (N_time, N_ant) `antenna_correlation()` score.
        ant_flags (np.ndarray): (N_ant,) antennas left out of the station median.

    Returns:
        stats (dict): 'ratio' (N_time, N_ant, 2), log10 power over the station median at each time;
            'power_var' (N_ant, 2), the robust standard deviation of that ratio over time, in dex; and
            'score_drop' (N_ant,), 1 - min / median of the antenna's score over time.
    """
    p = np.where(power > 0, power, np.nan).astype('float64')
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.log10(p / _nanmedian(p[:, ~ant_flags], axis=1)[:, None])
        var = 1.4826 * _nanmedian(np.abs(ratio - _nanmedian(ratio, axis=0)), axis=0)
        drop = 1 - np.nanmin(score, axis=0) / _nanmedian(score, axis=0)
    return {'ratio': ratio, 'power_var': var, 'score_drop': drop}


def holo_stats(meas_corr: np.ndarray, autos: np.ndarray, enu: np.ndarray, ant_flags: np.ndarray) -> dict:
    """Each antenna's holography amplitude, and its phase against its neighbours'.

    Args:
        meas_corr (np.ndarray): (N_ant, 2) X and Y correlation with the calibrator, as in
            `holography.jishnu_selfholo()`, from phased-up visibilities with raw amplitudes.
        autos (np.ndarray): (N_ant, 2) X and Y raw autocorrelation power, to cancel each antenna's gain.
        enu (np.ndarray): (N_ant, 3) antenna positions, m.
        ant_flags (np.ndarray): (N_ant,) antennas left out of the station median and the neighbours.

    Returns:
        stats (dict): (N_ant, 2) 'amp_db', the amplitude relative to the station median in dB, and
            'dphase_deg', the phase relative to the mean phasor of the N_NEIGHBOURS nearest unflagged
            antennas in deg.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        amp = np.abs(meas_corr) / np.sqrt(np.abs(autos))
        amp_db = 20 * np.log10(amp / _nanmedian(amp[~ant_flags], axis=0))
    good = np.flatnonzero(~ant_flags)
    d = np.linalg.norm(enu[:, None, :2] - enu[None, good, :2], axis=-1)
    d[good, np.arange(len(good))] = np.inf  # Not its own neighbour
    nbrs = good[np.argsort(d, axis=1)[:, :N_NEIGHBOURS]]  # (N_ant, k)
    with np.errstate(divide='ignore', invalid='ignore'):
        z = meas_corr / np.abs(meas_corr)
    ref = np.nansum(z[nbrs], axis=1)  # (N_ant, 2)
    dphase = np.rad2deg(np.angle(z * np.conj(ref)))
    return {'amp_db': amp_db, 'dphase_deg': np.where(np.isfinite(z), dphase, np.nan)}


#################
# Checks on one station: each returns (Findings, [(plot stem, caption)]) or raises NotRun
#################


def _autos(aa, t_idx=None, f_idx=None) -> np.ndarray:
    """|XX| and |YY| autocorrelations, (..., N_ant, 2), of every time or channel whose index is None."""
    d = aa.uvx.data
    auto_idx = np.flatnonzero(d.ant1.values == d.ant2.values)
    sel = (
        {'baseline': auto_idx}
        | ({} if t_idx is None else {'time': t_idx})
        | ({} if f_idx is None else {'frequency': f_idx})
    )
    p_idx = [int(np.flatnonzero(aa.p == p)[0]) for p in ('XX', 'YY')]
    return np.abs(d.isel(sel).values[..., p_idx])


def phase_only(cal):
    """Copy of a cal with unit-amplitude X and Y gains and no leakage terms.

    Antennas the cal zeroed get gain 1. An antenna's own phase does not change the amplitude of its
    correlation with the phased-up station, so they can still be measured.
    """
    cal = copy.deepcopy(cal)
    g = cal.gains.values
    with np.errstate(divide='ignore', invalid='ignore'):
        unit = np.where((g != 0) & np.isfinite(g), g / np.abs(g), 1)
    if g.shape[-1] == 4:
        unit[..., [1, 2]] = 0
    cal.gains.values = unit.astype(g.dtype)
    return cal


def _save(name: str, plotdir: Path, stem: str) -> str:
    stem = f'{name}_{stem}'
    plt.savefig(plotdir / f'{stem}.png', bbox_inches='tight')
    plt.close('all')
    return stem


def check_holography(aa, ants, ant_flags, cal_ok, thr, lim, plotdir):
    """Missing antennas, and phases that differ from the neighbours'; see `holo_stats()`."""
    sun = aa.coords.get_sun()
    alt = float(aa.coords.get_alt_az(sun).alt.deg)
    if alt < MIN_SUN_ALT:
        raise NotRun(f'the Sun is at {alt:.0f} deg, below {MIN_SUN_ALT} deg')
    pv = aa.coords.generate_phase_vector(sun, coplanar=True)[0]

    def meas_corr():
        v = aa.generate_vis_matrix('cal', include_autos=False)
        return np.einsum('i,ilp,l->ip', pv, v, np.conj(pv), optimize='optimal')[:, [0, 3]]

    # Raw phases are random, so the antennas add up on the Sun only once phased up. The phases come from
    # the cal run, or without one from a JishnuCal of this channel; the amplitudes stay raw, so that a
    # missing antenna stays missing, and the raw autos then cancel each antenna's gain.
    holo = aa.calibration.holography
    holo.set_cal_src(sun)
    orig_cal = aa.cal
    aa.set_cal(phase_only(aa.cal) if cal_ok else holo.run_jishnucal(do_magnitude_cal=False))
    s = holo_stats(meas_corr(), _autos(aa, aa.idx['t'], aa.idx['f']), aa.xyz_enu, ant_flags)
    # A JishnuCal fits the phases exactly, so without a cal run the phase residuals mean nothing
    dphase = s['dphase_deg'] if cal_ok else np.full_like(s['dphase_deg'], np.nan)
    z_phs = robust_z(np.abs(dphase), ant_flags)
    out = Findings(aa.name, ants, 'holography')
    missing, odd = np.zeros(aa.n_ant, bool), np.zeros(aa.n_ant, bool)
    for p, pol in enumerate(POLS):
        amp, dph = s['amp_db'][:, p], np.abs(dphase[:, p])
        m = (amp < lim['missing_db']) & ~ant_flags
        o = (z_phs[:, p] > thr) & (dph > lim['phase_deg']) & ~ant_flags & ~m
        out.add('missing', m, amp < lim['missing_db'] / 2, amp, pol, f'{lim["missing_db"]:g} dB', '.1f', ' dB')
        out.add(
            'phase', o, dph > lim['phase_deg'] / 2, dphase[:, p], pol, f'{lim["phase_deg"]:g} deg, {thr:g}σ', '.0f',
            ' deg',
        )  # fmt: skip
        missing |= m
        odd |= o

    holo.run_selfholo(vis='cal')
    holo.plot_aperture_xy(show_flags=True, phs_range=aperture_phase_range(holo.holo_dict))
    e, n = aa.xyz_enu[:, 0], -aa.xyz_enu[:, 1]  # The (E, -N) axes of holography.plot_aperture
    for ax in plt.gcf().axes:
        if ax.images:
            ax.scatter(e[missing], n[missing], marker='x', s=60, c='cyan', label='missing')
            ax.scatter(e[odd], n[odd], marker='o', s=80, facecolors='none', edgecolors='magenta', label='phase')
    if missing.any() or odd.any():
        plt.gcf().axes[0].legend(loc='upper right', fontsize=7)
    f_mhz = aa.f[aa.idx['f']].to_value('MHz')
    phases = 'the calibration run' if cal_ok else 'a JishnuCal'
    plt.suptitle(f'{aa.name} aperture at {f_mhz:.1f} MHz: raw amplitudes, phases from {phases}')
    stem = _save(aa.name, plotdir, 'aperture')
    aa.set_cal(orig_cal)
    caption = (
        f'{aa.name}. Cyan crosses mark missing antennas, magenta circles antennas whose phase differs from '
        "their neighbours', and red circles antennas flagged in the station configuration or the calibration."
    )
    return out, [(stem, caption)]


def check_uvdist(aa, ants, ant_flags, thr, plotdir):
    """Swapped polarisations, and low or excess correlation; see `uvdist_stats()`."""
    v = aa.generate_vis_matrix('data')  # (N_ant, N_ant, 4)
    auto = np.abs(np.diagonal(v, axis1=0, axis2=1)).T  # (N_ant, 4)
    i, j = np.triu_indices(aa.n_ant, k=1)
    pp = {0: (0, 0), 1: (0, 3), 2: (3, 0), 3: (3, 3)}  # Pol: (auto pol of antenna i, of antenna j)
    with np.errstate(divide='ignore', invalid='ignore'):
        rho = np.stack([np.abs(v[i, j, p]) / np.sqrt(auto[i, a] * auto[j, b]) for p, (a, b) in pp.items()], axis=1)
    rho[ant_flags[i] | ant_flags[j]] = np.nan
    bl = aa.bl_matrix[i, j]
    s = uvdist_stats(rho, bl, i, j, aa.n_ant)

    z_x, z_y, z_c = (robust_z(np.log(s[k]), ant_flags) for k in ('xx', 'yy', 'cross'))
    swap = (z_c > thr) & (s['cross'] > SWAP_RATIO) & ~ant_flags
    out = Findings(aa.name, ants, 'uvdist')
    out.add('swap', swap, s['cross'] > SWAP_RATIO / 2, s['cross'], 'XY', f'{SWAP_RATIO}, {thr:g}σ', '.0f')
    bad = swap.copy()
    for pol, z, val in (('X', z_x, s['xx']), ('Y', z_y, s['yy'])):
        lo, hi = (z < -thr) & ~ant_flags & ~swap, (z > thr) & ~ant_flags & ~swap
        out.add('low_corr', lo, z < -thr / 2, val, pol, f'{thr:g}σ', '.2f')
        out.add('high_corr', hi, z > thr / 2, val, pol, f'{thr:g}σ', '.2f')
        bad |= lo | hi

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    mid = (s['bin_edges'][1:] + s['bin_edges'][:-1]) / 2
    colors = plt.cm.tab10.colors
    for p, ax in enumerate(axes.ravel()):
        ax.scatter(bl, rho[:, p], s=1, c='0.6', alpha=0.2, rasterized=True)
        ax.plot(mid, s['bin_median'][:, p], 'k-', lw=1.5, label='median')
        # ponytail: the first 10 failed antennas are coloured; the rest are in the table
        for c, a in enumerate(np.flatnonzero(bad)[:10]):
            sel = (i == a) | (j == a)
            ax.scatter(bl[sel], rho[sel, p], s=3, color=colors[c], label=ants[a])
        ax.set_yscale('log')
        ax.set_title(str(aa.p[p]))
        ax.set_ylabel('Normalised correlation')
    for ax in axes[1]:
        ax.set_xlabel('UV distance [m]')
    axes[0, 0].legend(fontsize=7, markerscale=4)
    fig.suptitle(f'{aa.name} raw correlation against UV distance at {aa.f[aa.idx["f"]].to_value("MHz"):.1f} MHz')
    stem = _save(aa.name, plotdir, 'uvdist')
    caption = (
        f'{aa.name}. Grey points are baselines and the black line their median at each length; the baselines '
        'of failed antennas are coloured.'
    )
    return out, [(stem, caption)]


def check_bandpass(aa, ants, ant_flags, thr, lim, chan_flags, plotdir):
    """Bandpasses whose level or shape differs from the station's; see `bandpass_stats()`."""
    if len(aa.f) < 2:
        raise NotRun('the data have one channel')
    autos = _nanmedian(_autos(aa).astype('float64'), axis=0)  # Median over time, (N_freq, N_ant, 2)
    s = bandpass_stats(autos, chan_flags, ant_flags)
    out = Findings(aa.name, ants, 'bandpass')
    failed = np.zeros(aa.n_ant, bool)
    dex = lim['bandpass_dex']
    for p, pol in enumerate(POLS):
        level, ripple = s['level'][:, p], s['ripple'][:, p]
        dead = ~np.isfinite(level) & ~ant_flags
        out.add('no_signal', dead, dead, level, pol, '')
        lvl = (np.abs(robust_z(level, ant_flags)) > thr) & (np.abs(level) > dex) & ~ant_flags
        rip = (robust_z(ripple, ant_flags) > thr) & (ripple > dex) & ~ant_flags
        out.add('level', lvl, np.abs(level) > dex / 2, level, pol, f'{dex:g} dex, {thr:g}σ', '.2f', ' dex')
        out.add('shape', rip, ripple > dex / 2, ripple, pol, f'{dex:g} dex, {thr:g}σ', '.2f', ' dex')
        failed |= dead | lvl | rip

    f = aa.f.to_value('MHz')
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for p, pol in enumerate(POLS):
        ax = axes[0, p]
        im = ax.imshow(
            s['ratio'][..., p].T,
            aspect='auto',
            origin='lower',
            cmap='RdBu_r',
            vmin=-0.5,
            vmax=0.5,
            extent=(f[0], f[-1], -0.5, aa.n_ant - 0.5),
            interpolation='nearest',
        )
        ax.set(title=f'{pol}: log10(auto / station median)', xlabel='Frequency [MHz]', ylabel='Antenna index')
        fig.colorbar(im, ax=ax, label='dex')
        ax = axes[1, p]
        db = 10 * np.log10(np.where(chan_flags[:, None], np.nan, autos[..., p]))
        ax.plot(f, db[:, ~ant_flags & ~failed], c='0.7', lw=0.5)
        badness = np.nan_to_num(np.abs(s['level'][:, p]) + s['ripple'][:, p], nan=np.inf)
        worst = sorted(np.flatnonzero(failed), key=lambda a: -badness[a])
        for a in worst[:10]:  # ponytail: the 10 worst are coloured; the rest are in the table
            ax.plot(f, db[:, a], lw=1, label=ants[a])
        ax.set(title=f'{pol} autocorrelations', xlabel='Frequency [MHz]', ylabel='Power [dB]')
        if failed.any():
            ax.legend(fontsize=7)
    fig.suptitle(f'{aa.name} bandpasses')
    stem = _save(aa.name, plotdir, 'bandpass')
    caption = (
        f'{aa.name}. Above, each antenna against the station median; below, the spectra, with the worst failed '
        'antennas coloured. Channels with RFI or bad timestamps are blank.'
    )
    return out, [(stem, caption)]


def check_time(aa, ants, ant_flags, thr, lim, plotdir):
    """Antennas whose power or correlation changes between integrations; see `time_stats()`."""
    n_t = aa.uvx.data.sizes['time']
    if n_t < 2:
        raise NotRun('the data have one integration per channel')
    power = _autos(aa, f_idx=aa.idx['f']).astype('float64')  # (N_time, N_ant, 2)
    score = np.empty((n_t, aa.n_ant))
    for t in range(n_t):
        with aa.select(t=t):
            score[t] = antenna_correlation(aa, exclude=ant_flags)
    s = time_stats(power, score, ant_flags)
    out = Findings(aa.name, ants, 'time')
    dex = lim['time_dex']
    for p, pol in enumerate(POLS):
        var = s['power_var'][:, p]
        m = (robust_z(var, ant_flags) > thr) & (var > dex) & ~ant_flags
        out.add('power_var', m, var > dex / 2, var, pol, f'{dex:g} dex, {thr:g}σ', '.3f', ' dex')
    drop = s['score_drop']
    m = (robust_z(drop, ant_flags) > thr) & (drop > MIN_SCORE_DROP) & ~ant_flags
    out.add('corr_drop', m, drop > MIN_SCORE_DROP / 2, drop * 100, 'XY', f'{MIN_SCORE_DROP:.0%}, {thr:g}σ', '.0f', '%')

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for p, pol in enumerate(POLS):
        im = axes[p].imshow(
            s['ratio'][..., p].T,
            aspect='auto',
            origin='lower',
            cmap='RdBu_r',
            vmin=-0.3,
            vmax=0.3,
            interpolation='nearest',
        )
        axes[p].set(title=f'{pol}: log10(power / station median)', xlabel='Integration', ylabel='Antenna index')
        fig.colorbar(im, ax=axes[p], label='dex')
    im = axes[2].imshow(score.T, aspect='auto', origin='lower', interpolation='nearest')
    axes[2].set(title='Correlation with the station', xlabel='Integration', ylabel='Antenna index')
    fig.colorbar(im, ax=axes[2])
    fig.suptitle(f'{aa.name} over {n_t} integrations at {aa.f[aa.idx["f"]].to_value("MHz"):.1f} MHz')
    stem = _save(aa.name, plotdir, 'time')
    return out, [(stem, f'{aa.name}. A healthy antenna keeps the same colour along its row.')]


def check_cal(cal, name, ants, ant_flags, thr, lim, fit_kwargs, zero_phase0, flag_freqs, positions, plotdir):
    """Calibration flags, gain outliers, poor delay fits, and outlying delays and X-Y delay differences."""
    if cal is None:
        raise NotRun('no calibration run was chosen, or the run has no file for this station')
    p_idx = [0, 3] if cal.gains.sizes['polarization'] == 4 else [0, 1]
    g = np.abs(cal.gains.values[..., p_idx]).astype('float64')
    g[cal.flags.channel.values] = np.nan
    amp = _nanmedian(np.where(g > 0, g, np.nan), axis=0)  # (N_ant, 2)
    out = Findings(name, ants, 'cal')
    cal_flags = cal.flags.antenna.values & ~ant_flags
    out.add('cal_flag', cal_flags, cal_flags, np.zeros(len(ants)), 'XY', '', '.0f')
    excl = ant_flags | cal.flags.antenna.values
    for p, pol in enumerate(POLS):
        z = np.abs(robust_z(np.log(amp[:, p]), excl))
        out.add('gain', (z > thr) & ~excl, (z > thr / 2) & ~excl, amp[:, p], pol, f'{thr:g}σ')
    stems = []
    if cal.gains.sizes['frequency'] < 3:
        return out, stems

    flags = _resolve_channel_flags(cal.gains.frequency.values * u.Hz, flag_freqs)
    fit = fit_delays(cal, flags, **fit_kwargs)
    if zero_phase0:
        fit = pin_phase0(fit, cal.gains.frequency.values.astype('float64'), fit_kwargs['min_coherence'])
    delay_finding = 'delay_holo' if 'holo' in str(cal.method).lower() else 'delay'
    tau = fit['delay_ns']
    min_coh = fit_kwargs['min_coherence']
    for p, pol in enumerate(POLS):
        coh = fit['coherence'][:, p]
        poor = fit['flags'][:, p] & ~excl
        out.add('poor_fit', poor, (coh < (1 + min_coh) / 2) & ~excl, coh, pol, f'coherence {min_coh:g}', '.2f')
        ok = ~excl & ~fit['flags'][:, p]
        z = np.abs(robust_z(tau[:, p], ~ok))
        dev = tau[:, p] - _nanmedian(tau[ok, p])
        out.add(delay_finding, (z > thr) & ok, (z > thr / 2) & ok, dev, pol, f'{thr:g}σ', '.2f', ' ns')
    ok = ~excl & ~fit['flags'].any(axis=1)
    dxy = tau[:, 0] - tau[:, 1]
    dev = np.abs(dxy - _nanmedian(dxy[ok]))
    xy = lim['xy_delay_ns']
    out.add('xy_delay', (dev > xy) & ok, (dev > xy / 2) & ok, dev, 'XY', f'{xy:g} ns', '.2f', ' ns')
    stems.append((_plot_delays(name, {'fit': fit, 'cal': cal}, positions, plotdir), f'{name}. Fitted delays.'))
    return out, stems


#################
# Report
#################


def descriptions(lim: dict, thr: float) -> dict:
    """What each check measures and what its failures suggest, for the report's tabs."""
    return {
        'holography': (
            "The station is phased up on the Sun and each antenna's correlation with the Sun is measured. Its "
            'raw amplitude is divided by the square root of its autocorrelation, so that receiver gain does not '
            f'matter. An antenna more than {-lim["missing_db"]:g} dB below the station median is missing: it '
            'sees the sky but not the Sun, which usually means a dead or disconnected signal chain, or data '
            "from one input labelled as another.</p><p>With a calibration run chosen, each antenna's phase is "
            f'also compared with that of its {N_NEIGHBOURS} nearest neighbours. The aperture image draws every '
            'antenna at its surveyed position, so an antenna that is in the wrong place, or whose cable goes to '
            'the wrong input, does not appear displaced; it appears with the wrong phase. A phase more than '
            f"{lim['phase_deg']:g} deg from the neighbours' points to such an error. Without a calibration run "
            'the phases come from a JishnuCal of this channel, which fits them exactly, and this comparison is '
            'skipped. A calibration from a sky model shows position errors best, as a Sun holography '
            'calibration absorbs them into its phases.'
        ),
        'uvdist': (
            'This check uses the raw visibilities at the chosen channel, before any calibration, since a '
            "calibration that solves for leakage could hide a swap. Each baseline's correlation is divided by "
            "the two antennas' autocorrelations, which removes the receiver gains.</p><p>Correlation is not "
            'flat with baseline length. The diffuse sky makes short baselines correlate more strongly, and with '
            "the Sun up the curve follows the Sun's visibility, a sinc-like series of lobes. Each baseline is "
            'therefore compared with the median of baselines of about the same length, the black line in the '
            'plots, and each antenna is scored by the median of these ratios over all its baselines, which also '
            'averages over their orientations. A healthy antenna scores about one in each polarisation.</p><p>If '
            "an antenna's X and Y are swapped, its cross-polarisation products take the strength of the "
            'parallel ones, and the reverse. The ratio of cross to parallel correlation is then about a hundred '
            f'rather than one, and the check reports a swap above {SWAP_RATIO}. An antenna whose correlation '
            'is low in one polarisation has a dead, noisy or mis-wired input in it; one whose correlation is '
            'high may be picking up RFI or crosstalk.'
        ),
        'bandpass': (
            "Each antenna's autocorrelation spectrum, its median over integrations, is compared with the "
            'station median spectrum; channels with RFI or bad timestamps are left out. The level is the '
            'median ratio across the band and the shape is the scatter of the ratio about that level, both in '
            'dex, where 0.1 dex is 1 dB. An antenna fails when it is an outlier within the station and also '
            f'differs by more than {lim["bandpass_dex"]:g} dex. A low level suggests a weak amplifier or a lossy '
            'cable, a high one excess noise; a shape outlier suggests a reflection, a filter fault or '
            'narrowband interference in one signal chain.'
        ),
        'time': (
            'When the data hold more than one integration per channel, each antenna is followed across them '
            'at the chosen channel. An antenna fails if its power, relative to the station median, varies by '
            f'more than {lim["time_dex"]:g} dex, or if its correlation with the rest of the station falls by '
            f'more than {MIN_SCORE_DROP:.0%} of its usual value. Either points to an intermittent fault, such as '
            'a loose connector or a signal chain that drops out.'
        ),
        'cal': (
            'This check reads the chosen calibration run. It reports the antennas the calibration flagged, and '
            'antennas whose median gain amplitude is an outlier within the station. It then fits a delay to '
            "each antenna's gain phase across the band, as the Delay fit report does.</p><p>A poor fit means "
            'that the phase does not follow a straight line with frequency, as happens when a signal chain is '
            "broken or intermittent. A delay that stands out from the station's points to a cable of the wrong "
            'length; with a Sun holography calibration, it may also be an antenna out of position, since the '
            'geometric delay to the Sun is then part of the fitted delay. The difference between the X and Y '
            'delays is normally much the same on every antenna, so an antenna whose difference lies more than '
            f'{lim["xy_delay_ns"]:g} ns from the station median has a delay error in one polarisation.'
        ),
        'matrix': (
            'Each row is an antenna and each pair of columns a check, in X and Y. Green means the antenna '
            'passed. Yellow means it passed but is more than halfway to failing: past half of an absolute '
            f'limit or, where the check has none, more than {thr / 2:g}σ from the station median. Red means it '
            'failed; hover over the cell for the findings, which are also listed on the Summary tab. Light grey '
            'means the check did not run on this station, and dark grey that the antenna is flagged in the '
            'station configuration and was not checked.'
        ),
    }


def _run(check):
    """check(), or its NotRun if the data do not allow it."""
    try:
        return check()
    except NotRun as err:
        return err


def antenna_matrix(ants: np.ndarray, ant_flags: np.ndarray, grades: dict, rows: list, n_cols: int = 4) -> str:
    """Pass, warn and fail matrix of every antenna and check, in `n_cols` side-by-side blocks.

    Args:
        ants (np.ndarray): Antenna names.
        ant_flags (np.ndarray): Antennas flagged in the station configuration.
        grades (dict): Per check in CHECKS, a (N_ant, 2) array of `Findings` grades, or None if not run.
        rows (list): The station's issue rows, for the cells' tooltips.
        n_cols (int): Blocks of antennas side by side.
    """
    tips = {}
    for r in rows:
        for pol in POLS if r['pol'] == 'XY' else (r['pol'],):
            tips.setdefault((r['antenna'], r['check'], pol), []).append(f'{r["finding"]}: {r["value"]}')

    def cell(a, key, p):
        g = -1 if ant_flags[a] else (None if grades[key] is None else int(grades[key][a, p]))
        tip = '; '.join(tips.get((ants[a], key, POLS[p]), [])) or GRADE_TEXT[g]
        return f'<td style="background:{GRADE_HEX[g]}" title="{html.escape(f"{CHECKS[key]} {POLS[p]}: {tip}")}"></td>'

    head = (
        '<tr><th></th>' + ''.join(f'<th colspan="2">{CHECKS[k]}</th>' for k in CHECKS) + '</tr><tr><th>Antenna</th>'
        + '<th>X</th><th>Y</th>' * len(CHECKS) + '</tr>'
    )  # fmt: skip
    per_block = -(-len(ants) // n_cols)
    blocks = []
    for start in range(0, len(ants), per_block):
        body = ''.join(
            f'<tr><td class="amx-ant">{html.escape(ants[a])}</td>'
            + ''.join(cell(a, k, p) for k in CHECKS for p in (0, 1))
            + '</tr>'
            for a in range(start, min(start + per_block, len(ants)))
        )
        blocks.append(f'<table class="amx">{head}{body}</table>')
    return f'<div class="amx-wrap">{"".join(blocks)}</div>'


# Compact matrix cells; the report's table style pads every cell
MATRIX_CSS = (
    '<style>.amx-wrap{display:flex;flex-wrap:wrap;gap:1.5rem;align-items:flex-start}'
    'table.amx{width:auto;border-collapse:collapse;font-size:.72rem}'
    'table.amx td,table.amx th{padding:0 .25rem;height:1rem;border:1px solid #fff;text-align:center}'
    'table.amx td{min-width:1.1rem}table.amx td.amx-ant{text-align:left;font-family:monospace}</style>'
)


def diagnose(
    files: list[str],
    outdir: str = 'diagnose',
    freq_idx: int = DEFAULT_FREQ_IDX,
    caldir: str | None = None,
    sigma_thr: float = 5.0,
    limits: dict | None = None,
    fit_kwargs: dict | None = None,
    zero_phase0: bool = False,
    flag_freqs: list | None = None,
    ignore_static_flags: bool = False,
    cli_args: dict | None = None,
) -> Path:
    """Run the checks on each station and write `outdir/index.html` and `outdir/issues.csv`.

    Args:
        files (list): UVX files, one per station.
        outdir (str): Output directory.
        freq_idx (int): Channel for the holography, UV distance and time checks. It is clamped to the file's
            channels and moved to the nearest channel without RFI, a bad timestamp or a cal flag.
        caldir (str): Calibration run folder of `{station}.cal` files, or None to skip the cal check.
        sigma_thr (float): Robust outlier threshold, in sigma.
        limits (dict): Absolute limits, overriding those in `LIMITS`.
        fit_kwargs (dict): `fit_delays()` options: ref_ant, min_coherence and sigma_thr.
        zero_phase0 (bool): Pin the delay fit phase offsets to 0, see `pin_phase0()`.
        flag_freqs (list): (start, stop) Quantity ranges left out of the bandpass and delay fits.
        ignore_static_flags (bool): Ignore the static antenna flags stored in the UVX files.
        cli_args (dict): Command-line arguments for the Settings tab.

    Returns:
        index_fn (Path): The report page.
    """
    out = Path(outdir)
    plotdir = out / 'plots'
    plotdir.mkdir(parents=True, exist_ok=True)
    lim = LIMITS | (limits or {})
    fit_kwargs = {'ref_ant': 'auto', 'min_coherence': 0.5, 'sigma_thr': sigma_thr} | (fit_kwargs or {})
    all_issues, metas, failed, matrices = [], {}, {}, {}
    plots = {k: {} for k in CHECKS}
    notes = {k: [] for k in CHECKS}
    errors = dict.fromkeys(CHECKS, '')
    chans = {}

    # ponytail: stations run one at a time, holding one sweep in memory; run_in_parallel them if too slow
    for fn in files:
        try:
            aa = ApertureArray(read_station(fn, ignore_static_flags))
        except Exception as err:
            logger.exception(f'Cannot read {fn}')
            failed[Path(fn).name] = repr(err)
            continue
        aa.set_task_verbosity('silent')
        name = aa.name
        ants = np.asarray(aa.uvx.antennas.attrs['identifier'].values).astype(str)
        ant_flags = aa.antenna_flags.copy()
        metas[name] = uvx_metadata(fn, aa.uvx)

        cal, cal_ok = None, False
        cal_fn = Path(caldir) / f'{name}.cal' if caldir else None
        if cal_fn and cal_fn.exists():
            cal = read_cal(str(cal_fn))
            cal_ok = cal.gains.sizes['frequency'] == len(aa.f)
            if not cal_ok:
                logger.warning(
                    f'{name}: {cal_fn.name} has {cal.gains.sizes["frequency"]} channels and the data '
                    f'{len(aa.f)}, so it is not applied to the visibilities'
                )
        elif cal_fn:
            logger.warning(f'{name}: no {cal_fn}')

        # Channels with RFI, a bad timestamp or a cal flag are left out of the bandpass and not used for the
        # single-channel checks
        chan_flags = bad_channel_times(aa.uvx)
        if len(aa.f) > 2:
            chan_flags |= flag_channels(aa, antenna_flags=ant_flags, initial_flags=flag_freqs)
        if cal_ok:
            chan_flags |= cal.flags.channel.values
            aa.set_cal(cal)
        idx = min(freq_idx, len(aa.f) - 1)
        good = np.flatnonzero(~chan_flags)
        if good.size and idx not in good:
            idx = int(good[np.argmin(np.abs(good - idx))])
        if idx != freq_idx:
            logger.info(f'{name}: channel {freq_idx} is unavailable or flagged, so channel {idx} is used')
        aa.set_idx(f=idx)
        chans[name] = idx

        positions = {cal.telescope: dict(zip(ants, aa.xyz_enu, strict=True))} if cal is not None else None
        checks = {
            'holography': partial(check_holography, aa, ants, ant_flags, cal_ok, sigma_thr, lim, plotdir),
            'uvdist': partial(check_uvdist, aa, ants, ant_flags, sigma_thr, plotdir),
            'bandpass': partial(check_bandpass, aa, ants, ant_flags, sigma_thr, lim, chan_flags, plotdir),
            'time': partial(check_time, aa, ants, ant_flags, sigma_thr, lim, plotdir),
            'cal': partial(
                check_cal, cal, name, ants, ant_flags, sigma_thr, lim, fit_kwargs, zero_phase0, flag_freqs,
                positions, plotdir,
            ),
        }  # fmt: skip
        grades, rows = {}, []
        for key, check in checks.items():
            logger.info(f'{name}: {CHECKS[key]}')
            # A check that fails (e.g. on NaN data) shows as an error card, and the others still run
            res, ok = guarded(f'{CHECKS[key]} of {name}', lambda check=check: _run(check))
            grades[key] = None
            if not ok:
                errors[key] += res
            elif isinstance(res, NotRun):
                notes[key].append(f'{name}: not run, as {res}.')
            else:
                rows += res[0].rows
                grades[key] = res[0].grade
                plots[key][name] = res[1]
        all_issues += rows
        matrices[name] = antenna_matrix(ants, ant_flags, grades, rows)
        del aa, checks  # Free the sweep before reading the next

    if not metas:
        raise RuntimeError(f'Every station failed to load: {failed}')
    order = {k: i for i, k in enumerate(CHECKS)}
    all_issues.sort(key=lambda r: (station_key(r['station']), station_key(r['antenna']), order[r['check']]))
    with open(out / 'issues.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, ISSUE_COLS)
        w.writeheader()
        w.writerows(all_issues)

    stations = list(metas)
    counts = table(
        ['Station', *CHECKS.values(), 'Antennas'],
        [
            (
                s,
                *(len({r['antenna'] for r in all_issues if r['station'] == s and r['check'] == k}) for k in CHECKS),
                len({r['antenna'] for r in all_issues if r['station'] == s}),
            )
            for s in stations
        ],
    )
    issue_table = table(
        ['Station', 'Antenna', 'Pol', 'Check', 'Finding', 'Value', 'Limit', 'Likely cause'],
        [(*(html.escape(str(r[c])) for c in ISSUE_COLS[:3]), CHECKS[r['check']],
          *(html.escape(str(r[c])) for c in ISSUE_COLS[4:])) for r in all_issues],
    )  # fmt: skip
    summary = (
        '<div class="card"><h2>Failed antennas per station</h2><p>Each count is of antennas that failed the '
        'check in either polarisation; the last column counts antennas that failed any check.</p>'
        f'{counts}</div><div class="card"><h2>Findings</h2><p>Each row is an antenna that failed a check, with '
        'the measured value, the limit it broke and the most likely cause. Antennas flagged in the station '
        'configuration are not checked. The same list is in <a href="issues.csv">issues.csv</a>.</p>'
        f'{issue_table}</div>'
    )
    desc = descriptions(lim, sigma_thr)

    def check_tab(key):
        note = ''.join(f'<li>{html.escape(n)}</li>' for n in notes[key])
        body = f'<div class="card"><p>{desc[key]}</p>{f"<ul>{note}</ul>" if note else ""}</div>'
        return (key, CHECKS[key], body + errors[key] + cluster_tabs(key, plots[key]))

    matrix_tab = f'{MATRIX_CSS}<div class="card"><p>{desc["matrix"]}</p></div>' + subtabs(
        [(f'matrix-{s}', s, f'<div class="card">{m}</div>') for s, m in matrices.items()]
    )
    meta_fields = list(dict.fromkeys(k for m in metas.values() for k in m))
    meta_table = table(['Field', *stations], [(k, *(m.get(k, '') for m in metas.values())) for k in meta_fields])
    tabs = [
        ('summary', 'Summary', summary),
        ('antennas', 'Antennas', matrix_tab),
        *(check_tab(k) for k in CHECKS),
        ('metadata', 'Metadata', f'<div class="card"><h2>UVX file metadata</h2>{meta_table}</div>'),
        ('settings', 'Settings', f'<div class="card"><h2>Runtime arguments</h2>{args_table(cli_args or {})}</div>'),
    ]
    first = next(iter(metas.values()))
    n_ants = len({(r['station'], r['antenna']) for r in all_issues})
    info = [
        ('Stations', f'{len(stations)}: {", ".join(stations)}'),
        ('Channel', ', '.join(sorted({str(c) for c in chans.values()})) + f' (requested {freq_idx})'),
        ('UTC start', first['UTC start']),
        ('LST start', first['LST start']),
        ('Calibration run', html.escape(Path(caldir).name) if caldir else 'none'),
        ('Failed antennas', f'{n_ants}, with {len(all_issues)} findings'),
    ]
    badges = ''.join(f'<span class="badge">{html.escape(n)} could not be read</span>' for n in failed)
    index_fn = out / 'index.html'
    write_report(index_fn, 'Antenna Diagnosis', info, tabs, show_version(), card_extra=badges)
    return index_fn


def build_parser() -> argparse.ArgumentParser:
    """Command-line argument parser."""
    p = input_parser(__doc__, outdir='diagnose')
    io = p.add_argument_group('input and output')
    io.add_argument(
        '--caldir', default=None, help='Calibration run folder of {station}.cal files; without one, no cal check'
    )
    g = p.add_argument_group('checks')
    g.add_argument(
        '--freq-idx',
        type=int,
        default=DEFAULT_FREQ_IDX,
        help='Channel for the holography, UV distance and time checks; a flagged channel is replaced by the '
        'nearest clean one',
    )
    helps = {
        'missing_db': 'Holography: an antenna this many dB below the station median is missing',
        'phase_deg': "Holography: an antenna fails if its phase is this far from its neighbours', in deg",
        'bandpass_dex': 'Bandpass: an antenna fails if its level or ripple differs by this much, in dex '
        '(0.1 dex is 1 dB)',
        'time_dex': 'Time stability: an antenna fails if its power varies by this much, in dex',
        'xy_delay_ns': 'Calibration: an antenna fails if its X-Y delay difference is this far from the '
        'station median, in ns',
    }
    for k, v in LIMITS.items():
        g.add_argument(f'--{k.replace("_", "-")}', type=float, default=v, help=helps[k])
    add_fit_args(p)
    return p


def main():
    """CLI entry point: parse the arguments, run the checks and write the report."""
    p = build_parser()
    args = p.parse_args()
    flag_freqs = parse_flag_freqs(p, args.flag_freqs)
    files = find_station_files(p, args, AA1_GRID)
    index_fn = diagnose(
        files,
        outdir=args.outdir,
        freq_idx=args.freq_idx,
        caldir=args.caldir,
        sigma_thr=args.sigma_thr,
        limits={k: getattr(args, k) for k in LIMITS},
        fit_kwargs={
            'ref_ant': parse_ref_ant(args.ref_ant),
            'min_coherence': args.min_coherence,
            'sigma_thr': args.sigma_thr,
        },
        zero_phase0=args.zero_phase0,
        flag_freqs=flag_freqs,
        ignore_static_flags=args.ignore_static_flags,
        cli_args=vars(args),
    )
    print(f'Wrote {index_fn}')


if __name__ == '__main__':
    main()
