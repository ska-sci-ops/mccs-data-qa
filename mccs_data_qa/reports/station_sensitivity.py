#!/usr/bin/env python
"""Measure per-station SEFD and A/T from difference images of a two-integration frequency sweep.

Reads the per-channel `{station}.cal` files of a calibration run (run_stefcal.py or holo_suncal.py)
from --caldir. Channels flagged in the cal are left out. With --delay-caldir, the gain phases are the
delay model of a delay fit run (delay_fit_report.py) instead, with the amplitudes of --caldir.

Steps (run any subset with --steps):
    image  Healpix images at t=0 and t=1 and their difference, write `{station}_hpx_*.h5`.
    check  Noise-injection test of the Stokes I, Q, U, V noise prediction on the first station.
    sens   Element and station SEFD per pol and Stokes I, Q, U, V from the Jones AEP (Sutinjo et al.
           2022), at zenith and 45 deg from zenith on the X-Y diagonal, write
           `{station}_sensitivity.hkl/csv`. T_ant includes the Sun at its suncal flux. Q and U are in
           the station's Ludwig-3 frame (x = X dipole).
    plot   Per-station, all-station and array (AA*(mod), AA*, AA4) plots, write `plots/*.png`.
    html   Report webpage linking the plots and data files, write `index.html`.

Two SEFDs are reported (see station_sensitivity.md):
    element  One antenna. Sets the noise of single-station all-sky images.
    station  256-antenna station beamformed in that direction, including sky noise correlated between
             elements. This is the quantity for station-station interferometry (as in senscalc).
"""

import argparse
import gc
import glob
import os

import jax
import matplotlib

matplotlib.use('Agg')

import astropy.units as u
import h5py
import healpy as hp
import hickle as hkl
import numpy as np
import pandas as pd
import pylab as plt
from astropy.constants import c, k_B
from astropy.coordinates import AltAz, SkyCoord
from loguru import logger
from mccs_data_qa.reports.report import (
    CHANNEL_TIME_TOL_S,
    args_table,
    bad_channel_times,
    base_parser,
    figs,
    find_station_files,
    metadata_table,
    read_station,
    table,
    write_report,
)
from tqdm import tqdm

from ska_ost_low_uv import show_version
from ska_ost_low_uv.datamodel.cal import create_uvx_antenna_cal
from ska_ost_low_uv.io import read_cal, read_img, read_uvx, write_img
from ska_ost_low_uv.parallelize import run_in_parallel, task
from ska_ost_low_uv.postx import ApertureArray
from ska_ost_low_uv.postx.aeps import load_aep_jones
from ska_ost_low_uv.postx.coords.coord_utils import sky2hpix
from ska_ost_low_uv.postx.imaging.aa_imaging import n_unflagged_ant
from ska_ost_low_uv.postx.simulation._sim_utils import load_aep_for_sim
from ska_ost_low_uv.postx.simulation.gsm_sim import k_to_jysr
from ska_ost_low_uv.postx.sky_model import RadioSource
from ska_ost_low_uv.utils import AppendFileSink

STEPS = ['image', 'check', 'sens', 'plot', 'html']
AEP = 'ska_low_harp'
NSIDE_SIM = 128  # AEP sampling, must match the make_healpix n_side
NSIDE_COARSE = 8  # Cell size for local noise estimates (~7 deg)
NSIDE_JK = 4  # Jackknife cells (~15 deg), larger than the PSF at 50 MHz
FOV_DEG = 45  # Zenith cap used for the headline numbers
DIAG_ZA_DEG = 45  # Off-zenith SEFD direction, on the diagonal between the X and Y dipoles
MAX_ZA_DEG = 80  # B^-1 diverges towards the horizon
ETA = 0.962  # From SPRTS-516
RFI_BANDS_MHZ = [(130, 140), (240, 275), (280, 285), (330, 350)]
SPIKE_WINDOW = 15  # Running-median width (channels) for spike flagging
ZOOM_STATION_MHZ = (100, 350)
ZOOM_ALL_MHZ = (100, 330)
ARRAYS = {'AA*(mod)': 257, 'AA*': 307, 'AA4': 512}  # Number of stations
N_STATION_ANT = 256
TWO_K_JY = 2 * k_B.value * 1e26  # 2k in Jy m^2 K^-1
# Stokes selection matrices, S = tr(P_S e e^H) (Sutinjo et al. 2022, Eqs. 4-13)
P_S = {
    'i': np.eye(2),
    'q': np.diag([1.0, -1.0]),
    'u': np.array([[0, 1], [1, 0]]),
    'v': np.array([[0, -1j], [1j, 0]]),
}
POL_COLS = [
    'T_ant',
    'T_sys_auto',
    'T_rx',
    'sefd_el_img',
    'sefd_el',
    'sefd_st',
    'a_over_t_el',
    'a_over_t_st',
    'a_eff_el',
]
I_COLS = ['sefd_i_el_pred', 'sefd_i_el', 'a_over_t_i_el', 'sefd_i_st', 'a_over_t_i_st']
QUV = ('q', 'u', 'v')
_ST = '(256, beamformed, zenith)'
_AOT = '[m$^2$ K$^{-1}$]'
UNITS = {
    'T_ant': '$T_{ant}$ [K]',
    'T_sys_auto': '$T_{sys}$ (autos) [K]',
    'T_rx': '$T_{rx}$ [K]',
    'sefd_el_img': 'Element SEFD (difference-image noise) [Jy]',
    'sefd_el': 'Element SEFD (autos) [Jy]',
    'sefd_st': f'Station SEFD {_ST} [Jy]',
    'a_over_t_el': f'Element A/T {_AOT}',
    'a_over_t_st': f'Station A/T {_ST} {_AOT}',
    'a_eff_el': 'Element $A_{eff}$ [m$^2$]',
    'sefd_i_el_pred': 'Predicted element SEFD$_I$ [Jy]',
    'sefd_i_el': 'Element SEFD$_I$ [Jy]',
    'a_over_t_i_el': f'Element A/T$_I$ {_AOT}',
    'sefd_i_st': f'Station SEFD$_I$ {_ST} [Jy]',
    'a_over_t_i_st': f'Station A/T$_I$ {_ST} {_AOT}',
}


def delay_model_cal(cal, dcal):
    """Diagonal cal of the delay-model phases of `dcal` and the XX and YY gain amplitudes of `cal`.

    Args:
        cal (UVXAntennaCal): Calibration the delay fit was made to, which sets the flux scale.
        dcal (UVXAntennaCal): Unit-amplitude delay-model cal of that fit, from `delay_cal()`.

    Returns:
        cal (UVXAntennaCal): ('X', 'Y') gains, flagged where either cal is, with the provenance of `cal`.
    """
    p_idx = [0, 3] if cal.gains.sizes['polarization'] == 4 else [0, 1]
    gains = np.abs(cal.gains.values[..., p_idx]) * dcal.gains.values
    return create_uvx_antenna_cal(
        telescope=cal.telescope,
        method=f'{cal.method}+delay_fit',
        antenna_gains_arr=gains.astype('complex64'),
        antenna_flags_arr=cal.flags.antenna.values | dcal.flags.antenna.values,
        channel_flags_arr=cal.flags.channel.values | dcal.flags.channel.values,
        f=cal.gains.frequency.values * u.Hz,
        a=cal.gains.antenna.values,
        p=np.array(('X', 'Y')),
        provenance=cal.provenance | {'delay_fit': dcal.provenance['delay_fit']},
    )


def load_aa(
    fname: str, cal_path: str, ignore_static_flags: bool = False, delay_path: str | None = None
) -> ApertureArray:
    """Load a sweep file into a silent ApertureArray and apply its cal file, with delay-model phases if given."""
    logger.info(
        f'Loading {fname}, applying cal {cal_path}' + (f' with delays {delay_path}' if delay_path else '')
    )
    aa = ApertureArray(read_station(fname, ignore_static_flags))
    cal = read_cal(cal_path)
    aa.set_cal(delay_model_cal(cal, read_cal(delay_path)) if delay_path else cal)
    aa.set_task_verbosity('silent')
    return aa


def release():
    """Free jax device buffers and compiled executables before the next station."""
    gc.collect()
    jax.clear_caches()


def robust_std(x: np.ndarray) -> float:
    """Gaussian-equivalent std from the median absolute deviation, ignoring NaNs."""
    x = x[np.isfinite(x)]
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def n_samples(aa: ApertureArray) -> float:
    """Effective number of samples per visibility, eta * channel bandwidth * integration time."""
    return ETA * aa.uvx.data.frequency.attrs['channel_bandwidth'] * aa.uvx.data.time.attrs['resolution']


def zenith_disc(aa: ApertureArray, radius_deg: float, nside: int = NSIDE_SIM) -> tuple:
    """Healpix pixels within radius_deg of zenith, and their zenith angles in radians."""
    vec_zen = np.array(hp.pix2vec(nside, sky2hpix(nside, aa.coords.get_zenith())))
    pix = hp.query_disc(nside, vec_zen, radius=np.radians(radius_deg))
    za = np.arccos(np.clip(np.array(hp.pix2vec(nside, pix)).T @ vec_zen, -1, 1))
    return pix, za


def diagonal_pixel(aa: ApertureArray, vis_pix: np.ndarray, za_deg: float = DIAG_ZA_DEG) -> int:
    """Index into vis_pix of the direction za_deg from zenith at station-frame phi = 45 deg.

    phi = 45 deg is midway between the X (phi = 0) and Y (phi = 90) dipoles, at azimuth 45 deg + station rotation.
    """
    zen = aa.coords.get_zenith()
    az = 45 + float(aa.uvx.antennas.array_rotation_angle)
    altaz = AltAz(alt=(90 - za_deg) * u.deg, az=az * u.deg, obstime=zen.obstime, location=zen.location)
    return int(np.flatnonzero(vis_pix == sky2hpix(NSIDE_SIM, SkyCoord(altaz)))[0])


def to_coherency(x: np.ndarray) -> np.ndarray:
    """(npix, 4) XX, Re(XY), Im(XY), YY planes as (npix, 2, 2) Hermitian coherency matrices."""
    x = np.asarray(x, dtype='float64')
    xy = x[:, 1] + 1j * x[:, 2]
    return np.stack([np.stack([x[:, 0], xy], -1), np.stack([xy.conj(), x[:, 3]], -1)], -2)


def pol_coherency(v: np.ndarray) -> np.ndarray:
    """(..., 4) XX, XY, YX, YY visibilities as (..., 2, 2) coherency matrices."""
    return np.asarray(v).reshape(*np.shape(v)[:-1], 2, 2)


def pol_power(r: np.ndarray) -> np.ndarray:
    """Real XX and YY powers (..., 2) from (..., 2, 2) coherencies."""
    return np.real(np.diagonal(r, axis1=-2, axis2=-1))


def quad_trace(w: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """tr(W X W^H Y), batched over leading axes. For diagonal X and Y this is sum_ab |W_ab|^2 y_a x_b."""
    return np.einsum('...ab,...bc,...dc,...da->...', w, x, w.conj(), y).real


def load_jones(aa: ApertureArray, pix: np.ndarray) -> np.ndarray:
    """AEP Jones matrices (npix, 2, 2) at healpix pixels `pix`, for the current channel.

    The Y row is negated, which negates XY in J J^H to match the imager, as in load_eep_for_sim().
    """
    jones = load_aep_jones(
        aa.f[aa.idx['f']].to('MHz').value,
        NSIDE_SIM,
        AEP,
        sky_coord=aa.coords.get_zenith(),
        rot_angle=aa.uvx.antennas.array_rotation_angle,
        pix=pix,
    )
    jones[1] *= -1
    return np.moveaxis(jones, -1, 0).astype('complex128')


def stokes_weights(jones: np.ndarray) -> dict:
    """Per-pixel Stokes estimator weights M_S = J^-H P_S J^-1, so that S = tr(M_S C).

    Sutinjo, Ung & Sokolowski (2022), Eq. 18. M_I = (J J^H)^-1. SEFD_S follows from |M_S|^2 (Eq. 19).

    Args:
        jones (np.ndarray): (npix, 2, 2) Jones matrices.

    Returns:
        weights (dict): (npix, 2, 2) M_S for each of 'i', 'q', 'u', 'v'.
    """
    j_inv = np.linalg.inv(jones)
    return {s: j_inv.conj().swapaxes(-1, -2) @ p @ j_inv for s, p in P_S.items()}


def stokes_estimate(w: np.ndarray, img: np.ndarray) -> np.ndarray:
    """Estimate tr(W C) per pixel, e.g. Stokes S for W = M_S."""
    return np.einsum('nij,nji->n', w, to_coherency(img)).real


def stokes_noise(w: np.ndarray, p_auto: np.ndarray, n_samp: float) -> np.ndarray:
    """Predicted std of tr(W C) per pixel, summed over baselines as in sefd_from_power().

    Assumes noise is uncorrelated between antennas, and the imager's 1/N^2 gain.

    Args:
        w (np.ndarray): (npix, 2, 2) estimator weights, M_S for Stokes S.
        p_auto (np.ndarray): (n_ant, 2, 2) calibrated auto coherencies of unflagged antennas (Jy).
        n_samp (float): Effective number of samples per visibility.

    Returns:
        sigma (np.ndarray): (npix,) predicted image std (Jy/beam).
    """
    n = len(p_auto)
    s = p_auto.sum(axis=0)
    # sum over p != q of tr(W P_p W^H P_q) = tr(W S W^H S) - sum_p tr(W P_p W^H P_p)
    k = np.einsum('pbc,pda->bcda', p_auto, p_auto)
    var = quad_trace(w, s, s) - np.einsum('nab,ndc,bcda->n', w, w.conj(), k).real
    return np.sqrt(var / (n**4 * n_samp))


def image_noise(w: np.ndarray, p_bar: np.ndarray, s_img: np.ndarray, n: int, n_samp: float) -> np.ndarray:
    """Predicted std of tr(W C) per pixel, including sky noise correlated between antennas.

    Models the phased voltages as circular Gaussian with 2x2 covariance R_pq = P (p = q) or S (p != q), S the
    mean cross-correlation phased to that pixel, and sums over p != q as in the imager. Isserlis' theorem
    gives var = [(n-2) T(U, U) + T(P, P) + (n-1) T(S, S)] / (n^3 n_samp), with T(X, Y) = tr(W X W^H Y) and
    U = P + (n-1) S. With S = 0 this equals stokes_noise() for identical antennas.

    Args:
        w (np.ndarray): (npix, 2, 2) estimator weights, M_S for Stokes S.
        p_bar (np.ndarray): (2, 2) mean calibrated auto coherency of unflagged antennas (Jy).
        s_img (np.ndarray): (npix, 2, 2) mean cross-correlation coherency phased to each pixel (Jy).
        n (int): Number of unflagged antennas.
        n_samp (float): Effective number of samples per visibility.

    Returns:
        sigma (np.ndarray): (npix,) predicted image std (Jy/beam).
    """
    u = p_bar + (n - 1) * s_img
    var = (n - 2) * quad_trace(w, u, u) + quad_trace(w, p_bar, p_bar) + (n - 1) * quad_trace(w, s_img, s_img)
    return np.sqrt(var / (n**3 * n_samp))


def robust_std_jackknife(z: np.ndarray, cells: np.ndarray) -> tuple:
    """Robust std of z, and its jackknife error leaving out one sky cell at a time."""
    ids = np.unique(cells)
    loo = np.array([robust_std(z[cells != i]) for i in ids])
    n = len(ids)
    return robust_std(z), float(np.sqrt((n - 1) / n * np.sum((loo - loo.mean()) ** 2)))


def element_sefd(sigma: np.ndarray, n_ant: int, n_samp: float) -> np.ndarray:
    """Convert a Stokes image std to the antenna-averaged element SEFD (Jy)."""
    return sigma * n_ant**2 * np.sqrt(n_samp / (n_ant * (n_ant - 1)))


def sefd_from_power(w: np.ndarray, r: np.ndarray) -> float:
    """SEFD_S = sqrt(tr(M_S R M_S^H R)) for a 2x2 noise coherency R (Jy, Convention-1/2).

    Sutinjo, Ung & Sokolowski (2022) Eq. 19, sqrt(p^T |M_S|^2 p) with p = [P_X, P_Y], assumes noise
    uncorrelated between X and Y; this is its extension to correlated X and Y, and equals it for diagonal R.
    """
    return float(np.sqrt(quad_trace(w, r, r)))


def mean_cross_power(V: np.ndarray, good: np.ndarray) -> np.ndarray:
    """Mean 2x2 cross-correlation coherency over unflagged baselines: the sky correlated between elements."""
    v = V[np.ix_(good, good)]
    n = int(good.sum())
    return pol_coherency(v.sum(axis=(0, 1)) - np.einsum('iip->p', v)) / (n * (n - 1))


def station_power(p_el: np.ndarray, c_bar: np.ndarray, n: int = N_STATION_ANT) -> np.ndarray:
    """Beamformed noise coherency of an n-element station, normalised by n^2: P/n + (n-1)/n * C (Jy).

    C is the mean cross-correlation phased to the pointing direction. Assumes the value measured on the
    unflagged antennas holds for the full layout.
    """
    return p_el / n + (n - 1) / n * c_bar


def senscalc_sefd_i(path: str, lst_h: float, f_mhz: np.ndarray) -> np.ndarray:
    """Senscalc zenith station SEFD_I (Sokolowski et al. 2022) at the LST sample nearest lst_h, on f_mhz.

    SEFD_I = 1/2 sqrt(SEFD_X^2 + SEFD_Y^2), as in ska_ost_senscalc SEFDTable. The table is indexed
    (lst, azimuth, zenith_angle, frequency, pol); az = za = 0 is zenith.
    """
    with h5py.File(path, 'r') as h5:
        lst = h5['dimensions']['lst'][:]
        freqs = h5['dimensions']['frequency'][:]
        idx = np.argmin(np.abs((lst - lst_h + 12) % 24 - 12))
        sefd = h5['sefd'][idx, 0, 0]
    return np.interp(f_mhz, freqs, 0.5 * np.hypot(sefd[:, 0], sefd[:, 1]))


def coarse_reduce(values: np.ndarray, pix: np.ndarray, fn) -> np.ndarray:
    """Apply fn to `values` within each coarse healpix cell; NaN for empty cells."""
    cell = hp.vec2pix(NSIDE_COARSE, *hp.pix2vec(NSIDE_SIM, pix))
    out = np.full(hp.nside2npix(NSIDE_COARSE), np.nan)
    for idx in np.unique(cell):
        out[idx] = fn(values[cell == idx])
    return out


def calibrated_autos(aa: ApertureArray, V: np.ndarray) -> tuple:
    """Unflagged-antenna mask and (n_ant, 2, 2) auto coherencies; flagged antennas are zeroed by the cal."""
    diag = np.diag_indices(aa.n_ant)
    good = np.abs(V[diag][:, 0]) > 0
    assert good.sum() == n_unflagged_ant(aa)
    return good, pol_coherency(V[diag])


def run_image(aa: ApertureArray, prefix: str):
    """Write calibrated healpix cubes at t=0 and t=1, and their difference."""
    hpx = []
    for t in (0, 1):
        logger.info(f'[image] {aa.name}: healpix cube at t={t}')
        aa.set_idx(t=t)
        hpx.append(aa.imaging.make_healpix(vis='cal', include_autos=False, channel_range='all'))
        write_img(hpx[-1], f'{prefix}_hpx_t{t}.h5')
    write_img(hpx[0] - hpx[1], f'{prefix}_hpx_diff.h5')
    logger.info(f'[image] {aa.name}: wrote {prefix}_hpx_{{t0,t1,diff}}.h5')


def run_check(aa: ApertureArray, n_real: int = 10, seed: int = 42):
    """Image thermal noise drawn from the calibrated autos; S / sigma_pred should have unit std at all ZA."""
    logger.info(f'[check] {aa.name}: noise injection, {n_real} realisations')
    rng = np.random.default_rng(seed)
    good_ch = np.flatnonzero(~aa.cal.flags.channel.values)
    if not len(good_ch):
        logger.warning(f'[check] {aa.name}: every channel is flagged in the cal, skipping')
        return
    with aa.select(f=good_ch[len(good_ch) // 2]):
        n_samp = n_samples(aa)
        good, P = calibrated_autos(aa, np.asarray(aa.generate_vis_matrix(vis='cal', include_autos=True)))
        # Noise voltages v_p = L_p x with R_p = L_p L_p^H, so X and Y may be correlated
        L = np.zeros(P.shape, dtype='complex128')
        L[good] = np.linalg.cholesky(P[good])

        vis_pix, za = zenith_disc(aa, MAX_ZA_DEG)
        m = stokes_weights(load_jones(aa, vis_pix))
        sigma_pred = {s: stokes_noise(w, P[good], n_samp) for s, w in m.items()}
        iu = np.triu_indices(aa.n_ant, 1)
        diag = np.diag_indices(aa.n_ant)

        # Pixels are correlated over the PSF, so pool several realisations
        z = {s: [] for s in m}
        for _ in tqdm(range(n_real), desc='noise check'):
            # Baseline noise V_pq = L_p G L_q^H / sqrt(M), G unit-variance complex normal
            shape = (aa.n_ant, aa.n_ant, 2, 2)
            G = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2)
            V_noise = L[:, None] @ G @ L[None].conj().swapaxes(-1, -2) / np.sqrt(n_samp)
            V_noise[iu[1], iu[0]] = V_noise[iu[0], iu[1]].conj().swapaxes(-1, -2)  # Hermitian
            V_noise[diag] = 0
            V_noise = V_noise.reshape(aa.n_ant, aa.n_ant, 4)

            img = np.ma.filled(aa.imaging.make_healpix(vis=V_noise), np.nan)[vis_pix]
            for s, w in m.items():
                z[s].append(stokes_estimate(w, img) / sigma_pred[s])
    za = np.tile(za, n_real)

    for s, zs in z.items():
        zs = np.concatenate(zs)
        for za0, za1 in ((0, 20), (20, 40), (40, 60), (60, 80)):
            sel = (za >= np.radians(za0)) & (za < np.radians(za1))
            logger.info(
                f'[check] ZA {za0:2d}-{za1:2d} deg: std({s.upper()} / sigma_pred) = {robust_std(zs[sel]):.3f}'
            )


def run_sens(aa: ApertureArray, sky_model: str, prefix: str):
    """Element and station SEFD and A/T per channel, per pol and Stokes I, Q, U, V, write hkl and csv."""
    logger.info(f'[sens] {aa.name}: {len(aa.f)} channels')
    cal_flags = aa.cal.flags.channel.values.copy()
    if cal_flags.all():
        raise ValueError(f'[sens] {aa.name}: every channel is flagged in the cal; exclude it with --flag')
    df = {'xx': [], 'yy': [], 'i': []}
    rho_maps, sefd_maps, ratio_maps = [], [], []

    hpxd = read_img(f'{prefix}_hpx_diff.h5')
    assert hp.npix2nside(hpxd.shape[-2]) == NSIDE_SIM
    M = n_samples(aa)
    aa.set_idx(t=0)

    zen_pix, _ = zenith_disc(aa, FOV_DEG)
    # The sim integrates the AEP over above-horizon pixels only
    above, _ = zenith_disc(aa, 90)
    pix_area = hp.nside2pixarea(NSIDE_SIM)

    vis_pix, za = zenith_disc(aa, MAX_ZA_DEG)
    in_fov = za < np.radians(FOV_DEG)
    i_zen = np.argmin(za)
    i_diag = diagonal_pixel(aa, vis_pix)
    jk_cells = hp.vec2pix(NSIDE_JK, *hp.pix2vec(NSIDE_SIM, vis_pix[in_fov]))
    # The dirty sky images give the cross-correlation phased to each pixel
    h5_t = [h5py.File(f'{prefix}_hpx_t{t}.h5', 'r') for t in (0, 1)]
    # The model sky includes the Sun at the flux suncal fitted, so T_ant includes it
    sun = aa.cal.provenance.get('suncal')

    for ii in tqdm(range(aa.uvx.data.shape[1]), desc=prefix):
        if cal_flags[ii]:
            # Gains are zeroed in flagged channels; empty rows become NaN in the DataFrames
            for rows in df.values():
                rows.append({})
            for maps in (rho_maps, sefd_maps, ratio_maps):
                maps.append(None)
            continue
        with aa.select(f=ii):
            f_hz = aa.f[aa.idx['f']].to('Hz').value
            lam = c.value / f_hz

            V_cal = np.asarray(aa.generate_vis_matrix(vis='cal', include_autos=True))
            sun_src = None
            if sun is not None and np.isfinite(sun['sun_flux_jy'][ii]):
                sun_src = {
                    'Sun': RadioSource(
                        aa.coords.get_sun(),
                        mag=float(sun['sun_flux_jy'][ii]),
                        fwhm=float(sun['sun_fwhm_deg'][ii]),
                    )
                }
            V_sky = aa.simulation.sim_vis_gsm(
                sky_model=sky_model, as_xarray=False, units='Jysr', extended_sources=sun_src
            )
            aep = np.asarray(load_aep_for_sim(aa, NSIDE_SIM, AEP))
            m = stokes_weights(load_jones(aa, vis_pix))

            good, p_cal = calibrated_autos(aa, V_cal)
            p_auto = p_cal[good]
            p_sky = pol_coherency(np.asarray(V_sky)[np.diag_indices(aa.n_ant)][good])
            n_ant = int(good.sum())
            diff_img = hpxd[0, ii]

            b_zen = aep[vis_pix[i_zen], [0, 3]]

            # Noise coherency (Jy): one element, and the zenith-beamformed 256-element station
            p_el = p_auto.mean(axis=0)
            c_bar = mean_cross_power(V_cal, good)
            p_st = station_power(p_el, c_bar)
            pw_auto, pw_sky, pw_el, pw_st = (pol_power(x) for x in (p_auto, p_sky, p_el, p_st))

            for pol_name, p, pp in (('xx', 0, 0), ('yy', 3, 1)):
                # Per-pol element solid angle; k is the Jy-per-K auto response for I = XX + YY
                omega_e = aep[above, p].sum() * pix_area
                k = k_to_jysr(1.0, f_hz) / 2 * omega_e

                T_ant = pw_sky[:, pp].mean() / k
                T_sys_auto = pw_auto[:, pp].mean() / k

                # / sqrt(2): the t0 - t1 difference doubles the noise variance
                sigma = robust_std(np.asarray(diff_img[zen_pix, p].filled(np.nan))) / np.sqrt(2)
                sefd_el = 2 * pw_el[pp] / b_zen[pp]
                sefd_st = 2 * pw_st[pp] / b_zen[pp]

                df[pol_name].append(
                    {
                        'T_ant': float(T_ant),
                        'T_sys_auto': float(T_sys_auto),
                        'T_rx': float(T_sys_auto - T_ant),
                        'sigma': float(sigma),
                        # XX image is on an I = XX + YY scale (1 Jy -> 0.5 Jy/beam), so double sigma to
                        # put the SEFD on the Stokes-I flux scale of Macario Eqs. 2-3
                        'sefd_el_img': float(2 * sigma * np.sqrt(M) * n_ant),
                        'sefd_el': float(sefd_el),
                        'sefd_st': float(sefd_st),
                        'a_over_t_el': float(TWO_K_JY / sefd_el),
                        'a_over_t_st': float(TWO_K_JY / sefd_st),
                        'c_bar': float(c_bar[pp, pp].real),
                        'omega_e': float(omega_e),
                        # Cross-check against a_over_t_el * T_sys_auto
                        'a_eff_el': float(lam**2 / omega_e),
                    }
                )

            # t0 - t1 doubles the noise variance
            d = np.ma.filled(diff_img, np.nan)[vis_pix]
            sky = 0.5 * (h5_t[0]['img/data'][0, ii][vis_pix] + h5_t[1]['img/data'][0, ii][vis_pix])
            s_img = to_coherency(sky) * n_ant / (n_ant - 1)
            row = {}
            for st, w in m.items():
                z = stokes_estimate(w, d) / np.sqrt(2) / image_noise(w, p_el, s_img, n_ant, M)
                row[f'rho_{st}'], row[f'rho_{st}_err'] = robust_std_jackknife(z[in_fov], jk_cells)
                if st == 'i':
                    rho_maps.append(coarse_reduce(z, vis_pix, robust_std))
            sigma_pred = stokes_noise(m['i'], p_auto, M)
            row['rho_i_unc'] = robust_std((stokes_estimate(m['i'], d) / np.sqrt(2) / sigma_pred)[in_fov])

            # I and V noise estimates are nearly independent; half their difference is taken as a systematic
            rho_i, rho_v = row['rho_i'], row['rho_v']
            rho = (rho_i + rho_v) / 2
            rel_err = np.hypot(np.hypot(row['rho_i_err'], row['rho_v_err']) / 2, abs(rho_i - rho_v) / 2) / rho

            sefd_pred = element_sefd(sigma_pred, n_ant, M)
            sefd_sky = element_sefd(stokes_noise(m['i'], p_sky, M), n_ant, M)
            sefd_st_pred = sefd_from_power(m['i'][i_zen], p_st)
            sefd_i_el = rho * sefd_pred[i_zen]
            sefd_i_st = rho * sefd_st_pred
            sefd_maps.append(coarse_reduce(sefd_pred, vis_pix, np.nanmedian))

            # A/T_I = sqrt(2) k / SEFD_I, equal to the per-pol A/T for balanced pols
            a_el = TWO_K_JY / np.sqrt(2) / sefd_i_el
            a_st = TWO_K_JY / np.sqrt(2) / sefd_i_st
            row.update(
                {
                    'rho': float(rho),
                    'rel_err': float(rel_err),
                    'sefd_i_el_pred': float(sefd_pred[i_zen]),
                    'sefd_i_el_sky': float(sefd_sky[i_zen]),
                    'sefd_i_el': float(sefd_i_el),
                    'sefd_i_el_err': float(rel_err * sefd_i_el),
                    'a_over_t_i_el': float(a_el),
                    'a_over_t_i_el_err': float(rel_err * a_el),
                    'sefd_i_st_pred': float(sefd_st_pred),
                    'sefd_i_st': float(sefd_i_st),
                    'sefd_i_st_err': float(rel_err * sefd_i_st),
                    'a_over_t_i_st': float(a_st),
                    'a_over_t_i_st_err': float(rel_err * a_st),
                }
            )
            # Station beamformed towards the diagonal: the sky correlated between elements is the mean
            # cross-correlation phased to that pixel
            p_st_diag = station_power(p_el, s_img[i_diag])
            ratios = []
            for st in ('i', *QUV):
                pred = element_sefd(stokes_noise(m[st], p_auto, M), n_ant, M)
                if st != 'i':
                    ratios.append(coarse_reduce(pred / sefd_pred, vis_pix, np.nanmedian))
                    row[f'sefd_{st}_el'] = float(rho * pred[i_zen])
                    row[f'sefd_{st}_st'] = float(rho * sefd_from_power(m[st][i_zen], p_st))
                row[f'sefd_{st}_el_diag'] = float(rho * pred[i_diag])
                row[f'sefd_{st}_st_diag'] = float(rho * sefd_from_power(m[st][i_diag], p_st_diag))
            ratio_maps.append(ratios)
            df['i'].append({k: float(v) for k, v in row.items()})

    for h5 in h5_t:
        h5.close()
    out = {key: pd.DataFrame(rows) for key, rows in df.items()}
    for key, d in out.items():
        d.to_csv(f'{prefix}_sensitivity_{key}.csv')
    zen = aa.coords.get_zenith().galactic
    lsts = aa.t.sidereal_time('apparent', longitude=aa.uvx.origin.lon)
    out.update(
        rho_map=stack_nan(rho_maps),
        sefd_i_el_map=stack_nan(sefd_maps),
        # (n_chan, 3, npix): predicted element SEFD_Q, U, V over SEFD_I
        sefd_ratio_map=stack_nan(ratio_maps),
        f_mhz=aa.f.to('MHz').value,
        zen_lb=(zen.l.deg, zen.b.deg),
        obs_utc=[t.strftime('%Y-%m-%d %H:%M:%S') for t in aa.t],
        obs_lst=[str(lst.to_string(unit='hourangle', sep=':', precision=0, pad=True)) for lst in lsts],
        obs_lst_h=float(lsts[0].hour),
        time_flags=bad_channel_times(aa.uvx),
        cal_flags=cal_flags,
        # suncal channels where the GSM wasn't detected have no flux scale (older cals lack 'flux_scale')
        flux_flags=(
            ~np.asarray(sun.get('flux_scale', np.isfinite(sun['sun_flux_jy'])), dtype=bool)
            if sun is not None
            else np.zeros(len(aa.f), dtype=bool)
        ),
    )
    hkl.dump(out, f'{prefix}_sensitivity.hkl')
    i = out['i']
    logger.info(
        f'[sens] {aa.name}: median rho_I = {np.nanmedian(i["rho_i"]):.3f}, rho_V = {np.nanmedian(i["rho_v"]):.3f}, '
        f'median element SEFD_I = {np.nanmedian(i["sefd_i_el"]):.0f} Jy, '
        f'median station SEFD_I = {np.nanmedian(i["sefd_i_st"]):.0f} Jy'
    )
    logger.info(f'[sens] {aa.name}: wrote {prefix}_sensitivity.hkl and _sensitivity_{{xx,yy,i}}.csv')


def stack_nan(maps: list) -> np.ndarray:
    """Stack per-channel arrays, filling skipped (None) channels with NaN."""
    shape = np.shape(next(m for m in maps if m is not None))
    return np.array([np.full(shape, np.nan) if m is None else m for m in maps])


def band_flag(f_mhz: np.ndarray, d) -> np.ma.MaskedArray:
    """Mask channels inside the RFI_BANDS_MHZ ranges, and non-finite (spike-flagged) values."""
    d = np.asarray(d, dtype='float64')
    mask = ~np.isfinite(d)
    for f0, f1 in RFI_BANDS_MHZ:
        mask |= (f_mhz > f0) & (f_mhz < f1)
    return np.ma.array(d, mask=mask)


def spike_flags(y: np.ndarray, thr: float, window: int = SPIKE_WINDOW) -> np.ndarray:
    """Flag channels more than thr robust sigma above a running median, in log space.

    Log space keeps the threshold uniform across the band, where SEFD changes by an order of magnitude.
    """
    log_y = np.log10(np.asarray(y, dtype='float64'))
    resid = log_y - pd.Series(log_y).rolling(window, center=True, min_periods=1).median().to_numpy()
    return resid > thr * robust_std(resid)


def load_sensitivity(outdir: str, stations: list, spike_thr: float, senscalc_table: str | None = None) -> dict:
    """Load each station's sensitivity hkl, blanking flagged channels to NaN in xx/yy/i.

    Channels are flagged in the cal, for a bad timestamp, for having no flux scale (suncal found no
    GSM), and for spikes in sefd_i_el, found after the others are blanked.

    If `senscalc_table` is given, adds the senscalc zenith station SEFD_I at the observation LST
    as d['senscalc_sefd_i'], and the ratio d['i']['st_over_senscalc'].
    """
    dfs = {}
    for s in stations:
        d = hkl.load(os.path.join(outdir, f'{s}_sensitivity.hkl'))
        if 'sefd_ratio_map' not in d:
            raise ValueError(f'{s}: hkl predates the Stokes I, Q, U, V columns; rerun --steps sens')
        # Older hkl files have no cal_flags
        d['cal_flags'] = np.asarray(d.get('cal_flags', np.zeros(len(d['f_mhz']), dtype=bool)), dtype=bool)
        for key in ('xx', 'yy', 'i'):
            d[key].loc[d['time_flags'] | d['flux_flags'] | d['cal_flags']] = np.nan
        flags = spike_flags(d['i']['sefd_i_el'], spike_thr)
        for key in ('xx', 'yy', 'i'):
            d[key].loc[flags] = np.nan
        d['spike_flags'] = flags
        if senscalc_table:
            d['senscalc_sefd_i'] = senscalc_sefd_i(senscalc_table, d['obs_lst_h'], d['f_mhz'])
            d['i']['st_over_senscalc'] = d['i']['sefd_i_st'] / d['senscalc_sefd_i']
        d['i']['rel_err_pct'] = 100 * d['i']['rel_err']
        logger.info(
            f'{s}: spike-flagged {flags.sum()} channels at {np.round(d["f_mhz"][flags], 1).tolist()} MHz'
        )
        dfs[s] = d
    return dfs


def robust_top(arrays: list) -> float | None:
    """Upper y-limit that ignores isolated spikes: the 98th percentile plus the 2-98% range."""
    v = np.concatenate([np.ma.filled(np.ma.asarray(a, dtype='float64'), np.nan).ravel() for a in arrays])
    v = v[np.isfinite(v)]
    if not v.size:
        return None
    lo, hi = np.percentile(v, [2, 98])
    return hi + (hi - lo)


def plot_vs_freq(
    path: str,
    ylabel: str,
    lines: list,
    frange: tuple | None = None,
    ymin: float | None = None,
    ymax: float | None = None,
):
    """Plot band-flagged lines vs frequency and save to `path`.

    Args:
        path (str): Output PNG path.
        ylabel (str): Y-axis label.
        lines (list): (f_mhz, y, label, plot_kwargs) tuples.
        frange (tuple): (f0, f1) x-limits in MHz (default: full band).
        ymin (float): Floor on the lower y-limit; the axis starts at max(ymin, 0.95 * data min in frange).
        ymax (float): Cap on the upper y-limit; the axis tops out at min(ymax, 1.05 * data max in frange).
            The top is also capped at `robust_top()`, so isolated spikes don't set the scale.
    """
    plt.figure(figsize=(8, 5))
    bottom, top, shown = np.inf, -np.inf, []
    for f, y, label, kw in lines:
        y = band_flag(f, y)
        plt.plot(f, y, label=label, **kw)
        sel = np.ones_like(f, dtype=bool) if frange is None else (f >= frange[0]) & (f <= frange[1])
        if y[sel].count():
            bottom, top = min(bottom, np.ma.min(y[sel])), max(top, np.ma.max(y[sel]))
        shown.append(y[sel])
    if frange is not None:
        plt.xlim(*frange)
    if ymin is not None and np.isfinite(bottom):
        plt.ylim(bottom=max(ymin, 0.95 * bottom))
    tops = [t for t in (ymax, robust_top(shown)) if t is not None]
    if tops and np.isfinite(top):
        plt.ylim(top=min(1.05 * top, *tops))
    plt.legend()
    plt.xlabel('Frequency [MHz]')
    plt.ylabel(ylabel)
    plt.savefig(path)
    plt.close()


def run_plot(
    stations: list,
    outdir: str,
    chan: int,
    zoom_ylim: tuple,
    spike_thr: float,
    senscalc_table: str | None = None,
):
    """Per-station, all-station and array sensitivity plots, written to `{outdir}/plots/`."""
    logger.info(f'[plot] {len(stations)} stations: {stations}')
    dfs = load_sensitivity(outdir, stations, spike_thr, senscalc_table)
    pdir = os.path.join(outdir, 'plots')
    os.makedirs(pdir, exist_ok=True)
    d0 = next(iter(dfs.values()))
    sc_label = f'senscalc, LST {d0["obs_lst_h"]:.1f} h' if senscalc_table else None

    def path(name):
        return os.path.join(pdir, f'{name}.png')

    def senscalc_line(scale=1.0):
        if not senscalc_table:
            return []
        return [(d0['f_mhz'], scale * d0['senscalc_sefd_i'], sc_label, {'color': 'k', 'ls': '--'})]

    for s, d in tqdm(dfs.items(), desc='station plots'):
        f = d['f_mhz']
        for col in POL_COLS:
            lines = [(f, d[pol][col], pol.upper(), {}) for pol in ('xx', 'yy')]
            plot_vs_freq(path(f'{s}_pol_{col}'), f'{s} {UNITS[col]}', lines)
        for col in I_COLS:
            plot_vs_freq(path(f'{s}_i_{col}'), f'{s} {UNITS[col]}', [(f, d['i'][col], 'I', {})])
        plot_vs_freq(
            path(f'{s}_i_sefd_i_st_zoom'),
            f'{s} {UNITS["sefd_i_st"]}',
            [(f, d['i']['sefd_i_st'], 'I', {})] + senscalc_line(),
            frange=ZOOM_STATION_MHZ,
            ymin=zoom_ylim[0],
            ymax=zoom_ylim[1],
        )

        fig = plt.figure(figsize=(10, 5))
        rot = tuple(d['zen_lb'])
        hp.orthview(
            d['rho_map'][chan],
            rot=rot,
            half_sky=True,
            sub=121,
            fig=fig.number,
            min=0.7,
            max=1.3,
            cmap='RdBu_r',
            title=f'{s} $\\rho$, {f[chan]:.1f} MHz',
        )
        hp.orthview(
            d['sefd_i_el_map'][chan],
            rot=rot,
            half_sky=True,
            sub=122,
            fig=fig.number,
            title='Predicted element SEFD$_I$ [Jy]',
        )
        plt.savefig(path(f'{s}_maps_ch{chan}'))
        plt.close(fig)

        for kind, label in (('el', 'Element'), ('st', f'Station {_ST}')):
            # Solid: zenith. Dashed: DIAG_ZA_DEG from zenith on the X-Y diagonal.
            lines = []
            for n, st in enumerate(('i', *QUV)):
                lines += [
                    (f, d['i'][f'sefd_{st}_{kind}'], f'{st.upper()} zenith', {'color': f'C{n}'}),
                    (
                        f,
                        d['i'][f'sefd_{st}_{kind}_diag'],
                        f'{st.upper()} ZA {DIAG_ZA_DEG}°',
                        {'color': f'C{n}', 'ls': '--'},
                    ),
                ]
            plot_vs_freq(path(f'{s}_sefd_stokes_{kind}'), f'{s} {label} SEFD$_S$ [Jy]', lines)

        fig = plt.figure(figsize=(15, 5))
        for k, st in enumerate(QUV):
            hp.orthview(
                d['sefd_ratio_map'][chan, k],
                rot=rot,
                half_sky=True,
                sub=131 + k,
                fig=fig.number,
                title=f'{s} SEFD$_{st.upper()}$ / SEFD$_I$, {f[chan]:.1f} MHz',
            )
        plt.savefig(path(f'{s}_sefd_ratio_ch{chan}'))
        plt.close(fig)

    def per_pol(col):
        return [
            (d['f_mhz'], d[pol][col], f'{s} {pol.upper()}', {}) for s, d in dfs.items() for pol in ('xx', 'yy')
        ]

    plot_vs_freq(path('all_sefd_el_img'), UNITS['sefd_el_img'], per_pol('sefd_el_img'))

    # Element. Solid: measured. Dashed: radiometer prediction from calibrated autos. Dotted: sky-noise limit.
    lines = []
    for n, (s, d) in enumerate(dfs.items()):
        color = f'C{n}'
        f, i = d['f_mhz'], d['i']
        lines += [
            (f, i['sefd_i_el'], s, {'color': color}),
            (f, i['sefd_i_el_pred'], None, {'color': color, 'ls': '--'}),
            (f, i['sefd_i_el_sky'], None, {'color': color, 'ls': ':'}),
        ]
    plot_vs_freq(path('all_sefd_i_el'), UNITS['sefd_i_el'], lines)

    # Station. Solid: beamformed. Dotted: element / 256, i.e. no sky noise correlated between elements.
    lines = []
    for n, (s, d) in enumerate(dfs.items()):
        color = f'C{n}'
        f, i = d['f_mhz'], d['i']
        lines += [
            (f, i['sefd_i_st'], s, {'color': color}),
            (
                f,
                i['sefd_i_el'] / N_STATION_ANT,
                '256 × uncorrelated elements' if n == 0 else None,
                {'color': color, 'ls': ':', 'lw': 1},
            ),
        ]
    lines += senscalc_line()
    plot_vs_freq(path('all_sefd_i_st'), UNITS['sefd_i_st'], lines)
    plot_vs_freq(
        path('all_sefd_i_st_zoom'),
        UNITS['sefd_i_st'],
        lines,
        frange=ZOOM_ALL_MHZ,
        ymin=zoom_ylim[0],
        ymax=zoom_ylim[1],
    )

    aot_lines = [(d['f_mhz'], d['i']['a_over_t_i_st'], s, {}) for s, d in dfs.items()]
    if senscalc_table:
        aot_lines.append(
            (d0['f_mhz'], TWO_K_JY / np.sqrt(2) / d0['senscalc_sefd_i'], sc_label, {'color': 'k', 'ls': '--'})
        )
    plot_vs_freq(path('all_a_over_t_i_st'), UNITS['a_over_t_i_st'], aot_lines)
    # Solid: I. Dashed: V. Dash-dot: Q. Thin solid: U. Dotted: I against the uncorrelated-noise prediction.
    lines = []
    for n, (s, d) in enumerate(dfs.items()):
        color = f'C{n}'
        f, i = d['f_mhz'], d['i']
        lines += [
            (f, i['rho_i'], f'{s} I', {'color': color}),
            (f, i['rho_q'], f'{s} Q', {'color': color, 'ls': '-.', 'lw': 1}),
            (f, i['rho_u'], f'{s} U', {'color': color, 'lw': 0.5}),
            (f, i['rho_v'], f'{s} V', {'color': color, 'ls': '--'}),
            (f, i['rho_i_unc'], None, {'color': color, 'ls': ':', 'lw': 1}),
        ]
    plot_vs_freq(path('all_rho'), r'$\rho$ = measured / predicted noise', lines)

    # Array sensitivity from the station mean, assuming identical 256-antenna beamformed stations
    f = d0['f_mhz']

    def station_mean(col):
        # A spike-flagged (NaN) channel drops out of the mean; NaN only if flagged in every station
        return band_flag(
            f, np.ma.masked_invalid([d['i'][col] for d in dfs.values()]).mean(axis=0).filled(np.nan)
        )

    aot = station_mean('a_over_t_i_st')
    sefd = station_mean('sefd_i_st')
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 5))
    for name, n in ARRAYS.items():
        ax0.plot(f, aot * n, label=f'{name} ({n})')
        ax1.plot(f, sefd / n, label=f'{name} ({n})')
    ax0.set_ylabel('Array A/T$_I$ [m$^2$ K$^{-1}$]')
    ax1.set_ylabel('Array SEFD$_I$ [Jy]')
    for ax, arrays in ((ax0, [aot * n for n in ARRAYS.values()]), (ax1, [sefd / n for n in ARRAYS.values()])):
        top = robust_top(arrays)
        if top is not None:
            ax.set_ylim(top=min(ax.get_ylim()[1], top))
        ax.set_xlabel('Frequency [MHz]')
        ax.legend()
    fig.suptitle(
        f'Mean of {len(dfs)} beamformed stations (zenith, LST {d0["obs_lst_h"]:.1f} h): {", ".join(dfs)}'
    )
    fig.savefig(path('array_sensitivity'))
    plt.close(fig)
    logger.info(f'[plot] wrote {len(glob.glob(os.path.join(pdir, "*.png")))} PNGs to {pdir}')


def run_html(flist: list, stations: list, outdir: str, args: argparse.Namespace, version: str):
    """Write `{outdir}/index.html` linking the plots, a per-station summary table and the data files."""
    dfs = load_sensitivity(outdir, stations, args.spike_thr, args.senscalc_table)
    d0 = next(iter(dfs.values()))
    f = d0['f_mhz']

    obs_table = table(
        ['Station', 'Input file', 'UTC', 'LST (apparent)'],
        [
            (s, os.path.basename(fn), '<br>'.join(d['obs_utc']), '<br>'.join(d['obs_lst']))
            for fn, (s, d) in zip(flist, dfs.items(), strict=True)
        ],
    )
    settings_table = args_table(vars(args))
    rfi_table = table(['Start [MHz]', 'Stop [MHz]'], RFI_BANDS_MHZ)

    def flag_table(key):
        return table(
            ['Station', 'Channels', 'Frequencies [MHz]'],
            [(s, d[key].sum(), ', '.join(f'{x:.1f}' for x in f[d[key]])) for s, d in dfs.items()],
        )

    time_table, flux_table, cal_table, spike_table = (
        flag_table(k) for k in ('time_flags', 'flux_flags', 'cal_flags', 'spike_flags')
    )

    # Summary table: band-flagged medians within ZOOM_ALL_MHZ
    cols = {
        **{f'rho<sub>{st.upper()}</sub>': ('i', f'rho_{st}', '.3f') for st in ('i', *QUV)},
        'Element SEFD<sub>I</sub> [Jy]': ('i', 'sefd_i_el', '.0f'),
        **{f'Station SEFD<sub>{st.upper()}</sub> [Jy]': ('i', f'sefd_{st}_st', '.0f') for st in ('i', *QUV)},
        **{
            f'Station SEFD<sub>{st.upper()}</sub>, ZA {DIAG_ZA_DEG}° diagonal [Jy]': (
                'i',
                f'sefd_{st}_st_diag',
                '.0f',
            )
            for st in ('i', *QUV)
        },
        'Station A/T<sub>I</sub> [m²/K]': ('i', 'a_over_t_i_st', '.2f'),
        'Error per channel [%]': ('i', 'rel_err_pct', '.1f'),
        'T<sub>rx</sub> XX [K]': ('xx', 'T_rx', '.0f'),
        'T<sub>rx</sub> YY [K]': ('yy', 'T_rx', '.0f'),
    }
    if args.senscalc_table:
        cols['Station / senscalc'] = ('i', 'st_over_senscalc', '.2f')
    sel = (f >= ZOOM_ALL_MHZ[0]) & (f <= ZOOM_ALL_MHZ[1])
    summary_table = table(
        ['Station', *cols],
        [
            (s, *(f'{np.ma.median(band_flag(f, d[g][c])[sel]):{fmt}}' for g, c, fmt in cols.values()))
            for s, d in dfs.items()
        ],
    )

    # Tabs: (id, label, html)
    maps = {
        s: sorted(
            os.path.basename(m)[:-4]
            for pattern in (f'{s}_maps_ch*.png', f'{s}_sefd_ratio_ch*.png')
            for m in glob.glob(os.path.join(outdir, 'plots', pattern))
        )
        for s in stations
    }
    tabs = [
        ('array', 'AA projections', figs(['array_sensitivity'])),
        (
            'all',
            'All stations',
            f'<div class="card"><p class="subtitle">Medians over {ZOOM_ALL_MHZ[0]}&ndash;{ZOOM_ALL_MHZ[1]} MHz, '
            f'excluding RFI-flagged bands.</p>{summary_table}</div>'
            + figs(
                [
                    'all_sefd_i_st',
                    'all_sefd_i_st_zoom',
                    'all_a_over_t_i_st',
                    'all_sefd_i_el',
                    'all_rho',
                    'all_sefd_el_img',
                ]
            ),
        ),
    ]
    for s in stations:
        body = (
            '<h2>Stokes I</h2>'
            + figs([f'{s}_i_{c}' for c in I_COLS] + [f'{s}_i_sefd_i_st_zoom'])
            + '<h2>Stokes I, Q, U, V</h2>'
            + figs([f'{s}_sefd_stokes_el', f'{s}_sefd_stokes_st'])
            + '<h2>XX and YY</h2>'
            + figs([f'{s}_pol_{c}' for c in POL_COLS])
            + '<h2>Maps</h2>'
            + figs(maps[s])
        )
        tabs.append((s, s, body))
    data_files = sorted(
        {
            os.path.basename(fn)
            for pattern in ('*_sensitivity*', '*_hpx_*.h5', '*.log')
            for fn in glob.glob(os.path.join(outdir, pattern))
        }
    )
    links = ''.join(f'<li><a href="{fn}">{fn}</a></li>' for fn in data_files)
    tabs.append(
        (
            'data',
            'Data files',
            f'<div class="card"><h2>Observations</h2>{obs_table}<h2>Output files</h2><ul>{links}</ul></div>',
        )
    )
    meta_table = metadata_table(stations, flist, [read_uvx(fn) for fn in flist])
    tabs.append(('metadata', 'Metadata', f'<div class="card"><h2>UVX file metadata</h2>{meta_table}</div>'))
    tabs.append(
        (
            'settings',
            'Settings',
            f'<div class="card"><h2>Runtime arguments</h2>{settings_table}'
            f'<h2>RFI-flagged bands</h2>{rfi_table}'
            f'<h2>Spike-flagged channels (&gt;{args.spike_thr:g}&sigma; above running median of SEFD<sub>I</sub>)</h2>'
            f'{spike_table}'
            f'<h2>Channels flagged in the cal ({args.caldir}'
            f'{f", delay model {args.delay_caldir}" if args.delay_caldir else ""})</h2>{cal_table}'
            f'<h2>Channels with bad timestamps (off the sweep time step by &gt;{CHANNEL_TIME_TOL_S} s)</h2>'
            f'{time_table}'
            f'<h2>Channels without a flux scale (suncal did not detect the GSM, e.g. during solar bursts)</h2>'
            f'{flux_table}</div>',
        )
    )

    index_fn = os.path.join(outdir, 'index.html')
    info = [
        ('Stations', f'{len(stations)}: {", ".join(stations)}'),
        ('Frequency', f'{f.min():.1f}&ndash;{f.max():.1f} MHz ({len(f)} channels)'),
        ('UTC start', d0['obs_utc'][0]),
        ('LST start', d0['obs_lst'][0]),
    ]
    write_report(index_fn, 'Station Sensitivity Report', info, tabs, version)
    logger.info(f'[html] wrote {index_fn}')


def process_station(fname: str, key: str, args: argparse.Namespace, first: bool):
    """Run the per-station steps (image, check, sens) on one sweep file.

    Args:
        fname (str): Sweep file.
        key (str): Station name, for the cal file and outputs.
        args (argparse.Namespace): Parsed command-line arguments.
        first (bool): Run the check step, if requested, on this station.
    """
    prefix = os.path.join(args.outdir, key)
    if {'image', 'sens'} & set(args.steps) or ('check' in args.steps and first):
        delay = os.path.join(args.delay_caldir, f'{key}_delay.cal') if args.delay_caldir else None
        aa = load_aa(fname, os.path.join(args.caldir, f'{key}.cal'), args.ignore_static_flags, delay)
        if 'image' in args.steps:
            run_image(aa, prefix)
        if 'check' in args.steps and first:
            run_check(aa)
        if 'sens' in args.steps:
            run_sens(aa, args.sky_model, prefix)
        del aa
        release()


def build_parser() -> argparse.ArgumentParser:
    """Command-line argument parser."""
    p = base_parser(__doc__, outdir='sensitivity')
    p.add_argument('--caldir', required=True, help='Calibration run folder of {station}.cal files')
    p.add_argument(
        '--delay-caldir',
        default=None,
        help='Delay fit run folder of {station}_delay.cal files: use its delay-model phases with the gain '
        'amplitudes of --caldir',
    )
    p.add_argument('--steps', nargs='+', choices=STEPS, default=STEPS, help='Steps to run, in order')
    p.add_argument(
        '--flag',
        nargs='*',
        default=[],
        help='Stations to exclude, e.g. --flag s8-1 s9-2',
    )
    p.add_argument('--n-jobs', type=int, default=4, help='Stations processed in parallel (~3 GB RAM each)')
    p.add_argument(
        '--zoom-ymin', type=float, default=500, help='Lower y-limit floor for zoomed SEFD_I plots (Jy)'
    )
    p.add_argument(
        '--zoom-ymax', type=float, default=4500, help='Upper y-limit cap for zoomed SEFD_I plots (Jy)'
    )
    p.add_argument(
        '--spike-thr', type=float, default=5, help='Flag SEFD_I spikes above this many robust sigma (plot/html)'
    )
    p.add_argument('--plot-chan', type=int, default=150, help='Channel index for the rho / SEFD_I maps')
    p.add_argument(
        '--senscalc-table',
        default=None,
        help='ska_ost_senscalc ska_station_sensitivity_AAVS2.h5, to overlay senscalc at the observation LST',
    )
    return p


def main():
    """CLI entry point: parse args and run the selected steps."""
    p = build_parser()
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    log_path = os.path.join(args.outdir, 'station_sensitivity.log')
    logger.add(AppendFileSink(log_path), level='INFO')
    version = show_version()
    logger.info(version)
    logger.info(f'Arguments: {vars(args)}')
    logger.info(f'Logging to {log_path}')

    flist = find_station_files(p, args)
    uvxs = [read_uvx(f) for f in flist]
    keys = [uvx.name for uvx in uvxs]
    single = [k for k, uvx in zip(keys, uvxs, strict=True) if uvx.data.sizes['time'] < 2]
    if single:
        raise ValueError(f'Difference images need two integrations per channel; {single} have one')
    unknown = set(args.flag) - set(keys)
    if unknown:
        raise ValueError(f'Flagged stations not in file list: {sorted(unknown)}')
    flist = [f for f, k in zip(flist, keys, strict=True) if k not in args.flag]
    stations = [k for k in keys if k not in args.flag]
    if not stations:
        raise ValueError(f'Every station is excluded by --flag: {args.flag}')
    steps = set(args.steps)

    def require(folder: str, suffixes: tuple, hint: str):
        missing = [
            s for s in stations if not all(os.path.exists(os.path.join(folder, s + x)) for x in suffixes)
        ]
        if missing:
            raise FileNotFoundError(f'No {"/".join(suffixes)} in {folder} for {missing}; {hint}')

    per_station = bool({'image', 'check', 'sens'} & steps)
    if per_station:
        require(args.caldir, ('.cal',), 'run run_stefcal.py first')
        if args.delay_caldir:
            require(args.delay_caldir, ('_delay.cal',), 'run delay_fit_report.py first')
    if 'sens' in steps and 'image' not in steps:
        require(args.outdir, ('_hpx_t0.h5', '_hpx_t1.h5', '_hpx_diff.h5'), 'add the image step')
    if {'plot', 'html'} & steps and 'sens' not in steps:
        require(args.outdir, ('_sensitivity.hkl',), 'add the sens step')
    logger.info(f'Using {len(flist)} stations: {stations}')

    if per_station:
        # Each station is independent; a worker holds one station in memory (~3 GB peak)
        tasks = [
            task(process_station)(f, s, args, i == 0)
            for i, (f, s) in enumerate(zip(flist, stations, strict=True))
        ]
        run_in_parallel(tasks, n_workers=args.n_jobs, verbose=True)

    if 'plot' in args.steps:
        run_plot(
            stations,
            args.outdir,
            args.plot_chan,
            (args.zoom_ymin, args.zoom_ymax),
            args.spike_thr,
            args.senscalc_table,
        )
    if 'html' in args.steps:
        run_html(flist, stations, args.outdir, args, version)
    logger.info('Done')


if __name__ == '__main__':
    main()
