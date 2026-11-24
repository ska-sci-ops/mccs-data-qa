"""mccs_cal.py: Calibrate a station channel with the MCCS station calibration solver (ska-low-mccs-calibration).

The `--cal-model` choices mccs-gsm, mccs-sun and mccs-sunfast are the GSM, SUN and SUN_FAST calibration types of
the SFT 06.Calibration notebook, with its solver settings. SUN and GSM model the visibilities with the MCCS
embedded element patterns (EEPs), chosen with `--mccs-eep`: `regular` is the solver's npz EEPs, set with
$EEP_PATH, $EEP_FILE_NAME and $EEP_SUFFIX as in the SFT notebooks; `compressed-<name>` is `<name>.h5` (healpy alm,
the ska-ost-low-uv EEP format) in $POSTX_EEP_DIR, e.g. compressed-harp for ska_low_harp.h5. `--beam` doesn't apply.
"""

import os
import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from ska_ost_low_uv.datamodel.cal import UVXAntennaCal, create_uvx_antenna_cal
from ska_ost_low_uv.io.mccs_cal import POL4, QA_KEYS
from ska_ost_low_uv.postx.aeps.eep_models import EEP_DIR_ENV
from ska_ost_low_uv.utils import get_resource_path

with warnings.catch_warnings():
    # ska_sdp_datamodels.visibility raises a FutureWarning from xarray
    warnings.simplefilter('ignore', category=FutureWarning)
    from ska_low_mccs_calibration.calibration import calibrate_mccs_visibility
    from ska_low_mccs_calibration.utils import sdp_visibility_datamodel

# --cal-model: calibrate_mccs_visibility() options of the SFT CALIBRATION_TYPE
MCCS_CAL = {
    'mccs-gsm': {'skymodel': 'gsm', 'min_uv': 0},
    'mccs-sun': {'skymodel': 'sun', 'min_uv': 15},
    'mccs-sunfast': {'skymodel': 'sun', 'min_uv': 15, 'ignore_eeps': True},
}
# SFT solver settings, for every calibration type
SFT_KWARGS = {'refant': 1, 'niter': 10000, 'jones_niter': 500, 'gain_threshold': 0.25, 'adjust_solar_model': True}
EEP_DEFAULTS = {'EEP_FILE_NAME': 'HARP_SKALA41_randvogel_1m_256_elem_50ohm_', 'EEP_SUFFIX': '.npz'}


def _mccs_eeps() -> dict:
    """--mccs-eep choices: 'regular' (npz, $EEP_PATH), and 'compressed-<name>': file stem per .h5 in $POSTX_EEP_DIR."""
    eep_dir = os.environ.get(EEP_DIR_ENV)
    stems = sorted(p.stem for p in Path(eep_dir).glob('*.h5')) if eep_dir else []
    return {'regular': None} | {f'compressed-{s.removeprefix("ska_low_")}': s for s in stems}


MCCS_EEPS = _mccs_eeps()


def eep_kwargs(mccs_eep: str = 'regular') -> dict:
    """calibrate_mccs_visibility() EEP file options for a --mccs-eep choice.

    regular: from $EEP_PATH, $EEP_FILE_NAME and $EEP_SUFFIX. compressed-*: the .h5 file in $POSTX_EEP_DIR.
    """
    if MCCS_EEPS.get(mccs_eep):
        return {'eep_path': os.environ[EEP_DIR_ENV], 'eep_filebase': MCCS_EEPS[mccs_eep], 'eep_suffix': '.h5'}
    env = {k: os.environ.get(k, EEP_DEFAULTS.get(k, '')) for k in ('EEP_PATH', 'EEP_FILE_NAME', 'EEP_SUFFIX')}
    return {'eep_path': env['EEP_PATH'], 'eep_filebase': env['EEP_FILE_NAME'], 'eep_suffix': env['EEP_SUFFIX']}


def eep_missing(f_mhz, mccs_eep: str = 'regular', max_offset_mhz: float = 3.0) -> np.ndarray:
    """Bool (N_freq): channels with no EEP within max_offset_mhz in a compressed --mccs-eep file.

    The solver's load_eeps() raises for these, at its default 3 MHz limit, e.g. 247.5-277 MHz in ska_low_feko.h5.
    ponytail: regular (npz) EEPs are taken to cover the band; check their frequencies too if they ever have gaps.
    """
    f_mhz = np.atleast_1d(np.asarray(f_mhz, dtype=float))
    if not MCCS_EEPS.get(mccs_eep):
        return np.zeros(len(f_mhz), dtype=bool)
    kw = eep_kwargs(mccs_eep)
    with h5py.File(os.path.join(kw['eep_path'], kw['eep_filebase'] + '.h5'), 'r') as h5:
        # As load_eeps_h5(): ska-ost-low-uv >= 0.5.0 files hold frequencies in Hz, older ones in MHz
        freqs = h5['coords/frequency'][:] / 1e6 if 'alm' in h5 else h5['frequency'][:]
    return np.abs(f_mhz[:, None] - freqs[None, :]).min(axis=1) > max_offset_mhz


def mccs_cal_problem(cal_model: str, mccs_eep: str = 'regular') -> str:
    """Why an MCCS --cal-model can't run here with these EEPs, or '' if it can."""
    if MCCS_CAL[cal_model].get('ignore_eeps'):
        return ''
    if mccs_eep not in MCCS_EEPS:
        return f'No {mccs_eep} EEPs: set ${EEP_DIR_ENV} to the folder of .h5 EEPs before starting.'
    kw = eep_kwargs(mccs_eep)
    if MCCS_EEPS[mccs_eep]:
        fn = os.path.join(kw['eep_path'], kw['eep_filebase'] + '.h5')
        return '' if os.path.isfile(fn) else f'{mccs_eep}: {fn} not found.'
    if not os.path.isdir(kw['eep_path']):
        return (
            f'{cal_model} needs the MCCS EEPs: set $EEP_PATH ({kw["eep_path"] or "unset"}), use a compressed '
            f'--mccs-eep from ${EEP_DIR_ENV}, or use mccs-sunfast.'
        )
    return ''


def eep_order(aa) -> np.ndarray:
    """0-based EEP file row of each antenna, in correlator order.

    From the UVX `eep_index`, else the station's `eep_order.txt` in the ska-ost-low-uv config.

    Raises:
        ValueError: If neither is there, or the antenna names don't match.
    """
    if 'eep_index' in aa.uvx.antennas.attrs:
        return np.asarray(aa.uvx.antennas.attrs['eep_index'].values)
    fn = os.path.join(get_resource_path('config'), aa.name, 'eep_order.txt')
    if not os.path.exists(fn):
        raise ValueError(f'{aa.name}: no EEP order, neither eep_index in the UVX nor {fn}')
    eep = pd.read_csv(fn, sep=' ')
    if not np.array_equal(eep['name'].astype(str).values, aa.ant_names.values.astype(str)):
        raise ValueError(f"{aa.name}: antenna names in {fn} don't match the UVX")
    return eep['eep_index'].values


def to_sdp_vis(aa, eep_idx: np.ndarray):
    """The selected channel and time of a station as an ska-sdp-datamodels Visibility, antennas in EEP order.

    The UVX convention is the solver's: against a diffuse GSM at night, it fits with ~2x lower residuals than
    the raw HDF5 convention (UVX conjugated with XY and YX swapped). A baseline whose antennas swap order in
    EEP order is the reversed baseline, so it is conjugated with XY and YX swapped.

    Args:
        aa (ApertureArray): Station, at its selected channel and time.
        eep_idx (np.ndarray): EEP row of each antenna, see `eep_order()`.

    Returns:
        vis (xarray.Dataset): Visibility of one time and channel.
    """
    v = aa.uvx.data[aa.idx['t'], aa.idx['f']].values
    e1 = eep_idx[aa.uvx.data.coords['ant1'].values]
    e2 = eep_idx[aa.uvx.data.coords['ant2'].values]
    vis = np.where((e1 > e2)[:, None], np.conj(v[:, [0, 2, 1, 3]]), v)
    ant1, ant2 = np.minimum(e1, e2), np.maximum(e1, e2)
    order = np.lexsort((ant2, ant1))
    vis, ant1, ant2 = vis[order], ant1[order], ant2[order]
    bad = ~np.isfinite(vis)
    vis[bad] = 0

    enu = np.empty_like(aa.xyz_enu)
    enu[eep_idx] = aa.xyz_enu
    return sdp_visibility_datamodel(
        vis=vis.astype('complex64'),
        flags=bad,
        uvw=enu[ant1] - enu[ant2],
        ant1=ant1,
        ant2=ant2,
        location=aa.earthloc,
        antpos_enu=enu,
        time=aa._ws('t'),  # The channel's own capture time in a sweep
        int_time=1.0,  # ponytail: not used in calibration
        frequency_mhz=aa._ws('f').to_value('MHz'),
        station_name=aa.name,
        antenna_names=list(aa.ant_names.values[np.argsort(eep_idx)]),
    )


def mccs_calibrate_channel(
    aa,
    cal_model: str,
    leakage: str = 'solve',
    nside: int = 64,
    antenna_flags: np.ndarray | None = None,
    mccs_eep: str = 'regular',
) -> tuple[UVXAntennaCal, dict]:
    """Calibrate the selected channel of a station with the MCCS solver, as the SFT does.

    The solver's gains G (V_obs = G_i V G_j^H, EEP order) become the corrections C = G^-1 in correlator order
    that UVXAntennaCal holds, as in `ska_ost_low_uv.io.mccs_cal.read_mccs_cal()`. Antennas the solver masked
    get C = 0, and are flagged.

    Args:
        aa (ApertureArray): Station, at its selected channel.
        cal_model (str): A key of `MCCS_CAL`.
        leakage (str): 'solve' for the polarised (full-Jones) solve, 'none' for X and Y gains and an XY phase.
        nside (int): Healpix NSIDE of the GSM sky model.
        antenna_flags (np.ndarray): Bool (N_ant) antenna flags, added to the UVX static flags.
        mccs_eep (str): A key of `MCCS_EEPS`, see `eep_kwargs()`.

    Returns:
        cal (UVXAntennaCal): Full-Jones cal of the channel.
        info (dict): The solver's calibration info (corrcoeff, residual_std, xy_phase, ...), see `QA_KEYS`, and
            rel_residual, the residual rms relative to the model rms; per polarisation where the solver's is.
    """
    eep_idx = eep_order(aa)
    flags = aa.antenna_flags if antenna_flags is None else aa.antenna_flags | np.asarray(antenna_flags)
    with warnings.catch_warnings():
        # As in the SFT: the Sun-only model raises RuntimeWarnings, and sdp datamodels FutureWarnings
        warnings.simplefilter('ignore', category=RuntimeWarning)
        warnings.simplefilter('ignore', category=FutureWarning)
        gt, model, calibrated, masked, info = calibrate_mccs_visibility(
            to_sdp_vis(aa, eep_idx),
            masked_antennas=eep_idx[flags],
            nside=nside,
            jones_solve=leakage == 'solve',
            eep_rotation_deg=float(aa.uvx.antennas.attrs['array_rotation_angle'].values),
            **MCCS_CAL[cal_model],
            **SFT_KWARGS,
            **eep_kwargs(mccs_eep),
        )
    g = gt.gain.values[0, eep_idx, 0]  # (N_ant, 2, 2), correlator order
    bad = np.isin(eep_idx, masked) | ~np.isfinite(g).all(axis=(1, 2)) | (np.abs(np.linalg.det(g)) == 0)
    c = np.zeros_like(g)
    c[~bad] = np.linalg.inv(g[~bad])
    # corrcoeff, residual_std and residual_max are per polarisation
    info = {k: np.asarray(getattr(info, k), dtype=float) for k in QA_KEYS if hasattr(info, k)}
    # Residual rms relative to the model rms, per polarisation: a convergence measure for any sky model
    ok = ~calibrated.flags.values[0, :, 0]
    m, r = model.vis.values[0, :, 0], calibrated.vis.values[0, :, 0] - model.vis.values[0, :, 0]
    with np.errstate(divide='ignore', invalid='ignore'):  # NaN for cross-pols with no model (no EEPs)
        info['rel_residual'] = np.sqrt(np.nanmean(np.where(ok, np.abs(r) ** 2, np.nan), axis=0)) / np.sqrt(
            np.nanmean(np.where(ok, np.abs(m) ** 2, np.nan), axis=0)
        )
    cal = create_uvx_antenna_cal(
        telescope=aa.name,
        method=cal_model.replace('-', '_'),
        antenna_gains_arr=c.reshape(1, -1, 4),
        antenna_flags_arr=bad,
        f=aa.f[aa.idx['f'] : aa.idx['f'] + 1],
        a=aa.ant_names,
        p=POL4,
        provenance={'mccs': info},
    )
    return cal, info
