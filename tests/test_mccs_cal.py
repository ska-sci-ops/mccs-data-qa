"""test_mccs_cal: the MCCS solver wrapper recovers known antenna gains, in correlator order."""

import astropy.units as u
import numpy as np
import pytest
from astropy.time import Time

pytest.importorskip('ska_low_mccs_calibration')
from ska_ost_low_uv.postx import ApertureArray  # noqa: E402

from mccs_data_qa.reports import mccs_cal  # noqa: E402


def _jones(v):
    """(..., 4) XX, XY, YX, YY to (..., 2, 2)."""
    return v.reshape(*v.shape[:-1], 2, 2)


def test_mccs_calibrate_channel_recovers_gains():
    # s8-5 at 167 MHz with the Sun at 69 deg, from a real day set
    aa = ApertureArray(station_id='s8-5', sim_f=[167.1875] * u.MHz, sim_t=Time(['2026-10-13T03:20:18']))
    aa.set_task_verbosity('silent')
    eep = mccs_cal.eep_order(aa)
    a1, a2 = aa.uvx.data.coords['ant1'].values, aa.uvx.data.coords['ant2'].values
    data = aa.uvx.data.values
    data[...] = 1

    # The solver's own Sun model, in EEP order, mapped back to the UVX baselines (see to_sdp_vis)
    _, model, *_ = mccs_cal.calibrate_mccs_visibility(
        mccs_cal.to_sdp_vis(aa, eep), **mccs_cal.MCCS_CAL['mccs-sunfast'], **mccs_cal.SFT_KWARGS
    )
    m = {(i, j): v for i, j, v in zip(model.antenna1.values, model.antenna2.values, model.vis.values[0, :, 0])}
    e1, e2 = eep[a1], eep[a2]
    mv = np.array([m[min(i, j), max(i, j)] for i, j in zip(e1, e2)])
    mv = np.where((e1 > e2)[:, None], np.conj(mv[:, [0, 2, 1, 3]]), mv)

    # Known diagonal gains per antenna, in correlator order: V_ab = G_a M G_b^H
    rng = np.random.default_rng(1)
    g = np.zeros((aa.n_ant, 2, 2), complex)
    g[:, [0, 1], [0, 1]] = rng.uniform(0.5, 2, (aa.n_ant, 2)) * np.exp(2j * np.pi * rng.uniform(size=(aa.n_ant, 2)))
    v = g[a1] @ _jones(mv) @ np.conj(np.swapaxes(g[a2], -1, -2))
    data[0, 0] = v.reshape(-1, 4)

    cal, info = mccs_cal.mccs_calibrate_channel(aa, 'mccs-sunfast', leakage='none')
    c = _jones(cal.gains.values[0])
    ok = ~cal.flags.antenna.values
    assert ok.sum() > 0.9 * aa.n_ant
    # C_a G_a is the same for every antenna: a common amplitude and phase per pol (the Y phase is free
    # without cross-pol model visibilities)
    cg = (c @ g)[ok][:, [0, 1], [0, 1]]
    np.testing.assert_allclose(cg / cg[:1], 1, atol=1e-3)
    assert info['rel_residual'][[0, 3]].max() < 1e-3


def test_eep_kwargs_regular_and_compressed(tmp_path, monkeypatch):
    (tmp_path / 'ska_low_harp.h5').touch()
    monkeypatch.setenv('POSTX_EEP_DIR', str(tmp_path))
    monkeypatch.setenv('EEP_PATH', '/eeps')
    monkeypatch.setattr(mccs_cal, 'MCCS_EEPS', mccs_cal._mccs_eeps())
    assert list(mccs_cal.MCCS_EEPS) == ['regular', 'compressed-harp']
    assert mccs_cal.eep_kwargs('compressed-harp') == {
        'eep_path': str(tmp_path),
        'eep_filebase': 'ska_low_harp',
        'eep_suffix': '.h5',
    }
    assert mccs_cal.eep_kwargs('regular')['eep_path'] == '/eeps'
    assert mccs_cal.eep_kwargs('regular')['eep_suffix'] == '.npz'
    assert mccs_cal.mccs_cal_problem('mccs-gsm', 'compressed-harp') == ''
    assert 'POSTX_EEP_DIR' in mccs_cal.mccs_cal_problem('mccs-gsm', 'compressed-feko')


def test_eep_missing_flags_channels_far_from_any_eep(tmp_path, monkeypatch):
    # A gap like ska_low_feko.h5's 244.53-280 MHz; frequencies in Hz as in ska-ost-low-uv >= 0.5.0 files
    import h5py

    with h5py.File(tmp_path / 'ska_low_feko.h5', 'w') as h5:
        h5['alm'] = np.zeros((3, 1, 1), 'int32')
        h5['coords/frequency'] = np.array([241.40625, 244.53125, 280.0]) * 1e6
    monkeypatch.setenv('POSTX_EEP_DIR', str(tmp_path))
    monkeypatch.setattr(mccs_cal, 'MCCS_EEPS', mccs_cal._mccs_eeps())
    f = [244.5, 247.5, 262.0, 276.5, 277.5]
    assert mccs_cal.eep_missing(f, 'compressed-feko').tolist() == [False, False, True, True, False]
    assert not mccs_cal.eep_missing(f, 'regular').any()
