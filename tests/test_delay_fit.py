"""Delay fit with the phase offset pinned to 0."""

import astropy.units as u
import numpy as np
from mccs_data_qa.reports.delay_fit_report import pin_phase0

from ska_ost_low_uv.datamodel.cal import create_uvx_antenna_cal
from ska_ost_low_uv.postx.calibration.delay_fit import LIGHT_SPEED, fit_delays


def test_pin_phase0():
    """Pinned fit recovers delays of zero-offset phases, and gives phase0 = 0 where the free fit doesn't."""
    rng = np.random.default_rng(1)
    f = np.linspace(50e6, 350e6, 384)
    delay_ns = rng.uniform(-20, 20, (16, 2))
    phs = 2 * np.pi * f[:, None, None] * delay_ns[None] * 1e-9 + rng.normal(0, 0.2, (len(f), 16, 2))
    cal = create_uvx_antenna_cal(
        telescope='test',
        method='test',
        antenna_gains_arr=np.exp(1j * phs).astype('complex64'),
        antenna_flags_arr=np.zeros(16, dtype=bool),
        f=f * u.Hz,
        a=np.arange(16),
        p=np.array(('X', 'Y')),
    )
    fit = pin_phase0(fit_delays(cal), f)
    assert fit['phase0_pinned'] and not fit['phase0'].any()
    assert not fit['flags'].any()
    # Within 5 sigma of the fit's own Cramer-Rao error (~7 ps here)
    assert np.all(np.abs(fit['delay_ns'] - delay_ns) < 5 * fit['delay_err'] / LIGHT_SPEED * 1e9)
    assert np.all(fit['coherence'] > 0.95)
