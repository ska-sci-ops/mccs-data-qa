import numpy as np
from mccs_data_qa.reports.holo_suncal import aperture_phase_range

RNG = np.random.default_rng(2)
N, OSAMP = 99, 3


def holo(phase_deg, mag=None):
    """A synthetic run_selfholo() dict: unit magnitude, the given phases, in all four pols."""
    mag = np.ones((N, N)) if mag is None else mag
    img = mag * np.exp(1j * np.deg2rad(phase_deg))
    return {'aperture_img': np.repeat(img[..., None], 4, axis=-1), 'oversample_factor': OSAMP}


def test_aperture_phase_range():
    assert aperture_phase_range(holo(RNG.normal(0, 1, (N, N)))) == (-5, 5)  # Very good delays
    assert aperture_phase_range(holo(RNG.normal(0, 3, (N, N)))) in ((-10, 10), (-15, 15))
    assert aperture_phase_range(holo(RNG.uniform(-60, 60, (N, N)))) == (-90, 90)
    assert aperture_phase_range(holo(RNG.uniform(-180, 180, (N, N)))) == (-180, 180)
    # A few bad pixels saturate rather than widen the scale
    phs = RNG.normal(0, 3, (N, N))
    phs[N // 2, N // 2 - 5 : N // 2 + 5] = 170
    assert aperture_phase_range(holo(phs)) in ((-10, 10), (-15, 15))
    # Masked pixels (more than 20 dB below the peak) and pixels outside the shown crop don't count
    mag = np.full((N, N), 1e-3)
    mag[N // 2 - 5 : N // 2 + 5, N // 2 - 5 : N // 2 + 5] = 1
    phs = RNG.uniform(-180, 180, (N, N))
    phs[N // 2 - 5 : N // 2 + 5, N // 2 - 5 : N // 2 + 5] = RNG.normal(0, 3, (10, 10))
    assert aperture_phase_range(holo(phs, mag)) in ((-10, 10), (-15, 15))
