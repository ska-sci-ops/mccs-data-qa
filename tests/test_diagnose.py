import numpy as np
from mccs_data_qa.reports.diagnose import (
    CHECKS,
    GRADE_HEX,
    Findings,
    antenna_matrix,
    bandpass_stats,
    holo_stats,
    robust_z,
    time_stats,
    uvdist_stats,
)

RNG = np.random.default_rng(1)


def test_robust_z():
    z = robust_z(np.array([1.0, 2, 3, 4, 100]))
    assert z[2] == 0 and z[4] > 50 and abs(z[0]) < 2
    # Masked entries are left out of the median and MAD, but still scored
    z = robust_z(np.array([1.0, 1, 1, 50]), mask=np.array([False, False, False, True]))
    assert z[3] == np.inf and np.all(z[:3] == 0)


def test_uvdist_flags_pol_swap():
    n_ant = 20
    i, j = np.triu_indices(n_ant, k=1)
    bl = RNG.uniform(1, 40, len(i))
    # Parallel pols fall with baseline length (sky structure), cross pols ~10x lower
    par = np.exp(-bl / 20) * RNG.uniform(0.9, 1.1, len(i))
    cross = par / 10 * RNG.uniform(0.8, 1.2, (2, len(i)))  # Independent noise, as in real data
    rho = np.stack([par, cross[0], cross[1], par], axis=1)
    swapped = (i == 7) | (j == 7)  # Antenna 7's X and Y swapped: parallel and cross pols trade places
    rho[swapped] = rho[swapped][:, [1, 0, 3, 2]]
    s = uvdist_stats(rho, bl, i, j, n_ant)
    z = robust_z(np.log(s['cross']))
    assert z[7] > 5 and np.all(np.abs(np.delete(z, 7)) < 5)
    assert robust_z(np.log(s['xx']))[7] < -5


def test_bandpass_flags_level_and_ripple():
    n_f, n_ant = 100, 30
    spec = np.linspace(1, 3, n_f)[:, None, None]
    autos = spec * RNG.uniform(0.95, 1.05, (n_f, n_ant, 2))
    autos[:, 3, 0] *= 0.1  # Low X level
    autos[:, 5, 1] *= 1 + 0.5 * np.sin(np.arange(n_f) / 3)  # Rippled Y
    autos[:, 8] = 0  # Dead
    chan_flags = np.zeros(n_f, bool)
    chan_flags[50] = True
    autos[50] = 1e6  # A flagged RFI channel doesn't count
    s = bandpass_stats(autos, chan_flags, np.zeros(n_ant, bool))
    assert np.abs(robust_z(s['level'][:, 0])[3]) > 5
    assert robust_z(s['ripple'][:, 1])[5] > 5
    assert np.all(np.isnan(s['level'][8]))
    good = np.setdiff1d(np.arange(n_ant), [3, 5, 8])
    assert np.all(np.abs(robust_z(s['level'][:, 0])[good]) < 5)


def test_time_flags_unstable_antenna():
    n_t, n_ant = 10, 30
    power = RNG.uniform(0.95, 1.05, (n_t, n_ant, 2))
    power[::2, 4, :] *= 3  # Jumping power
    score = RNG.uniform(0.45, 0.55, (n_t, n_ant))
    score[6, 9] = 0.05  # Correlation drop
    s = time_stats(power, score, np.zeros(n_ant, bool))
    assert robust_z(s['power_var'][:, 0])[4] > 5
    assert robust_z(s['score_drop'])[9] > 5


def test_holo_flags_missing_and_phase():
    # 8 x 8 grid, 2 m spacing; healthy antennas share a phase
    e, n = np.meshgrid(np.arange(8) * 2.0, np.arange(8) * 2.0)
    enu = np.column_stack([e.ravel(), n.ravel(), np.zeros(64)])
    mc = np.full((64, 2), 1 + 0j) * RNG.uniform(0.9, 1.1, (64, 2)) * np.exp(1j * RNG.normal(0, 0.05, (64, 2)))
    autos = np.ones((64, 2))
    mc[10] *= 0.01  # Missing
    mc[30] *= np.exp(1j * 2.0)  # Phase error
    flags = np.zeros(64, bool)
    flags[40] = True
    mc[40] = 0  # Flagged antennas don't count as neighbours
    s = holo_stats(mc, autos, enu, flags)
    assert np.all(s['amp_db'][10] < -30)
    assert np.all(np.abs(s['dphase_deg'][30]) > 100)
    z = robust_z(np.abs(s['dphase_deg']), flags)
    assert np.all(z[30] > 5) and np.isfinite(s['dphase_deg'][41]).all()


def test_findings_grade_matrix():
    ants = np.array(['a1', 'a2', 'a3', 'a4'])
    f = Findings('s8-1', ants, 'uvdist')
    f.add('low_corr', [True, False, False, False], [True, True, False, False], np.arange(4.0), 'X', '5σ')
    f.add('swap', [False, False, True, False], [False, False, True, False], np.arange(4.0), 'XY', '10')
    # Fail in X only, warn in X only, fail in both pols, pass
    assert f.grade.tolist() == [[2, 0], [1, 0], [2, 2], [0, 0]]
    assert [(r['antenna'], r['pol']) for r in f.rows] == [('a1', 'X'), ('a3', 'XY')]
    grades = dict.fromkeys(CHECKS) | {'uvdist': f.grade}
    flags = np.array([False, False, False, True])
    m = antenna_matrix(ants, flags, grades, f.rows)
    assert m.count(GRADE_HEX[2]) == 3 and m.count(GRADE_HEX[1]) == 1
    assert m.count(GRADE_HEX[-1]) == 2 * len(CHECKS)  # The flagged antenna, every check
    assert 'X and Y swapped' in m
