"""compare_beams.py: Test whether per-antenna EEPs calibrate SKA-Low stations better than the AEP.

For each station and every --chan-step'th channel, the sky model (--sky-model, plus the Sun when it is
up) is built with each --beams element pattern and full-Jones gains are solved against it, fitting the
Sun's flux on the sky model's flux scale as suncal() does. The page compares:

* the in-sample residual |V_cal - M| / |V_cal| over unflagged cross-correlations;
* the residual on held-out baselines, with gains solved on the other half;
* the residual on the second integration, with gains solved on the first, against the noise floor
  |V_cal(t0) - V_cal(t1)| / sqrt(2);
* the gain magnitude scatter and number of flagged antennas.

At --controls-freq-idx the EEP model is also run with the antennas shuffled, shuffled within radius
rings, replaced by their mean, and rotated by an offset. The true EEP should beat all of these, with
the residual smallest at zero rotation offset. If calibration runs (run_stefcal.py) and delay fits
(delay_fit_report.py) exist for both beams, the spread of the fitted Sun flux between stations and the
per-antenna delay-fit residuals are compared too.

Usage:
    python -m mccs_data_qa.reports.compare_beams --datadir DIR --eb '*'
"""

import contextlib
import glob
import json
import os
from pathlib import Path

import numpy as np
import pylab as plt
from mccs_data_qa.reports.report import (
    AA1_GRID,
    BEAMS,
    DEFAULT_SIGMA_THR,
    add_suncal_args,
    args_table,
    base_parser,
    cal_variant,
    figs,
    find_station_files,
    read_station,
    station_key,
    table,
    write_report,
)
from scipy.stats import median_abs_deviation

from ska_ost_low_uv import show_version
from ska_ost_low_uv.io import read_cal
from ska_ost_low_uv.parallelize import run_in_parallel, task
from ska_ost_low_uv.postx import ApertureArray
from ska_ost_low_uv.postx.aeps.eep_models import load_eep_jones
from ska_ost_low_uv.postx.calibration.simple_cal import _build_apparent_vis_model, simple_stefcal_4pol
from ska_ost_low_uv.postx.calibration.vecmat import _pol4_to_jones
from ska_ost_low_uv.postx.simulation import _sim_utils, gsm_sim
from ska_ost_low_uv.postx.sky_model import RadioSource

N_FLUX_ITER = 4  # Gain solves alternating with the Sun flux fit
N_RINGS = 6  # Radius bins for the ring-shuffle control
ROT_OFFSETS = np.arange(-20, 20.1, 2.5)  # deg
METRICS = {
    'res_in': 'In-sample residual',
    'res_bl': 'Held-out baseline residual',
    'res_t1': 'Second-integration residual',
    'excess_t1': 'Second-integration residual above noise',
    'gain_scatter': 'Gain |g| scatter (MAD / median, XX and YY)',
    'n_flag': 'Flagged antennas',
}


def holdout_mask(n_ant: int, seed: int = 0) -> np.ndarray:
    """Random symmetric split of the cross-correlations: True for the half used to solve gains."""
    rng = np.random.default_rng(seed)
    m = np.triu(rng.random((n_ant, n_ant)) < 0.5, k=1)
    return m | m.T


def _fit(aa, D: np.ndarray, Mg: np.ndarray, Ms: np.ndarray | None, flags, sigma_thr: float, train: np.ndarray):
    """Solve full-Jones gains against Mg + S Ms on the `train` baselines, fitting S as suncal() does.

    The first solve is against the Sun alone, which needs no flux. Without Ms it is one solve against Mg.

    Returns:
        fit (dict): 'cal' (UVXAntennaCal), 'ok' unflagged antennas, 'a' and 'b' model scales
            (calibrated data = a Mg + b Ms), and the last StefCal 'dG'.
    """
    keep = train[..., None]
    M = Mg if Ms is None else Ms
    for _ in range(1 if Ms is None else N_FLUX_ITER + 1):
        cal = simple_stefcal_4pol(
            aa,
            antenna_flags=flags,
            sigma_thr=sigma_thr,
            vis_meas_override=np.where(keep, D, 0),
            vis_model_override=np.where(keep, M, 0),
        )
        ok = ~cal.flags.antenna.values
        sel = train & ok[:, None] & ok[None, :]
        ms = None if Ms is None else _pol4_to_jones(Ms)[sel]
        a, b = _scales(_apply(cal, D)[sel], _pol4_to_jones(Mg)[sel], ms)
        if Ms is not None:
            M = Mg + (b / a) * Ms
    return {'cal': cal, 'ok': ok, 'a': a, 'b': b, 'dG': cal.provenance['stefcal_4pol']['dG']}


def _apply(cal, D: np.ndarray) -> np.ndarray:
    """Calibrated (n_ant, n_ant, 2, 2) visibilities C_a D_ab C_b^H."""
    return _pol4_to_jones(cal.apply(D))


def _scales(y: np.ndarray, mg: np.ndarray, ms: np.ndarray | None) -> tuple[float, float]:
    """Real least-squares (a, b) in y = a mg + b ms."""
    X = np.stack([mg.ravel()] + ([] if ms is None else [ms.ravel()]), axis=1)
    A = np.concatenate([X.real, X.imag])
    coef = np.linalg.lstsq(A, np.concatenate([y.ravel().real, y.ravel().imag]), rcond=None)[0]
    return float(coef[0]), float(coef[1]) if ms is not None else 0.0


def _residual(fit: dict, Vc: np.ndarray, Mg: np.ndarray, Ms: np.ndarray | None, sel: np.ndarray) -> float:
    """|Vc - (a Mg + b Ms)| / |Vc| on the baselines in `sel`."""
    M = fit['a'] * _pol4_to_jones(Mg) + (0 if Ms is None else fit['b'] * _pol4_to_jones(Ms))
    return float(np.linalg.norm((Vc - M)[sel]) / np.linalg.norm(Vc[sel]))


def _models(aa, beam: str, sun: dict | None, sky_model: str) -> tuple[np.ndarray, np.ndarray | None]:
    """Diffuse sky (Jy/sr) and 1 Jy Sun apparent visibility models with this beam, at the selected channel."""
    kw = BEAMS[beam]
    Mg = _build_apparent_vis_model(aa, diffuse_model=sky_model, diffuse_model_units='Jysr', **kw)
    Ms = (
        None
        if sun is None
        else _build_apparent_vis_model(aa, extended_model=sun, diffuse_model_units='Jysr', **kw)
    )
    return Mg, Ms


def _evaluate(aa, D0, D1, Mg, Ms, flags, sigma_thr) -> dict:
    """All per-channel metrics for one model."""
    n = aa.n_ant
    full = ~np.eye(n, dtype=bool)
    fit = _fit(aa, D0, Mg, Ms, flags, sigma_thr, full)
    ok = fit['ok']
    sel = full & ok[:, None] & ok[None, :]
    Vc0 = _apply(fit['cal'], D0)
    out = {'res_in': _residual(fit, Vc0, Mg, Ms, sel), 'dG': fit['dG'], 'n_flag': int((~ok).sum())}

    train = holdout_mask(n)
    fit_bl = _fit(aa, D0, Mg, Ms, flags, sigma_thr, train)
    ok_bl = fit_bl['ok']
    test = ~train & full & ok_bl[:, None] & ok_bl[None, :]
    out['res_bl'] = _residual(fit_bl, _apply(fit_bl['cal'], D0), Mg, Ms, test)

    if D1 is not None:
        Vc1 = _apply(fit['cal'], D1)
        out['res_t1'] = _residual(fit, Vc1, Mg, Ms, sel)
        noise = float(np.linalg.norm((Vc0 - Vc1)[sel]) / np.sqrt(2) / np.linalg.norm(Vc1[sel]))
        out['noise'] = noise
        out['excess_t1'] = float(np.sqrt(max(out['res_t1'] ** 2 - noise**2, 0)))
    g = np.abs(fit['cal'].gains.values[0][:, [0, 3]])[ok]
    out['gain_scatter'] = float(np.mean([median_abs_deviation(g[:, p]) / np.median(g[:, p]) for p in (0, 1)]))
    return out


@contextlib.contextmanager
def _eep_override(eep_index: np.ndarray | None = None, rot_offset: float = 0, mean: bool = False):
    """Temporarily replace the EEP loader used by the GSM and Sun sims, for the null controls."""
    orig = gsm_sim.load_eep_for_sim

    def load(aa, nside, eep, sky_coord=None, pix=None):
        idx = _sim_utils._eep_index(aa) if eep_index is None else eep_index
        jones = load_eep_jones(
            aa.f[aa.idx['f']].to('MHz').value,
            nside,
            eep=eep,
            antennas=idx,
            sky_coord=sky_coord,
            rot_angle=float(aa.uvx.antennas.array_rotation_angle) + rot_offset,
            pix=pix,
            normalize=True,
        )
        jones[:, 1] *= -1  # As load_eep_for_sim()
        if mean:
            jones[:] = jones.mean(axis=0, keepdims=True)
        return jones

    gsm_sim.load_eep_for_sim = load
    try:
        yield
    finally:
        gsm_sim.load_eep_for_sim = orig


def _controls(aa, eep_beam: str, D0, D1, sun, flags, sigma_thr, sky_model) -> dict:
    """In-sample and held-out baseline residuals of the EEP null controls at the selected channel."""
    idx = _sim_utils._eep_index(aa)
    rng = np.random.default_rng(1)
    r = np.hypot(*aa.xyz_enu[:, :2].T)
    rings = np.digitize(r, np.quantile(r, np.linspace(0, 1, N_RINGS + 1)[1:-1]))
    ring_idx = idx.copy()
    for k in range(N_RINGS):
        sel = np.where(rings == k)[0]
        ring_idx[sel] = rng.permutation(idx[sel])
    cases = {f'shuffled ({s + 1})': {'eep_index': rng.permutation(idx)} for s in range(3)}
    cases |= {'shuffled in rings': {'eep_index': ring_idx}, 'mean EEP': {'mean': True}}
    cases |= {f'rot {d:+.1f}': {'rot_offset': float(d)} for d in ROT_OFFSETS}
    out = {}
    for name, kw in cases.items():
        with _eep_override(**kw):
            res = _evaluate(aa, D0, D1, *_models(aa, eep_beam, sun, sky_model), flags, sigma_thr)
        out[name] = {k: res[k] for k in ('res_in', 'res_bl')}
    return out


def _compare_station(
    fn: str,
    beams: list,
    chan_step: int,
    controls_idx: int,
    sigma_thr: float,
    sun_fwhm: float,
    min_sun_alt: float,
    ignore_static_flags: bool,
    sky_model: str,
) -> dict:
    """Metrics per beam and channel, and the EEP controls, for one station sweep.

    The controls are run at the last channel if `controls_idx` is beyond the sweep.
    """
    aa = ApertureArray(read_station(fn, ignore_static_flags))
    aa.set_task_verbosity('silent')
    n_chan = aa.uvx.data.frequency.size
    controls_idx = min(controls_idx, n_chan - 1)
    chans = sorted(set(range(0, n_chan, chan_step)) | {controls_idx})
    res = {'station': aa.name, 'f_mhz': [], 'beams': {b: [] for b in beams}, 'controls': None}
    eeps = [b for b in beams if 'eep' in BEAMS[b]]
    for f_idx in chans:
        aa.set_idx(f=f_idx, t=0)
        flags = aa.calibration.flagging.run_flagging()
        D0 = aa.generate_vis_matrix(vis='data', include_autos=False)
        D1 = None
        if aa.uvx.data.time.size > 1:
            aa.set_idx(t=1)
            D1 = aa.generate_vis_matrix(vis='data', include_autos=False)
            aa.set_idx(t=0)
        sun_alt = float(aa.coords.get_alt_az(aa.coords.get_sun()).alt.deg)
        sun = (
            {'Sun': RadioSource(aa.coords.get_sun(), mag=1.0, fwhm=sun_fwhm)} if sun_alt > min_sun_alt else None
        )
        res['f_mhz'].append(float(aa.f[f_idx].to('MHz').value))
        for b in beams:
            res['beams'][b].append(_evaluate(aa, D0, D1, *_models(aa, b, sun, sky_model), flags, sigma_thr))
        if f_idx == controls_idx and eeps:
            res['controls'] = _controls(aa, eeps[0], D0, D1, sun, flags, sigma_thr, sky_model)
            res['controls_f_mhz'] = res['f_mhz'][-1]
            res['controls_true'] = {b: res['beams'][b][-1] for b in beams}
    return res


def _series(res: dict, beam: str, key: str) -> np.ndarray:
    return np.array([m.get(key, np.nan) for m in res['beams'][beam]], dtype=float)


def _plot_metric(results: list, beams: list, key: str, outdir: Path) -> str:
    """Metric against frequency, one panel per station, one line per beam (plus the noise floor)."""
    fig, axs = plt.subplots(1, len(results), figsize=(5 * len(results), 3.5), squeeze=False, sharey=True)
    for ax, res in zip(axs[0], results, strict=True):
        for b in beams:
            ax.plot(res['f_mhz'], _series(res, b, key), '.-', label=b)
        if key == 'res_t1':
            ax.plot(res['f_mhz'], _series(res, beams[0], 'noise'), 'k:', label='noise floor')
        ax.set_title(res['station'])
        ax.set_xlabel('Frequency [MHz]')
    axs[0, 0].set_ylabel(METRICS[key])
    axs[0, 0].legend()
    fig.tight_layout()
    fn = f'{key}.png'
    fig.savefig(outdir / fn, dpi=100)
    plt.close(fig)
    return fn


def _plot_controls(results: list, beams: list, outdir: Path) -> str:
    """EEP null controls (bars) and the rotation-offset scan, against the true EEP and the AEP."""
    eep = next(b for b in beams if 'eep' in BEAMS[b])
    aep = next((b for b in beams if 'aep' in BEAMS[b]), None)
    fig, axs = plt.subplots(2, len(results), figsize=(5 * len(results), 7), squeeze=False)
    for j, res in enumerate(results):
        c = res['controls']
        names = [n for n in c if not n.startswith('rot ')]
        vals = {'true EEP': res['controls_true'][eep]['res_bl']} | {n: c[n]['res_bl'] for n in names}
        if aep:
            vals['AEP'] = res['controls_true'][aep]['res_bl']
        axs[0, j].barh(list(vals), list(vals.values()), color=['C2'] + ['C7'] * len(names) + ['C0'] * bool(aep))
        axs[0, j].set_title(f'{res["station"]}, {res["controls_f_mhz"]:.2f} MHz')
        axs[0, j].set_xlabel('Held-out baseline residual')
        rot = [c[f'rot {d:+.1f}']['res_bl'] for d in ROT_OFFSETS]
        axs[1, j].plot(ROT_OFFSETS, rot, 'o-')
        axs[1, j].set_xlabel('Rotation offset [deg]')
        axs[1, j].set_ylabel('Held-out baseline residual')
    fig.tight_layout()
    fn = 'controls.png'
    fig.savefig(outdir / fn, dpi=100)
    plt.close(fig)
    return fn


def _sun_flux_spread(caldir: Path, beams: list, sky_model: str) -> dict:
    """Per-channel std / mean of the suncal Sun flux across stations, per beam, from run_stefcal.py runs."""
    out = {}
    for b in beams:
        fns = sorted(glob.glob(str(caldir / cal_variant('sun+sky', 'solve', b, sky_model) / '*.cal')))
        cals = [read_cal(fn) for fn in fns if not fn.endswith('_delay.cal')]
        fluxes = [np.asarray(c.provenance.get('suncal', {}).get('sun_flux_jy', []), dtype=float) for c in cals]
        if len(fluxes) < 2 or not all(len(f) for f in fluxes):
            continue
        s = np.vstack(fluxes)
        with np.errstate(invalid='ignore'):
            out[b] = (cals[0].gains.frequency.values / 1e6, np.nanstd(s, 0) / np.nanmean(s, 0))
    return out


def _delay_resid(delaydir: Path, beams: list, sky_model: str) -> dict:
    """Median delay-fit phase residual rms (rad) per beam and station, from delay_fit_report.py runs."""
    out = {}
    for b in beams:
        for fn in glob.glob(str(delaydir / cal_variant('sun+sky', 'solve', b, sky_model) / '*_delay.cal')):
            fit = read_cal(fn).provenance['delay_fit']
            rms = np.asarray(fit['resid_rms'], dtype=float)[~np.asarray(fit['flags'], dtype=bool)]
            out.setdefault(b, {})[Path(fn).name.removesuffix('_delay.cal')] = float(np.nanmedian(rms))
    return out


def compare_beams(
    files: list,
    outdir: str,
    beams: list,
    chan_step: int = 8,
    controls_freq_idx: int = 77,
    sigma_thr: float = DEFAULT_SIGMA_THR,
    sun_fwhm: float = 0.57,
    min_sun_alt: float = 10,
    sky_model: str = 'gsm16',
    caldir: str | None = None,
    delaydir: str | None = None,
    n_jobs: int = 3,
    ignore_static_flags: bool = False,
    reuse_results: bool = False,
    cli_args: dict | None = None,
) -> Path:
    """Compare element patterns on each station sweep and write the comparison page.

    Args:
        files (list): UVX files, one per station.
        outdir (str): Output folder for index.html, the plots and results.json.
        beams (list): Element patterns to compare, keys of `report.BEAMS`.
        chan_step (int): Compare every chan_step'th channel.
        controls_freq_idx (int): Channel for the EEP null controls.
        sigma_thr (float): Antenna gain outlier threshold.
        sun_fwhm (float): FWHM of the Gaussian Sun model (deg).
        min_sun_alt (float): Sun elevation (deg) below which the model is the diffuse sky only.
        sky_model (str): Diffuse sky model.
        caldir (str): run_stefcal.py output folder, for the Sun flux spread between stations.
        delaydir (str): delay_fit_report.py output folder, for the delay-fit residuals.
        n_jobs (int): Stations compared in parallel.
        ignore_static_flags (bool): Ignore the static antenna flags stored in the UVX files.
        reuse_results (bool): Rebuild the page from an existing results.json instead of recomputing.
        cli_args (dict): Command-line arguments for the Settings tab.

    Returns:
        index_fn (Path): The page written.
    """
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    if reuse_results and (out / 'results.json').exists():
        results = json.loads((out / 'results.json').read_text())
    else:
        tasks = [
            task(_compare_station)(
                fn,
                beams,
                chan_step,
                controls_freq_idx,
                sigma_thr,
                sun_fwhm,
                min_sun_alt,
                ignore_static_flags,
                sky_model,
            )
            for fn in files
        ]
        results = sorted(
            run_in_parallel(tasks, n_workers=n_jobs, verbose=True), key=lambda r: station_key(r['station'])
        )
        (out / 'results.json').write_text(json.dumps(results, indent=1))

    base = beams[0]
    rows = []
    for key, label in METRICS.items():
        for res in results:
            ref = _series(res, base, key)
            if np.all(np.isnan(ref)):
                continue
            row = [label, res['station'], f'{np.nanmedian(ref):.4g}']
            for b in beams[1:]:
                v = _series(res, b, key)
                ratio = np.nanmedian(v / ref) if key != 'n_flag' else np.nanmedian(v - ref)
                wins = np.nanmean(v < ref) if key != 'n_flag' else np.nanmean(v <= ref)
                row += [f'{np.nanmedian(v):.4g}', f'{ratio:.3g}', f'{100 * wins:.0f}%']
            rows.append(row)
    header = ['Metric', 'Station', base]
    for b in beams[1:]:
        header += [b, f'{b} / {base}' + ' (median)', f'channels {b} better']
    verdict = table(header, rows)

    plots = [_plot_metric(results, beams, k, out) for k in ('res_in', 'res_bl', 'res_t1', 'gain_scatter')]
    captions = [METRICS[k] for k in ('res_in', 'res_bl', 'res_t1', 'gain_scatter')]
    body = f'<div class="card"><h2>Medians over channels</h2>{verdict}</div>'
    body += figs([p.removesuffix('.png') for p in plots], prefix='', captions=captions)
    tabs = [('summary', 'Summary', body)]

    if all(r['controls'] for r in results):
        fn = _plot_controls(results, beams, out)
        ctl = figs(
            [fn.removesuffix('.png')],
            prefix='',
            captions=[
                'EEP null controls at one channel: held-out baseline residual, and against rotation offset'
            ],
        )
        tabs.append(('controls', 'Controls', ctl))

    extra = ''
    spread = _sun_flux_spread(Path(caldir), beams, sky_model) if caldir else {}
    if spread:
        fig, ax = plt.subplots(figsize=(8, 3.5))
        for b, (f, s) in spread.items():
            ax.plot(f, s, '.', ms=3, label=f'{b} (median {100 * np.nanmedian(s):.1f}%)')
        ax.set_xlabel('Frequency [MHz]')
        ax.set_ylabel('Sun flux std / mean across stations')
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / 'sun_flux_spread.png', dpi=100)
        plt.close(fig)
        extra += figs(
            ['sun_flux_spread'], prefix='', captions=['Spread of the fitted Sun flux between stations']
        )
    dres = _delay_resid(Path(delaydir), beams, sky_model) if delaydir else {}
    if dres:
        stations = sorted({s for v in dres.values() for s in v}, key=station_key)
        dtab = table(
            ['Beam', *stations], [[b, *(f'{v.get(s, np.nan):.3f}' for s in stations)] for b, v in dres.items()]
        )
        extra += f'<div class="card"><h2>Median delay-fit phase residual rms [rad]</h2>{dtab}</div>'
    if extra:
        tabs.append(('sweep', 'Sweep cal', extra))
    tabs.append(('settings', 'Settings', f'<div class="card">{args_table(cli_args or {})}</div>'))

    info = [
        ('Stations', ', '.join(r['station'] for r in results)),
        ('Beams', ', '.join(beams)),
        ('Channels', f'{len(results[0]["f_mhz"])}, every {chan_step}'),
    ]
    index_fn = out / 'index.html'
    write_report(index_fn, 'Element Pattern Comparison', info, tabs, show_version())
    return index_fn


def build_parser():
    """Command-line argument parser."""
    parser = base_parser(__doc__, outdir='')
    parser.add_argument('--beams', nargs='+', choices=list(BEAMS), default=['aep-harp', 'eep-harp'])
    parser.add_argument('--chan-step', type=int, default=8, help='Compare every Nth channel')
    parser.add_argument('--controls-freq-idx', type=int, default=77, help='Channel for the EEP null controls')
    parser.add_argument(
        '--sigma-thr', type=float, default=DEFAULT_SIGMA_THR, help='Antenna gain outlier threshold'
    )
    add_suncal_args(parser, ('sun_fwhm', 'min_sun_alt'))
    parser.add_argument(
        '--caldir', default='', help='run_stefcal.py output (default <datadir>/reports/calibration)'
    )
    parser.add_argument(
        '--delaydir', default='', help='delay_fit_report.py output (default <datadir>/reports/delay_fit)'
    )
    parser.add_argument('--n-jobs', type=int, default=3, help='Stations compared in parallel')
    parser.add_argument(
        '--reuse-results', action='store_true', help='Rebuild the page from results.json in the output folder'
    )
    return parser


def main():
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()
    files = find_station_files(parser, args, AA1_GRID)
    reports = os.path.join(args.datadir, 'reports')
    index_fn = compare_beams(
        files,
        outdir=args.outdir or os.path.join(reports, 'beam_comparison'),
        beams=args.beams,
        chan_step=args.chan_step,
        controls_freq_idx=args.controls_freq_idx,
        sigma_thr=args.sigma_thr,
        sun_fwhm=args.sun_fwhm,
        min_sun_alt=args.min_sun_alt,
        sky_model=args.sky_model,
        caldir=args.caldir or os.path.join(reports, 'calibration'),
        delaydir=args.delaydir or os.path.join(reports, 'delay_fit'),
        n_jobs=args.n_jobs,
        reuse_results=args.reuse_results,
        ignore_static_flags=args.ignore_static_flags,
        cli_args=vars(args),
    )
    print(f'Wrote {index_fn}')


if __name__ == '__main__':
    main()
