# Station sensitivity: conventions, method and comparisons

This note goes with `mccs_data_qa/reports/station_sensitivity.py`. It sets out the definitions the script uses, the two SEFDs it reports (element and station, both at zenith), and how they compare with the SKA-Low sensitivity calculator (senscalc) and with AAVS2 measurements.

Example numbers are from the 2026-05-05 sweep (`eb-t0001-20260505-0000*`; stations s8-1, s8-2, s8-3; UTC 19:46:08, LST 18.46 h, zenith at Galactic $(l, b) \approx (6.2^\circ, -6.8^\circ)$). At that LST the Galactic centre is about $9^\circ$ from zenith.

## 1. Conventions

### 1.1 Visibility flux scale (Smirnov 2011, Sect. 7.2)

The Stokes parameters of the incident field are
$$
I = \langle|e_x|^2\rangle + \langle|e_y|^2\rangle, \qquad
\langle \mathbf{e}\mathbf{e}^H \rangle = \frac{1}{2}\begin{pmatrix} I+Q & U-jV \\ U+jV & I-Q \end{pmatrix}.
$$
The correlator's response to a 1 Jy unpolarised source is a matter of convention:

| Convention | $V_{XX}$ for 1 Jy | Stokes I from the image | Used by |
|---|---|---|---|
| ½ | 0.5 | $I = XX + YY$ | `ska_ost_low_uv` sim and calibration, Hamaker et al. (1996), Sutinjo et al. |
| 1 | 1 | $I = (XX + YY)/2$ | AIPS, CASA, MIRIAD (so Macario et al. 2022) |

`ska_ost_low_uv` uses **Convention-½**. `simulate_visibilities_gsm` puts $I/2$ into each of XX and YY (`gsm_sim.py`), with $I = 2kT/\lambda^2$ from `k_to_jysr`. Point sources get amplitude $\sqrt{S/2}$ (`simple_sim.py`). Calibrated autos and images are on the same scale, so a 1 Jy source reads 0.5 Jy/beam in the XX image. Mixing "$I = XX+YY$" with "$I = (XX+YY)/2$" gives a factor-of-2 error in flux.

### 1.2 SEFD and A/T

A single polarisation $p$:
$$
\mathrm{SEFD}_p = \frac{2 k T_{\mathrm{sys},p}}{A_{\mathrm{eff},p}}, \qquad
\left(\frac{A}{T}\right)_p = \frac{2k}{\mathrm{SEFD}_p}, \qquad 2k = 2760\ \mathrm{Jy\,m^2\,K^{-1}}.
$$
In Convention-½, a calibrated autocorrelation $P_p$ (Jy) corresponds to $\mathrm{SEFD}_p = 2P_p / \hat{B}_{pp}$, where $\hat{B}_{pp}$ is the peak-normalised element response in that direction.

Stokes I, at zenith and for unpolarised sky (Sutinjo et al. 2021; Sokolowski et al. 2022, Eq. 5):
$$
\mathrm{SEFD}_I = \frac{1}{2}\sqrt{\mathrm{SEFD}_X^2 + \mathrm{SEFD}_Y^2}, \qquad
\left(\frac{A}{T}\right)_I = \frac{\sqrt{2}\,k}{\mathrm{SEFD}_I}.
$$
For balanced polarisations, $\mathrm{SEFD}_I = \mathrm{SEFD}_p/\sqrt{2}$, so $(A/T)_I = (A/T)_p$. Senscalc uses the same A/T factor (`2760 * sqrt(2) / 2 / sefd`).

### 1.3 Radiometer equation

Per baseline, with $M = \eta\,\Delta\nu\,\tau$ effective samples ($\eta = 0.962$, SPRTS-516):
$$
\sigma\!\left[\mathrm{Re}\,V_{pq,aa}\right] = \sqrt{\frac{P_{a,p}\,P_{a,q}}{2M}}.
$$
For an array of $N$ identical elements, the Stokes-I image noise is $\sigma_I = \mathrm{SEFD}_I / \sqrt{N(N-1)M}$. That is the same form as senscalc's `_sefd_to_sensitivity`, with stations in place of elements.

## 2. Element SEFD (measured)

### 2.1 Stokes estimators (Sutinjo, Ung & Sokolowski 2022)

The Jones AEP (`ska_low_harp_jones.h5`, Ludwig-3 basis, loaded with `load_aep_jones`) gives $\mathbf{J}$ per pixel. With the image coherency $\mathbf{C}$ per pixel, each Stokes parameter is
$$
\tilde{S} = \mathrm{tr}\!\left(\mathbf{M}_S\,\mathbf{C}\right), \qquad
\mathbf{M}_S = \mathbf{J}^{-H}\mathbf{P}_S\mathbf{J}^{-1}, \qquad S \in \{I, Q, U, V\},
$$
with $\mathbf{P}_I = \mathbb{1}$, $\mathbf{P}_Q = \mathrm{diag}(1, -1)$, $\mathbf{P}_U = \begin{pmatrix}0 & 1\\ 1 & 0\end{pmatrix}$ and $\mathbf{P}_V = j\begin{pmatrix}0 & -1\\ 1 & 0\end{pmatrix}$ (their Eqs. 4–13, 18). $\mathbf{M}_I = (\mathbf{J}\mathbf{J}^H)^{-1} = \mathbf{B}^{-1}$, so Stokes I needs only the beam coherency, but Q, U and V need $\mathbf{J}$ itself. The Y row of $\mathbf{J}$ is negated so that the XY sign of $\mathbf{J}\mathbf{J}^H$ matches the imager, as in `load_eep_for_sim`; this does not change $|\mathbf{M}_S|$.

Q and U are in the station's Ludwig-3 frame (x along the X dipole), Sutinjo's instrument basis, not the celestial frame. SEFD$_I$ and SEFD$_V$ do not change under a real rotation of the frame; SEFD$_Q$ and SEFD$_U$ do. For ideal crossed short dipoles the code reproduces their Eqs. 30 and 32: SEFD$_I$ = SEFD$_Q \propto \sqrt{\cos^{-4}\theta + 1}$ and SEFD$_U$ = SEFD$_V \propto \sqrt{2}/\cos\theta$.

### 2.2 Predicted image noise

Summing Sutinjo's Eq. 19 over baselines, for the imager's $1/N^2$ gain and noise uncorrelated between antennas and between X and Y:
$$
\sigma_I^2(\hat{n}) = \frac{1}{N^4 M}\sum_{a,b\in\{X,Y\}} \left|B^{-1}_{ab}(\hat{n})\right|^2
\left( s_a s_b - \sum_p P_{a,p}P_{b,p} \right), \qquad s_a = \sum_p P_{a,p}.
$$
For identical antennas this gives the element SEFD (their Eq. 19)
$$
\mathrm{SEFD}_{S,\mathrm{el}}(\hat{n}) = \sqrt{\mathbf{p}^T\,|\mathbf{M}_S(\hat{n})|^{\circ 2}\,\mathbf{p}}, \qquad \mathbf{p} = [\bar{P}_X, \bar{P}_Y]^T,
$$
where $\circ 2$ is the element-wise square, and the same with $|\mathbf{M}_S|^{\circ 2}$ in place of $|\mathbf{B}^{-1}|^{\circ 2}$ above for Q, U and V. At zenith with $\mathbf{B} \approx \mathbb{1}$ this reduces to $\sqrt{\bar{P}_X^2 + \bar{P}_Y^2} = \tfrac{1}{2}\sqrt{\mathrm{SEFD}_X^2 + \mathrm{SEFD}_Y^2}$. It depends on direction and includes the $P_X P_Y$ cross terms.

**X–Y correlated noise.** Sutinjo's Eq. 19 assumes the X and Y noise of an element is uncorrelated. The script instead carries each noise power as a $2\times 2$ coherency $\mathbf{R}$ (from the calibrated XX, XY, YX, YY autos, cross-correlations and phased images) and uses
$$
\mathrm{SEFD}_S = \sqrt{\mathrm{tr}\!\left(\mathbf{M}_S\,\mathbf{R}\,\mathbf{M}_S^H\,\mathbf{R}\right)},
$$
the variance of $\mathbf{v}_q^H\mathbf{M}_S\mathbf{v}_p$ for independent antennas with noise coherency $\mathbf{R}$. For diagonal $\mathbf{R}$ this is exactly Eq. 19, and every sum $\sum_{ab}|W_{ab}|^2 x_a y_b$ in Sects. 2.2 and 2.4 becomes $\mathrm{tr}(\mathbf{W}\mathbf{X}\mathbf{W}^H\mathbf{Y})$. The measured X–Y correlation is 3–10%. It changes the zenith SEFDs by under 0.3%, but SEFD$_I$ and SEFD$_U$ on the $45^\circ$ diagonal by 2–6%, where $\mathbf{M}_I$ and $\mathbf{M}_U$ have large off-diagonal terms. Noise injection with $|r_{XY}| = 0.5$ through the imager gives $\tilde{S}/\sigma_S$ with std 0.98–1.02 for all four Stokes parameters, against 0.87–1.12 if the correlation is ignored.

### 2.3 Measurement: difference images and $\rho$

Consecutive integrations are imaged (`make_healpix`, no autos), differenced, and beam-inverted pixel by pixel. Then
$$
\rho_I = \mathrm{std}_{\mathrm{robust}}\!\left[\frac{\tilde{I}_{\mathrm{diff}}(\hat{n})}{\sqrt{2}\,\sigma_I(\hat{n})}\right]_{\theta < 45^\circ},
$$
with $\sigma_I(\hat{n})$ from Sect. 2.4 (correlated sky included). $\rho_Q$, $\rho_U$ and $\rho_V$ are the same for the other Stokes parameters. Sutinjo et al. validate SEFD$_{Q,U,V}$/SEFD$_I$ on difference images in the same way, so $\rho_Q \approx \rho_U \approx \rho_V \approx \rho_I$ tests the ratios. The script reports
$$
\rho = \tfrac{1}{2}(\rho_I + \rho_V), \qquad
\mathrm{SEFD}_{I,\mathrm{el}}^{\mathrm{meas}} = \rho\,\mathrm{SEFD}_{I,\mathrm{el}}(\hat{z}), \qquad
\mathrm{SEFD}_{I,\mathrm{st}}^{\mathrm{meas}} = \rho\,\mathrm{SEFD}_{I,\mathrm{st}},
$$
and the same $\rho$ times the Q, U and V predictions.
The $\sqrt{2}$ accounts for differencing two independent images. $\rho = 1$ means the image noise is what the autocorrelations, the correlated sky and $M$ predict. Values above 1 point to sky change between integrations, calibration errors, RFI, or noise correlated between antennas beyond the sky term (e.g. mutual coupling of receiver noise).

Checks:
- A Monte-Carlo with random non-diagonal $\mathbf{B}$ matches the formula to within sampling error (about 1.3%).
- Noise injection through the real imager and AEP (10 realisations, s8-3, 167 MHz): $\tilde{I}/\sigma_I$ has std 0.99, 1.00, 1.00 and 1.02 in the zenith-angle bands 0–20°, 20–40°, 40–60° and 60–80°.
- $\rho$ from one pair of integrations is limited by pixel correlation over the PSF. Jackknife errors per channel are about 2% above 100 MHz and 5–6% at 50–100 MHz (Sect. 2.5).
- Against the uncorrelated prediction of Sect. 2.2 (`rho_i_unc`), $\rho$ rises above 1 below about 85 MHz, because diffuse sky is correlated between elements (Sect. 2.4).

The per-pol element SEFD from image noise, $\mathrm{SEFD}_p = 2\sigma_p\sqrt{M}\,N$, is kept as an independent estimate. The factor 2 converts the Convention-½ XX image to the Stokes-I flux scale.

### 2.4 Noise correlated between antennas

The prediction in Sect. 2.2 assumes noise is uncorrelated between antennas. Diffuse sky emission is correlated between elements on short baselines, so the noise in a pixel follows the beamformed power in that direction. With $\mathbf{a} = \sum_p c_p \mathbf{v}_p$ the beamformed voltage towards pixel $\hat{n}$, and excluding autocorrelations as the imager does,
$$
\sigma^2(\hat{n}) = \frac{1}{M}\sum_{a,b} q_{ab}(\hat{n})\left[\left(\frac{\bar{P}_a}{N} + S_a(\hat{n})\right)\left(\frac{\bar{P}_b}{N} + S_b(\hat{n})\right) - \frac{\bar{P}_a\bar{P}_b}{N^3}\right],
$$
where $q = |\mathbf{B}^{-1}|^{\circ 2}$ for Stokes I, and $S_a(\hat{n}) = \frac{N}{N-1}\times$ the dirty-image value (no autos) of polarisation $a$ at $\hat{n}$, i.e. the mean cross-correlation phased to $\hat{n}$. This is the large-$N$ form; the script (`image_noise`) uses the exact finite-$N$ expression, which differs in the $PS$ and $S^2$ terms by $O(1/N)$. With $S = 0$ both reduce to Sect. 2.2 for identical antennas. $\bar{P}_a/N + S_a(\hat{n})$ is the station-beam power towards $\hat{n}$ (Sect. 3), so this is the same physics as the station SEFD.

Without this term, $\rho$ rises below about 85 MHz (median 1.3–1.45 over 50–100 MHz; 2.0–3.0 at 55 MHz), where $S/(\bar{P}/N)$ reaches 2–5 near the Galactic plane. With it, $\rho = 1.00$–$1.05$ across the band (Sect. 2.5). The script uses this prediction for $\rho_I$ and $\rho_V$, and keeps the uncorrelated $\rho$ as `rho_i_unc`.

### 2.5 Stokes V as a cross-check, and error estimates

On a difference image, Stokes V is a nearly independent estimate of the same noise: near zenith $\mathbf{M}_V$ weights mostly the XY plane, which is independent of XX and YY. The script uses the beam-corrected $\tilde{V} = \mathrm{tr}(\mathbf{M}_V\mathbf{C})$ and the prediction of Sect. 2.4 with $q = |\mathbf{M}_V|^{\circ 2}$. The tables below were computed with the raw instrumental $\mathrm{tr}(\mathbf{P}_V\mathbf{C})$ rather than $\tilde{V}$.

Differencing removes most of the usual objections to V:

| Concern | On a snapshot | On a difference image |
|---|---|---|
| Leakage from I into V | Can match the thermal noise (sky-to-noise $10^2$–$10^3$) | Cancels; only leakage × sky drift remains (second order) |
| Circularly polarised sources | Mostly compact; solar bursts | Static ones cancel; bursts and RFI affect I equally |
| Sky drift between integrations | Not applicable | Residual × leakage, so V is cleaner than I here |
| Noise correlated between antennas | Enters V through the product of the X- and Y-beamformed powers, just as it enters I | Same |
| X–Y correlation (cross-polar coupling, polarised signal, XY term of $\mathbf{B}$) | Enters I (as $+2|\langle ab^*\rangle|^2$) and V differently | Same; this is what $\rho_V/\rho_I$ tests |

At zenith with $\mathbf{B} \approx \mathbb{1}$, $\sigma_V^2 \propto 2P_XP_Y$ and $\sigma_I^2 \propto P_X^2 + P_Y^2$. These are equal for balanced polarisations; the measured $T_{\mathrm{rx}}$ imbalance (up to 72 K against 57 K) changes this by well under 1%.

**Measurement** (every second channel, $\theta < 45^\circ$, prediction of Sect. 2.4). Errors are jackknife over $\sim 15^\circ$ HEALPix cells (NSIDE 4), which accounts for pixel correlation over the PSF. The channel-to-channel scatter of $\rho_V/\rho_I$ is $\sqrt{2}$ times the per-channel jackknife error, so the jackknife errors are realistic.

| Station | Band [MHz] | $\rho_I$ (uncorrelated model) | $\rho_I$ | $\rho_V$ | $\rho_V/\rho_I$ | Scatter of $\rho_V/\rho_I$ | Jackknife error per channel |
|---|---|---|---|---|---|---|---|
| s8-1 | 50–100 | 1.45 | 1.02 | 1.00 | 0.96 | 15% | 6% |
| s8-1 | 100–330 | 0.99 | 1.02 | 1.00 | 0.99 | 3.6% | 2.1% |
| s8-2 | 50–100 | 1.32 | 1.05 | 1.01 | 0.95 | 12% | 5% |
| s8-2 | 100–330 | 0.97 | 1.00 | 1.00 | 1.00 | 2.8% | 2.0% |
| s8-3 | 50–100 | 1.41 | 1.02 | 0.99 | 0.97 | 8% | 5% |
| s8-3 | 100–330 | 0.98 | 1.01 | 1.00 | 0.99 | 3.1% | 2.1% |

(Medians over channels, RFI bands excluded.)

**Interpretation.** With correlated sky included, the difference-image RMS in both I and V matches the radiometer prediction from the calibrated autos to within about 2% above 100 MHz and about 5% below. V agrees with I to within 1% above 100 MHz, so X–Y correlation and residual sky drift are negligible there. Below 100 MHz, V is 3–5% below I. That could be residual drift in the I difference, which is larger where the Galactic plane is up, or X–Y correlation. It is comparable to the per-channel error.

**Error budget for SEFD$_I$ and A/T reported from the RMS**, i.e. $\mathrm{SEFD}_I = \rho\,\mathrm{SEFD}_{I,\mathrm{pred}}$ for element or station (relative, 1σ):

| Term | 100–330 MHz | 50–100 MHz | Notes |
|---|---|---|---|
| Statistical, per channel (jackknife) | 2% | 5–6% | One pair of integrations; falls as $1/\sqrt{N_{\mathrm{pairs}}}$ |
| Statistical, band median (~117 / 33 channels) | 0.2% | 1% | |
| I against V | ≤1% | 3–5% | Take $\tfrac{1}{2}(\rho_I + \rho_V)$ and quote half the difference as systematic |
| Noise model (Sect. 2.4 approximation, $\rho_I - 1$) | 1–2% | 2–5% | |
| Quantisation efficiency convention | 2% | 2% | $M = \eta\Delta\nu\tau$ against $\sigma \propto 1/\eta$ |
| Absolute flux scale (GSM16 calibration, AEP model) | ~10% | ~10–15% | Not tested by the RMS; Haslam $\beta = -2.55$ differs by 5–18% |
| Station only: $\bar{C}$ from the unflagged subset stands for 256 antennas | few % | larger | Not tested |

Excluding the flux scale, a single channel is good to about 3% above 100 MHz and about 7% below. The absolute flux scale dominates the total. The RMS tests the radiometer equation and the noise model, including the correlated-sky term that sets the station SEFD, but it cannot test the Jy scale, because the prediction and the measurement share it.

## 3. Station SEFD (beamformed, zenith)

A 256-element station beamformed at zenith with uniform weights has power, normalised by $N^2$,
$$
P_{\mathrm{st},a} = \frac{1}{N^2}\sum_{p,q} V_{pq,aa} = \frac{\bar{P}_a}{N} + \frac{N-1}{N}\,\bar{C}_a, \qquad
\bar{C}_a = \frac{\mathrm{Re}\sum_{p\neq q} V_{pq,aa}}{N(N-1)}.
$$
A source $S$ at zenith gives $P_{\mathrm{st},a} = \hat{B}_{aa}S/2$, so
$$
\mathrm{SEFD}_{p,\mathrm{st}} = \frac{2P_{\mathrm{st},a}}{\hat{B}_{aa}}, \qquad
\mathrm{SEFD}_{S,\mathrm{st}} = \sqrt{\mathbf{p}_{\mathrm{st}}^T\,|\mathbf{M}_S(\hat{z})|^{\circ 2}\,\mathbf{p}_{\mathrm{st}}}, \qquad N = 256.
$$
At zenith the beamformed station has the element Jones matrix, so $\mathbf{M}_S$ is the same as for one element (Sutinjo et al. 2022b, Paper III).

At zenith $\mathbf{J} \approx \mathbb{1}$, so SEFD$_Q$ = SEFD$_I$ and SEFD$_U$ = SEFD$_V$ = $\sqrt{2P_XP_Y}$, which differs from SEFD$_I$ only to second order in the X/Y imbalance (about 0.3% for a 10% imbalance). The Stokes parameters separate off zenith, so the script also reports every SEFD$_S$ at $45^\circ$ from zenith on the diagonal between the X and Y dipoles (station-frame $\phi = 45^\circ$, azimuth $45^\circ$ + station rotation), where the projected dipoles are furthest from orthogonal. For the station there, $\bar{C}_a$ is replaced by the mean cross-correlation phased to that direction, $S_a(\hat{n})$ of Sect. 2.4.
$\bar{C}_a$ is the sky correlated between elements. It makes the station's $T_{\mathrm{ant}}$ the station-beam-weighted sky (Sokolowski et al. 2022, Eq. 2) rather than the element-weighted sky. The element-scaled value $\mathrm{SEFD}_{I,\mathrm{el}}/256$ drops this term. That is correct for noise in single-station all-sky images, but not for the SEFD of a station in station-to-station interferometry.

Assumptions:
- $\bar{C}_a$ measured on the unflagged subset stands for the full 256-element layout.
- Uniform weights and zenith pointing.
- $\rho$ is applied to the station SEFD as well. The noise prediction it is measured against includes the same correlated-sky term (Sect. 2.4), so $\rho \approx 1$ checks both.

Size of the term (s8-3, $\bar{C}_X$ relative to $\bar{P}_X/256$): 2.0 at 70 MHz, 0.77 at 100 MHz, 0.38 at 120 MHz, 0.07 at 160 MHz, and about 0 (±0.05) above 200 MHz. It is large where the station is dense and its beam wide, and where the beam sits on the bright Galactic plane.

## 4. Comparisons

### 4.1 Senscalc versus the gist

The gist ([telegraphic/7b168fd3](https://gist.github.com/telegraphic/7b168fd3376d20cd0dbdd965617075b5)) plots `SEFDTable.lookup_stokes_i_sefd(0, 90, 0, 12, f)`: zenith, **averaged over LST 0–12 h** (cold sky). The table (`ska_station_sensitivity_AAVS2.h5`, Sokolowski et al. 2022) varies strongly with LST. Station SEFD$_I$ at zenith, in Jy:

| MHz | LST 0–12 h (gist) | LST 18.5 h |
|---|---|---|
| 69 | 3291 | 19961 |
| 100 | 1603 | 8246 |
| 160 | 1346 | 3620 |
| 200 | 1404 | 2995 |
| 300 | 1712 | 2914 |

The factor of about 2 between the report and the gist is mostly this LST difference, not a Stokes-I convention.

### 4.2 Senscalc at matched LST

Station SEFD$_I$ (256 antennas, zenith) in Jy from `station_sensitivity.py`, against senscalc at LST 18.5 h (interpolated to the channel frequency):

| MHz | Senscalc | s8-3 element/256 | s8-3 station | Station ÷ senscalc (s8-1 / s8-2 / s8-3) |
|---|---|---|---|---|
| 70 | 19295 | 4942 | 15634 ± 634 | 0.96 / 0.84 / 0.81 |
| 100 | 8222 | 3888 | 7059 ± 322 | 0.94 / 0.84 / 0.86 |
| 120 | 5626 | 3210 | 4535 ± 190 | 0.90 / 0.77 / 0.81 |
| 160 | 3618 | 2919 | 3136 ± 56 | 0.86 / 0.84 / 0.87 |
| 200 | 2992 | 2498 | 2407 ± 58 | 0.86 / 0.78 / 0.80 |
| 230 | 3069 | 2846 | 2722 ± 46 | 0.92 / 0.87 / 0.89 |
| 300 | 2910 | 2749 | 2608 ± 55 | 0.94 / 0.88 / 0.90 |

Element/256 ÷ senscalc runs from 0.26 (70 MHz) to 0.94 (300 MHz). Adding the correlated-sky term takes it to 0.77–0.96, with no trend in frequency. Station values are $\rho$ × prediction with $\pm$ the per-channel error, excluding the flux scale.

Known differences in the senscalc model (Sokolowski et al. 2022):
- **Sky model:** Haslam 408 MHz scaled with $\beta = -2.55$. Element-beam-weighted at this LST, that is 18%, 17%, 15%, 11%, 9%, 8% and 5% above GSM16 at 70, 100, 120, 160, 200, 230 and 300 MHz. Our flux scale is tied to GSM16 through calibration.
- **Efficiency:** radiation efficiency $\eta = 1$.
- **Receiver:** AAVS2 $T_{\mathrm{rcv}}$ from lab measurements.
- **Station beam:** array factor times the isolated SKALA4.1 pattern, on the AAVS2 layout.
- **Stokes I:** Eq. 5 is exact at zenith but not away from the cardinal planes (Sutinjo et al. 2021).

### 4.3 Macario et al. (2022), AAVS2

Macario et al. measure AAVS2 sensitivity from MIRIAD difference images (Convention-1, $I = (XX+YY)/2$), with Sun-based calibration, 0.14 s integrations and 0.78 MHz bandwidth. Their Eq. 2,
$$
\mathrm{SEFD}_{s,p} \simeq \frac{\sigma_p}{\sqrt{2}}\sqrt{\eta\,t_i\,B}\;\frac{N}{256},
$$
is **element-scaled**: it scales the raw image RMS by $N/256$. Our equivalent is the element prediction times $\rho$ against the uncorrelated model, divided by 256 (`sefd_i_el_pred` × `rho_i_unc` / 256). That keeps whatever correlated sky the image noise contains, as Macario's does. It is not `sefd_i_el` / 256, which is the single-element SEFD with the correlated-sky contribution to image noise modelled out. Their Table 4 gives SKA (512-station) A/T averaged over LST 0–8 h. We divided by 512 and scaled to LST 18.46 h by the element-beam $T_{\mathrm{sys}}$ ratio: GSM16 weighted by the AEP plus our measured $T_{\mathrm{rx}}$, averaged as $\langle 1/T_{\mathrm{sys}}\rangle$ over LST 0–8 h. The modelled $T_{\mathrm{ant}}$ matches our measured $T_{\mathrm{ant}}$ to within 2%.

A/T in m²/K per station:

| MHz | Macario, LST 0–8 h | $T_{\mathrm{sys}}$, 0–8 h → 18.5 h | Macario scaled to 18.5 h | Ours, image-noise element-scaled $(A/T)_I$ (3-station mean) | Ours ÷ Macario |
|---|---|---|---|---|---|
| 55 | 0.29 | 5616 → 10715 K | 0.15 | 0.14 | 0.91 |
| 70 | 0.41 | 2365 → 5064 K | 0.19 | 0.26 | 1.37 |
| 110 | 1.23 | 728 → 1717 K | 0.52 | 0.66 | 1.27 |
| 160 | 1.58 | 251 → 662 K | 0.60 | 0.74 | 1.23 |
| 230 | 1.37 | 148 → 306 K | 0.66 | 0.67 | 1.02 |
| 320 | 1.58 | 92 → 165 K | 0.88 | 0.78 | 0.89 |

These agree to within about ±25%. Macario et al. note that their calibration probably makes their sensitivity an underestimate, and that at 70 MHz their XX is about 2× worse than YY.

### 4.4 Summary

| Quantity | Script column | Sky noise | Matches |
|---|---|---|---|
| Element SEFD$_I$ (one antenna) | `sefd_i_el` | Element-beam $T_{\mathrm{ant}}$ | Sutinjo prediction from autos ($\rho \approx 1.00$) |
| Image-noise element/256 | `sefd_i_el_pred` × `rho_i_unc` / 256 | Image noise, uncorrelated model | Macario et al. (2022), 0.89–1.37× |
| Element/256 | `sefd_i_el` / 256 | Uncorrelated between elements | Lower bound on station SEFD |
| Station SEFD$_I$ (256, beamformed, zenith) | `sefd_i_st` | Station-beam $T_{\mathrm{ant}}$ | Senscalc at matched LST, 0.77–0.96× |

## 5. Caveats and open items

- **Absolute scale:** tied to GSM16 through diffuse-sky calibration with the peak-normalised `ska_low` AEP.
- **Noise correlated between antennas:** correlated sky is modelled (Sect. 2.4). Mutual coupling of receiver noise is not; it would show as $\rho > 1$ in both I and V.
- **Q, U frame:** SEFD$_Q$ and SEFD$_U$ are in the station Ludwig-3 frame (Sect. 2.1). A celestial-frame value needs the parallactic rotation per pixel.
- **Beam model:** $\mathbf{M}_S$ uses the Jones AEP, while $T_{\mathrm{ant}}$, $\Omega_e$ and the calibration use the coherency AEP. Their $\mathbf{J}\mathbf{J}^H$ agree to within 2.5% of peak (worst at 50 MHz).
- **Quantisation efficiency:** the script uses $M = \eta\,\Delta\nu\,\tau$. The $\sigma \propto 1/\eta$ convention would change SEFD by about 2%.

## 6. Output columns

`{station}_sensitivity_i.csv` (Stokes I, zenith):

| Column | Meaning |
|---|---|
| `rho_i`, `rho_i_err` | Measured / predicted Stokes-I noise in the difference image, $\theta < 45^\circ$, correlated sky included; jackknife error (Sects. 2.3–2.5) |
| `rho_q`, `rho_u`, `rho_v` (and `_err`) | Same for Stokes Q, U and V, with $\mathbf{M}_S$ from the Jones AEP |
| `rho_i_unc` | $\rho_I$ against the uncorrelated prediction (Sect. 2.2), for reference |
| `rho` | $\tfrac{1}{2}(\rho_I + \rho_V)$, the scale applied to the predictions |
| `rel_err` | Relative error per channel: $\sqrt{(\tfrac{1}{2}\sqrt{\epsilon_I^2 + \epsilon_V^2})^2 + (\tfrac{1}{2}|\rho_I - \rho_V|)^2}/\rho$. Excludes the flux scale (Sect. 2.5) |
| `sefd_i_el_pred`, `sefd_i_el_sky` | Predicted element SEFD$_I$ from the calibrated autos, and from simulated autos (sky only) [Jy] |
| `sefd_i_el`, `sefd_i_el_err` | Measured element SEFD$_I$ = `rho` × `sefd_i_el_pred`, and its error [Jy] |
| `a_over_t_i_el`, `a_over_t_i_el_err` | Element $(A/T)_I = \sqrt{2}k/\mathrm{SEFD}_{I,\mathrm{el}}$ [m² K⁻¹] |
| `sefd_i_st_pred` | Predicted station SEFD$_I$, 256 antennas, beamformed at zenith (Sect. 3) [Jy] |
| `sefd_i_st`, `sefd_i_st_err` | Measured station SEFD$_I$ = `rho` × `sefd_i_st_pred` [Jy] |
| `a_over_t_i_st`, `a_over_t_i_st_err` | Station $(A/T)_I$ [m² K⁻¹] |
| `sefd_{q,u,v}_el`, `sefd_{q,u,v}_st` | Measured element and station SEFD$_S$ = `rho` × Sutinjo prediction, Ludwig-3 frame for Q and U [Jy] |
| `sefd_{i,q,u,v}_el_diag`, `sefd_{i,q,u,v}_st_diag` | Same, $45^\circ$ from zenith on the X–Y diagonal; the station is beamformed towards that direction (Sect. 3) [Jy] |

`{station}_sensitivity_{xx,yy}.csv` (per polarisation, zenith):

| Column | Meaning |
|---|---|
| `T_ant`, `T_sys_auto`, `T_rx` | Element antenna temperature (sim), system temperature (calibrated autos), and their difference [K] |
| `sigma` | Robust std of the difference image within 45° of zenith, divided by $\sqrt{2}$ [Jy/beam, Convention-½] |
| `sefd_el_img` | Element SEFD$_p$ from image noise, $2\sigma\sqrt{M}N$ [Jy] |
| `sefd_el` | Element SEFD$_p = 2\bar{P}_p/\hat{B}_{pp}$ from the autos [Jy] |
| `sefd_st` | Station SEFD$_p = 2P_{\mathrm{st},p}/\hat{B}_{pp}$ [Jy] |
| `a_over_t_el`, `a_over_t_st` | $2k/\mathrm{SEFD}_p$, element and station [m² K⁻¹] |
| `c_bar` | Mean cross-correlation $\bar{C}_p$, the sky correlated between elements [Jy] |
| `omega_e`, `a_eff_el` | Element solid angle of the peak-normalised AEP [sr], and $\lambda^2/\Omega_e$ [m²] |

The dotted lines on `all_sefd_i_st.png` are `sefd_i_el / 256`, 256 elements with no sky noise correlated between them. The value comparable to Macario et al. (2022) is `sefd_i_el_pred` × `rho_i_unc` / 256 (Sect. 4.3). The hkl file also holds `rho_map`, `sefd_i_el_map` and `sefd_ratio_map` (predicted element SEFD$_{Q,U,V}$/SEFD$_I$, as in Sutinjo et al. Figs. 2–4) on coarse HEALPix (NSIDE 8), `obs_lst_h`, and the observation times.

Usage:

```bash
python -m mccs_data_qa.reports.station_sensitivity --datadir DIR --eb 'eb-t0001-20260505-0000*' \
    --caldir calibration/sun-gsm16_leakage_aep-harp --outdir outputs/sefd-20260505 \
    --senscalc-table <ska-ost-senscalc>/src/ska_ost_senscalc/static/lookups/ska_station_sensitivity_AAVS2.h5
```

## References

- Carozzi, T. D., direction-dependent Jones estimation from all-sky images, Astronomy and Computing, doi:10.1016/j.ascom.2014.11.002.
- Carozzi, T. D. & Woan, G. (2011), IXR polarimetric figure of merit.
- Haslam, C. G. T. et al. (1982), 408 MHz all-sky survey.
- Macario, G. et al. (2022), JATIS 8, 011014; arXiv:2109.11983. *Characterization of the SKA1-Low prototype station AAVS2.*
- Smirnov, O. M. (2011), A&A 527, A106; arXiv:1101.1764. *Revisiting the RIME I*, Sect. 7.2.
- Sokolowski, M. et al. (2022), PASA; arXiv:2204.05873. *What is the SKA-Low sensitivity for your favourite radio source?*
- Sutinjo, A. T. et al. (2021), A&A 646, A143. SEFD of a dual-polarised interferometer (Paper I).
- Sutinjo, A. T., Kovaleva, M. & Xu, Y. (2022a), PASP 134, 014502. SEFD of a polarimetric tripole interferometer.
- Sutinjo, A. T. et al. (2022b), A&A 660, A134. SEFD of phased arrays (Paper III).
- Sutinjo, A. T., Ung, D. C. X. & Sokolowski, M. (2022), A&A 664, A102. SEFD of Stokes I, Q, U, V.
- Zheng, H. et al. (2017), MNRAS 464, 3486. GSM2016.
- SPRTS-516: quantisation efficiency $\eta = 0.962$.
