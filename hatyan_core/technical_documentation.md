# Technical documentation — hatyan_core

## Introduction

Tidal water levels at a fixed location can be described as a superposition of sinusoidal oscillations, each driven by a specific astronomical forcing. The method of **harmonic tidal analysis** decomposes an observed record $h(t)$ into a finite set of $N$ constituents, each characterised by an angular frequency $\omega_i$, an amplitude $A_i$, and a phase $\phi_i$. The model takes the form

$$h(t) = \sum_{i=1}^{N} f_i(t)\, A_i \cos\!\bigl(\omega_i\,\Delta t + v_{0,i} + u_i(t) - \phi_i\bigr),$$

where $\Delta t$ is time elapsed since a reference epoch, $v_{0,i}$ is the astronomical argument at the start of the period (the initial phase of the forcing), and $f_i(t)$ and $u_i(t)$ are the **nodal correction factor** and **nodal phase correction**, respectively, which account for the slow 18.6-year modulation of the lunar orbit. The frequencies $\omega_i$ and the nodal corrections are fully determined by celestial mechanics; only the amplitudes $A_i$ and phases $\phi_i$ are estimated from observations by least-squares regression. This package implements two variants of the standard approach: the **Schureman (1941)** [[1]](#ref1) method, which derives nodal corrections analytically from astronomical constants, and the **Foreman (2004)** [[2]](#ref2) method, which obtains them from tabulated satellite data.

## Doodson numbers and tidal frequencies

The existence of discrete tidal frequencies with well-defined periods is a direct consequence of the periodic nature of the Earth–Moon–Sun system. The tide-generating force arises from the difference between the gravitational attraction of the Moon (or Sun) and the centrifugal force associated with the Earth's orbital motion around their common centre of mass. Expanding this force potential in a series of spherical harmonics, and expressing the orbital positions of the Moon and Sun in terms of their slowly varying Keplerian elements — mean longitude, longitude of perigee, longitude of the ascending node, and so on — yields a sum of cosine terms whose frequencies are integer linear combinations of a small set of fundamental astronomical angular speeds [[4]](#ref4). This is the mathematical origin of the Doodson number system. The leading harmonic degree (degree 2 of the Legendre expansion) produces three distinct groups of constituents according to how the tidal potential depends on latitude and longitude: a **long-period** group ($n_1 = 0$) from the zonal term, a **diurnal** group ($n_1 = 1$) from the tesseral term, and a **semi-diurnal** group ($n_1 = 2$) from the sectoral term. Higher degrees give rise to the shallow-water overtides.

The letter codes in constituent names reflect the astronomical body and the species. **M** (from Latin *Luna*) denotes the Moon, **S** denotes the Sun, **K** indicates a luni-solar constituent where lunar and solar frequencies nearly coincide, **N** denotes a lunar elliptic constituent, and **O** the principal lunar diurnal. The trailing digit gives the species: **1** for diurnal, **2** for semi-diurnal, **4** for quarter-diurnal, and so on. Thus **M2** is the principal lunar semi-diurnal constituent (two cycles per lunar day, period ≈ 12.42 hr), **S2** is the principal solar semi-diurnal (period exactly 12.00 hr), **K1** is the luni-solar declinational diurnal (period ≈ 23.93 hr), and **M4** is the first shallow-water overtide of M2 (period ≈ 6.21 hr).

The frequency $\omega_i$ of each tidal constituent is expressed as an integer linear combination of six slowly varying astronomical angular speeds. Following Doodson (1921), these six fundamental arguments are:

| Argument | Symbol | Physical meaning | Angular speed (°/hr) | Period |
|---|---|---|---|---|
| Mean lunar time | $\tau$ | Hour angle of the mean Moon | 14.4921 | 24.84 hr |
| Mean lunar longitude | $s$ | Mean longitude of the Moon | 0.5490 | 27.32 days |
| Mean solar longitude | $h$ | Mean longitude of the Sun | 0.0411 | 365.24 days |
| Lunar perigee longitude | $p$ | Longitude of the lunar perigee | 0.0046 | 8.85 yr |
| Lunar node longitude | $N'$ | Negative longitude of the ascending lunar node | 0.0022 | 18.61 yr |
| Solar perigee longitude | $p_1$ | Longitude of the solar perigee (perihelion) | 0.000002 | ~21,000 yr |

The angular frequency of constituent $i$ is then

$$\omega_i = n_1\,\omega_\tau + n_2\,\omega_s + n_3\,\omega_h + n_4\,\omega_p + n_5\,\omega_{N'} + n_6\,\omega_{p_1},$$

where $(n_1, n_2, n_3, n_4, n_5, n_6)$ is the integer **Doodson number** of that constituent. The leading coefficient $n_1$ is called the **species**: $n_1 = 0$ for long-period tides, $n_1 = 1$ for diurnal tides, $n_1 = 2$ for semi-diurnal tides, and higher values for shallow-water overtides. The first argument $\tau$ is related to the solar hour angle $T$ (advancing at exactly 15°/hr) by $\tau = T + h - s$, which allows the six-argument Doodson representation to be converted to an equivalent solar-time form used internally in this package.

The table below lists the Doodson numbers and nominal periods for the constituents most commonly encountered in tidal analysis. Shallow-water constituents (M4, MS4, M6) do not have independent Doodson numbers; their frequency and phase are derived from the linear combination of harmonic constituents indicated in the last column.

| Constituent | $n_1$ | $n_2$ | $n_3$ | $n_4$ | $n_5$ | $n_6$ | Period (hr) | Description |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| SA   |  0 |  0 |  1 |  0 |  0 | −1 | 8766.2 | Solar annual |
| SSA  |  0 |  0 |  2 |  0 |  0 |  0 | 4382.9 | Solar semi-annual |
| Mm   |  0 |  1 |  0 | −1 |  0 |  0 |  661.3 | Lunar monthly |
| MSf  |  0 |  2 | −2 |  0 |  0 |  0 |  354.4 | Lunisolar synodic fortnightly |
| Mf   |  0 |  2 |  0 |  0 |  0 |  0 |  327.9 | Lunar fortnightly |
| Q1   |  1 | −2 |  0 |  1 |  0 |  0 |   26.87 | Larger elliptic diurnal |
| O1   |  1 | −1 |  0 |  0 |  0 |  0 |   25.82 | Principal lunar diurnal |
| P1   |  1 |  1 | −2 |  0 |  0 |  0 |   24.07 | Principal solar diurnal |
| K1   |  1 |  1 |  0 |  0 |  0 |  0 |   23.93 | Luni-solar declinational diurnal |
| N2   |  2 | −1 |  0 |  1 |  0 |  0 |   12.66 | Larger lunar elliptic semi-diurnal |
| M2   |  2 |  0 |  0 |  0 |  0 |  0 |   12.42 | Principal lunar semi-diurnal |
| S2   |  2 |  2 | −2 |  0 |  0 |  0 |   12.00 | Principal solar semi-diurnal |
| K2   |  2 |  2 |  0 |  0 |  0 |  0 |   11.97 | Luni-solar declinational semi-diurnal |
| M4   |  — |  — |  — |  — |  — |  — |    6.21 | Shallow water overtide: $2 \times \mathrm{M2}$ |
| MS4  |  — |  — |  — |  — |  — |  — |    6.10 | Shallow water compound: $\mathrm{M2} + \mathrm{S2}$ |
| M6   |  — |  — |  — |  — |  — |  — |    4.14 | Shallow water overtide: $3 \times \mathrm{M2}$ |

Note that $N'$ (column $n_5$) is zero for all principal constituents in this table. The 18.61-year modulation associated with $\omega_{N'}$ enters the tidal signal not through the carrier frequency but through the slowly varying nodal factors $f_i$ and $u_i$, which are covered in a later section.

The constituent names, speeds, and XDO codes in this section follow the standard list maintained by the IHO Tide, Water Level and Current Working Group (TWCWG) [[3]](#ref3).

### Shallow-water constituents: overtides and compound tides

In the open ocean the tidal response is well described by the linear superposition of astronomically forced constituents. In shallow coastal waters, estuaries, and tidal channels, however, the governing hydrodynamic equations become nonlinear. Nonlinear terms arise from several sources: the advective acceleration $u\,\partial u/\partial x$ in the momentum equation, the depth-dependent wave celerity in the continuity equation (the tidal wave propagates faster at high water than at low water), and quadratic bottom friction. These nonlinearities act as a source of energy at frequencies that are integer sums and differences of the frequencies already present in the signal, generating a family of **overtides** (integer multiples of a single constituent's frequency) and **compound tides** (combinations of two or more different constituents).

The mechanism can be illustrated with the product of the M2 and S2 signals. A nonlinear term proportional to the product of two tidal components gives

$$\cos\!\bigl(\omega_{\mathrm{M2}}\,t - \phi_{\mathrm{M2}}\bigr)\cdot\cos\!\bigl(\omega_{\mathrm{S2}}\,t - \phi_{\mathrm{S2}}\bigr) = \tfrac{1}{2}\cos\!\bigl((\omega_{\mathrm{M2}}+\omega_{\mathrm{S2}})\,t - (\phi_{\mathrm{M2}}+\phi_{\mathrm{S2}})\bigr) + \tfrac{1}{2}\cos\!\bigl((\omega_{\mathrm{M2}}-\omega_{\mathrm{S2}})\,t - (\phi_{\mathrm{M2}}-\phi_{\mathrm{S2}})\bigr),$$

where the product-to-sum identity has been applied. The first term oscillates at the **sum frequency** $\omega_{\mathrm{MS4}} = \omega_{\mathrm{M2}} + \omega_{\mathrm{S2}}$ with **sum phase** $\phi_{\mathrm{MS4}} = \phi_{\mathrm{M2}} + \phi_{\mathrm{S2}}$; this is the compound constituent MS4 (period ≈ 6.10 hr). The second term oscillates at the **difference frequency** $\omega_{\mathrm{M2}} - \omega_{\mathrm{S2}}$, corresponding to a long-period modulation with the spring–neap beat period (≈ 14.8 days). Similarly, a term proportional to $\cos^2(\omega_{\mathrm{M2}}\,t - \phi_{\mathrm{M2}})$ expands to a constant plus $\cos(2\omega_{\mathrm{M2}}\,t - 2\phi_{\mathrm{M2}})$, generating the overtide M4 at twice the M2 frequency with twice its phase. The general rule is:

$$\omega_{\mathrm{compound}} = \sum_k n_k\,\omega_k, \qquad \phi_{\mathrm{compound}} = \sum_k n_k\,\phi_k,$$

where $n_k$ are signed integers. Because shallow-water constituents have no independent astronomical forcing their Doodson numbers are not defined; instead their frequencies and phases are fully determined by those of their parent harmonic constituents through the relations above. The nodal corrections follow the same logic: $f_{\mathrm{compound}} = \prod_k f_k^{|n_k|}$ and $u_{\mathrm{compound}} = \sum_k n_k\,u_k$.

## Harmonic analysis

In practice the amplitudes $A_i$ and phases $\phi_i$ in the tidal model are unknown and must be estimated from an observed water-level record, typically a time series of measurements from a tide gauge. Given observations $h(t_j)$ at times $t_1, \ldots, t_M$, and a pre-selected set of $N$ constituents with known frequencies and nodal corrections, the goal of **harmonic analysis** is to find the values of $A_i$ and $\phi_i$ that minimise the misfit between the model and the observations. Because the amplitude and phase enter the cosine through the nonlinear combination $A_i\cos(\cdots - \phi_i)$, direct optimisation would require a nonlinear solver. The standard approach avoids this by applying the trigonometric identity

$$f_i(t)\,A_i\cos\!\bigl(\theta_i(t) - \phi_i\bigr) = f_i(t)\!\left[a_i\cos\theta_i(t) + b_i\sin\theta_i(t)\right],$$

where $\theta_i(t) = \omega_i\,\Delta t + v_{0,i} + u_i(t)$ collects all known astronomical arguments and $a_i = A_i\cos\phi_i$, $b_i = A_i\sin\phi_i$ are the new unknowns. The full model then becomes

$$h(t) = \sum_{i=1}^{N} f_i(t)\!\left[a_i\cos\theta_i(t) + b_i\sin\theta_i(t)\right],$$

which is **linear** in the coefficients $a_i$ and $b_i$. This allows the analysis to be cast as an ordinary least-squares problem and solved efficiently with standard linear algebra. Once the coefficients are obtained, the amplitude and phase are recovered as

$$A_i = \sqrt{a_i^2 + b_i^2}, \qquad \phi_i = \arctan\!\left(\frac{b_i}{a_i}\right),$$

where the four-quadrant arctangent is used to place $\phi_i$ in the correct half-plane.

### Least-squares matrix formulation

Stack the $M$ observations into a column vector $\mathbf{h} = [h(t_1), \ldots, h(t_M)]^T$ and collect the $2N$ unknowns into $\mathbf{x} = [a_1, b_1, \ldots, a_N, b_N]^T$. The linear model can then be written compactly as

$$\mathbf{A}\mathbf{x} = \mathbf{h},$$

where $\mathbf{A}$ is the $M \times 2N$ **design matrix** whose $j$-th row contains the nodal-corrected basis functions evaluated at observation time $t_j$:

$$\mathbf{A}_{j,\,2i-1} = f_i(t_j)\cos\theta_i(t_j), \qquad \mathbf{A}_{j,\,2i} = f_i(t_j)\sin\theta_i(t_j).$$

For a typical tide-gauge record the system is highly overdetermined ($M \gg 2N$), so there is no exact solution and we seek instead the coefficient vector that minimises the sum of squared residuals $\|\mathbf{A}\mathbf{x} - \mathbf{h}\|^2$. Differentiating with respect to $\mathbf{x}$ and setting the gradient to zero yields the **normal equations**

$$\mathbf{A}^T\mathbf{A}\,\mathbf{x} = \mathbf{A}^T\mathbf{h},$$

with solution

$$\mathbf{x} = (\mathbf{A}^T\mathbf{A})^{-1}\mathbf{A}^T\mathbf{h}.$$

The $2N \times 2N$ matrix $\mathbf{A}^T\mathbf{A}$ is called the **normal matrix**. Because tidal basis functions at different frequencies are nearly orthogonal over long records — their inner products integrate to approximately zero — the normal matrix is close to diagonal and well conditioned, provided all included constituents can be distinguished from one another by the record length. This last condition is formalised by the Rayleigh criterion below.

### Resolvability and the Rayleigh criterion

Not all constituents can be estimated from an arbitrarily short record. Two constituents with frequencies $\omega_i$ and $\omega_j$ can only be distinguished if the observation period $T$ is long enough for their signals to accumulate a measurable phase difference. The **Rayleigh criterion** states that two constituents are resolvable when

$$T \geq \frac{1}{|\omega_i - \omega_j|},$$

where frequencies are expressed in cycles per unit time so that the right-hand side is the **beat period** — the time it takes for the two oscillations to complete one full cycle relative to each other. For constituent pairs that are close in frequency the beat period can be very long, requiring correspondingly long records. For example, M2 and S2 differ by approximately 0.00282 cycles/hr, giving a beat period of roughly 354 hr (≈ 15 days, the spring–neap cycle); separating them therefore requires at least two weeks of data. M2 and N2 are closer still (beat period ≈ 662 hr ≈ 28 days), requiring about a month, while the pairs K1/P1 and K2/S2 have beat periods of approximately 182 days, so six months of observations are needed to resolve them. As a rule of thumb, a **one-year record** is sufficient to separate all constituents in common use. In practice the Rayleigh criterion is sometimes applied with a safety factor greater than one, and constituents that cannot be resolved from the available record length are either omitted or their amplitude ratio is fixed from theory (inference).

### Nodal corrections

A particularly important application of this inference principle concerns the **nodal satellites** — groups of minor constituents that cluster around each major constituent at frequency offsets corresponding to the 18.61-year period of the lunar node. These satellites arise because the Moon's orbital plane precesses slowly, modulating the tidal potential at a rate of one full cycle every 18.61 years. Resolving the satellites individually from the main constituent would require records spanning many years, far beyond what is typically available. Instead, it is assumed that the ocean responds linearly to the tidal potential and in essentially the same manner to the main constituent and its nearby satellites. Under this assumption the amplitude ratio and phase difference between each satellite and its parent constituent are the same in the ocean response as they are in the theoretical tidal potential. The combined effect of the main constituent and all its satellites can therefore be written as a single oscillation at the main frequency but with a slowly varying amplitude factor $f_i(t)$ and a slowly varying phase correction $u_i(t)$ — exactly the nodal correction factors that appear in the prediction formula in the introduction. The two methods differ in how these corrections are computed: the **Schureman** method derives $f_i$ and $u_i$ analytically from the astronomical constants of the lunar orbit [[1]](#ref1), while the **Foreman** method computes them by explicitly summing the contributions of tabulated satellite terms, including a small latitude-dependent correction that accounts for differences in the ocean's response between nodal species [[2]](#ref2).

### Satellite altimetry and tidal aliasing

Tide gauges sample the sea surface continuously or at sub-hourly intervals, but satellite altimeters measure sea-surface height only when the satellite passes overhead. Because a satellite follows a fixed ground track, a given ocean location is revisited only once per **repeat cycle** $T_r$ — typically of the order of 10 days for dedicated altimetry missions. This sparse, irregular sampling introduces a fundamental limitation: the **Nyquist frequency** $f_N = 1/(2T_r)$ is far below the frequencies of all major tidal constituents. For a 10-day repeat cycle $f_N \approx 0.05$ cycles/day, corresponding to a Nyquist period of 20 days, while semi-diurnal constituents oscillate at periods of roughly 12 hours. All tidal constituents are therefore severely **aliased**.

Aliasing occurs because a sinusoid at true frequency $f$ sampled at interval $T_r$ is mathematically indistinguishable from sinusoids at frequencies $f + k/T_r$ for any integer $k$. The observed alias frequency is obtained by subtracting the nearest integer multiple of the sampling rate $1/T_r$ from $f$ and taking the absolute value:

$$f_{\mathrm{alias}} = \left| f - \frac{\mathrm{round}(f\,T_r)}{T_r} \right|.$$

The alias period $T_{\mathrm{alias}} = 1/f_{\mathrm{alias}}$ is the period at which the constituent appears to oscillate in the altimeter data. For the TOPEX/Poseidon and Jason series of satellites ($T_r \approx 9.916$ days), M2 aliases to a period of approximately 62 days and S2 to approximately 59 days — both well above the Nyquist period and in principle resolvable from a multi-year altimeter record. The repeat period is by design not a rational multiple of the tidal periods, so each constituent aliases to a distinct frequency. This is not accidental: the orbit was specifically chosen so that the aliases of the dominant tidal constituents are well separated in frequency, allowing harmonic analysis to be applied to the time series of sea-surface heights at each ground-track location in exactly the same way as to a tide-gauge record.

The critical subtlety is that the **Rayleigh criterion now applies to the alias frequencies**, not to the original tidal frequencies. Two constituents may be well separated in their true frequencies yet have aliases that are nearly identical, making them indistinguishable from the altimeter data regardless of record length. Conversely, two constituents whose true frequencies are too close to resolve from a short gauge record (such as K1 and P1, which require roughly six months) may alias to frequencies that are far enough apart to be separated from altimetry. The resolvability of constituents from satellite altimetry therefore depends entirely on the relationship between the orbital repeat period and the tidal frequencies, and the choice of repeat cycle is a key design constraint for missions intended to observe the global ocean tide.

## References

<a name="ref1">[1]</a> Schureman, P. (1941). *Manual of Harmonic Analysis and Prediction of Tides*. Special Publication No. 98. U.S. Coast and Geodetic Survey, U.S. Government Printing Office, Washington, D.C. Available at: [https://tidesandcurrents.noaa.gov/publications/SpecialPubNo98.pdf](https://tidesandcurrents.noaa.gov/publications/SpecialPubNo98.pdf)

<a name="ref2">[2]</a> Foreman, M.G.G. (2004). *Manual for Tidal Heights Analysis and Prediction*. Pacific Marine Science Report 77-10. Institute of Ocean Sciences, Patricia Bay, Victoria, B.C. (originally published 1977; revised 2004). Available at: [https://waves-vagues.dfo-mpo.gc.ca/Library/54866.pdf](https://waves-vagues.dfo-mpo.gc.ca/Library/54866.pdf)

<a name="ref3">[3]</a> Simon, B. (SHOM) and Page, J. (UKHO) (2017). *Tidal Constituents*. IHO Tide, Water Level and Current Working Group (TWCWG), updated 8 May 2017. Available at: [https://iho.int/mtg_docs/com_wg/IHOTC/IHOTC_Misc/TWCWG_Constituent_list.pdf](https://iho.int/mtg_docs/com_wg/IHOTC/IHOTC_Misc/TWCWG_Constituent_list.pdf)

<a name="ref4">[4]</a> Pugh, D., Woodworth, P. L., & Woodworth, P. (2014). *Sea-Level Science: Understanding Tides, Surges, Tsunamis and Mean Sea-Level Changes*. Cambridge University Press.
