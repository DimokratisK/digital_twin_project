# CFD Boundary Conditions — Literature Review Summary

This document summarises the key parameters extracted from published studies
on left atrium (LA) computational fluid dynamics, used to validate and
configure the boundary conditions in our OpenFOAM pipeline.

---

## References

| ID  | Authors | Title | Journal | Year | DOI |
|-----|---------|-------|---------|------|-----|
| [1] | Otani T, Al-Issa A, Pourmorteza A, McVeigh ER, Wada S, Ashikaga H | A Computational Framework for Personalized Blood Flow Analysis in the Human Left Atrium | Ann Biomed Eng 44(11):3284-3294 | 2016 | 10.1007/s10439-016-1590-x |
| [2] | Koizumi R, Funamoto K, Hayase T, Kanke Y, Shibata M, Shiraishi Y, Yambe T | Numerical analysis of hemodynamic changes in the left atrium due to atrial fibrillation | J Biomech 48(3):472-478 | 2015 | 10.1016/S0021-9290(14)00679-4 |
| [3] | Masci A, Alessandrini M, Forti D, Menghini F, Dede L, Tommasi C, Quarteroni A, Corsi C | A Patient-Specific Computational Fluid Dynamics Model of the Left Atrium in Atrial Fibrillation: Development and Initial Evaluation | FIMH 2017, LNCS 10263, pp. 392-400 | 2017 | 10.1007/978-3-319-59448-4_37 |
| [4] | Balzotti C, Siena P, Girfoglio M, Stabile G, Duenas-Pamplona J, Sierra-Pallares J, Amat-Santos I, Rozza G | A reduced order model formulation for left atrium flow: an atrial fibrillation case | Biomech Model Mechanobiol 23:1411-1429 | 2024 | 10.1007/s10237-024-01847-1 |
| [5] | Nguyen TD, Do TCN, Pham TH, Pham VS | Hemodynamics in Coronary Arteries: using Open-Source Software-SimVascular | Vietnam J Sci Technol 62(3) | 2024 | 10.15625/2525-2518/18503 |

---

## Blood Properties

| Parameter | [1] Otani | [2] Koizumi | [3] Masci | [4] Balzotti | [5] Nguyen | **Adopted** |
|-----------|-----------|-------------|-----------|--------------|------------|------------|
| Density (kg/m3) | 1050 | 1050 | not stated | 1050 | 1060 | **1050** |
| Dynamic viscosity (Pa.s) | 0.0035 | 0.0035 | not stated | 0.0035 | 0.004 | **0.0035** |
| Rheology model | Newtonian | Newtonian | — | Newtonian + Casson | Newtonian | **Newtonian** |
| Flow regime | Laminar (Re~1100-2400) | Laminar | Laminar | Laminar (DNS) | Newtonian | **Laminar** |

Note: [5] studies coronary arteries (not LA) and uses slightly different blood properties.
The adopted values reflect the consensus from the four LA-specific studies [1]-[4].

---

## Solver Configuration

| Parameter | [1] Otani | [2] Koizumi | [3] Masci | [4] Balzotti | **Adopted** |
|-----------|-----------|-------------|-----------|--------------|------------|
| Solver | OpenFOAM 2.3.1 | FLUENT 6.3 | LifeV (FE) | OpenFOAM 2206 | **OpenFOAM** |
| Wall treatment | Moving (4D CT) | Moving (prescribed) | Moving (ALE) | **Rigid** | **Rigid** |
| Time step | 1e-4 s | 3.3e-3 s | not stated | 0.01 s (max, Co<0.8) | **adaptive, Co<0.5** |
| Cardiac cycles simulated | 5 (last 3 analysed) | 4 (last 1 analysed) | 3 (last 1 analysed) | 4 (last 1 analysed) | **5 (last 3 analysed)** |
| Convergence criterion | vel<1e-5, p<1e-8 | residual 1e-4 | — | — | **1e-5** |

Rigid-wall justification: Balzotti et al. [4] (citing Garcia-Isla 2018 and Duenas-Pamplona 2021)
showed that rigid and moving wall simulations produce very similar flow patterns and residence
time distributions in cases of impaired atrial function (AF), especially when both the reservoir
and booster functions are decreased.

---

## Mesh

| Parameter | [1] Otani | [2] Koizumi | [3] Masci | [4] Balzotti | **Adopted** |
|-----------|-----------|-------------|-----------|--------------|------------|
| Element type | Tetrahedral | Tetrahedral | Tetrahedral | Tetrahedral | **Hex-dominant (snappyHexMesh)** |
| Element count | 367k-549k | 149k | ~1.04M | 543k | **~500k-1.5M** |
| Base element size | 1.5 mm | not stated | not stated | GCI-verified | **1.5 mm target** |

---

## Boundary Conditions

| Parameter | [1] Otani | [2] Koizumi | [3] Masci | [4] Balzotti | **Adopted** |
|-----------|-----------|-------------|-----------|--------------|------------|
| BC approach | Zero-gradient pressure at PVs; MV flow from LV volume change | Constant pressure (PV 10 mmHg, MV 8 mmHg during diastole) | Mass-balance: PV flows from MV Doppler + volume conservation | MV Doppler velocity scaled by factor; equal PV flow split | **Pulsatile velocity inlet at PVs; pressure outlet at MV** |
| PV flow split | From flow direction | Not individually specified | Proportional to PV cross-sectional area | **Equal across 4 PVs** | **Equal (default) or area-proportional** |
| MV treatment | Closed during systole (wall), open during diastole | On/off switch | Natural BC with backflow penalisation | Velocity waveform | **Zero velocity during systole, pulsatile during diastole** |

---

## Hemodynamic Reference Values

| Parameter | [2] Koizumi (healthy) | [4] Balzotti | Notes |
|-----------|-----------------------|--------------|-------|
| TAWSS on LAA (Pa) | 0.31 | ~1e-4 to 3e-4 | Balzotti reports on LAA surface only; Koizumi reports spatial average |
| OSI on LAA | 0.12 | 0 to 0.5 | |
| RRT on LAA (1/Pa) | 6.38 | — | |
| MV E-wave peak velocity (cm/s) | ~60 (from flow ~140 mL/s) | 65-70 [3] | |
| MV A-wave peak velocity (cm/s) | ~43 (from flow ~100 mL/s) | 35-40 [3] | Absent in AF |
| PV peak velocity (cm/s) | — | 20-40 [3] | During systole ~20, during filling ~40 |
| MV peak flow rate (mL/s) | ~140 (E), ~100 (A) | ~230 (f=1.0) [4] | |

---

## Key Takeaways for Our Pipeline

1. **Blood density = 1050 kg/m3** — consensus from [1], [2], [4]. Value 1060 sometimes
   used for coronary studies [5] but 1050 is standard for LA simulations.

2. **Dynamic viscosity = 0.0035 Pa.s** — unanimous across LA studies [1], [2], [4].

3. **Rigid wall is justified for AF** — explicitly validated by [4] citing multiple studies.
   Our static segmentation from nnU-Net provides geometry without wall motion, making
   the rigid-wall approach the natural choice.

4. **Laminar flow** — all studies assume laminar; [4] explicitly tested and found negligible
   difference between turbulent and non-turbulent mean flow fields.

5. **Newtonian rheology is acceptable** — [4] compared Newtonian and Casson models,
   finding similar results for most indices. Non-Newtonian effects are secondary.

6. **Number of cardiac cycles**: 3-5 total, with first 1-3 discarded as transient.
   We simulate 5, discard first 2, analyse last 3 — consistent with [1].

7. **PV flow split**: equal split [4], area-proportional [3], or fixed percentages are
   all used in the literature. Equal split is the simplest justified approach.

8. **Blood age and washout** metrics ([4]) are important for stroke risk assessment
   and should be added to our post-processing pipeline in the future.
