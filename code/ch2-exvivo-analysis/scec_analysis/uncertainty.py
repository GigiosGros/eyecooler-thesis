"""Measurement-uncertainty budget (GUM), absolute and differential.

Absolute: uncertainty of a single temperature reading. Differential: uncertainty
of a within-trace difference such as dT, where static common-mode terms
(calibration reference offset, interchangeability, self-heating, lead resistance,
static amplifier offsets) cancel. Type B terms are rectangular (half-width/sqrt 3)
unless stated; the self-heating residual is one-sided (bound/sqrt 12).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

from .calibration import dT_dV, human_plateau_levels, plateau_levels

_AWG_OHM_PER_M = {20: 0.0334, 22: 0.0531, 24: 0.0842, 26: 0.134, 28: 0.213, 30: 0.338}
_MATERIAL = {"copper": 1.0, "tinned_copper": 1.02, "nickel": 6.9, "steel": 11.0}


def R_ntc(T_C, u):
    return u["R25_ohm"] * np.exp(u["beta_K"] * (1 / (T_C + 273.15) - 1 / (u["T_ref_C"] + 273.15)))


def T_from_R(R, u, beta):
    return 1 / (1 / (u["T_ref_C"] + 273.15) + np.log(R / u["R25_ohm"]) / beta) - 273.15


def _v_out(T_C, u):
    R = R_ntc(T_C, u)
    Rp = u["R_parallel_ohm"]
    Reff = R * Rp / (R + Rp)
    I = (u["Vpos"] - u["Vneg"]) / (u["R_top_ohm"] + Reff + u["R_bottom_ohm"])
    return u["opamp_gain"] * (u["Vpos"] - I * u["R_top_ohm"])


def _sens(T_C, u, d=0.001):
    return 2 * d / (_v_out(T_C + d, u) - _v_out(T_C - d, u))


def plateau_noise(calibration: dict, sh: dict, cfg: dict) -> float:
    """Type A: median over calibration plateaus of 2 x sensitivity x SD."""
    vals = []
    for eye, df in calibration.items():
        for ch in cfg["channels"]["temperature"]:
            if ch not in df.columns or ch not in sh.get(eye, {}):
                continue
            data = df[ch].dropna()
            for c in human_plateau_levels(data, cfg):
                pts = data[(data >= c - 1.0) & (data <= c + 1.0)]
                if len(pts) > 30:
                    vals.append(2 * abs(dT_dV(c, sh[eye][ch])) * float(pts.std(ddof=1)))
    return float(np.median(vals)) if vals else cfg["uncertainty"]["noise_95CI_fallback_C"]


def absolute_budget(u: dict, noise_95: float):
    T = u["cal_temps_C"]
    worst = max(abs(_sens(t, u)) for t in T)
    G = u["opamp_gain"]
    rows = [
        ("Calibrator accuracy", u["well_accuracy_C"] / np.sqrt(3)),
        ("Calibrator stability", u["well_stability_C"] / np.sqrt(3)),
        ("DAQ dc accuracy", u["daq_dc_accuracy_V"] * worst / np.sqrt(3)),
        ("DAQ quantisation", (u["daq_fullscale_V"] / 2 ** u["daq_bits"] / 2) * worst / np.sqrt(3)),
    ]
    T_mid = float(np.mean(T))
    R_mid = R_ntc(T_mid, u)
    R_eff = R_mid * u["R_parallel_ohm"] / (R_mid + u["R_parallel_ohm"])
    R_src = u["R_top_ohm"] * (R_eff + u["R_bottom_ohm"]) / (u["R_top_ohm"] + R_eff + u["R_bottom_ohm"])
    rows += [
        ("Op-amp offset drift", u["OpAmp_Vos_drift_uV_C"] * 1e-6 * u["ambient_delta_C"] * G * worst / np.sqrt(3)),
        ("Op-amp bias current x source R", u["OpAmp_Ibias_max_A"] * R_src * G * worst / np.sqrt(3)),
        ("Op-amp gain temperature coefficient",
         (max(T) - min(T)) * G * np.sqrt(2) * 100e-6 * u["ambient_delta_C"] / np.sqrt(3)),
        ("Sensor interchangeability", u["interchangeability_C"] / np.sqrt(3)),
    ]
    dB = u["beta_K"] * u["beta_tol_pct"] / 100
    beta = max(abs(T_from_R(R_ntc(t, u), u, u["beta_K"] + dB) - T_from_R(R_ntc(t, u), u, u["beta_K"] - dB)) / 2
               for t in T)
    rows.append(("Thermistor beta tolerance", beta / np.sqrt(3)))
    rows.append(("Self-heating residual (one-sided, bath-ceiling bound)",
                 u["selfheat_residual_bound_C"] / np.sqrt(12)))
    rows.append(("Dynamic lag", u["thermistor_tau_s"] * u["max_cooling_rate_C_per_s"] / np.sqrt(3)))
    R_wire = 2 * 0.676 * 6.9 + 2 * _AWG_OHM_PER_M.get(u["wire_gauge_awg"], 0.0842) \
        * _MATERIAL.get(u["wire_material"], 1.0) * u["wire_length_m"]
    R_tot = u["R_top_ohm"] + R_eff + u["R_bottom_ohm"]
    rows.append(("Lead resistance", abs((u["Vpos"] - u["Vneg"]) * (R_eff + u["R_bottom_ohm"])
                                        / R_tot ** 2 * R_wire) * G * worst))
    rows.append(("Voltage noise (Type A, plateau scatter)", noise_95 / 1.96))
    rows.append(("Steinhart-Hart interpolation", u["interpolation_error_C"] / np.sqrt(3)))
    rows.append(("Stem conduction along leads", u["stem_conduction_C"] / np.sqrt(3)))
    df = pd.DataFrame(rows, columns=["source", "u_C"])
    uc = float(np.sqrt((df.loc[df["u_C"] > 1e-6, "u_C"] ** 2).sum()))
    return df, uc, 2 * uc


def differential_budget(u: dict, noise_95: float):
    T = u["cal_temps_C"]
    dB = u["beta_K"] * u["beta_tol_pct"] / 100
    R = R_ntc(float(np.mean(T)), u)
    rows = [
        ("Voltage noise x sqrt 2 (two readings)", np.sqrt(2) * noise_95 / 1.96),
        ("Gain temperature coefficient x span",
         (max(T) - min(T)) * np.sqrt(2) * 100e-6 * u["ambient_delta_C"] / np.sqrt(3)),
        ("Thermistor beta tolerance (slope)",
         abs(T_from_R(R, u, u["beta_K"] + dB) - T_from_R(R, u, u["beta_K"] - dB)) / 2 / np.sqrt(3)),
        ("Steinhart-Hart interpolation", u["interpolation_error_C"] / np.sqrt(3)),
        ("Stem conduction along leads", u["stem_conduction_C"] / np.sqrt(3)),
        ("Bath stability", u["bath_stability_C"] / np.sqrt(3)),
    ]
    df = pd.DataFrame(rows, columns=["source", "u_C"])
    uc = float(np.sqrt((df["u_C"] ** 2).sum()))
    return df, uc, 2 * uc


# ── Budget from measured quantities (uncertainty.method = "measured") ─────────────────────
# The notebook budget above takes its sensitivities from a circuit model whose resistor
# values do not reproduce the calibration data. This budget uses measured quantities
# instead: the calibration slope, the plateau scatter, the drift logged after calibration,
# and the linearity of the 2022 eleven-temperature bench table. Sources are in config.toml.

def plateau_noise_sd(calibration: dict, cfg: dict) -> float:
    """Median SD (degC) of the settled calibration plateaus: samples within
    +/- plateau_window_V of each plateau level (the notebook used +/- 1 V, which takes in
    the ramps between plateaus). A calibration file shared by several eyes counts once."""
    u = cfg["uncertainty"]
    vals, seen = [], set()
    for df in calibration.values():
        for ch in cfg["channels"]["temperature"]:
            if ch not in df.columns:
                continue
            data = df[ch].dropna()
            key = (ch, len(data), round(float(data.iloc[0]), 6), round(float(data.iloc[-1]), 6))
            if key in seen:
                continue
            seen.add(key)
            for c in human_plateau_levels(data, cfg):
                pts = data[(data >= c - u["plateau_window_V"]) & (data <= c + u["plateau_window_V"])]
                if len(pts) > 30:
                    vals.append(u["slope_C_per_V"] * float(pts.std(ddof=1)))
    return float(np.median(vals)) if vals else np.nan


def run_scatter_sd(runs: dict, cfg: dict) -> float:
    """Median over the included tissue traces of the SD of (calibrated - filtered) over the
    final window: the scatter of a single sample during a run (relay ripple included)."""
    w, device, vals = cfg["qc"]["final_window_s"], cfg["channels"]["device"], []
    for r in runs.values():
        t = pd.to_numeric(pd.Series(r.temp.index), errors="coerce").to_numpy()
        m = t >= np.nanmax(t) - w
        for ch in cfg["channels"]["temperature"]:
            if ch in device or r.exclusions.get(ch) or ch not in r.temp.columns:
                continue
            d = (pd.to_numeric(r.temp[ch], errors="coerce").to_numpy() - r.filtered[ch].to_numpy(float))[m]
            if np.isfinite(d).sum() > 30:
                vals.append(float(np.nanstd(d, ddof=1)))
    return float(np.median(vals)) if vals else np.nan


def human_line_terms(calibration: dict, cfg: dict) -> dict:
    """Linear-conversion half-widths of the HUMAN calibration lines
    (2026-10-19), from the calibration files themselves rather than the 2022 bench table:
      abs:   largest miss of a line at its own three plateaus (all channels and sessions);
      small: largest error of a small excursion (dT_small_C) on the primary-endpoint channel,
             taking the true chain between 20.5 and 37 C as that channel's chord plus the
             curvature of the 2022 bench table about its own 20.5-37 C chord (bench_table_*);
      large: largest spread of a device-channel line's three plateau misses.
    A calibration file shared by several eyes counts once."""
    u, t_ref = cfg["uncertainty"], np.asarray(cfg["calibration"]["reference_temps_C"], float)
    bT, bV = np.asarray(u["bench_table_C"], float), np.asarray(u["bench_table_V"], float)
    o = np.argsort(bV)
    b37, b20 = bV[np.argmin(abs(bT - 37))], bV[np.argmin(abs(bT - 20.5))]
    curv = lambda v: PchipInterpolator(bV[o], bT[o])(v) - (20.5 + (v - b20) * 16.5 / (b37 - b20))
    i37, i20 = int(np.argmin(abs(t_ref - 37))), int(np.argmin(abs(t_ref - 20.5)))
    lo, hi = sorted(u["dT_small_C"])
    out, seen = dict(abs=0.0, small=0.0, large=0.0), set()
    for df in calibration.values():
        for ch in cfg["channels"]["temperature"]:
            if ch not in df.columns:
                continue
            data = df[ch].dropna()
            key = (ch, len(data), round(float(data.iloc[0]), 6), round(float(data.iloc[-1]), 6))
            if key in seen:
                continue
            seen.add(key)
            lv = human_plateau_levels(data, cfg)
            if len(lv) != 3:
                continue
            m, c = np.polyfit(np.asarray(lv, float), t_ref, 1)
            res = m * np.asarray(lv, float) + c - t_ref
            out["abs"] = max(out["abs"], float(np.abs(res).max()))
            if ch == cfg["channels"]["primary_endpoint"]:
                v = lambda T: lv[i20] + (T - 20.5) * (lv[i37] - lv[i20]) / 16.5
                true_dT = (hi - lo) + curv(v(hi)) - curv(v(lo))
                out["small"] = max(out["small"], float(abs(m * (v(hi) - v(lo)) - true_dT)))
            if ch in cfg["channels"]["device"]:
                out["large"] = max(out["large"], float(res.max() - res.min()))
    return out


def _power_ratio(T_C, u):
    """Thermistor power up to a constant, P ~ R / (R + g)^2, g = resistance the thermistor
    sees (fitted from the calibration plateaus; no circuit values needed)."""
    R = R_ntc(T_C, u)
    return R / (R + u["source_resistance_ohm"]) ** 2


def measured_absolute_budget(u: dict, noise_sd: float):
    """Single temperature reading. Self-heating: the reading can only be LOW (the bead runs
    cooler in tissue than in the calibration block), by between 0 and the bound; no
    correction is applied, so u = bound / sqrt(3) (rectangular on [0, bound], uncorrected)."""
    s3 = np.sqrt(3)
    T_hot = max(u["cal_temps_C"])
    dRdT = R_ntc(T_hot, u) * u["beta_K"] / (T_hot + 273.15) ** 2
    rows = [
        ("Calibrator accuracy", u["well_accuracy_C"] / s3),
        ("Calibrator stability", u["well_stability_C"] / s3),
        ("Calibrator block uniformity", u["well_uniformity_C"] / s3),
        ("DAQ dc accuracy", u["daq_dc_accuracy_V"] * u["slope_C_per_V"] / s3),
        ("Drift after calibration (measured)", u["drift_after_calibration_V"] * u["slope_C_per_V"] / s3),
        ("Sensor interchangeability", u["interchangeability_C"] / s3),
        ("Self-heating (one-sided: reading low; not corrected)", u["selfheat_residual_bound_C"] / s3),
        ("Dynamic lag", u["thermistor_tau_s"] * u.get("max_cooling_rate_measured_C_per_s", u["max_cooling_rate_C_per_s"]) / s3),
        ("Lead resistance", u["sensor_lead_ohm"] / dRdT / s3),
        ("Reading noise (calibration plateau scatter)", noise_sd),
        (f"Linear-conversion error ({u.get('linear_source', '2022 bench table')})", u["linear_interpolation_error_C"] / s3),
    ]   # stem conduction removed from the measured budget (2026-10-08): the 0.30 C fin-model value could not be re-derived
    df = pd.DataFrame(rows, columns=["source", "u_C"])
    uc = float(np.sqrt((df["u_C"] ** 2).sum()))
    return df, uc, 2 * uc


def _beta_dT_error(hi, lo, u):
    """Half-width of the error in an excursion hi -> lo from the beta tolerance of the
    implanted thermistor (2026-10-19): the beads are not the calibration set,
    so a bead whose beta differs from the set's reads a slightly different slope."""
    dB = u["beta_K"] * u["beta_tol_pct"] / 100

    def ex(beta):
        return (T_from_R(R_ntc(hi, u), u, beta) - T_from_R(R_ntc(lo, u), u, beta)) - (hi - lo)
    return abs(ex(u["beta_K"] + dB) - ex(u["beta_K"] - dB)) / 2


def measured_differential_budget(u: dict, noise_sd: float, run_sd: float, cal_span_C=None):
    """dT = T_initial - T_final of one trace, for a small excursion (optic nerve) and a
    large one (ring). Static offsets cancel; what remains depends on the size of dT:
    the tilt of the calibration line, the curvature the line ignores, the change of
    self-heating with temperature, and the beta tolerance of the implanted bead.
    u_C is the small-excursion budget (primary endpoint); u_C_large the large-excursion
    one. cal_span_C: the temperature span between the calibration set points (default
    the human 4-37 C; the porcine transfer function was trimmed at 20.5 and 37 C, a
    span of 16.5 C, which doubles the tilt term)."""
    s3 = np.sqrt(3)
    span = cal_span_C if cal_span_C is not None else (max(u["cal_temps_C"]) - min(u["cal_temps_C"]))
    slope_rel = np.sqrt(2) * np.sqrt(u["well_accuracy_C"] ** 2 + u["well_uniformity_C"] ** 2) / s3 / span
    cases = []
    for (hi, lo), interp in ((u["dT_small_C"], u["linear_interpolation_dT_small_C"]),
                             (u["dT_large_C"], u["linear_interpolation_dT_large_C"])):
        dp = abs(_power_ratio(hi, u) - _power_ratio(lo, u)) / _power_ratio(u["selfheat_reference_C"], u)
        cases.append([
            ("Reading scatter (start sample)", max(noise_sd, run_sd)),
            ("Calibration-line tilt (calibrator error differs between set points)", slope_rel * (hi - lo)),
            (f"Linear-conversion error on a difference ({u.get('linear_source', '2022 bench table')})", interp / s3),
            ("Self-heating change with temperature", u["selfheat_residual_bound_C"] * dp / s3),
            ("Thermistor beta tolerance (implanted bead, not the calibration set)", _beta_dT_error(hi, lo, u) / s3),
            ("Drift during the run (measured after calibration)",
             u["drift_after_calibration_V"] * u["slope_C_per_V"] / s3),
            ("Bath stability", u["bath_stability_C"] / s3),
        ])
    df = pd.DataFrame(cases[0], columns=["source", "u_C"])
    df["u_C_large"] = [v for _, v in cases[1]]
    uc = float(np.sqrt((df["u_C"] ** 2).sum()))
    df.attrs["U_large"] = 2 * float(np.sqrt((df["u_C_large"] ** 2).sum()))
    return df, uc, 2 * uc
