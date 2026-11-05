"""Extra read-out points: where the sensors actually sat in the nine human donor eyes.

Source: Ex-Vivo/CT Segmentation/outputs/sensor_picks_in_eye_frame.csv (micro-CT
sensor picks).  Each sensor is placed on the model globe at the same polar
angle from the anterior pole and the same fraction of the half axial length,
turned into the mirror plane (azimuth dropped): scleral and vitreous sensors on
the +x side, the optic-nerve sensor on the side of the nerve.

Correction 2026-10-22: this is NOT the mapping of the
thesis figure script code/figures/aim2_probe_locations.py, which places
scleral and vitreous sensors by axial fraction plus micro-CT depth, the
optic-nerve sensor relative to the model nerve, and leaves screened sensors
out of the mean.  The only shared constant is Z_POSTERIOR.  Known defects of
the points below (they are kept unchanged because the stored runs used them):
  * the half-axial fraction puts most scleral points outside the model globe
    (rho > 1 for eyes wider than long); use the "(on sclera)" twins;
  * the optic-nerve points (mean and 6 of 9 eyes) lie outside the ex-vivo mesh
    and were read by silent extrapolation in replica.fem.locate; ex-vivo
    read-outs now come from dev/exvivo_reread_probes.py, which refuses them
    (dev/probe_guard.py) and places the sensor on the model nerve;
  * the "mean" point averages all nine eyes, not the eyes included in the
    measured mean; dev/exvivo_reread_probes.py writes included-eye means.

The model globe is not the donor eyes, so these points reproduce angular
position and relative depth only.
"""
import csv
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PICKS = os.path.normpath(os.path.join(HERE, "..", "..", "Ex-Vivo", "CT Segmentation", "outputs",
                                      "sensor_picks_in_eye_frame.csv"))
Z_APEX, Z_POSTERIOR = 0.0, -25.9            # model globe on the optical axis, mm
CENTRE_Z, HALF_AXIAL = (Z_APEX + Z_POSTERIOR) / 2, (Z_APEX - Z_POSTERIOR) / 2
SITES = {"anterior": ("Anterior sclera", 1), "equatorial": ("Equatorial sclera", 1), "vitreous": ("Vitreous", 1),
         "posterior": ("Posterior sclera", 1), "optic": ("Proximal optic nerve", -1)}


def _point(polar_deg, rho, side):
    th = np.radians(polar_deg)
    return [side * rho * HALF_AXIAL * np.sin(th) * 1e-3, 0.0, (CENTRE_Z + rho * HALF_AXIAL * np.cos(th)) * 1e-3]


def sensor_probes(path=PICKS, group="human", individual=True):
    """{probe name: dict(kind="point", xyz=[m])}: the mean position of each site, and every eye's sensor."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["group"] == group]
    out = {}
    for key, (label, side) in SITES.items():
        rs = [r for r in rows if r["sensor"] == key]
        if not rs:
            continue
        pol = np.array([float(r["polar_from_anterior_pole_deg"]) for r in rs])
        rho = np.array([float(r["radius_over_half_axial"]) for r in rs])
        out[f"CT sensor mean - {label}"] = dict(kind="point", xyz=_point(pol.mean(), rho.mean(), side))
        if individual:
            for r, a, b in zip(rs, pol, rho):
                out[f"CT sensor {r['eye']} - {label}"] = dict(kind="point", xyz=_point(a, b, side))
    return out


def snap_to_sclera(prob, probes):
    """For each scleral sensor point, the nearest point of the model's sclera in the mirror plane.

    The donor globes are not the model globe, so a point mapped by angle and
    relative radius can land just outside the model sclera (in the fat or the
    ring).  The snapped point is the centre of the nearest sclera triangle of
    the mirror-plane section, so it lies inside the scleral wall.  Returns
    {name + " (on sclera)": probe}.
    """
    dom = [d for d, n in prob.domain_names.items() if n == "Sclera"]
    P = prob.points[:prob.n_original]
    t = prob.tris[(prob.tri_up == 0) & np.isin(prob.tri_down, dom)]
    t = t[(np.abs(P[t][:, :, 1]) < 1e-7).all(1)]
    cen = P[t].mean(1)
    out = {}
    for n, p in probes.items():
        if "sclera" not in n.lower():
            continue
        c = cen[np.linalg.norm(cen - np.asarray(p["xyz"]), axis=1).argmin()]
        out[n + " (on sclera)"] = dict(kind="point", xyz=[float(c[0]), 0.0, float(c[2])],
                                       moved_mm=float(np.linalg.norm(c - np.asarray(p["xyz"])) * 1e3))
    return out


if __name__ == "__main__":
    for n, p in sensor_probes(individual=False).items():
        print(f"{n:40s} x = {p['xyz'][0] * 1e3:7.2f} mm   z = {p['xyz'][2] * 1e3:7.2f} mm")
