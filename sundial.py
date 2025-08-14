
#!/usr/bin/env python3
"""
Sundial Designer — Vertical Wall with Arbitrary Declination
Adds full-scale PDF export and analemmas (equation-of-time hour curves).

Usage examples:
  python sundial_designer.py --lat 37.8 --decl 20 --style-length 150 --units mm \
      --plate-width 300 --plate-height 400 --pdf --half-hours --analemmas \
      --lon -122.4 --tz-std-meridian -120 --outdir ./out

Notes:
- 'style-length' and all geometric outputs are in the chosen --units (mm|cm|in). If you set style-length=150 and units=mm,
  the rod is 150 mm long and the drawing coordinates are in mm.
- Analemmas require longitude/time zone to convert mean time -> apparent solar time via equation of time.
  If you omit --lon/--tz-std-meridian, analemmas default to true solar time (no mean-time correction).
"""
import argparse
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numpy.linalg import norm


# -----------------------------
# Linear algebra & geometry
# -----------------------------

def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = norm(v)
    if n == 0:
        return v
    return v / n


def rotation_about_z(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[ c, -s, 0.0],
                     [ s,  c, 0.0],
                     [0.0, 0.0, 1.0]], dtype=float)


@dataclass
class DialPlane:
    """Defines the vertical wall plane and its in-plane axes (u right, v up)."""
    n: np.ndarray  # unit outward normal (in ENU coords)
    u: np.ndarray  # unit axis along the wall to the right
    v: np.ndarray  # unit axis along the wall upward

    @staticmethod
    def for_vertical_wall(declination_rad: float) -> "DialPlane":
        # ENU axes: x east, y north, z up.
        # A wall facing due south has outward normal (0,-1,0).
        # Positive declination rotates this normal toward the west about +z.
        south = np.array([0.0, -1.0, 0.0])
        Rz = rotation_about_z(declination_rad)
        n = unit(Rz @ south)

        # In-plane u to the right (horizontal) and v upward.
        zhat = np.array([0.0, 0.0, 1.0])
        u = unit(np.cross(zhat, n))
        v = unit(np.cross(n, u))
        # Ensure v points mostly up
        if v[2] < 0:
            u, v = -u, -v
        return DialPlane(n=n, u=u, v=v)

    def project_uv(self, P: np.ndarray) -> Tuple[float, float]:
        return float(np.dot(P, self.u)), float(np.dot(P, self.v))


# -----------------------------
# Solar & style (gnomon) geometry
# -----------------------------

def sun_dir_ENU(phi: float, dec: float, H: float) -> np.ndarray:
    """
    Sun unit direction in ENU frame (from observer toward Sun).
    phi: latitude (+N), dec: solar declination, H: hour angle (0 at solar noon, + in afternoon).
    """
    cp, sp = math.cos(phi), math.sin(phi)
    cd, sd = math.cos(dec), math.sin(dec)
    cH, sH = math.cos(H), math.sin(H)

    x = cd * sH
    y = cp * cd * cH - sp * sd
    z = sp * cd * cH + cp * sd
    return unit(np.array([x, y, z], dtype=float))


def polar_style_dir(phi: float) -> np.ndarray:
    """Direction of a polar-style gnomon (points to NCP): az=0°, alt=phi => (0, cosφ, sinφ)."""
    return unit(np.array([0.0, math.cos(phi), math.sin(phi)], dtype=float))


def intersect_ray_plane(P0: np.ndarray, d: np.ndarray, n: np.ndarray, Q0: np.ndarray) -> np.ndarray:
    """
    Ray-plane intersection: P(t) = P0 + t d; plane n·(X - Q0) = 0.
    Returns the 3D point; if nearly parallel, returns NaNs.
    """
    denom = float(np.dot(n, d))
    if abs(denom) < 1e-12:
        return np.array([math.nan, math.nan, math.nan])
    t = float(np.dot(n, (Q0 - P0))) / denom
    return P0 + t * d


def fit_line_uv(UV: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fit a straight line to 2D points via PCA: returns (centroid, unit direction)."""
    if UV.shape[0] < 2:
        return np.array([math.nan, math.nan]), np.array([math.nan, math.nan])
    C = UV.mean(axis=0)
    A = UV - C
    cov = A.T @ A
    w, V = np.linalg.eigh(cov)
    d = unit(V[:, np.argmax(w)])
    # orient with positive v for consistency
    if d[1] < 0:
        d = -d
    return C, d


# Solar declination & equation of time approximations
def declination_from_daynum(N: int) -> float:
    # Jan 1 = 1 (approximation)
    return math.radians(-23.44 * math.cos(2 * math.pi * (N + 10) / 365.0))


def equation_of_time_minutes(N: int) -> float:
    # NOAA empirical approximation
    B = 2 * math.pi * (N - 81) / 364.0
    return 9.87 * math.sin(2 * B) - 7.53 * math.cos(B) - 1.5 * math.sin(B)


# -----------------------------
# Dial construction
# -----------------------------

def build_vertical_declining_dial(latitude_deg: float,
                                  wall_declination_deg: float,
                                  style_length: float,
                                  hours: List[float]):
    phi = math.radians(latitude_deg)
    d = math.radians(wall_declination_deg)
    plane = DialPlane.for_vertical_wall(d)

    style_dir = polar_style_dir(phi)
    style_foot = np.array([0.0, 0.0, 0.0])
    style_tip = style_foot + style_length * style_dir

    # Hour lines: sample across a range of declinations to define straight lines
    hour_records = []
    hour_samples: Dict[float, Dict[str, np.ndarray]] = {}

    decs = np.radians(np.linspace(-23.44, 23.44, 61))
    for Hh in hours:
        H = math.radians(15.0 * Hh)
        pts = []
        uv_pts = []
        for dec in decs:
            s = sun_dir_ENU(phi, dec, H)
            if float(np.dot(plane.n, s)) >= 0.0:  # wall not lit
                continue
            P = intersect_ray_plane(style_tip, -s, plane.n, style_foot)
            if not np.any(np.isnan(P)):
                pts.append(P)
                uv_pts.append(plane.project_uv(P))

        if len(uv_pts) >= 2:
            UV = np.array(uv_pts)
            C, dvec = fit_line_uv(UV)
            angle_from_vertical = math.degrees(math.atan2(dvec[0], dvec[1]))
            hour_records.append({
                "hour": Hh,
                "visible": True,
                "angle_from_vertical_deg": angle_from_vertical,
                "dir_u": dvec[0],
                "dir_v": dvec[1]
            })
            hour_samples[Hh] = {
                "dec_deg": np.degrees(decs),
                "UV": UV
            }
        else:
            hour_records.append({
                "hour": Hh,
                "visible": False,
                "angle_from_vertical_deg": math.nan,
                "dir_u": math.nan, "dir_v": math.nan
            })
            hour_samples[Hh] = {"dec_deg": np.degrees(decs), "UV": np.empty((0, 2))}

    # Date lines (declinations): equinox and solstices as curves across the dial
    date_lines = []
    for dec_deg, label in [(0.0, "Equinox"),
                           (23.44, "June solstice"),
                           (-23.44, "December solstice")]:
        dec = math.radians(dec_deg)
        UVs = []
        Hhs = []
        for Hh in np.linspace(-6.0, 6.0, 121):  # solar hours across the day
            H = math.radians(15.0 * Hh)
            s = sun_dir_ENU(phi, dec, H)
            if float(np.dot(plane.n, s)) >= 0.0:
                continue
            P = intersect_ray_plane(style_tip, -s, plane.n, style_foot)
            if not np.any(np.isnan(P)):
                UVs.append(plane.project_uv(P))
                Hhs.append(Hh)
        date_lines.append({"label": label,
                           "dec_deg": dec_deg,
                           "Hh": np.array(Hhs),
                           "UV": np.array(UVs)})

    # Noon line (H=0): direction for shadow-length scale
    decs_noon = np.radians(np.linspace(-23.44, 23.44, 61))
    noon_pts = []
    for dec in decs_noon:
        s = sun_dir_ENU(phi, dec, 0.0)
        if float(np.dot(plane.n, s)) >= 0.0:
            continue
        P = intersect_ray_plane(style_tip, -s, plane.n, style_foot)
        if not np.any(np.isnan(P)):
            noon_pts.append(P)
    noon_UV = np.array([plane.project_uv(P) for P in noon_pts])
    noon_C, noon_dir = fit_line_uv(noon_UV)

    # Monthly noon shadow-length ticks (distance along noon line from origin)
    monthly = []
    month_mid = [15, 46, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349]  # ~mid-month DOY
    month_name = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    for m, N in enumerate(month_mid):
        dec = declination_from_daynum(N)
        s = sun_dir_ENU(phi, dec, 0.0)
        if float(np.dot(plane.n, s)) >= 0.0:
            monthly.append({"month": month_name[m], "noon_t": math.nan, "decl_deg": math.degrees(dec)})
            continue
        P = intersect_ray_plane(style_tip, -s, plane.n, style_foot)
        UV = np.array(plane.project_uv(P))
        t = float(np.dot(UV, noon_dir))  # signed distance along noon line
        monthly.append({"month": month_name[m], "noon_t": t, "decl_deg": math.degrees(dec)})

    # Angle of the rod relative to the wall plane
    angle_style_normal = math.degrees(
        math.acos(max(-1.0, min(1.0, float(np.dot(unit(style_dir), plane.n)))))
    )
    angle_style_with_wall = 90.0 - angle_style_normal

    return {
        "plane": plane,
        "style_dir": style_dir,
        "style_length": style_length,
        "hour_records": hour_records,
        "hour_samples": hour_samples,
        "date_lines": date_lines,
        "noon_dir": noon_dir,
        "monthly_noon": monthly,
        "angle_style_with_wall_deg": angle_style_with_wall
    }


# -----------------------------
# Analemmas
# -----------------------------

def build_analemmas(latitude_deg: float,
                    wall_declination_deg: float,
                    style_length: float,
                    hours_for_analemmas: List[float],
                    lon_deg: float | None,
                    tz_std_meridian_deg: float | None,
                    dst_minutes: float = 0.0):
    """
    Build analemma curves (figure-8) for specified clock hours.
    If lon_deg and tz_std_meridian_deg are None, uses true solar time (no mean-time correction).
    """
    phi = math.radians(latitude_deg)
    d = math.radians(wall_declination_deg)
    plane = DialPlane.for_vertical_wall(d)
    style_dir = polar_style_dir(phi)
    style_foot = np.array([0.0, 0.0, 0.0])
    style_tip = style_foot + style_length * style_dir

    # Longitude/time-zone correction (minutes): + if location east of standard meridian
    if lon_deg is None or tz_std_meridian_deg is None:
        lon_correction_min = 0.0
    else:
        lon_correction_min = 4.0 * (lon_deg - tz_std_meridian_deg)

    analemmas = []  # list of dict per hour
    # sample days of year
    days = list(range(1, 366, 3))  # every ~3 days
    for Hclock in hours_for_analemmas:
        UVs = []
        Ns = []
        for N in days:
            dec = declination_from_daynum(N)
            eot_min = equation_of_time_minutes(N)
            # apparent solar time offset from mean time (minutes)
            time_offset_min = eot_min + lon_correction_min - dst_minutes
            H_app_hours = Hclock + time_offset_min / 60.0
            H = math.radians(15.0 * H_app_hours)
            s = sun_dir_ENU(phi, dec, H)
            if float(np.dot(plane.n, s)) >= 0.0:
                continue
            P = intersect_ray_plane(style_tip, -s, plane.n, style_foot)
            if not np.any(np.isnan(P)):
                UVs.append(plane.project_uv(P))
                Ns.append(N)
        analemmas.append({"hour_clock": Hclock, "daynum": np.array(Ns), "UV": np.array(UVs)})
    return analemmas


# -----------------------------
# Plotting and export
# -----------------------------

def _units_to_inch_factor(units: str) -> float:
    units = units.lower()
    if units == "mm":
        return 1.0 / 25.4
    if units == "cm":
        return 1.0 / 2.54
    if units in ("in", "inch", "inches"):
        return 1.0
    raise ValueError("units must be one of: mm, cm, in")


def plot_and_export(result: dict,
                    analemmas: List[dict] | None,
                    latitude_deg: float,
                    wall_declination_deg: float,
                    style_length: float,
                    outdir: str,
                    prefix: str = "sundial_vertical_declining"):
    os.makedirs(outdir, exist_ok=True)
    hour_records = result["hour_records"]
    date_lines = result["date_lines"]
    noon_dir = result["noon_dir"]
    monthly = result["monthly_noon"]

    wall_extent = 2.0 * max(1.0, style_length)
    fig, ax = plt.subplots(figsize=(8.5, 11))

    ax.axhline(0, linewidth=1)
    ax.axvline(0, linewidth=1)

    for rec in hour_records:
        if not rec["visible"]:
            continue
        du, dv = rec["dir_u"], rec["dir_v"]
        t = np.linspace(-wall_extent, wall_extent, 2)
        ax.plot(t * du, t * dv, linewidth=1)
        label_u = 1.02 * wall_extent * du / (abs(du) + abs(dv) + 1e-9)
        label_v = 1.02 * wall_extent * dv / (abs(dv) + abs(du) + 1e-9)
        hour_label = f"{int(12 + rec['hour'])%24:02d}:{'30' if abs(rec['hour']%1)>1e-6 else '00'}"
        ax.text(label_u, label_v, hour_label, fontsize=8, ha='center', va='center')

    for dl in date_lines:
        UV = dl["UV"]
        if UV.shape[0] >= 2:
            ax.plot(UV[:, 0], UV[:, 1], linewidth=1, linestyle="--")
            idx = np.argmax(UV[:, 0])
            ax.text(UV[idx, 0], UV[idx, 1], dl["label"], fontsize=8, ha='left', va='bottom')

    if not (np.isnan(noon_dir).any()):
        tline = np.linspace(-wall_extent, wall_extent, 2)
        ax.plot(tline * noon_dir[0], tline * noon_dir[1], linewidth=1, linestyle=":")
        perp = np.array([-noon_dir[1], noon_dir[0]])
        tick_size = 0.05 * wall_extent
        for m in monthly:
            if not np.isfinite(m["noon_t"]):
                continue
            tu = m["noon_t"] * noon_dir[0]
            tv = m["noon_t"] * noon_dir[1]
            ax.plot([tu - tick_size * perp[0], tu + tick_size * perp[0]],
                    [tv - tick_size * perp[1], tv + tick_size * perp[1]], linewidth=1)
            ax.text(tu, tv, m["month"], fontsize=8, ha='center', va='bottom')

    # Analemmas overlay
    if analemmas:
        for a in analemmas:
            UV = a["UV"]
            if UV.shape[0] >= 2:
                ax.plot(UV[:, 0], UV[:, 1], linewidth=0.8, linestyle="-")
                # label with hour at last point
                u_last, v_last = UV[-1, 0], UV[-1, 1]
                hour_label = f"{int(12 + a['hour_clock'])%24:02d}:{'30' if abs(a['hour_clock']%1)>1e-6 else '00'}"
                ax.text(u_last, v_last, f"Analemma {hour_label}", fontsize=7, ha='left', va='center')

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("u (right along wall)")
    ax.set_ylabel("v (up along wall)")
    ax.set_title(
        f"Vertical Sundial — lat {latitude_deg}°, wall decl {wall_declination_deg}°  |  polar style = {style_length} units"
    )
    ax.grid(True)

    svg_path = os.path.join(outdir, f"{prefix}.svg")
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.close(fig)

    # CSV exports
    df_hours = pd.DataFrame([{
        "hour_solar": rec["hour"],
        "visible": rec["visible"],
        "angle_from_vertical_deg": rec["angle_from_vertical_deg"],
        "dir_u": rec["dir_u"],
        "dir_v": rec["dir_v"]
    } for rec in hour_records])
    df_hours.to_csv(os.path.join(outdir, "hour_lines.csv"), index=False)

    df_noon = pd.DataFrame(result["monthly_noon"])
    df_noon.to_csv(os.path.join(outdir, "noon_shadow_scale.csv"), index=False)

    rows = []
    for dl in date_lines:
        UV = dl["UV"]
        Hh = dl["Hh"]
        for (u, v), h in zip(UV, Hh):
            rows.append({"label": dl["label"], "declination_deg": dl["dec_deg"], "hour_solar": h, "u": u, "v": v})
    pd.DataFrame(rows).to_csv(os.path.join(outdir, "date_lines.csv"), index=False)

    if analemmas:
        rows = []
        for a in analemmas:
            UV = a["UV"]
            Ns = a["daynum"]
            for (u, v), N in zip(UV, Ns):
                rows.append({"hour_clock": a["hour_clock"], "daynum": int(N), "u": u, "v": v})
        pd.DataFrame(rows).to_csv(os.path.join(outdir, "analemmas.csv"), index=False)

    return svg_path


def export_fullscale_pdf(result: dict,
                         analemmas: List[dict] | None,
                         plate_width: float,
                         plate_height: float,
                         units: str,
                         outdir: str,
                         prefix: str = "sundial_vertical_declining_fullscale",
                         margin: float = 0.0,
                         origin: str = "center"):
    """
    Create a 1:1 scale PDF for printing. Coordinates are in the given units.
    The origin (style foot) is placed at the center of the plate by default.
    """
    os.makedirs(outdir, exist_ok=True)
    inch_factor = _units_to_inch_factor(units)

    # Figure size (inches)
    fig_w_in = (plate_width + 2 * margin) * inch_factor
    fig_h_in = (plate_height + 2 * margin) * inch_factor
    fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in))

    # Drawing extents in UV units (same numeric units as style_length)
    # Place origin at center unless specified
    if origin == "center":
        umin = -plate_width / 2
        umax = plate_width / 2
        vmin = -plate_height / 2
        vmax = plate_height / 2
    else:
        # bottom-left origin with margin
        umin = 0.0
        vmin = 0.0
        umax = plate_width
        vmax = plate_height

    # Border rectangle (plate outline)
    ax.plot([umin, umax, umax, umin, umin],
            [vmin, vmin, vmax, vmax, vmin],
            linewidth=0.8)

    # Draw hour lines, date lines, noon line with monthly ticks
    hour_records = result["hour_records"]
    date_lines = result["date_lines"]
    noon_dir = result["noon_dir"]
    monthly = result["monthly_noon"]

    # Draw many-length lines for visibility
    wall_extent = max(plate_width, plate_height) * 1.1
    for rec in hour_records:
        if not rec["visible"]:
            continue
        du, dv = rec["dir_u"], rec["dir_v"]
        t = np.linspace(-wall_extent, wall_extent, 2)
        ax.plot(t * du, t * dv, linewidth=0.8)
        # Hour label near plate edge in direction of the line
        label_u = (0.48 * plate_width) * du / (abs(du) + abs(dv) + 1e-9)
        label_v = (0.48 * plate_height) * dv / (abs(du) + abs(dv) + 1e-9)
        hour_label = f"{int(12 + rec['hour'])%24:02d}:{'30' if abs(rec['hour']%1)>1e-6 else '00'}"
        ax.text(label_u, label_v, hour_label, fontsize=8, ha='center', va='center')

    # Date lines (equinox/solstices)
    for dl in date_lines:
        UV = dl["UV"]
        if UV.shape[0] >= 2:
            ax.plot(UV[:, 0], UV[:, 1], linewidth=0.8, linestyle="--")
            idx = np.argmax(UV[:, 0])
            ax.text(UV[idx, 0], UV[idx, 1], dl["label"], fontsize=7, ha='left', va='bottom')

    # Noon line & monthly ticks
    if not (np.isnan(noon_dir).any()):
        tline = np.linspace(-wall_extent, wall_extent, 2)
        ax.plot(tline * noon_dir[0], tline * noon_dir[1], linewidth=0.8, linestyle=":")
        perp = np.array([-noon_dir[1], noon_dir[0]])
        tick_size = 0.02 * max(plate_width, plate_height)
        for m in monthly:
            if not np.isfinite(m["noon_t"]):
                continue
            tu = m["noon_t"] * noon_dir[0]
            tv = m["noon_t"] * noon_dir[1]
            ax.plot([tu - tick_size * perp[0], tu + tick_size * perp[0]],
                    [tv - tick_size * perp[1], tv + tick_size * perp[1]], linewidth=0.8)
            ax.text(tu, tv, m["month"], fontsize=6, ha='center', va='bottom')

    # Analemmas
    if analemmas:
        for a in analemmas:
            UV = a["UV"]
            if UV.shape[0] >= 2:
                ax.plot(UV[:, 0], UV[:, 1], linewidth=0.6, linestyle="-")
                # label
                u_last, v_last = UV[-1, 0], UV[-1, 1]
                hour_label = f"{int(12 + a['hour_clock'])%24:02d}:{'30' if abs(a['hour_clock']%1)>1e-6 else '00'}"
                ax.text(u_last, v_last, hour_label, fontsize=6, ha='left', va='center')

    # Mark origin (style foot)
    ax.plot([0], [0], marker='o', markersize=3)

    # Axis formatting
    ax.set_xlim(umin - margin, umax + margin)
    ax.set_ylim(vmin - margin, vmax + margin)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(f"u ({units})  — right along wall")
    ax.set_ylabel(f"v ({units})  — up along wall")
    ax.set_title("Full-scale sundial plate")

    pdf_path = os.path.join(outdir, f"{prefix}.pdf")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return pdf_path


# -----------------------------
# CLI
# -----------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Vertical-declining sundial designer with full-scale PDF & analemmas.")
    p.add_argument("--lat", type=float, required=True, help="Latitude in degrees (+N).")
    p.add_argument("--decl", type=float, required=True, help="Wall declination in degrees: + west of south, - east of south.")
    p.add_argument("--style-length", type=float, default=100.0, help="Rod length in chosen units (default 100).")
    p.add_argument("--hours-start", type=float, default=-6.0, help="Start hour relative to solar noon (default -6).")
    p.add_argument("--hours-end", type=float, default=6.0, help="End hour relative to solar noon (default +6).")
    p.add_argument("--half-hours", action="store_true", help="Include half-hour lines.")
    p.add_argument("--outdir", type=str, default=".", help="Output directory (default current).")
    p.add_argument("--prefix", type=str, default="sundial_vertical_declining", help="Output filename prefix for SVG/PDF.")
    # Full-scale PDF options
    p.add_argument("--pdf", action="store_true", help="Export full-scale PDF.")
    p.add_argument("--plate-width", type=float, default=300.0, help="Plate width (units).")
    p.add_argument("--plate-height", type=float, default=400.0, help="Plate height (units).")
    p.add_argument("--units", type=str, default="mm", choices=["mm", "cm", "in"], help="Units for geometry and PDF scaling.")
    p.add_argument("--margin", type=float, default=0.0, help="Extra margin around plate in units (for page bleed).")
    # Analemmas
    p.add_argument("--analemmas", action="store_true", help="Add analemmas (equation-of-time hour curves).")
    p.add_argument("--lon", type=float, default=None, help="Longitude in degrees (East positive).")
    p.add_argument("--tz-std-meridian", type=float, default=None, help="Time zone standard meridian in degrees (East positive, multiples of 15).")
    p.add_argument("--dst-minutes", type=float, default=0.0, help="DST offset in minutes (usually 0 or 60).")
    return p.parse_args()


def main():
    args = parse_args()
    step = 0.5 if args.half_hours else 1.0
    num = int(round((args.hours_end - args.hours_start) / step)) + 1
    hours = [args.hours_start + i * step for i in range(num)]

    result = build_vertical_declining_dial(latitude_deg=args.lat,
                                           wall_declination_deg=args.decl,
                                           style_length=args.style_length,
                                           hours=hours)
    # Optional analemmas
    analemmas = None
    if args.analemmas:
        analemmas = build_analemmas(latitude_deg=args.lat,
                                    wall_declination_deg=args.decl,
                                    style_length=args.style_length,
                                    hours_for_analemmas=hours,
                                    lon_deg=args.lon,
                                    tz_std_meridian_deg=args.tz_std_meridian,
                                    dst_minutes=args.dst_minutes)

    svg_path = plot_and_export(result, analemmas,
                               latitude_deg=args.lat,
                               wall_declination_deg=args.decl,
                               style_length=args.style_length,
                               outdir=args.outdir,
                               prefix=args.prefix)

    print(f"SVG saved to: {svg_path}")
    print(f"Hour lines, noon scale, date lines CSV files written in: {args.outdir}")

    if analemmas:
        print("Analemmas CSV written (analemmas.csv).")

    if args.pdf:
        pdf_path = export_fullscale_pdf(result, analemmas,
                                        plate_width=args.plate_width,
                                        plate_height=args.plate_height,
                                        units=args.units,
                                        outdir=args.outdir,
                                        prefix=f"{args.prefix}_fullscale",
                                        margin=args.margin,
                                        origin="center")
        print(f"Full-scale PDF saved to: {pdf_path}")
        print("NOTE: The PDF is 1:1 in the chosen units. Make sure to print at 100% scale (no 'fit to page').")
        print(f"Style (rod) angle with wall ≈ {result['angle_style_with_wall_deg']:.2f} degrees.")

if __name__ == "__main__":
    main()
