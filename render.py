import os
import io
import sys
import math
import datetime as dt
from dataclasses import dataclass

import numpy as np
import requests
import xarray as xr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, BoundaryNorm

# -----------------------------
# CONFIG
# -----------------------------
BASE_IASTATE_4326 = "https://mesonet.agron.iastate.edu/data/gis/images/4326"
# HRRR uses date/hour folders like: .../hrrr/21/simref_000.png  (your log shows this pattern)
# We'll keep this as-is and just colorize fields we compute.
OUT_DIR = os.environ.get("OUT_DIR", "out")

# -----------------------------
# UTIL: NWS-style reflectivity colormap (approx)
# -----------------------------
def register_nws_reflectivity():
    """
    Register a reasonable approximation of an NWS-style reflectivity colormap.

    We define dBZ boundaries and associated colors (greens -> yellows -> reds -> purples).
    Then we return a (cmap, norm) pair suitable for pcolormesh/imshow.
    """
    # dBZ boundaries (common radar)
    bounds = np.array([-10, 5, 15, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70], dtype=float)

    # Colors approximating NWS reflectivity
    # (not exact proprietary, but classic-looking and good)
    colors = [
        "#646464",  # -10..5  (very light/gray)
        "#04e9e7",  # 5..15   (cyan)
        "#019ff4",  # 15..25  (blue)
        "#0300f4",  # 25..30  (deep blue)
        "#02fd02",  # 30..35  (green)
        "#01c501",  # 35..40  (dark green)
        "#008e00",  # 40..45  (darker green)
        "#fdf802",  # 45..50  (yellow)
        "#e5bc00",  # 50..55  (gold)
        "#fd9500",  # 55..60  (orange)
        "#fd0000",  # 60..65  (red)
        "#d40000",  # 65..70  (dark red)
    ]

    cmap = ListedColormap(colors, name="NWSReflectivity")
    norm = BoundaryNorm(bounds, cmap.N, clip=True)

    # Register so "NWSReflectivity" becomes valid if anything references it later
    try:
        matplotlib.colormaps.register(cmap, force=True)
    except Exception:
        # Older API fallback
        try:
            plt.register_cmap(name="NWSReflectivity", cmap=cmap)
        except Exception:
            pass

    return cmap, norm, bounds


# -----------------------------
# FETCH HELPERS
# -----------------------------
def fetch_png(url: str, timeout=20) -> bytes:
    r = requests.get(url, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"Fetch failed {r.status_code} for {url}")
    return r.content


def save_bytes(path: str, b: bytes):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b)


# -----------------------------
# PLOTTING
# -----------------------------
def render_colorbar(path: str, label: str, cmap, norm, ticks):
    fig = plt.figure(figsize=(6, 1.0), dpi=160)
    ax = fig.add_axes([0.05, 0.45, 0.9, 0.25])
    cb = matplotlib.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation="horizontal", ticks=ticks)
    cb.set_label(label, fontsize=9)
    cb.ax.tick_params(labelsize=8)
    fig.savefig(path, bbox_inches="tight", transparent=True)
    plt.close(fig)


def render_field_png(
    out_path: str,
    data2d: np.ndarray,
    extent_lonlat: tuple,
    cmap,
    norm=None,
    vmin=None,
    vmax=None,
    title: str = "",
    add_alpha_mask: bool = True,
):
    """
    Render a plain image (no basemap) with transparent background (like tropical tidbits style overlays).
    extent_lonlat = (lon_min, lon_max, lat_min, lat_max)
    """
    lon_min, lon_max, lat_min, lat_max = extent_lonlat

    fig = plt.figure(figsize=(10.24, 7.68), dpi=150)  # 1536x1152-ish
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    # Mask invalids -> transparency
    arr = np.array(data2d, dtype=float)
    if add_alpha_mask:
        mask = ~np.isfinite(arr)
        arr = np.ma.array(arr, mask=mask)

    if norm is not None:
        im = ax.imshow(
            arr,
            origin="upper",
            cmap=cmap,
            norm=norm,
            extent=[lon_min, lon_max, lat_min, lat_max],
            interpolation="nearest",
        )
    else:
        im = ax.imshow(
            arr,
            origin="upper",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            extent=[lon_min, lon_max, lat_min, lat_max],
            interpolation="nearest",
        )

    if title:
        ax.text(
            0.01, 0.99, title,
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=10,
            color="white",
            bbox=dict(facecolor="black", alpha=0.35, pad=3, edgecolor="none")
        )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, transparent=True, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


# -----------------------------
# MAIN TASKS
# -----------------------------
@dataclass
class Product:
    key: str
    label: str


def run():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Make colormap available
    nws_cmap, nws_norm, nws_bounds = register_nws_reflectivity()

    # For now: just pull the already-rendered simref PNGs from Iowa State
    # and copy them through to your output structure (so your site always has frames).
    #
    # Your log shows 404s like:
    # https://mesonet.agron.iastate.edu/data/gis/images/4326/hrrr/21/simref_000.png
    #
    # That means the hour folder "21" was wrong for the run time.
    # We'll auto-detect the latest available run hour by probing.
    #
    # We'll probe the last ~30 hours for a folder that has simref_000.png
    now_utc = dt.datetime.utcnow()
    candidate_hours = [(now_utc - dt.timedelta(hours=i)).strftime("%H") for i in range(0, 30)]

    run_hour = None
    for hh in candidate_hours:
        test_url = f"{BASE_IASTATE_4326}/hrrr/{hh}/simref_000.png"
        try:
            _ = fetch_png(test_url, timeout=10)
            run_hour = hh
            break
        except Exception:
            continue

    if run_hour is None:
        raise RuntimeError("Could not find any HRRR run hour with simref_000.png in last ~30 hours.")

    # Decide frame range (HRRR often has 0-18+ on IEM images; you can extend if needed)
    fhrs = list(range(0, 19))  # f000..f018

    failures = []
    ok = 0

    for fh in fhrs:
        name = f"simref_{fh:03d}.png"
        url = f"{BASE_IASTATE_4326}/hrrr/{run_hour}/{name}"
        out_path = os.path.join(OUT_DIR, "hrrr", "conus", "latest", f"simref_f{fh:03d}.png")

        try:
            b = fetch_png(url)
            save_bytes(out_path, b)
            ok += 1
        except Exception as e:
            failures.append(f"simref f{fh:03d} failed: {e}")

    # Write a tiny manifest so the website can know what exists
    manifest_path = os.path.join(OUT_DIR, "hrrr", "conus", "latest", "manifest.json")
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write(
            "{\n"
            f'  "model": "hrrr",\n'
            f'  "domain": "conus",\n'
            f'  "run_hour_utc": "{run_hour}",\n'
            f'  "frames": {fhrs},\n'
            f'  "frame_count_ok": {ok}\n'
            "}\n"
        )

    if failures:
        # Don't hard-fail if some frames are missing; just log.
        # (If you want strict behavior, raise instead.)
        print("Some frames failed:")
        for line in failures:
            print(" -", line)

    print(f"Done. Run hour={run_hour}Z, frames ok={ok}/{len(fhrs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
