#!/usr/bin/env python3
import os
import json
import time
import math
import shutil
import tempfile
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple, List

import requests
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.feature as cfeature

import xarray as xr

from supabase import create_client


# -----------------------------
# Config (env)
# -----------------------------
SUPABASE_URL = os.environ.get("SUPABASE_URL", "").strip()
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip()
SUPABASE_BUCKET = os.environ.get("SUPABASE_BUCKET", "wx-plots").strip()

# How many forecast hours we render per run (keep small at first; raise later)
HRRR_MAX_FHR = int(os.environ.get("HRRR_MAX_FHR", "18"))   # 0..18 step 1
GFS_MAX_FHR  = int(os.environ.get("GFS_MAX_FHR", "48"))    # 0..48 step 3

# Networking
UA = "wx-plots-renderer/1.0 (+github actions)"
TIMEOUT = 30


# -----------------------------
# Utility
# -----------------------------
def log(msg: str) -> None:
    print(msg, flush=True)

def must_env(name: str) -> str:
    v = os.environ.get(name, "").strip()
    if not v:
        raise SystemExit(f"Missing required env var: {name}")
    return v

def http_head_ok(url: str) -> bool:
    try:
        r = requests.head(url, timeout=TIMEOUT, headers={"User-Agent": UA})
        return r.status_code == 200
    except Exception:
        return False

def http_get_stream(url: str, out_path: str) -> None:
    with requests.get(url, stream=True, timeout=TIMEOUT, headers={"User-Agent": UA}) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


# -----------------------------
# NOMADS URL builders
# -----------------------------
def hrrr_url(run_dt: datetime, cycle: int, fhr: int, domain: str = "conus") -> str:
    """
    HRRR surface file (wrfsfc) from NOMADS.
    Example:
      https://nomads.ncep.noaa.gov/pub/data/nccf/com/hrrr/prod/hrrr.YYYYMMDD/conus/hrrr.tHHz.wrfsfcfFFF.grib2
    """
    ymd = run_dt.strftime("%Y%m%d")
    hh = f"{cycle:02d}"
    fff = f"{fhr:02d}" if fhr < 100 else f"{fhr:03d}"
    return f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/hrrr/prod/hrrr.{ymd}/{domain}/hrrr.t{hh}z.wrfsfcf{fff}.grib2"

def gfs_url(run_dt: datetime, cycle: int, fhr: int, res: str = "0p25") -> str:
    """
    GFS pgrb2 file from NOMADS.
    Example:
      https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/gfs.YYYYMMDD/HH/atmos/gfs.tHHz.pgrb2.0p25.fFFF
    """
    ymd = run_dt.strftime("%Y%m%d")
    hh = f"{cycle:02d}"
    fff = f"{fhr:03d}"
    return f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/gfs.{ymd}/{hh}/atmos/gfs.t{hh}z.pgrb2.{res}.f{fff}"


# -----------------------------
# Cycle resolution (latest available)
# -----------------------------
def find_latest_hrrr_cycle(domain: str = "conus") -> Tuple[datetime, int]:
    """
    Try recent dates/cycles and pick the newest cycle that has f00 available.
    """
    now = datetime.now(timezone.utc)
    # HRRR cycles hourly; try last 30 hours
    for back_h in range(0, 30):
        candidate = now - timedelta(hours=back_h)
        run_dt = candidate.replace(minute=0, second=0, microsecond=0)
        cyc = run_dt.hour
        url = hrrr_url(run_dt, cyc, 0, domain=domain)
        if http_head_ok(url):
            return (run_dt, cyc)
    raise RuntimeError("Could not find an available HRRR cycle on NOMADS in last 30 hours.")

def find_latest_gfs_cycle() -> Tuple[datetime, int]:
    """
    GFS cycles 00/06/12/18; try last ~2 days.
    """
    now = datetime.now(timezone.utc)
    cycles = [18, 12, 6, 0]
    for back_d in range(0, 3):
        day = (now - timedelta(days=back_d)).replace(hour=0, minute=0, second=0, microsecond=0)
        # pick likely latest first based on current time
        for cyc in cycles:
            # only consider cycles that are not "in the future" for today
            candidate_dt = day.replace(hour=cyc)
            if candidate_dt > now + timedelta(hours=1):
                continue
            url = gfs_url(candidate_dt, cyc, 0)
            if http_head_ok(url):
                return (candidate_dt, cyc)
    raise RuntimeError("Could not find an available GFS cycle on NOMADS in last 3 days.")


# -----------------------------
# GRIB reading
# -----------------------------
def open_grib_field(
    grib_path: str,
    short_name: str,
    type_of_level: str,
    level: int,
) -> xr.Dataset:
    """
    Open a GRIB2 dataset filtered by keys.
    """
    # cfgrib filter keys: shortName, typeOfLevel, level
    ds = xr.open_dataset(
        grib_path,
        engine="cfgrib",
        backend_kwargs={
            "filter_by_keys": {
                "shortName": short_name,
                "typeOfLevel": type_of_level,
                "level": int(level),
            }
        },
    )
    return ds


def pick_data_var(ds: xr.Dataset) -> xr.DataArray:
    # Pick first non-coordinate variable.
    for v in ds.data_vars:
        return ds[v]
    raise RuntimeError("No data variables found in dataset.")


def get_latlon(da: xr.DataArray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns lat, lon as 2D arrays.
    cfgrib typically provides latitude/longitude as 2D coords.
    """
    if "latitude" in da.coords and "longitude" in da.coords:
        lat = da["latitude"].values
        lon = da["longitude"].values
        return lat, lon
    # sometimes on dataset
    if "latitude" in da.to_dataset(name="x").coords and "longitude" in da.to_dataset(name="x").coords:
        lat = da.to_dataset(name="x")["latitude"].values
        lon = da.to_dataset(name="x")["longitude"].values
        return lat, lon
    raise RuntimeError("Could not locate latitude/longitude coordinates in GRIB dataset.")


# -----------------------------
# Plotting
# -----------------------------
def conus_extent():
    # (lon_min, lon_max, lat_min, lat_max)
    return (-126.0, -66.0, 22.0, 50.5)

def make_axes(domain: str, model: str):
    if model == "gfs":
        proj = ccrs.Robinson()
        ax = plt.axes(projection=proj)
        ax.set_global()
        return ax, proj
    # HRRR CONUS
    proj = ccrs.LambertConformal(central_longitude=-97.0, central_latitude=38.0)
    ax = plt.axes(projection=proj)
    ax.set_extent(conus_extent(), crs=ccrs.PlateCarree())
    return ax, proj

def add_map_features(ax):
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.6)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.5)
    try:
        ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.35)
    except Exception:
        pass

def render_pcolormesh(
    out_png: str,
    da: xr.DataArray,
    lat: np.ndarray,
    lon: np.ndarray,
    title: str,
    units: str,
    cmap: str = "turbo",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    model: str = "hrrr",
    domain: str = "conus",
):
    plt.figure(figsize=(12.8, 7.2), dpi=120)
    ax, _ = make_axes(domain=domain, model=model)
    add_map_features(ax)

    # Normalize lon to [-180,180] for global plotting sanity
    lon_plot = lon.copy()
    if lon_plot.max() > 180:
        lon_plot = ((lon_plot + 180) % 360) - 180

    mesh = ax.pcolormesh(
        lon_plot,
        lat,
        da.values,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        shading="auto",
        vmin=vmin,
        vmax=vmax,
    )
    cb = plt.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.03, fraction=0.045)
    cb.set_label(units)

    plt.title(title, fontsize=12, pad=10)
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()


def render_contours_mslp(
    out_png: str,
    da: xr.DataArray,
    lat: np.ndarray,
    lon: np.ndarray,
    title: str,
    levels_hpa: List[int],
    model: str = "gfs",
    domain: str = "global",
):
    plt.figure(figsize=(12.8, 7.2), dpi=120)
    ax, _ = make_axes(domain=domain, model=model)
    add_map_features(ax)

    lon_plot = lon.copy()
    if lon_plot.max() > 180:
        lon_plot = ((lon_plot + 180) % 360) - 180

    # Convert Pa -> hPa if needed
    vals = da.values
    if np.nanmean(vals) > 20000:
        vals = vals / 100.0

    cs = ax.contour(
        lon_plot,
        lat,
        vals,
        levels=levels_hpa,
        colors="black",
        linewidths=0.7,
        transform=ccrs.PlateCarree(),
    )
    ax.clabel(cs, inline=True, fontsize=7, fmt="%d")

    plt.title(title, fontsize=12, pad=10)
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()


# -----------------------------
# Supabase upload
# -----------------------------
def supabase_client():
    url = must_env("SUPABASE_URL")
    key = must_env("SUPABASE_SERVICE_ROLE_KEY")
    return create_client(url, key)

def upload_file(sb, bucket: str, object_path: str, file_path: str, content_type: str = "image/png"):
    with open(file_path, "rb") as f:
        data = f.read()
    # upsert=True overwrites existing objects at the same path
    res = sb.storage.from_(bucket).upload(
        path=object_path,
        file=data,
        file_options={"content-type": content_type, "upsert": "true"},
    )
    return res


# -----------------------------
# Recipes (v1)
# -----------------------------
def run_recipe_field(model: str, grib_path: str, item: Dict[str, Any]) -> Tuple[xr.DataArray, np.ndarray, np.ndarray]:
    g = item["grib"]
    ds = open_grib_field(
        grib_path=grib_path,
        short_name=g["shortName"],
        type_of_level=g["typeOfLevel"],
        level=int(g["level"]),
    )
    da = pick_data_var(ds)
    lat, lon = get_latlon(da)
    return da, lat, lon


# -----------------------------
# Main
# -----------------------------
def main():
    # Load registry
    with open("products.json", "r", encoding="utf-8") as f:
        registry = json.load(f)

    sb = None
    if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
        sb = supabase_client()
        log(f"Supabase configured. Bucket={SUPABASE_BUCKET}")
    else:
        log("Supabase not configured (SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY missing). Will render locally only.")

    tmp = tempfile.mkdtemp(prefix="wxplots_")
    log(f"Working dir: {tmp}")

    try:
        # --- HRRR
        if "hrrr" in registry["models"]:
            run_dt, cyc = find_latest_hrrr_cycle(domain="conus")
            log(f"HRRR latest cycle: {run_dt:%Y-%m-%d} {cyc:02d}Z")

            hrrr_model = registry["models"]["hrrr"]
            domain_cfg = hrrr_model["domains"]["conus"]

            max_fhr = min(HRRR_MAX_FHR, int(domain_cfg["fhr_range"]["end"]))
            fhrs = list(range(0, max_fhr + 1, int(domain_cfg["fhr_range"]["step"])))

            for group_key, group in hrrr_model["products"].items():
                for item in group["items"]:
                    item_id = item["id"]
                    label = item["label"]
                    recipe = item["recipe"]

                    # v1 only supports "field" right now (we’ll add CAPE/shear recipes next step)
                    if recipe != "field":
                        log(f"[HRRR] SKIP (recipe not implemented yet): {item_id} ({recipe})")
                        continue

                    for fhr in fhrs:
                        url = hrrr_url(run_dt, cyc, fhr, domain="conus")
                        grib_local = os.path.join(tmp, f"hrrr_conus_{cyc:02d}_f{fhr:02d}.grib2")

                        # Download GRIB2 (once per fhr; reuse across products by caching filename)
                        if not os.path.exists(grib_local):
                            log(f"[HRRR] Download f{fhr:02d}: {url}")
                            http_get_stream(url, grib_local)

                        da, lat, lon = run_recipe_field("hrrr", grib_local, item)

                        plot_cfg = item.get("plot", {})
                        units = plot_cfg.get("units", "")
                        cmap = plot_cfg.get("cmap", "turbo")
                        vmin, vmax = None, None
                        if "range" in plot_cfg and isinstance(plot_cfg["range"], list) and len(plot_cfg["range"]) == 2:
                            vmin, vmax = float(plot_cfg["range"][0]), float(plot_cfg["range"][1])

                        title = f"HRRR CONUS {cyc:02d}Z • f{fhr:02d} • {label}"
                        out_png = os.path.join(tmp, f"hrrr_conus_{item_id}_f{fhr:03d}.png")

                        render_pcolormesh(
                            out_png=out_png,
                            da=da,
                            lat=lat,
                            lon=lon,
                            title=title,
                            units=units,
                            cmap=cmap,
                            vmin=vmin,
                            vmax=vmax,
                            model="hrrr",
                            domain="conus",
                        )

                        object_path = f"hrrr/conus/latest/{item_id}_f{fhr:03d}.png"
                        if sb:
                            upload_file(sb, SUPABASE_BUCKET, object_path, out_png)
                            log(f"[HRRR] Uploaded: {object_path}")
                        else:
                            log(f"[HRRR] Rendered: {out_png}")

        # --- GFS
        if "gfs" in registry["models"]:
            run_dt, cyc = find_latest_gfs_cycle()
            log(f"GFS latest cycle: {run_dt:%Y-%m-%d} {cyc:02d}Z")

            gfs_model = registry["models"]["gfs"]
            domain_cfg = gfs_model["domains"]["global"]

            max_fhr = min(GFS_MAX_FHR, int(domain_cfg["fhr_range"]["end"]))
            step = int(domain_cfg["fhr_range"]["step"])
            fhrs = list(range(0, max_fhr + 1, step))

            for group_key, group in gfs_model["products"].items():
                for item in group["items"]:
                    item_id = item["id"]
                    label = item["label"]
                    recipe = item["recipe"]

                    if recipe != "field":
                        log(f"[GFS] SKIP (recipe not implemented yet): {item_id} ({recipe})")
                        continue

                    for fhr in fhrs:
                        url = gfs_url(run_dt, cyc, fhr)
                        grib_local = os.path.join(tmp, f"gfs_{cyc:02d}_f{fhr:03d}.grib2")

                        if not os.path.exists(grib_local):
                            log(f"[GFS] Download f{fhr:03d}: {url}")
                            http_get_stream(url, grib_local)

                        da, lat, lon = run_recipe_field("gfs", grib_local, item)

                        title = f"GFS {cyc:02d}Z • f{fhr:03d} • {label}"
                        out_png = os.path.join(tmp, f"gfs_global_{item_id}_f{fhr:03d}.png")

                        plot_cfg = item.get("plot", {})
                        kind = plot_cfg.get("kind", "pcolormesh")

                        if item_id == "mslp" or kind == "contour":
                            levels = plot_cfg.get("levels_hpa", [960, 968, 976, 984, 992, 1000, 1008, 1016, 1024, 1032, 1040])
                            render_contours_mslp(
                                out_png=out_png,
                                da=da,
                                lat=lat,
                                lon=lon,
                                title=title,
                                levels_hpa=list(levels),
                                model="gfs",
                                domain="global",
                            )
                        else:
                            units = plot_cfg.get("units", "")
                            cmap = plot_cfg.get("cmap", "turbo")
                            vmin, vmax = None, None
                            if "range" in plot_cfg and isinstance(plot_cfg["range"], list) and len(plot_cfg["range"]) == 2:
                                vmin, vmax = float(plot_cfg["range"][0]), float(plot_cfg["range"][1])

                            render_pcolormesh(
                                out_png=out_png,
                                da=da,
                                lat=lat,
                                lon=lon,
                                title=title,
                                units=units,
                                cmap=cmap,
                                vmin=vmin,
                                vmax=vmax,
                                model="gfs",
                                domain="global",
                            )

                        object_path = f"gfs/global/latest/{item_id}_f{fhr:03d}.png"
                        if sb:
                            upload_file(sb, SUPABASE_BUCKET, object_path, out_png)
                            log(f"[GFS] Uploaded: {object_path}")
                        else:
                            log(f"[GFS] Rendered: {out_png}")

        log("Done.")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
