#!/usr/bin/env python3
from __future__ import annotations

import io
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote

import numpy as np
import requests
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------
# Settings
# -----------------------

IEM_BASE = "https://mesonet.agron.iastate.edu/data/gis/images/4326/hrrr"
IEM_META = f"{IEM_BASE}/refd_1080.json"  # contains model_init_utc often

NOMADS_BASE = "https://nomads.ncep.noaa.gov/cgi-bin/filter_hrrr_2d.pl"

DEFAULT_DOMAIN = "conus"
DEFAULT_MAX_FH = 18

# quick CONUS-ish extent (for imshow)
EXTENT = (-130.0, -65.0, 22.0, 52.0)  # (lonW, lonE, latS, latN)

MS_TO_KT = 1.9438444924406048

@dataclass
class Settings:
    supabase_url: str
    supabase_key: str
    bucket: str
    domain: str
    max_fh: int


def die(msg: str, code: int = 1):
    print(msg, file=sys.stderr)
    sys.exit(code)


def env_required(name: str) -> str:
    v = os.environ.get(name, "").strip()
    if not v:
        die(f"Missing required env var: {name}")
    return v


def load_settings() -> Settings:
    return Settings(
        supabase_url=env_required("SUPABASE_URL").rstrip("/"),
        supabase_key=env_required("SUPABASE_SERVICE_ROLE_KEY"),
        bucket=os.environ.get("SUPABASE_BUCKET", "wx-plots").strip() or "wx-plots",
        domain=os.environ.get("HRRR_DOMAIN", DEFAULT_DOMAIN).strip() or DEFAULT_DOMAIN,
        max_fh=int(os.environ.get("MAX_FH", str(DEFAULT_MAX_FH))),
    )


def http_get_bytes(url: str, timeout=(10, 60), retries: int = 3) -> bytes:
    headers = {
        "User-Agent": "wx-plots-renderer/2.0 (GitHub Actions)",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }
    last = None
    for a in range(1, retries + 1):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            if r.status_code == 200 and r.content:
                return r.content
            last = RuntimeError(f"HTTP {r.status_code} for {url}")
        except Exception as e:
            last = e
        time.sleep(0.7 * a)
    raise RuntimeError(f"Fetch failed after {retries} tries: {last}")


def parse_iso_z(s: str) -> datetime:
    s = s.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s).astimezone(timezone.utc)


def get_hrrr_init_stamp() -> Tuple[str, str]:
    """
    Returns (init_iso_utc, init_stamp_YYYYMMDDHH)
    """
    meta = json.loads(http_get_bytes(IEM_META).decode("utf-8"))
    init_iso = meta.get("model_init_utc") or meta.get("init_utc") or ""
    if init_iso:
        dt = parse_iso_z(init_iso)
    else:
        dt = datetime.now(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ"), dt.strftime("%Y%m%d%H")


# -----------------------
# Supabase upload (REST)
# -----------------------

def supabase_upload_bytes(settings: Settings, remote_path: str, data: bytes, content_type: str):
    remote_path = remote_path.lstrip("/")
    url = f"{settings.supabase_url}/storage/v1/object/{settings.bucket}/{quote(remote_path)}"
    headers = {
        "Authorization": f"Bearer {settings.supabase_key}",
        "apikey": settings.supabase_key,
        "x-upsert": "true",
        "Content-Type": content_type,
    }
    r = requests.post(url, headers=headers, data=data, timeout=(10, 60))
    if r.status_code in (200, 201):
        return
    r2 = requests.put(url, headers=headers, data=data, timeout=(10, 60))
    if r2.status_code in (200, 201):
        return
    raise RuntimeError(f"Supabase upload failed {remote_path} (POST {r.status_code}, PUT {r2.status_code}) "
                       f"POST: {r.text[:250]} PUT: {r2.text[:250]}")


# -----------------------
# IEM SimRef (REFD) fetch
# -----------------------

def iem_refd_url(fh: int) -> str:
    minutes = fh * 60
    return f"{IEM_BASE}/refd_{minutes:04d}.png"


# -----------------------
# NOMADS HRRR GRIB fetch (subset)
# -----------------------

def nomads_url_wrfsfc(init_stamp: str, fh: int, want_cape=True, want_10m=True) -> str:
    """
    wrfsfc file: hrrr.tHHz.wrfsfcfFF.grib2
    """
    ymd = init_stamp[:8]
    hh = init_stamp[8:10]
    ff = f"{fh:02d}"

    params = []
    if want_cape:
        params += ["var_CAPE=on"]
    if want_10m:
        params += ["var_UGRD=on", "var_VGRD=on", "lev_10_m_above_ground=on"]

    qs = "&".join(params)
    return (f"{NOMADS_BASE}?file=hrrr.t{hh}z.wrfsfcf{ff}.grib2"
            f"&dir=%2Fhrrr.{ymd}%2Fconus&{qs}")


def nomads_url_wrfprs(init_stamp: str, fh: int) -> str:
    """
    wrfprs file: hrrr.tHHz.wrfprsfFF.grib2  (pressure levels)
    """
    ymd = init_stamp[:8]
    hh = init_stamp[8:10]
    ff = f"{fh:02d}"
    # request only U/V at 850 mb
    qs = "var_UGRD=on&var_VGRD=on&lev_850_mb=on"
    return (f"{NOMADS_BASE}?file=hrrr.t{hh}z.wrfprsf{ff}.grib2"
            f"&dir=%2Fhrrr.{ymd}%2Fconus&{qs}")


def open_grib_bytes(grib_bytes: bytes) -> xr.Dataset:
    """
    Read GRIB2 bytes with cfgrib via a temp in-memory file approach.
    """
    # cfgrib needs a real file-like path; write to temp file in /tmp
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".grib2", delete=True) as f:
        f.write(grib_bytes)
        f.flush()
        ds = xr.open_dataset(f.name, engine="cfgrib")
    return ds


def get_latlon(ds: xr.Dataset) -> Tuple[np.ndarray, np.ndarray]:
    # cfgrib usually provides latitude/longitude variables
    if "latitude" in ds and "longitude" in ds:
        lat = ds["latitude"].values
        lon = ds["longitude"].values
        # sometimes lon is 0..360
        lon = np.where(lon > 180, lon - 360, lon)
        return lat, lon
    # fallback: coords might be named "lat" / "lon"
    for a, b in [("lat", "lon"), ("latitude", "longitude")]:
        if a in ds.coords and b in ds.coords:
            lat = ds.coords[a].values
            lon = ds.coords[b].values
            lon = np.where(lon > 180, lon - 360, lon)
            return lat, lon
    raise RuntimeError("Could not find lat/lon in dataset")


# -----------------------
# Plotting
# -----------------------

def save_field_png(field: np.ndarray, lat: np.ndarray, lon: np.ndarray,
                   title: str, out_w: int = 1200, out_h: int = 720) -> bytes:
    """
    Very fast "Tidbits-like" static plot: imshow with extent. No Cartopy to keep Actions light.
    """
    lonW, lonE, latS, latN = EXTENT

    # mask outside extent (keeps plot clean)
    mask = (lon < lonW) | (lon > lonE) | (lat < latS) | (lat > latN)
    plot = np.array(field, dtype=np.float32)
    plot = np.where(mask, np.nan, plot)

    fig = plt.figure(figsize=(out_w/100, out_h/100), dpi=100)
    ax = plt.gca()
    ax.set_facecolor("#101010")

    # imshow needs regular grid; HRRR is near-regular on lat/lon arrays here
    # We plot by extent; distortion is acceptable for MVP.
    im = ax.imshow(plot, origin="upper",
                   extent=(lon.min(), lon.max(), lat.min(), lat.max()),
                   aspect="auto")

    ax.set_xlim(lonW, lonE)
    ax.set_ylim(latS, latN)
    ax.set_title(title, color="white", fontsize=14, pad=10)
    ax.tick_params(colors="#cfcfcf", labelsize=9)
    ax.grid(color="white", alpha=0.08, linewidth=0.8)

    cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.ax.tick_params(labelsize=9, colors="#cfcfcf")

    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


# -----------------------
# Main render
# -----------------------

def main():
    settings = load_settings()
    init_iso, init_stamp = get_hrrr_init_stamp()

    latest_prefix = f"hrrr/{settings.domain}/latest"
    run_prefix = f"hrrr/{settings.domain}/runs/{init_stamp}"

    manifest: Dict = {
        "model": "hrrr",
        "domain": settings.domain,
        "init_utc": init_iso,
        "init_stamp": init_stamp,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "products": {},
    }

    hours = list(range(0, settings.max_fh + 1))
    manifest["forecast_hours"] = hours

    # ---- 1) SimRef from IEM (REFD)
    simref_ok = []
    for fh in hours:
        try:
            png = http_get_bytes(iem_refd_url(fh))
            name = f"simref_f{fh:03d}.png"
            supabase_upload_bytes(settings, f"{latest_prefix}/{name}", png, "image/png")
            supabase_upload_bytes(settings, f"{run_prefix}/{name}", png, "image/png")
            simref_ok.append(fh)
            time.sleep(0.12)
        except Exception as e:
            print(f"[simref] fh{fh:03d} failed: {e}", file=sys.stderr)

    manifest["products"]["simref"] = {"ok_hours": simref_ok, "source": "IEM REFD PNG"}

    # ---- 2) CAPE + 0–850 bulk shear from NOMADS HRRR GRIB
    cape_ok, shear_ok = [], []

    for fh in hours:
        try:
            # wrfsfc: CAPE + 10m wind
            sfc_url = nomads_url_wrfsfc(init_stamp, fh, want_cape=True, want_10m=True)
            sfc_bytes = http_get_bytes(sfc_url, timeout=(15, 120), retries=2)
            ds_sfc = open_grib_bytes(sfc_bytes)

            # CAPE variable name can vary; look for anything containing "CAPE"
            cape_var = None
            for v in ds_sfc.data_vars:
                if "CAPE" in v.upper():
                    cape_var = v
                    break
            if cape_var is None:
                raise RuntimeError("CAPE var not found in wrfsfc subset")

            cape = ds_sfc[cape_var].values.astype(np.float32)

            # 10m wind
            u10 = None
            v10 = None
            for v in ds_sfc.data_vars:
                up = v.upper()
                if up.startswith("UGRD"):
                    u10 = ds_sfc[v].values.astype(np.float32)
                if up.startswith("VGRD"):
                    v10 = ds_sfc[v].values.astype(np.float32)
            if u10 is None or v10 is None:
                raise RuntimeError("10m U/V not found in wrfsfc subset")

            lat, lon = get_latlon(ds_sfc)

            # wrfprs: 850mb wind
            prs_url = nomads_url_wrfprs(init_stamp, fh)
            prs_bytes = http_get_bytes(prs_url, timeout=(15, 120), retries=2)
            ds_prs = open_grib_bytes(prs_bytes)

            u850 = None
            v850 = None
            for v in ds_prs.data_vars:
                up = v.upper()
                if up.startswith("UGRD"):
                    u850 = ds_prs[v].values.astype(np.float32)
                if up.startswith("VGRD"):
                    v850 = ds_prs[v].values.astype(np.float32)
            if u850 is None or v850 is None:
                raise RuntimeError("850mb U/V not found in wrfprs subset")

            # shear magnitude in knots
            shear = np.sqrt((u850 - u10) ** 2 + (v850 - v10) ** 2) * MS_TO_KT

            # render CAPE + shear
            cape_png = save_field_png(
                cape, lat, lon,
                title=f"HRRR CONUS SBCAPE (J/kg)  init {init_stamp}  f{fh:02d}"
            )
            shear_png = save_field_png(
                shear, lat, lon,
                title=f"HRRR CONUS 0–850mb Bulk Shear (kt)  init {init_stamp}  f{fh:02d}"
            )

            cape_name = f"cape_f{fh:03d}.png"
            shr_name = f"shr0_850_f{fh:03d}.png"

            supabase_upload_bytes(settings, f"{latest_prefix}/{cape_name}", cape_png, "image/png")
            supabase_upload_bytes(settings, f"{run_prefix}/{cape_name}", cape_png, "image/png")
            cape_ok.append(fh)

            supabase_upload_bytes(settings, f"{latest_prefix}/{shr_name}", shear_png, "image/png")
            supabase_upload_bytes(settings, f"{run_prefix}/{shr_name}", shear_png, "image/png")
            shear_ok.append(fh)

            time.sleep(0.25)

        except Exception as e:
            print(f"[cape/shear] fh{fh:03d} failed: {e}", file=sys.stderr)

    manifest["products"]["cape"] = {"ok_hours": cape_ok, "source": "NOMADS HRRR GRIB (wrfsfc)"}
    manifest["products"]["shr0_850"] = {"ok_hours": shear_ok, "source": "NOMADS HRRR GRIB (wrfsfc+wrfprs)"}

    # ---- upload manifest
    mbytes = json.dumps(manifest, indent=2).encode("utf-8")
    supabase_upload_bytes(settings, f"{latest_prefix}/manifest.json", mbytes, "application/json")
    supabase_upload_bytes(settings, f"{run_prefix}/manifest.json", mbytes, "application/json")

    print("Done ✅")


if __name__ == "__main__":
    main()
