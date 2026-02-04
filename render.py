#!/usr/bin/env python3
"""
Render HRRR "simulated reflectivity" frames by downloading IEM's HRRR REFD PNG rasters,
then upload them to Supabase Storage.

Why:
- The old /hrrr/<cycle>/simref_###.png path 404s.
- IEM publishes current HRRR reflectivity rasters as:
  https://mesonet.agron.iastate.edu/data/gis/images/4326/hrrr/refd_0000.png
  ... refd_0060.png ... refd_1080.png (18h)

Outputs (Supabase bucket = wx-plots by default):
- hrrr/conus/latest/simref_f000.png ... simref_f018.png
- hrrr/conus/runs/<YYYYMMDDHH>/simref_f000.png ... simref_f018.png
- hrrr/conus/latest/manifest.json
- hrrr/conus/runs/<YYYYMMDDHH>/manifest.json
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

import requests


# -----------------------
# Config
# -----------------------

IEM_BASE = "https://mesonet.agron.iastate.edu/data/gis/images/4326/hrrr"
# Metadata JSON exists for each frame; using the 18h one to grab init time.
IEM_META_JSON = f"{IEM_BASE}/refd_1080.json"

DEFAULT_BUCKET = "wx-plots"
DEFAULT_DOMAIN = "conus"
DEFAULT_MAX_FH = 18  # 0..18 inclusive
DEFAULT_STEP_HOURS = 1  # 60-min steps

OUT_DIR = Path("out")


@dataclass
class Settings:
    supabase_url: str
    supabase_service_role_key: str
    bucket: str
    domain: str
    max_fh: int
    step_hours: int


def env_required(name: str) -> str:
    v = os.environ.get(name, "").strip()
    if not v:
        print(f"Missing required env var: {name}", file=sys.stderr)
        sys.exit(1)
    return v


def load_settings() -> Settings:
    return Settings(
        supabase_url=env_required("SUPABASE_URL").rstrip("/"),
        supabase_service_role_key=env_required("SUPABASE_SERVICE_ROLE_KEY"),
        bucket=os.environ.get("SUPABASE_BUCKET", DEFAULT_BUCKET).strip() or DEFAULT_BUCKET,
        domain=os.environ.get("HRRR_DOMAIN", DEFAULT_DOMAIN).strip() or DEFAULT_DOMAIN,
        max_fh=int(os.environ.get("MAX_FH", str(DEFAULT_MAX_FH))),
        step_hours=int(os.environ.get("STEP_HOURS", str(DEFAULT_STEP_HOURS))),
    )


# -----------------------
# IEM download helpers
# -----------------------

def http_get_bytes(url: str, timeout: Tuple[int, int] = (10, 40), retries: int = 3) -> bytes:
    last_err = None
    headers = {
        "User-Agent": "wx-plots-renderer/1.0 (GitHub Actions)",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            if r.status_code == 200 and r.content:
                return r.content
            last_err = RuntimeError(f"HTTP {r.status_code} for {url}")
        except Exception as e:
            last_err = e
        time.sleep(0.6 * attempt)
    raise RuntimeError(f"Fetch failed after {retries} tries for {url}: {last_err}")


def parse_iso_z(s: str) -> datetime:
    # Handles "2026-02-04T21:00:00Z"
    s = s.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s).astimezone(timezone.utc)


def get_model_init_stamp() -> Tuple[str, str]:
    """
    Returns:
      init_iso: e.g. "2026-02-04T21:00:00Z"
      init_stamp: e.g. "2026020421"  (folder-safe)
    """
    meta = json.loads(http_get_bytes(IEM_META_JSON).decode("utf-8"))
    init_iso = meta.get("model_init_utc") or meta.get("init_utc") or ""
    if not init_iso:
        # fallback: "now"
        dt = datetime.now(timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ"), dt.strftime("%Y%m%d%H")

    dt = parse_iso_z(init_iso)
    init_iso_norm = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    init_stamp = dt.strftime("%Y%m%d%H")
    return init_iso_norm, init_stamp


def iem_refd_url(forecast_hour: int) -> str:
    minutes = forecast_hour * 60
    return f"{IEM_BASE}/refd_{minutes:04d}.png"


# -----------------------
# Supabase Storage upload (REST)
# -----------------------

def supabase_upload_bytes(settings: Settings, remote_path: str, data: bytes, content_type: str) -> None:
    """
    Upload/Upsert to:
      POST {SUPABASE_URL}/storage/v1/object/{bucket}/{remote_path}
    with x-upsert: true
    """
    url = f"{settings.supabase_url}/storage/v1/object/{settings.bucket}/{remote_path.lstrip('/')}"
    headers = {
        "Authorization": f"Bearer {settings.supabase_service_role_key}",
        "apikey": settings.supabase_service_role_key,
        "x-upsert": "true",
        "Content-Type": content_type,
    }

    r = requests.post(url, headers=headers, data=data, timeout=(10, 60))
    if r.status_code in (200, 201):
        return

    # Some setups prefer PUT; try it as a fallback.
    r2 = requests.put(url, headers=headers, data=data, timeout=(10, 60))
    if r2.status_code in (200, 201):
        return

    raise RuntimeError(
        f"Supabase upload failed for {remote_path} (POST {r.status_code}, PUT {r2.status_code}). "
        f"POST body: {r.text[:300]} PUT body: {r2.text[:300]}"
    )


def supabase_upload_file(settings: Settings, remote_path: str, file_path: Path, content_type: str) -> None:
    supabase_upload_bytes(settings, remote_path, file_path.read_bytes(), content_type)


# -----------------------
# Main
# -----------------------

def main() -> None:
    settings = load_settings()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    init_iso, init_stamp = get_model_init_stamp()
    print(f"HRRR init: {init_iso} (stamp {init_stamp})")

    # Forecast hours list
    hours = list(range(0, settings.max_fh + 1, settings.step_hours))
    failed: List[str] = []

    # Download all frames first (so uploads don’t partially run if fetch is broken)
    local_files: List[Tuple[int, Path]] = []
    for fh in hours:
        url = iem_refd_url(fh)
        local_name = f"simref_f{fh:03d}.png"
        local_path = OUT_DIR / local_name
        try:
            print(f"Downloading fh={fh:03d} -> {url}")
            data = http_get_bytes(url)
            local_path.write_bytes(data)
            local_files.append((fh, local_path))
            time.sleep(0.15)  # be polite
        except Exception as e:
            failed.append(f"simref f{fh:03d}: {e}")

    if failed:
        print("\nSome frames failed:\n")
        for line in failed:
            print(f" - {line}")
        # Fail hard so you notice (and don't publish a broken run).
        sys.exit(2)

    # Upload frames
    run_prefix = f"hrrr/{settings.domain}/runs/{init_stamp}"
    latest_prefix = f"hrrr/{settings.domain}/latest"

    for fh, local_path in local_files:
        fname = f"simref_f{fh:03d}.png"
        rp_latest = f"{latest_prefix}/{fname}"
        rp_run = f"{run_prefix}/{fname}"

        print(f"Uploading -> {rp_latest}")
        supabase_upload_file(settings, rp_latest, local_path, "image/png")

        print(f"Uploading -> {rp_run}")
        supabase_upload_file(settings, rp_run, local_path, "image/png")

    # Upload manifest
    manifest = {
        "model": "hrrr",
        "domain": settings.domain,
        "product": "simref",
        "init_utc": init_iso,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "forecast_hours": hours,
        "source": {
            "provider": "IEM",
            "refd_base": IEM_BASE,
        },
    }
    manifest_bytes = json.dumps(manifest, indent=2).encode("utf-8")

    print("Uploading -> latest/manifest.json")
    supabase_upload_bytes(settings, f"{latest_prefix}/manifest.json", manifest_bytes, "application/json")

    print("Uploading -> run manifest.json")
    supabase_upload_bytes(settings, f"{run_prefix}/manifest.json", manifest_bytes, "application/json")

    print("\nDone ✅")


if __name__ == "__main__":
    main()
