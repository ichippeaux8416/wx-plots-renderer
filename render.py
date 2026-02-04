#!/usr/bin/env python3
import os
import json
import time
from datetime import datetime, timezone
from urllib.parse import quote

import requests

# -------- CONFIG (env) --------
SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
SUPABASE_BUCKET = os.environ.get("SUPABASE_BUCKET", "wx-plots")

# where in the bucket we store "latest" images
OUT_PREFIX = os.environ.get("OUT_PREFIX", "hrrr/conus/latest").strip("/")

# how many forecast hours to try (HRRR is hourly; keep this reasonable for Actions)
FHR_MAX = int(os.environ.get("FHR_MAX", "18"))  # 0..18 by default
TIMEOUT = int(os.environ.get("HTTP_TIMEOUT", "20"))

STRICT = os.environ.get("STRICT", "0") == "1"  # if 1, fail job if we can't fetch anything

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    raise SystemExit("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY env vars.")

BASE_STORAGE_API = f"{SUPABASE_URL}/storage/v1"
HEADERS_AUTH = {
    "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
    "apikey": SUPABASE_SERVICE_ROLE_KEY,
}

# -------- SOURCE DISCOVERY --------
# We try a few common Iowa State Mesonet / IEM-style patterns.
# If one works for f000, we keep using that pattern for that product.
SOURCE_TEMPLATES = [
    # Most common pattern people use for IEM GIS images
    "https://mesonet.agron.iastate.edu/data/gis/images/4326/hrrr/conus/{stem}_f{fhr:03d}.png",
    "https://mesonet.agron.iastate.edu/data/gis/images/3857/hrrr/conus/{stem}_f{fhr:03d}.png",
    # Sometimes products are in a "hrrr/{domain}/" folder but still similar
    "https://mesonet.agron.iastate.edu/data/gis/images/4326/hrrr/{stem}/{stem}_f{fhr:03d}.png",
]

# “Products” we want to attempt.
# For each product, we provide possible stems because naming varies by source.
PRODUCTS = [
    ("simref", ["simref", "simref_comp", "refc"]),
    ("cape",   ["mucape", "cape", "sbcape"]),
    ("shr0_850", ["shr0_850", "shr_sfc_850", "shear_0_850", "sfc_850_shear", "shr850"]),  # will skip if none exist
]


def http_get(url: str) -> requests.Response:
    return requests.get(url, timeout=TIMEOUT, headers={"User-Agent": "wx-plots-renderer/1.0"}, stream=True)


def find_working_template(stems):
    """
    Returns (template, stem) that works for fhr=0, else (None, None).
    """
    for stem in stems:
        for tpl in SOURCE_TEMPLATES:
            test_url = tpl.format(stem=stem, fhr=0)
            try:
                r = http_get(test_url)
                if r.status_code == 200 and "image" in (r.headers.get("content-type", "").lower()):
                    return tpl, stem
            except Exception:
                pass
    return None, None


def upload_bytes(object_path: str, content: bytes, content_type: str) -> bool:
    # Supabase Storage upload endpoint:
    # /storage/v1/object/<bucket>/<path>
    # We try PUT first, then POST fallback (some setups accept one or the other).
    object_path = object_path.lstrip("/")
    url = f"{BASE_STORAGE_API}/object/{SUPABASE_BUCKET}/{quote(object_path)}"

    headers = {
        **HEADERS_AUTH,
        "Content-Type": content_type,
        "x-upsert": "true",
        "Cache-Control": "no-store",
    }

    # Try PUT
    r = requests.put(url, headers=headers, data=content, timeout=TIMEOUT)
    if r.status_code in (200, 201):
        return True

    # Fallback to POST (some APIs prefer POST)
    r2 = requests.post(url, headers=headers, data=content, timeout=TIMEOUT)
    if r2.status_code in (200, 201):
        return True

    print(f"[upload] Failed {object_path}: PUT {r.status_code} / POST {r2.status_code}")
    # print response bodies (truncated)
    try:
        print("PUT body:", (r.text or "")[:500])
    except Exception:
        pass
    try:
        print("POST body:", (r2.text or "")[:500])
    except Exception:
        pass
    return False


def main():
    print("== wx-plots-renderer ==")
    print("UTC now:", datetime.now(timezone.utc).isoformat())
    print("Bucket:", SUPABASE_BUCKET)
    print("Prefix:", OUT_PREFIX)
    print("FHR_MAX:", FHR_MAX)

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "bucket": SUPABASE_BUCKET,
        "prefix": OUT_PREFIX,
        "products": {},
    }

    any_uploaded = False

    for prod_key, stems in PRODUCTS:
        tpl, stem = find_working_template(stems)
        if not tpl:
            print(f"[{prod_key}] No working source found (skipping). Tried stems={stems}")
            manifest["products"][prod_key] = {"ok": False, "reason": "no_source_found", "hours": []}
            continue

        print(f"[{prod_key}] Using stem='{stem}' with template='{tpl}'")
        hours_ok = []

        for fhr in range(0, FHR_MAX + 1):
            src_url = tpl.format(stem=stem, fhr=fhr)
            try:
                r = http_get(src_url)
                if r.status_code != 200:
                    # skip missing frames
                    continue
                ctype = r.headers.get("content-type", "image/png").split(";")[0].strip() or "image/png"
                if "image" not in ctype:
                    continue

                content = r.content
                if not content:
                    continue

                out_name = f"{prod_key}_f{fhr:03d}.png"
                out_path = f"{OUT_PREFIX}/{out_name}"

                ok = upload_bytes(out_path, content, ctype)
                if ok:
                    any_uploaded = True
                    hours_ok.append(fhr)
                    print(f"  ✓ uploaded {out_name}")
                else:
                    print(f"  ✗ upload failed {out_name}")

                # be gentle to upstream
                time.sleep(0.2)

            except Exception as e:
                print(f"  ! error f{fhr:03d}: {e}")

        manifest["products"][prod_key] = {
            "ok": len(hours_ok) > 0,
            "source_stem": stem,
            "source_template": tpl,
            "hours": hours_ok,
        }

    # Upload manifest.json so the front-end can know what exists
    manifest_bytes = json.dumps(manifest, indent=2).encode("utf-8")
    m_ok = upload_bytes(f"{OUT_PREFIX}/manifest.json", manifest_bytes, "application/json")
    print("manifest upload:", "OK" if m_ok else "FAILED")

    if not any_uploaded and STRICT:
        raise SystemExit("No products uploaded (STRICT=1).")

    print("Done.")


if __name__ == "__main__":
    main()
