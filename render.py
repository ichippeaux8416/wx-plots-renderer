import os
import sys
import time
import json
import requests

# ---------------------------
# Config via env vars
# ---------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "wx-plots")
OUT_PREFIX = os.getenv("OUT_PREFIX", "hrrr/conus/latest").strip("/")
FHR_MAX = int(os.getenv("FHR_MAX", "18"))
STRICT = os.getenv("STRICT", "0") == "1"

# Iowa State Mesonet HRRR (CONUS) image base
# NOTE: This is the same domain you've been using for HRRR images.
ISU_BASE = "https://mesonet.agron.iastate.edu"

# What we are rendering right now:
# - Simulated Reflectivity (composite)
PRODUCTS = [
    {
        "key": "simref",
        "label": "Simulated Reflectivity",
        # Adjust the source path template if your earlier script used a different endpoint.
        # This one is a common Iowa State GIS image endpoint.
        "src_template": ISU_BASE + "/data/gis/images/4326/hrrr/{cycle}/simref_{fhr:03d}.png",
        "dst_name": "simref_f{fhr:03d}.png",
    }
]

# ---------------------------
# Helpers
# ---------------------------
def die(msg: str, code: int = 1):
    print(msg, file=sys.stderr)
    sys.exit(code)

def need_env():
    if not SUPABASE_URL:
        die("Missing SUPABASE_URL env var.")
    if not SUPABASE_SERVICE_ROLE_KEY:
        die("Missing SUPABASE_SERVICE_ROLE_KEY env var.")

def utc_cycle_hour(now=None):
    """Return latest completed HRRR cycle hour (00-23) in UTC, as string 'HH'."""
    if now is None:
        now = time.gmtime()
    # HRRR runs hourly; pick current UTC hour (often available shortly after top of hour).
    return f"{now.tm_hour:02d}"

def supabase_put_object(path: str, content: bytes, content_type: str):
    """
    Upload to Supabase Storage using service role key.
    POST /storage/v1/object/{bucket}/{path}
    """
    url = f"{SUPABASE_URL}/storage/v1/object/{SUPABASE_BUCKET}/{path}"
    headers = {
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Content-Type": content_type,
        # Overwrite if exists
        "x-upsert": "true",
    }
    r = requests.post(url, headers=headers, data=content, timeout=60)
    if r.status_code >= 300:
        raise RuntimeError(f"Supabase upload failed {r.status_code}: {r.text}")
    return r

def fetch_png(url: str) -> bytes:
    r = requests.get(url, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"Fetch failed {r.status_code} for {url}")
    ct = r.headers.get("content-type", "")
    if "png" not in ct and not url.lower().endswith(".png"):
        # still allow, but warn
        print(f"Warning: unexpected content-type '{ct}' for {url}")
    return r.content

def write_latest_pointer(prefix: str, cycle: str):
    """
    Optional: write a tiny JSON file to point to the cycle used
    """
    obj_path = f"{prefix}/latest.json"
    payload = json.dumps({"cycle": cycle, "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}).encode("utf-8")
    supabase_put_object(obj_path, payload, "application/json")
    print(f"Uploaded {obj_path}")

def main():
    need_env()

    cycle = utc_cycle_hour()
    print(f"Using cycle (UTC hour): {cycle}")

    any_success = False
    failures = []

    for prod in PRODUCTS:
        key = prod["key"]
        print(f"\n== {prod['label']} ({key}) ==")

        for fhr in range(0, FHR_MAX + 1):
            src = prod["src_template"].format(cycle=cycle, fhr=fhr)
            dst = f"{OUT_PREFIX}/{key}_f{fhr:03d}.png"

            try:
                print(f"Fetching: {src}")
                png = fetch_png(src)

                # Upload to Supabase
                supabase_put_object(dst, png, "image/png")
                print(f"Uploaded: {dst}")

                any_success = True

            except Exception as e:
                msg = f"{key} f{fhr:03d} failed: {e}"
                print(msg, file=sys.stderr)
                failures.append(msg)
                if STRICT:
                    raise

    # update pointer file
    if any_success:
        write_latest_pointer(OUT_PREFIX, cycle)

    if failures:
        print("\nSome frames failed:")
        for m in failures:
            print(" - " + m)

    if not any_success:
        die("No frames rendered successfully. Check source URLs / permissions.", 2)

if __name__ == "__main__":
    main()
