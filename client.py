
import requests
import json

# === CONFIG: change these to test ===
BASE_URL = "http://localhost:8000"   # or your Render URL
SYSTEM = "sun-mars"                 # e.g., "earth-moon", "sun-earth", etc.
FAMILY = "butterfly"                       # any valid family
LIBR = None                               # e.g., 1-5 if needed
BRANCH = 'N'                           # e.g., "N", "S", "E", "W" if needed

# === Flexible Query Function ===
def query_orbits(sys, family, libr=None, branch=None):
    payload = {
        "sys": sys,
        "family": family
    }
    if libr is not None:
        payload["libr"] = libr
    if branch is not None:
        payload["branch"] = branch

    r = requests.post(f"{BASE_URL}/orbits/query", json=payload)
    r.raise_for_status()
    result_bundle = r.json()
    print(f"✅ Queried orbits: system={sys}, family={family}, libr={libr}, branch={branch}")
    return result_bundle

def get_family_info(result_bundle):
    r = requests.post(f"{BASE_URL}/orbits/info", json=result_bundle)
    r.raise_for_status()
    print("✅ Retrieved metadata.")
    print(json.dumps(r.json(), indent=2))

def select_orbit(index, result_bundle, dimensionless=False):
    payload = {
        "index": index,
        "dimensionless": dimensionless,
        "result_bundle": result_bundle
    }
    r = requests.post(f"{BASE_URL}/orbits/select", json=payload)
    r.raise_for_status()
    print(f"✅ Selected orbit index {index}.")
    return r.json()

def plot_single(index, result_bundle):
    payload = {
        "result_bundle": result_bundle,
        "index": index,
        "dimensionless": False,
        "azim": -60,
        "elev": 30
    }
    r = requests.post(f"{BASE_URL}/plot/single", json=payload)
    with open("orbit_single.png", "wb") as f:
        f.write(r.content)
    print("✅ Saved single orbit plot as orbit_single.png.")

def plot_family(key, result_bundle):
    payload = {
        "result_bundle": result_bundle,
        "key": key,
        "val_range": None,
        "dimensionless": False,
        "max_display": 25,
        "azim": -60,
        "elev": 30
    }
    r = requests.post(f"{BASE_URL}/plot/family", json=payload)
    with open("orbit_family.png", "wb") as f:
        f.write(r.content)
    print("✅ Saved family plot as orbit_family.png.")

def export_csv(result_bundle):
    payload = {
        "result_bundle": result_bundle,
        "index": 0
    }
    r = requests.post(f"{BASE_URL}/export/csv", json=payload)
    print("✅ CSV Export Path:", r.json())

def export_cfg(index, result_bundle):
    payload = {
        "index": index,
        "result_bundle": result_bundle
    }
    r = requests.post(f"{BASE_URL}/export/cfg", json=payload)
    print("✅ CFG Export Path:", r.json())

def export_trajectory(index, result_bundle):
    payload = {
        "index": index,
        "result_bundle": result_bundle
    }
    r = requests.post(f"{BASE_URL}/export/trajectory", json=payload)
    print("✅ Trajectory Export Path:", r.json())

if __name__ == "__main__":
    # Use the config variables at the top
    bundle = query_orbits(SYSTEM, FAMILY, LIBR, BRANCH)
    get_family_info(bundle)
    select_orbit(0, bundle)
    plot_single(0, bundle)
    plot_family("jacobi", bundle)
    export_csv(bundle)
    export_cfg(0, bundle)
    export_trajectory(0, bundle)
