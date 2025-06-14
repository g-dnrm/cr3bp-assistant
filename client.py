# === Flexible CR3BP Orbit Client ===
import requests
import json
import argparse

# === Config ===
BASE_URL = "https://cr3bp-proxy.onrender.com"  # or your Render proxy URL

def query_orbits(args):
    payload = {
        "sys": args.system,
        "family": args.family.lower()
    }
    # Optional parameters
    if args.libr is not None:
        payload["libr"] = args.libr
    if args.branch is not None:
        payload["branch"] = args.branch
    if args.jacobimin is not None:
        payload["jacobimin"] = args.jacobimin
    if args.jacobimax is not None:
        payload["jacobimax"] = args.jacobimax
    if args.periodmin is not None:
        payload["periodmin"] = args.periodmin
    if args.periodmax is not None:
        payload["periodmax"] = args.periodmax
    if args.stabmin is not None:
        payload["stabmin"] = args.stabmin
    if args.stabmax is not None:
        payload["stabmax"] = args.stabmax

    r = requests.post(f"{BASE_URL}/orbits/query", json=payload)
    r.raise_for_status()
    result_bundle = r.json()
    print(f"✅ Queried: {json.dumps(payload, indent=2)}")
    return result_bundle

def get_family_info(bundle):
    r = requests.post(f"{BASE_URL}/orbits/info", json=bundle)
    r.raise_for_status()
    print(json.dumps(r.json(), indent=2))

def select_orbit(index, bundle, dimensionless=False):
    payload = {
        "index": index,
        "dimensionless": dimensionless,
        "result_bundle": bundle
    }
    r = requests.post(f"{BASE_URL}/orbits/select", json=payload)
    r.raise_for_status()
    print(json.dumps(r.json(), indent=2))

def plot_single(index, bundle, dimensionless=False):
    payload = {
        "result_bundle": bundle,
        "index": index,
        "dimensionless": dimensionless,
        "azim": -60,
        "elev": 30
    }
    r = requests.post(f"{BASE_URL}/plot/single", json=payload)
    with open("orbit_single.png", "wb") as f:
        f.write(r.content)
    print("✅ Saved: orbit_single.png")

def plot_family(key, bundle, val_range, max_display):
    payload = {
        "result_bundle": bundle,
        "key": key,
        "val_range": val_range if val_range else None,
        "dimensionless": False,
        "max_display": max_display,
        "azim": -60,
        "elev": 30
    }
    r = requests.post(f"{BASE_URL}/plot/family", json=payload)
    with open("orbit_family.png", "wb") as f:
        f.write(r.content)
    print("✅ Saved: orbit_family.png")

def export_csv(bundle):
    payload = {
        "result_bundle": bundle,
        "index": 0
    }
    r = requests.post(f"{BASE_URL}/export/csv", json=payload)
    print("✅ CSV Export Path:", r.json())

def export_cfg(index, bundle):
    payload = {
        "index": index,
        "result_bundle": bundle
    }
    r = requests.post(f"{BASE_URL}/export/cfg", json=payload)
    print("✅ CFG Export Path:", r.json())

def export_trajectory(index, bundle):
    payload = {
        "index": index,
        "result_bundle": bundle
    }
    r = requests.post(f"{BASE_URL}/export/trajectory", json=payload)
    print("✅ Trajectory Export Path:", r.json())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flexible CR3BP Orbit Client")
    parser.add_argument("--system", required=True, help="e.g., earth-moon, sun-earth, sun-mars, etc.")
    parser.add_argument("--family", required=True, help="e.g., halo, lyapunov, vertical, axial, butterfly, etc.")
    parser.add_argument("--libr", type=int, help="1, 2, 3, 4, 5 if needed")
    parser.add_argument("--branch", help="N, S, E, W if needed")

    # Optional filters
    parser.add_argument("--jacobimin", type=float)
    parser.add_argument("--jacobimax", type=float)
    parser.add_argument("--periodmin", type=float)
    parser.add_argument("--periodmax", type=float)
    parser.add_argument("--stabmin", type=float)
    parser.add_argument("--stabmax", type=float)

    # Actions
    parser.add_argument("--info", action="store_true", help="Show system/family info")
    parser.add_argument("--select", type=int, help="Select orbit index")
    parser.add_argument("--plot_single", type=int, help="Plot single orbit by index")
    parser.add_argument("--plot_family", action="store_true", help="Plot family")
    parser.add_argument("--family_key", default="jacobi", help="Key for family plot (jacobi, period, stability)")
    parser.add_argument("--val_range", nargs=2, type=float, help="Value range for family plot: min max")
    parser.add_argument("--max_display", type=int, default=25, help="Max orbits to plot")
    parser.add_argument("--export_csv", action="store_true")
    parser.add_argument("--export_cfg", type=int, help="Export CFG for index")
    parser.add_argument("--export_trajectory", type=int, help="Export trajectory CSV for index")

    args = parser.parse_args()

    bundle = query_orbits(args)

    if args.info:
        get_family_info(bundle)

    if args.select is not None:
        select_orbit(args.select, bundle)

    if args.plot_single is not None:
        plot_single(args.plot_single, bundle)

    if args.plot_family:
        val_range = args.val_range if args.val_range else None
        plot_family(args.family_key, bundle, val_range, args.max_display)

    if args.export_csv:
        export_csv(bundle)

    if args.export_cfg is not None:
        export_cfg(args.export_cfg, bundle)

    if args.export_trajectory is not None:
        export_trajectory(args.export_trajectory, bundle)
