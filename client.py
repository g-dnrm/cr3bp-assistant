# === Flexible CR3BP Orbit Client ===
"""
CR3BP Orbit Catalog Command-Line Client

This CLI lets you:
- Query periodic orbits with precise filters (Jacobi, period, stability)
- Retrieve orbit family/system metadata
- Select specific orbits by index
- Plot single or family orbits to PNG files
- Export results as CSV, CFG, or full propagated trajectories

Endpoints used:
- /orbits/info  — get metadata & valid ranges for a family
- /orbits/filter — get filtered orbits matching user constraints
- /orbits/select — get detailed initial conditions for a chosen index
- /plot/single, /plot/family — get plots as PNGs
- /export/csv, /export/cfg, /export/trajectory — get exportable files

Example:
    python client.py --system sun-earth --family halo --libr 1 --branch N --jacobimin 3.0 --jacobimax 3.2 --periodunits TU --plot_family --family_key jacobi --max_display 10
"""

import requests
import json
import argparse

# === Config ===
BASE_URL = "https://cr3bp-proxy.onrender.com"  # Replace with your Render or local proxy URL

# === Functions ===
def get_family_info(args):
    """
    Calls /orbits/info and downloads the .json file with the orbit family info.
    """
    payload = {
        "sys": args.system,
        "family": args.family.lower(),
        "periodunits": args.periodunits
    }
    if args.libr is not None:
        payload["libr"] = args.libr
    if args.branch is not None:
        payload["branch"] = args.branch

    # Send request and receive raw metadata directly
    r = requests.post(f"{BASE_URL}/orbits/info", json=payload)
    r.raise_for_status()
    result = r.json()

    # Display result directly (no download step needed)
    print("✅ Family Info (inline mode):")
    print(json.dumps(result, indent=2))

def filter_orbits(args):
    """
    Calls /orbits/filter to apply user filters and get orbits.
    """
    payload = {
        "sys": args.system,
        "family": args.family.lower(),
        "periodunits": args.periodunits
    }
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

    r = requests.post(f"{BASE_URL}/orbits/filter", json=payload)
    r.raise_for_status()
    result_bundle = r.json()
    print(f"✅ Filtered Query: {json.dumps(payload, indent=2)}")
    return result_bundle


def select_orbit(index, bundle, dimensionless=False):
    """
    Calls /orbits/select to get detailed ICs for a selected index.
    """
    payload = {
        "index": index,
        "dimensionless": dimensionless,
        "result_bundle": bundle
    }
    r = requests.post(f"{BASE_URL}/orbits/select", json=payload)
    r.raise_for_status()
    print("✅ Selected Orbit:")
    print(json.dumps(r.json(), indent=2))


def plot_single(index, bundle, dimensionless=False):
    """
    Calls /plot/single to generate a PNG for a single orbit.
    """
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
    """
    Calls /plot/family to generate a PNG for a family of orbits.
    """
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
    """
    Calls /export/csv to get a CSV for the filtered orbits.
    """
    payload = {
        "result_bundle": bundle,
        "index": 0  # not used for CSV
    }
    r = requests.post(f"{BASE_URL}/export/csv", json=payload)
    print("✅ CSV Export Path:", r.json())


def export_cfg(index, bundle):
    """
    Calls /export/cfg to get CFG initial conditions for one orbit.
    """
    payload = {
        "index": index,
        "result_bundle": bundle
    }
    r = requests.post(f"{BASE_URL}/export/cfg", json=payload)
    print("✅ CFG Export Path:", r.json())


def export_trajectory(index, bundle):
    """
    Calls /export/trajectory to get full propagated trajectory for one orbit.
    """
    payload = {
        "index": index,
        "result_bundle": bundle
    }
    r = requests.post(f"{BASE_URL}/export/trajectory", json=payload)
    print("✅ Trajectory Export Path:", r.json())


# === Main ===

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Flexible CR3BP Orbit Client: search, filter, plot, export JPL periodic orbits."
    )
    parser.add_argument("--system", required=True,
                        help="Three-body system, e.g., earth-moon, sun-earth, sun-mars")
    parser.add_argument("--family", required=True,
                        help="Orbit family: halo, lyapunov, vertical, axial, butterfly, dragonfly, etc.")
    parser.add_argument("--libr", type=int, help="Libration point: 1, 2, 3, 4, 5")
    parser.add_argument("--branch", help="Branch: N, S, E, W if applicable")

    # Filter options
    parser.add_argument("--jacobimin", type=float, help="Min Jacobi constant")
    parser.add_argument("--jacobimax", type=float, help="Max Jacobi constant")
    parser.add_argument("--periodmin", type=float, help="Min period")
    parser.add_argument("--periodmax", type=float, help="Max period")
    parser.add_argument("--stabmin", type=float, help="Min stability index")
    parser.add_argument("--stabmax", type=float, help="Max stability index")
    parser.add_argument("--periodunits", choices=["s", "h", "d", "TU"], default="TU",
                        help="Units for period filter: s, h, d, TU (default: TU)")

    # Actions
    parser.add_argument("--info", action="store_true", help="Show system/family metadata")
    parser.add_argument("--select", type=int, help="Select orbit by index")
    parser.add_argument("--plot_single", type=int, help="Plot single orbit by index")
    parser.add_argument("--plot_family", action="store_true", help="Plot orbit family")
    parser.add_argument("--family_key", default="jacobi", help="Key for family plot (jacobi, period, stability)")
    parser.add_argument("--val_range", nargs=2, type=float,
                        help="Value range for family plot: min max")
    parser.add_argument("--max_display", type=int, default=25,
                        help="Max number of orbits to plot for family")
    parser.add_argument("--export_csv", action="store_true", help="Export filtered orbits to CSV")
    parser.add_argument("--export_cfg", type=int, help="Export CFG for selected orbit index")
    parser.add_argument("--export_trajectory", type=int, help="Export propagated trajectory for selected index")

    args = parser.parse_args()

    # === Flow ===
    if args.info:
        get_family_info(args)

    # If user asks for filtering or follow-up actions, do filter
    bundle = None
    if not args.info or any([
        args.jacobimin, args.jacobimax,
        args.periodmin, args.periodmax,
        args.stabmin, args.stabmax,
        args.select, args.plot_single, args.plot_family,
        args.export_csv, args.export_cfg, args.export_trajectory
    ]):
        bundle = filter_orbits(args)

    if bundle:
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