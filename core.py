import requests
import json
import csv
import configparser
import numpy as np
from scipy.integrate import solve_ivp
from numba import njit
import matplotlib.pyplot as plt
import time
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Dict


# === API Query Class ===
class CR3BPOrbitAPI:
    """
    Handles querying the NASA/JPL Three-Body Periodic Orbits API.

    Families Supported:
        - L1, L2, L3: Lyapunov, Halo (N/S), Vertical, Axial
        - L4, L5: Long Period, Short Period, Vertical, Axial
        - Global: Butterfly (N/S), Dragonfly (N/S), Distant Retrograde, Distant Prograde,
                  Low Prograde (East/West), Resonant

    Parameters Support Matrix:
        - 'libr' is only valid for families associated with L1, L2, L3, L4, or L5
        - 'branch' is used for families with directional characteristics (N/S, E/W)

    Returns:
        dict: API response with appended metadata: family, libration_point, branch, and system
    """

    # Metadata for validation and downstream annotation
    FAMILY_META = {
        "lyapunov": {"libration_points": [1, 2, 3]},
        "halo": {"libration_points": [1, 2, 3], "branches": ["N", "S"]},
        "vertical": {"libration_points": [1, 2, 3, 4, 5]},
        "axial": {"libration_points": [1, 2, 3, 4, 5]},
        "long-period": {"libration_points": [4, 5]},
        "short-period": {"libration_points": [4, 5]},
        "butterfly": {"branches": ["N", "S"]},
        "dragonfly": {"branches": ["N", "S"]},
        "distant-retrograde": {},
        "distant-prograde": {},
        "low-prograde": {"branches": ["E", "W"]},
        "resonant": {}
    }

    def __init__(self, use_proxy=False):
        """
        Initializes the API with optional proxy routing.
        """
        self.BASE_URL = (
            "https://your-server.com/query_orbits"
            if use_proxy else
            "https://ssd-api.jpl.nasa.gov/periodic_orbits.api"
        )
        self.session = requests.Session()

    def query(self, **params):
        """
        Executes a query to the CR3BP periodic orbit API
        with fallback defaults and metadata enhancement.
        """
        sys = params.get("sys")
        family = params.get("family")
        libr = params.get("libr")
        branch = params.get("branch")

        if not sys or not family:
            raise ValueError("Both 'sys' and 'family' must be specified.")

        # Defensive: default periodunits to TU if missing
        params.setdefault("periodunits", "TU")

        # Family name for downstream use
        family = family.lower()
        meta = self.FAMILY_META.get(family, {})

        try:
            if "ssd-api.jpl.nasa.gov" in self.BASE_URL:
                response = self.session.get(self.BASE_URL, params=params)
            else:
                response = self.session.post(self.BASE_URL, json=params)

            if response.status_code != 200:
                return {"error": f"HTTP {response.status_code}: {response.reason}"}

            data = response.json()

            # Augment metadata for clarity
            sys_metadata = data.get("system", {})
            data.update({
                "system": {
                    "name": sys,
                    "lunit": sys_metadata.get("lunit"),
                    "tunit": sys_metadata.get("tunit"),
                    "mass_ratio": sys_metadata.get("mass_ratio"),
                    "L1": sys_metadata.get("L1"),
                    "L2": sys_metadata.get("L2"),
                    "L3": sys_metadata.get("L3"),
                    "L4": sys_metadata.get("L4"),
                    "L5": sys_metadata.get("L5"),
                },
                "family": family,
                "libration_point": libr,
                "branch": branch,
                "meta": meta
            })

            if "warning" in data:
                return {"warning": data["warning"], **data}
            if str(data.get("count", "0")) == "0":
                return {"warning": "Query succeeded but returned 0 matching records.", **data}
            if data.get("signature", {}).get("version") != "1.0":
                return {"error": "API version mismatch."}

            return data
        except requests.RequestException as e:
            return {"error": f"Request failed: {str(e)}"}

# === Intelligent Filtered Query Wrapper ===
class CR3BPQueryBuilder:
    """
    Builds and executes CR3BP periodic orbit catalog queries using the CR3BPOrbitAPI.

    Relies on upstream validation from the proxy server for family, libr, and branch.
    """

    def __init__(self, api):
        self.api = api

    def _standardize_family(self, family):
        return family.strip().lower()

    def fetch_with_filters(self, sys, family, libr=None, branch=None,
                           jacobi_override=None, period_override=None,
                           stability_override=None, periodunits="TU"):
        """
        Performs a filtered query to the CR3BP orbit catalog.

        Args:
            sys (str): System name (e.g., "earth-moon").
            family (str): Orbit family name.
            libr (int, optional): Libration point (1-5) if needed.
            branch (str, optional): Branch identifier if needed.
            jacobi_override (tuple, optional): (min, max) Jacobi filter.
            period_override (tuple, optional): (min, max) Period filter.
            stability_override (tuple, optional): (min, max) Stability filter.
            periodunits (str): "TU", "s", "h", or "d".

        Returns:
            dict: A dictionary containing the filters and the API result.
        """

        # Base validated query parameters (all guaranteed by the proxy)
        base_params = {
            "sys": sys,
            "family": self._standardize_family(family)
        }
        if libr is not None:
            base_params["libr"] = libr
        if branch is not None:
            base_params["branch"] = branch

        # Initial query to get full limits
        result = self.api.query(**base_params)
        if "error" in result or "warning" in result:
            return {"filters": base_params, "result": result}

        # Load limits & user filters
        filter_manager = CR3BPFilterManager()
        filter_manager.load_limits_from_response(result)

        if jacobi_override:
            filter_manager.set_filter_range("jacobi", jacobi_override)
        if period_override:
            filter_manager.set_filter_range("period", period_override)
        if stability_override:
            filter_manager.set_filter_range("stability", stability_override)

        # Set periodunits on period filter explicitly
        if "period" in filter_manager.filters:
            filter_manager.filters["period"].periodunits = periodunits

        # Generate final filter query
        filters = filter_manager.generate_filter_query()

        # Final query with filters applied
        final_query = {**base_params, **filters, "periodunits": periodunits}
        refined_result = self.api.query(**final_query)

        return {
            "filters": final_query,
            "result": refined_result
        }
    def query_raw_info(self, sys, family, libr=None, branch=None, periodunits="TU"):  # 🆕 Added method
        """
        Fetches raw, unfiltered family information directly from the NASA API.

        Args:
            sys (str): System name.
            family (str): Orbit family name.
            libr (int, optional): Libration point.
            branch (str, optional): Branch identifier.
            periodunits (str): Period units ("TU", "s", "h", or "d").

        Returns:
            dict: Raw API result without filtering.
        """
        base_params = {
            "sys": sys,
            "family": self._standardize_family(family),
            "periodunits": periodunits
        }
        if libr is not None:
            base_params["libr"] = libr
        if branch is not None:
            base_params["branch"] = branch
        full_data = self.api.query(**base_params)

        # Strip heavy orbit data
        return {
            "signature": full_data.get("signature", {}),
            "system": full_data.get("system", {}),
            "limits": full_data.get("limits", {}),
            "count": full_data.get("count"),
            "family": full_data.get("family"),
            "libration_point": full_data.get("libration_point"),
            "branch": full_data.get("branch")
        }

# === Filter Base and Subclasses ===
class BaseFilter:
    def __init__(self, name, limits):
        self.name = name
        self.min_val, self.max_val = limits
        self.user_range = None

    def set_user_range(self, val_range):
        if not isinstance(val_range, (list, tuple)) or len(val_range) != 2:
            raise ValueError(f"{self.name} range must be a (min, max) tuple.")
        if val_range[0] > val_range[1]:
            raise ValueError(f"{self.name} range must be min ≤ max.")
        self.user_range = val_range

    def get_query_params(self):
        raise NotImplementedError

class JacobiFilter(BaseFilter):
    def get_query_params(self):
        if not self.user_range:
            return {}
        return {
            "jacobimin": self.user_range[0],
            "jacobimax": self.user_range[1]
        }

class PeriodFilter(BaseFilter):
    def __init__(self, limits, periodunits="TU"):
        super().__init__("period", limits)
        self.periodunits = periodunits  # by design, always overwritten upstream

    def get_query_params(self):
        if not self.user_range:
            return {}
        # Defensive: optional guard
        if self.periodunits not in {"s", "h", "d", "TU"}:
            raise ValueError(f"Invalid periodunits: {self.periodunits}")
        return {
            "periodmin": self.user_range[0],
            "periodmax": self.user_range[1],
            "periodunits": self.periodunits
        }

class StabilityFilter(BaseFilter):
    def get_query_params(self):
        if not self.user_range:
            return {}
        return {
            "stabmin": self.user_range[0],
            "stabmax": self.user_range[1]
        }


# === Refactored Manager ===
class CR3BPFilterManager:
    """
    Manages Jacobi, Period, and Stability filters using modular filter objects.

    Usage:
        1. Initialize from response using `load_limits_from_response`.
        2. Use `set_filter_range` to override filter ranges.
        3. Generate query dictionary using `generate_filter_query`.
    """

    def __init__(self):
        self.filters = {}

    def load_limits_from_response(self, response_json):
        """
        Initializes filters from API response limits.
        """
        try:
            limits = response_json["limits"]
            self.filters["jacobi"] = JacobiFilter("jacobi", limits["jacobi"])
            self.filters["period"] = PeriodFilter(limits["period"])
            self.filters["stability"] = StabilityFilter("stability", limits["stability"])

        except (KeyError, ValueError) as e:
            raise RuntimeError("Failed to load limits from response.") from e

    def set_filter_range(self, key, val_range):
        """
        Updates the user-defined range for a specific filter.
        """
        if key not in self.filters:
            raise ValueError(f"Filter '{key}' not available.")
        self.filters[key].set_user_range(val_range)

    def generate_filter_query(self):
        """
        Builds final query dictionary using active filter ranges.
        """
        params = {}
        for f in self.filters.values():
            params.update(f.get_query_params())
        return params

# === Result Interpreter ===
class CR3BPResultInterpreter:
    """
    Interprets and formats results from a CR3BP periodic orbit API query.

    Provides:
    - Access to physical units and system metadata via properties
    - Conversion of dimensionless orbit data to physical units on demand
    - Display utilities for system information and orbit characteristics

    Attributes:
        response (dict): Raw API response containing metadata, orbits, and limits.
        data (list): Raw orbit data (LU/TU units)
        fields (list): Orbit variable names
    """

    def __init__(self, response_json: Dict, periodunits: str = "TU"):
        """
        Initializes the interpreter from raw JSON returned by the API.

        Args:
            response_json (dict): JSON response from the API.
            periodunits (str): Desired output units for period: TU, s, h, d
        """
        self.response = response_json
        self.data = response_json.get("data", [])
        self.fields = response_json.get("fields", [])
        self._periodunits = periodunits.upper()

    @property
    def system(self):
        return self.response["system"]

    @property
    def lunit(self) -> float:
        return float(self.system["lunit"])

    @property
    def tunit(self) -> float:
        return float(self.system["tunit"])

    @property
    def mass_ratio(self) -> float:
        return float(self.system["mass_ratio"])

    @property
    def system_name(self) -> str:
        return self.system.get("name", "unknown")

    @property
    def libration_points(self) -> Dict[str, List[float]]:
        return {k: list(map(float, v)) for k, v in self.system.items() if k.startswith("L")}

    @property
    def orbits(self) -> List[Dict]:
        converted = []
        for row in self.data:
            orbit = dict(zip(self.fields, row))
            for k in ["x", "y", "z"]:
                orbit[k] = float(orbit[k]) * self.lunit
            for k in ["vx", "vy", "vz"]:
                orbit[k] = float(orbit[k]) * (self.lunit / self.tunit)
            orbit["jacobi"] = float(orbit["jacobi"])
            orbit["stability"] = float(orbit["stability"])

            # 🔑 Period conversion:
            period_TU = float(orbit["period"])
            if self._periodunits == "TU":
                orbit["period"] = period_TU
            elif self._periodunits == "S":
                orbit["period"] = period_TU * self.tunit
            elif self._periodunits == "H":
                orbit["period"] = period_TU * self.tunit / 3600
            elif self._periodunits == "D":
                orbit["period"] = period_TU * self.tunit / 86400
            else:
                orbit["period"] = period_TU  # fallback

            converted.append(orbit)
        return converted

    def get_family_label(self, hyphenated: bool = False) -> str:
        family = self.response.get("family", "").capitalize()
        libr = self.response.get("libration_point")
        branch = self.response.get("branch", "").upper()
        branch_map = {"N": "Northern", "S": "Southern", "E": "Eastern", "W": "Western"}
        sep = "-" if hyphenated else " "
        if libr in {1, 2, 3}:
            base = f"L{libr}"
            if family in {"Halo", "Butterfly", "Dragonfly"}:
                direction = branch_map.get(branch, "")
                return sep.join(filter(None, [base, direction, family]))
            return sep.join([base, family])
        elif libr in {4, 5}:
            return sep.join([f"L{libr}", family])
        elif family.lower() in {"resonant", "distant retrograde", "distant prograde", "low prograde"}:
            direction = branch_map.get(branch, "")
            return sep.join(filter(None, [direction, family]))
        return family

    def show_system_info(self):
        print("🪐 System Physical Properties:")
        print(f"- Length unit (LU): {self.lunit:.2f} km")
        print(f"- Time unit (TU): {self.tunit:.2f} sec")
        print(f"- Mass ratio: {self.mass_ratio:.5e}")
        for label, coords in self.libration_points.items():
            print(f"- {label}: x = {coords[0]:.6f} LU, y = {coords[1]:.6f}, z = {coords[2]:.6f}")

    def show_limits(self):
        print("\n📊 Limits for Active Family:")
        for key in ["jacobi", "period", "stability"]:
            if key in self.response.get("limits", {}):
                low, high = self.response["limits"][key]
                unit = ""
                if key == "period":
                    unit = f" ({self._periodunits})"
                print(f"- {key.capitalize()}{unit}: [{low}, {high}]")

    def show_orbit_table(self, count=10):
        print(f"\n📋 Showing first {min(count, len(self.orbits))} orbits (in km, km/s, period in {self._periodunits}):")
        for i, orbit in enumerate(self.orbits[:count]):
            print(f"ID {i+1}: Jacobi={orbit['jacobi']:.6f}, Period={orbit['period']:.6f} {self._periodunits}, Stability={orbit['stability']:.2f}")

    def select_orbit_by_index(self, idx, dimensionless=False) -> Dict:
        if idx < 0 or idx >= len(self.data):
            raise IndexError("Orbit index out of range.")
        raw = dict(zip(self.fields, map(float, self.data[idx])))
        converted = self.orbits[idx]
        print(f"\n🔍 Full Orbit Details [ID {idx + 1}]")
        for k in self.fields:
            val = raw[k] if dimensionless else converted[k]
            unit = ""
            if k in ["x", "y", "z"]:
                unit = "LU" if dimensionless else "km"
            elif k in ["vx", "vy", "vz"]:
                unit = "LU/TU" if dimensionless else "km/s"
            elif k == "period":
                unit = "TU" if dimensionless else self._periodunits
            elif k == "jacobi":
                unit = "(LU²/TU²)"
            print(f"{k}: {val:.6f} {unit}")
        return raw if dimensionless else converted

    def display_filter_summary(self, filters_dict):
        label = self.get_family_label()
        print("🔍 Query Filters:")
        print(f"- System: {self.system_name}")
        print(f"- Family: {label}")
        if "jacobimin" in filters_dict and "jacobimax" in filters_dict:
            print(f"- Jacobi Constant: {filters_dict['jacobimin']} to {filters_dict['jacobimax']}")
        if "periodmin" in filters_dict and "periodmax" in filters_dict:
            print(f"- Period ({self._periodunits}): {filters_dict['periodmin']} to {filters_dict['periodmax']}")
        if "stabmin" in filters_dict and "stabmax" in filters_dict:
            print(f"- Stability Index: {filters_dict['stabmin']} to {filters_dict['stabmax']}")

    def display_result_count(self):
        count = self.response.get("count", "0")
        print(f"\n📦 Found {count} orbits matching filter criteria.")


class CR3BPExporter:
    """
    Handles exporting and selection of CR3BP periodic orbit data.

    Supports:
    - Exporting selected orbit initial conditions to a .cfg file (legacy feature)
    - Exporting filtered orbits to a JPL-style CSV with full metadata
    - Selecting orbits matching a specified property (Jacobi, period, stability)

    Attributes:
        fields (list): Field names from API result.
        system_info (dict): System-level metadata.
        query_info (dict): Original query parameters (for header context).
        data (list): Raw orbit data rows from API.
    """

    def __init__(self, fields, system_info, query_info, data):
        self.fields = fields
        self.system_info = system_info
        self.query_info = query_info
        self.data = data

    def to_csv(self, filename=None):
        """
        Exports all available orbit fields to a CSV file with physical units and a metadata header.

        Enhancements:
        - Formatted Metadata Block: Includes system name, family label, and high-precision mass ratio.
        - Auto Filename: Automatically generates a descriptive filename using system, family, libration point, and branch.
        - Units: Column headers include physical units (LU, TU).
        - Derived Quantity: Adds orbit period in days.
        - Removed Redundancy: Mass ratio appears only in the metadata header, not in each data row.

        Args:
            filename (str): Optional filename for output. If None, an informative name is generated.
        """
        if not self.data:
            print("⚠️ No orbit data to export.")
            return

        # Extract and format components
        system_name = self.system_info.get("name", "").replace(" ", "-").lower()
        family = self.query_info.get("family", "").replace(" ", "-").capitalize()
        libr = self.query_info.get("libr", "")
        branch = self.query_info.get("branch", "").upper()
        branch_map = {"N": "Northern", "S": "Southern", "E": "Eastern", "W": "Western"}
        direction = branch_map.get(branch, branch)

        libr_str = f"L{libr}" if str(libr).isdigit() else None
        family_label_parts = [libr_str, direction if direction else None, family]
        family_label = "-".join(filter(None, family_label_parts)).replace("--", "-")

        if not filename:
            filename = f"{system_name}_{family_label}_ICs.csv"

        tunit = float(self.system_info.get("tunit", 1))
        mu = float(self.system_info.get("mass_ratio", 0))

        # Define descriptive column labels
        label_map = {
            "x": "x0 (LU)", "y": "y0 (LU)", "z": "z0 (LU)",
            "vx": "vx0 (LU/TU)", "vy": "vy0 (LU/TU)", "vz": "vz0 (LU/TU)",
            "jacobi": "Jacobi constant (LU^2/TU^2)", "period": "Period (TU)", "stability": "Stability index"
        }
        headers = ["Id"] + [label_map.get(k, k) for k in self.fields] + ["Period (days)"]

        # Write to file
        with open(filename, mode="w", newline="") as f:
            # Metadata block
            f.write(f"# System: {self.system_info.get('name', '')}\n")
            f.write(f"# Family: {family_label.replace('-', ' ')}\n")
            f.write(f"# Mass ratio: {mu:.10f}\n")
            for key in ["jacobimin", "jacobimax", "periodmin", "periodmax", "stabmin", "stabmax"]:
                if key in self.query_info:
                    f.write(f"# {key}: {self.query_info[key]}\n")

            writer = csv.writer(f)
            writer.writerow(headers)
            for idx, row in enumerate(self.data):
                orbit = dict(zip(self.fields, row))
                period_days = float(orbit["period"]) * tunit / 86400
                writer.writerow([idx + 1] + [orbit[k] for k in self.fields] + [period_days])

        print(f"✅ Filtered orbit data exported to: {filename}")
        return filename

    def to_cfg(self, index, filename=None):
        """
        Exports a single orbit's initial conditions and metadata to a .cfg file.

        Args:
            index (int): Index of the orbit in the data list.
            filename (str): Optional output filename. If not provided, a descriptive name is used.
        """
        if index < 0 or index >= len(self.data):
            print("⚠️ Invalid orbit index.")
            return

        orbit = dict(zip(self.fields, self.data[index]))
        config = configparser.ConfigParser()

        mu = self.system_info.get("mass_ratio", "")
        lunit = self.system_info.get("lunit", "")
        tunit = self.system_info.get("tunit", "")

        config["system"] = {
            "name": self.system_info.get("name", ""),
            "mass_ratio": mu,
            "lunit_km": lunit,
            "tunit_sec": tunit,
            "note": "To convert: [LU] * lunit_km => km; [LU/TU] * lunit_km / tunit_sec => km/s; Jacobi in LU^2/TU^2"
        }

        # Clean family section with optional libration point
        family_section = {"name": self.query_info.get("family", "")}
        libr = self.query_info.get("libr")
        if libr is not None:
            family_section["libration_point"] = str(libr)
        branch = self.query_info.get("branch", "")
        if branch:
            family_section["branch"] = branch
        config["family"] = family_section

        # Orbit fields with formatted values
        orbit_section = {
            "x (LU)": orbit["x"],
            "y (LU)": orbit["y"],
            "z (LU)": orbit["z"],
            "vx (LU/TU)": orbit["vx"],
            "vy (LU/TU)": orbit["vy"],
            "vz (LU/TU)": orbit["vz"],
            "jacobi (LU^2/TU^2)": orbit["jacobi"],
            "period (TU)": orbit["period"],
            "stability": orbit["stability"],
            "id": str(index + 1)
        }
        config["orbit"] = {
            k: f"{v:.17g}" if isinstance(v, float) else str(v)
            for k, v in orbit_section.items()
        }

        if not filename:
            system = self.system_info.get("name", "").replace(" ", "-").lower()
            family = self.query_info.get("family", "").replace(" ", "-")
            branch_map = {"N": "Northern", "S": "Southern", "E": "Eastern", "W": "Western"}
            branch_label = branch_map.get(branch.upper(), branch.upper())
            libr_label = f"L{libr}" if libr is not None else ""
            label_parts = [libr_label, branch_label, family]
            label = "-".join(part for part in label_parts if part).replace("--", "-")
            filename = f"{system}_{label}_ID{index + 1}_ics.cfg"

        with open(filename, "w") as configfile:
            config.write(configfile)

        print(f"✅ Configuration file exported: {filename}")
        return filename

    def export_propagated_orbit_csv(self, index, filename=None, num_points=500):
        """
        Exports the propagated trajectory of a selected orbit to a CSV file.

        Args:
            index (int): Index of the orbit to propagate.
            filename (str): Optional filename. Auto-generated if None.
            num_points (int): Number of points in the trajectory.
        """
        if index < 0 or index >= len(self.data):
            print("⚠️ Invalid orbit index.")
            return

        orbit = dict(zip(self.fields, map(float, self.data[index])))
        state0 = [orbit[k] for k in ["x", "y", "z", "vx", "vy", "vz"]]
        period = orbit["period"]
        mu = float(self.system_info.get("mass_ratio", 0))
        system = self.system_info.get("name", "").replace(" ", "-").lower()

        # Build label
        family = self.query_info.get("family", "").replace(" ", "-")
        libr = self.query_info.get("libr")
        branch = self.query_info.get("branch", "").upper()
        branch_map = {"N": "Northern", "S": "Southern", "E": "Eastern", "W": "Western"}
        branch_label = branch_map.get(branch, branch)
        libr_label = f"L{libr}" if libr is not None else ""
        label_parts = [libr_label, branch_label, family]
        label = "-".join(part for part in label_parts if part).replace("--", "-")

        if not filename:
            filename = f"{system}_{label}_ID{index + 1}_trajectory.csv"

        propagator = CR3BPPropagator.from_memory(state0, mu, period)
        sol = propagator.propagate()

        # Sample trajectory
        t_vals = np.linspace(0, period, num_points)
        states = sol.sol(t_vals).T  # shape (num_points, 6)

        # Write CSV
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            # Header
            f.write(f"# System: {system}\n")
            f.write(f"# Family: {label.replace('-', ' ')}\n")
            f.write(f"# Mass ratio: {mu:.10f}\n")
            f.write(f"# Jacobi (LU^2/TU^2) = {orbit['jacobi']:.17g}\n")
            f.write(f"# Period (TU) = {orbit['period']:.17g}\n")
            f.write("# Time (TU), X (LU), Y (LU), Z (LU), VX (LU/TU), VY (LU/TU), VZ (LU/TU)\n")
            for t, row in zip(t_vals, states):
                writer.writerow([f"{t:.17g}"] + [f"{x:.17g}" for x in row])

        print(f"✅ Propagated orbit trajectory saved to: {filename}")
        return filename

    def find_orbits_by_property(self, key, target_value, tol=1e-6):
        """
        Finds orbits where a specified property approximately matches a target value.

        Args:
            key (str): Must be 'jacobi', 'period', or 'stability'.
            target_value (float): Value to search for.
            tol (float): Absolute tolerance for approximate matching.

        Returns:
            list of dict: Orbit dictionaries matching the condition.
        """
        if key not in {"jacobi", "period", "stability"}:
            raise ValueError("Invalid key. Must be 'jacobi', 'period', or 'stability'.")

        matching = []
        for row in self.data:
            orbit = dict(zip(self.fields, row))
            if abs(float(orbit[key]) - target_value) < tol:
                matching.append(orbit)

        if not matching:
            print(f"⚠️ No orbits found for {key} ≈ {target_value} (tol={tol}).")
        else:
            print(f"✅ Found {len(matching)} orbit(s) matching {key} ≈ {target_value}")
        return matching

@njit(cache=True)
def _cr3bp_eom_numba(t, state, mu):
    """
    JIT-compiled function to compute the derivatives of the CR3BP equations of motion.

    Args:
        t (float): Time (not used explicitly as CR3BP is autonomous).
        state (ndarray): State vector [x, y, z, vx, vy, vz].
        mu (float): Mass ratio of the two primaries.

    Returns:
        ndarray: Derivative of the state vector.
    """
    x, y, z, vx, vy, vz = state

    r1 = ((x + mu)**2 + y**2 + z**2)**0.5
    r2 = ((x - 1 + mu)**2 + y**2 + z**2)**0.5

    ax = 2*vy + x - (1 - mu)*(x + mu)/r1**3 - mu*(x - 1 + mu)/r2**3
    ay = -2*vx + y - (1 - mu)*y/r1**3 - mu*y/r2**3
    az = -(1 - mu)*z/r1**3 - mu*z/r2**3

    return np.array([vx, vy, vz, ax, ay, az])

class CR3BPPropagator:
    """
    Propagates orbits in the Circular Restricted Three-Body Problem (CR3BP) framework.

    Attributes:
        mu (float): Mass ratio of the two primaries.
        state0 (list): Initial state vector [x, y, z, vx, vy, vz] in dimensionless units.
        period (float): Orbital period (dimensionless time units).
    """
    def __init__(self):
        self.mu = None
        self.state0 = None
        self.period = None

    @staticmethod
    def cr3bp_eom_numba(t, state, mu):
        """
        Wrapper for the JIT-compiled CR3BP equations of motion.

        Args:
            t (float): Time (not used explicitly).
            state (ndarray): Current state vector.
            mu (float): Mass ratio.

        Returns:
            ndarray: Derivative of the state.
        """
        return _cr3bp_eom_numba(t, state, mu)

    @classmethod
    def from_cfg(cls, cfg_path):
        """
        Constructs a CR3BPPropagator from a .cfg file.

        Args:
            cfg_path (str): Path to the configuration file.

        Returns:
            CR3BPPropagator: Initialized instance.
        """
        instance = cls()
        config = configparser.ConfigParser()
        config.read(cfg_path)

        sys_cfg = config["system"]
        orbit_cfg = config["orbit"]

        instance.mu = float(sys_cfg["mass_ratio"])
        instance.period = float(orbit_cfg["period_tu"])
        instance.state0 = [
            float(orbit_cfg["x0"]), float(orbit_cfg["y0"]), float(orbit_cfg["z0"]),
            float(orbit_cfg["vx0"]), float(orbit_cfg["vy0"]), float(orbit_cfg["vz0"])
        ]
        return instance

    @classmethod
    def from_memory(cls, state0, mu, period):
        """
        Constructs a CR3BPPropagator from in-memory parameters.

        Args:
            state0 (list): Initial state vector.
            mu (float): Mass ratio.
            period (float): Period of the orbit.

        Returns:
            CR3BPPropagator: Initialized instance.
        """
        instance = cls()
        instance.state0 = state0
        instance.mu = mu
        instance.period = period
        return instance

    def propagate(self, n_periods=1, rtol=1e-6, atol=1e-8):
        """
        Numerically integrates the orbit over a specified number of periods.

        Args:
            n_periods (int): Number of orbital periods to propagate.
            rtol (float): Relative tolerance for the integrator.
            atol (float): Absolute tolerance for the integrator.

        Returns:
            OdeResult: SciPy integration result containing state history.
        """
        t_span = [0, self.period * n_periods]
        sol = solve_ivp(
            lambda t, y: self.cr3bp_eom_numba(t, y, self.mu),
            t_span, self.state0, method='RK45',
            rtol=rtol, atol=atol, dense_output=True
        )
        return sol

    def check_periodicity(self, sol):
        """
        Computes the Euclidean distance between final and initial states.

        Args:
            sol (OdeResult): Output of the integration.

        Returns:
            float: Norm of the state difference.
        """
        final_state = sol.y[:, -1]
        delta = np.linalg.norm(np.array(self.state0) - final_state)
        print(f"\n🔁 Periodicity Check: |Δfinal-initial| = {delta:.3e} (LU, LU/TU)")
        return delta


class CR3BPPlotter:
    """
    Handles 3D visualization of CR3BP orbits using interpreter data.

    Requires:
        - An instance of CR3BPResultInterpreter with parsed API response.

    Supports:
        - Plotting a unique orbit by dynamical property
        - Plotting a family of orbits colored by a property

    Notes:
        - Propagation always uses raw LU/TU units (dimensionless)
        - Display scaling is applied only if dimensionless=False
    """

    def __init__(self, interpreter):
        """
        Initializes the plotter with system metadata and raw orbits.

        Args:
            interpreter (CR3BPResultInterpreter): Parsed API result handler.
        """
        self.interpreter = interpreter
        self.system = interpreter.system_name
        self.mu = interpreter.mass_ratio
        self.lunit = interpreter.lunit
        self.tunit = interpreter.tunit
        self.fields = interpreter.fields
        self.raw_data = interpreter.data
        self.family_label = interpreter.get_family_label()
        self.libration_coords = interpreter.libration_points

    def _propagate_orbit(self, orbit_raw, num_points=None):
        """
        Propagates an orbit in dimensionless units (LU/TU).

        Args:
            orbit_raw (dict): Orbit in LU and LU/TU format.
            num_points (int, optional): If provided, samples trajectory using interpolation.

        Returns:
            tuple: Arrays (x, y, z) of orbit coordinates.
        """
        state0 = [orbit_raw[k] for k in ["x", "y", "z", "vx", "vy", "vz"]]
        period = orbit_raw["period"]
        prop = CR3BPPropagator.from_memory(state0, self.mu, period)
        sol = prop.propagate()  # always with dense_output

        if num_points:
            t_vals = np.linspace(0, period, num_points)
            xyz = sol.sol(t_vals)
            x, y, z = xyz[0], xyz[1], xyz[2]
        else:
            x, y, z = sol.y[0], sol.y[1], sol.y[2]

        return x, y, z

    def _autoscale_axes(self, ax, x, y, z, padding=0.05):
        """
        Sets equal aspect ratio and scaling for a 3D plot.

        Args:
            ax (Axes3D): The 3D matplotlib axis.
            x, y, z (np.ndarray): Orbit coordinates.
            padding (float): Fractional padding to add to box size.
        """
        all_xyz = np.vstack([x, y, z])
        mins = np.min(all_xyz, axis=1)
        maxs = np.max(all_xyz, axis=1)
        centers = (mins + maxs) / 2
        half_range = np.max(maxs - mins) / 2 * (1 + padding)

        ax.set_xlim(centers[0] - half_range, centers[0] + half_range)
        ax.set_ylim(centers[1] - half_range, centers[1] + half_range)
        ax.set_zlim(centers[2] - half_range, centers[2] + half_range)

    def _plot_libration_point(self, ax, dimensionless):
        """
        Plots only the relevant Lagrange point (e.g., L1, L2, L3).

        Args:
            ax (Axes3D): The 3D axis to annotate.
            dimensionless (bool): If True, uses LU; otherwise converts to km.
        """
        libr_id = self.interpreter.response.get("libration_point")
        if not libr_id:
            return  # No specific L-point assigned

        name = f"L{libr_id}"
        coords = self.libration_coords.get(name)
        if not coords:
            return  # Missing coordinate data for the requested point

        lx, ly, lz = coords
        if not dimensionless:
            lx *= self.lunit
            ly *= self.lunit
            lz *= self.lunit

        ax.scatter([lx], [ly], [lz], s=60, c='black', marker='o', label=name)
        ax.text(lx, ly, lz, f' {name}', fontsize=10)

    def plot_orbit_by_property(self, key, value, dimensionless=False, save_path=None,
                               azim=-60, elev=30):
        """
        Finds and plots a single orbit matching a given property.

        Args:
            key (str): Property to match ("jacobi", "period", or "stability").
            value (float): Value to match.
            dimensionless (bool): Plot in LU/TU if True, else km/km/s.
            save_path (str, optional): If provided, saves the plot to this path instead of showing it.
            azim (int, optional): Azimuth angle for 3D view. Defaults to -60.
            elev (int, optional): Elevation angle for 3D view. Defaults to 30.
        """
        tol = 1e-12
        raw_orbits = [
            dict(zip(self.fields, map(float, row)))
            for row in self.raw_data
        ]
        best_orbit = min(raw_orbits, key=lambda o: abs(o[key] - value))
        if abs(best_orbit[key] - value) > tol:
            print(f"⚠️ No orbit found for {key} ≈ {value} (tol={tol})")
            return

        x, y, z = self._propagate_orbit(best_orbit, num_points=1000)
        if not dimensionless:
            x, y, z = x * self.lunit, y * self.lunit, z * self.lunit

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(x, y, z, label="Orbit", alpha=0.8)
        ax.scatter([x[0]], [y[0]], [z[0]], s=50, color=ax.lines[-1].get_color(), marker='o', label='Initial Condition')
        self._plot_libration_point(ax, dimensionless)
        self._autoscale_axes(ax, x, y, z)

        unit = "LU" if dimensionless else "km"
        ax.set_xlabel(f"x ({unit})")
        ax.set_ylabel(f"y ({unit})")
        ax.set_zlabel(f"z ({unit})")
        ax.view_init(elev=elev, azim=azim)

        if key == "period":
            period_days = value * self.tunit / 86400
            title = f"{self.system} - {self.family_label}: Period = {period_days:.5f} days"
        else:
            title = f"{self.system} - {self.family_label}: {key.title()} = {value:.12g}"

        ax.set_title(title)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            plt.close(fig)
            return save_path
        else:
            plt.show()
            return None

    def plot_family_by_range(self, key, val_range=None, dimensionless=False, max_display=25, azim=-60, elev=30,
                             save_path=None):
        """
        Plots a sample of orbits in the family colored by a property.

        Args:
            key (str): Property to color by ("jacobi", "period", or "stability").
            val_range (tuple): Optional (min, max) range for filtering.
            dimensionless (bool): Plot in LU/TU if True, else km/km/s.
            max_display (int): Max number of orbits to show.
            azim (int): Azimuth angle for 3D view.
            elev (int): Elevation angle for 3D view.
            save_path (str): If provided, saves the plot to this path.
        """
        raw_orbits = [
            dict(zip(self.fields, map(float, row)))
            for row in self.raw_data
        ]
        if val_range:
            raw_orbits = [o for o in raw_orbits if val_range[0] <= o[key] <= val_range[1]]
        if not raw_orbits:
            print("⚠️ No orbits in selected range.")
            return

        raw_orbits.sort(key=lambda o: o[key])
        step = max(1, len(raw_orbits) // max_display)
        selected = raw_orbits[::step]

        values = [o[key] for o in selected]
        norm = Normalize(min(values), max(values))
        cmap = plt.colormaps["viridis"]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        all_xyz = []

        for orbit in selected:
            x, y, z = self._propagate_orbit(orbit, num_points=1000)
            if not dimensionless:
                x, y, z = x * self.lunit, y * self.lunit, z * self.lunit
            all_xyz.append(np.vstack([x, y, z]))
            ax.plot(x, y, z, color=cmap(norm(orbit[key])), alpha=0.8)

        self._plot_libration_point(ax, dimensionless)
        all_xyz = np.hstack(all_xyz)
        self._autoscale_axes(ax, all_xyz[0], all_xyz[1], all_xyz[2], padding=0.2)

        unit = "LU" if dimensionless else "km"
        ax.set_xlabel(f"x ({unit})")
        ax.set_ylabel(f"y ({unit})")
        ax.set_zlabel(f"z ({unit})")
        ax.set_title(f"{self.system} - {self.family_label} - Colored by {key.title()}")
        ax.view_init(elev=elev, azim=azim)

        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label=key.title(), fraction=0.02, pad=0.1)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            plt.close(fig)
            return save_path
        else:
            plt.show()
            return None
