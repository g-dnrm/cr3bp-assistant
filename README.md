
# 🚀 CR3BP Orbit Catalog Assistant

**A robust, cloud-connected Python client and proxy server to access, filter, plot, and export periodic orbits from NASA's Three-Body Problem catalog.**  
This tool lets mission designers, researchers, or students interactively query all known periodic orbit families in 7 major three-body dynamical systems.

---

## 🌌 Supported Three-Body Dynamical Systems

| System                | Description |
|-----------------------|--------------|
| Earth-Moon            | Earth-Moon CR3BP |
| Sun-Earth             | Sun-Earth CR3BP |
| Sun-Mars              | Sun-Mars CR3BP |
| Mars-Phobos           | Mars-Phobos CR3BP |
| Jupiter-Europa        | Jupiter-Europa CR3BP |
| Saturn-Titan          | Saturn-Titan CR3BP |
| Saturn-Enceladus      | Saturn-Enceladus CR3BP |

---

## 🪐 Supported Families of Periodic Orbits

| Families | |
| --- | --- |
| L1, L2, L3 Lyapunov orbits | (62,755 orbits) |
| L1, L2, L3 Northern/Southern Halo orbits | (87,049 orbits) |
| L1, L2, L3 Vertical orbits | (99,766 orbits) |
| L1, L2, L3 Axial orbits | (41,146 orbits) |
| L4, L5 Long Period orbits | (15,948 orbits) |
| L4, L5 Short Period orbits | (66,512 orbits) |
| L4, L5 Vertical orbits | (31,260 orbits) |
| L4, L5 Axial orbits | (31,840 orbits) |
| Northern/Southern Butterfly orbits | (21,872 orbits) |
| Northern/Southern Dragonfly orbits | (12,534 orbits) |
| Distant Retrograde orbits | (28,298 orbits) |
| Distant Prograde orbits | (20,761 orbits) |
| Eastern/Western Low Prograde orbits | (13,370 orbits) |
| Resonant orbits | (194,663 orbits) |

---

## ⚙️ Features

- Query the orbit catalog with flexible filters: system, family, libration point, branch, Jacobi constant, period, stability index.
- Retrieve family metadata and parameter ranges.
- Select specific orbits by index and inspect their details.
- Plot a single orbit in 3D.
- Plot an entire orbit family with optional value range filters.
- Export orbit initial conditions to `.csv` and `.cfg` files.
- Export propagated trajectory for a selected orbit.

---

## 🏗️ Requirements

- Python 3.8+
- `requests`, `argparse`, `fastapi`, `pydantic`, `matplotlib`

Install dependencies:

```bash
pip install requests fastapi pydantic matplotlib
```

---

## 🚀 Running the Proxy Server

Your FastAPI proxy server handles API routing and plotting:

```bash
uvicorn proxy_server:app --reload --port 8000
```

By default, the client uses `http://localhost:8000`.

---

## 🖥️ Using the Python Client

**Run `client.py` with command-line arguments:**

```bash
python client.py --system earth-moon --family halo --libr 1 --branch N --info --plot_family
```

### ✅ Available CLI Flags

| Flag | Purpose |
|------|---------|
| `--system` | (Required) The 3-body system, e.g. `earth-moon` |
| `--family` | (Required) The orbit family, e.g. `halo` |
| `--libr` | Libration point (1–5), if required |
| `--branch` | Branch direction (N, S, E, W), if required |
| `--jacobimin` / `--jacobimax` | Filter: Jacobi constant range |
| `--periodmin` / `--periodmax` | Filter: Period range |
| `--stabmin` / `--stabmax` | Filter: Stability index range |
| `--info` | Show family metadata & limits |
| `--select` | Select orbit by index and show details |
| `--plot_single` | Plot a single orbit by index |
| `--plot_family` | Plot family orbits |
| `--family_key` | Key for family plot (jacobi, period, stability) |
| `--val_range` | Value range for family plot (two floats) |
| `--max_display` | Max orbits to plot in family (default: 25) |
| `--export_csv` | Export full filtered orbits to CSV |
| `--export_cfg` | Export .cfg for an orbit by index |
| `--export_trajectory` | Export propagated trajectory for an orbit by index |

---

## 📦 Example Scenarios

**1️⃣ Query & plot a Sun-Mars L1 Northern Halo family:**

```bash
python client.py --system sun-mars --family halo --libr 1 --branch N --plot_family
```

**2️⃣ Query Earth-Moon Lyapunov orbits with Jacobi constant between 3.1–3.5:**

```bash
python client.py --system earth-moon --family lyapunov --libr 1 --jacobimin 3.1 --jacobimax 3.5 --info --export_csv
```

**3️⃣ Select a specific Butterfly orbit and export its trajectory:**

```bash
python client.py --system saturn-titan --family butterfly --branch N --select 0 --export_trajectory 0
```

---

## 📁 Outputs

- `orbit_family.png` — 3D plot of selected orbit family.
- `orbit_single.png` — 3D plot of a selected single orbit.
- `{system}_{family}_ICs.csv` — CSV file with initial conditions for all filtered orbits.
- `{system}_{family}_IDX.cfg` — Config file for a selected orbit.
- `{system}_{family}_IDX_trajectory.csv` — Propagated trajectory data.

---

## ⚠️ Notes

- The client works for **all supported systems & families** with no hardcoding.
- You must provide `libr` or `branch` when required (e.g. Halo or Butterfly families).
- For technical details, consult `proxy_server.py` and `core.py` (or contact the developer).

---

**Enjoy advanced mission design with full three-body periodic orbit catalog access!**
