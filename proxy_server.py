from fastapi import FastAPI, Query, Body, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, Tuple
import tempfile
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

from core import (
    CR3BPOrbitAPI,
    CR3BPQueryBuilder,
    CR3BPResultInterpreter,
    CR3BPExporter,
    CR3BPPlotter
)

app = FastAPI(title="CR3BP Periodic Orbits Assistant")

# ==== Models ====
class QueryRequest(BaseModel):
    sys: str
    family: str
    libr: Optional[int] = None
    branch: Optional[str] = None
    jacobimin: Optional[float] = None
    jacobimax: Optional[float] = None
    periodmin: Optional[float] = None
    periodmax: Optional[float] = None
    stabmin: Optional[float] = None
    stabmax: Optional[float] = None
    periodunits: Optional[str] = "TU"

class OrbitSelectRequest(BaseModel):
    index: int
    dimensionless: bool = False
    result_bundle: Dict[str, Any]

class ExportRequest(BaseModel):
    index: int
    result_bundle: Dict[str, Any]

class OrbitPlotRequest(BaseModel):
    result_bundle: Dict[str, Any]
    index: Optional[int] = None
    jacobi: Optional[float] = None
    dimensionless: bool = False
    azim: int = -60
    elev: int = 30

class OrbitFamilyPlotRequest(BaseModel):
    result_bundle: Dict[str, Any]
    key: str
    val_range: Optional[Tuple[float, float]] = None
    dimensionless: bool = False
    max_display: int = 30
    azim: int = -60
    elev: int = 30

# ==== Endpoints ====
@app.post("/orbits/query")
def query_orbits(req: QueryRequest):
    api = CR3BPOrbitAPI(use_proxy=False)
    builder = CR3BPQueryBuilder(api)
    try:
        jacobi = (req.jacobimin, req.jacobimax) if req.jacobimin and req.jacobimax else None
        period = (req.periodmin, req.periodmax) if req.periodmin and req.periodmax else None
        stab = (req.stabmin, req.stabmax) if req.stabmin and req.stabmax else None

        result_bundle = builder.fetch_with_filters(
            sys=req.sys, family=req.family, libr=req.libr, branch=req.branch,
            jacobi_override=jacobi, period_override=period,
            stability_override=stab, periodunits=req.periodunits
        )
        return result_bundle
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/orbits/info")
def get_orbit_metadata(result_bundle: Dict[str, Any] = Body(...)):
    try:
        result = result_bundle["result"]
        interpreter = CR3BPResultInterpreter(result)
        metadata = {
            "system_info": {
                "name": interpreter.system_name,
                "lunit": interpreter.lunit,
                "tunit": interpreter.tunit,
                "mass_ratio": interpreter.mass_ratio,
                "libration_points": interpreter.libration_points,
            },
            "limits": result.get("limits", {}),
            "count": result.get("count", 0),
            "sample_orbits": interpreter.orbits[:5]
        }
        return metadata
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/orbits/select")
def select_orbit(req: OrbitSelectRequest):
    try:
        interpreter = CR3BPResultInterpreter(req.result_bundle["result"])
        selected = interpreter.select_orbit_by_index(req.index, dimensionless=req.dimensionless)
        return selected
    except IndexError:
        raise HTTPException(status_code=404, detail="Orbit index out of range")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/export/csv")
def export_csv(req: ExportRequest):
    try:
        interpreter = CR3BPResultInterpreter(req.result_bundle["result"])
        exporter = CR3BPExporter(
            fields=interpreter.fields,
            system_info=interpreter.response["system"],
            query_info=req.result_bundle["filters"],
            data=interpreter.response["data"]
        )
        file_path = exporter.to_csv(return_path=True)
        return {"export_path": file_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/export/cfg")
def export_cfg(req: ExportRequest):
    try:
        interpreter = CR3BPResultInterpreter(req.result_bundle["result"])
        exporter = CR3BPExporter(
            fields=interpreter.fields,
            system_info=interpreter.response["system"],
            query_info=req.result_bundle["filters"],
            data=interpreter.response["data"]
        )
        file_path = exporter.to_cfg(index=req.index, return_path=True)
        return {"export_path": file_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/export/trajectory")
def export_trajectory(req: ExportRequest):
    try:
        interpreter = CR3BPResultInterpreter(req.result_bundle["result"])
        exporter = CR3BPExporter(
            fields=interpreter.fields,
            system_info=interpreter.response["system"],
            query_info=req.result_bundle["filters"],
            data=interpreter.response["data"]
        )
        file_path = exporter.export_propagated_orbit_csv(index=req.index, return_path=True)
        return {"export_path": file_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/plot/single")
def plot_single_orbit(plot_req: OrbitPlotRequest):
    try:
        interpreter = CR3BPResultInterpreter(plot_req.result_bundle["result"])
        plotter = CR3BPPlotter(interpreter)

        orbit_idx = plot_req.index
        orbit_jacobi = plot_req.jacobi

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
            plt.ioff()
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            if orbit_idx is not None:
                orbit = dict(zip(interpreter.fields, map(float, interpreter.data[orbit_idx])))
            elif orbit_jacobi is not None:
                all_orbits = [
                    dict(zip(interpreter.fields, map(float, row)))
                    for row in interpreter.data
                ]
                orbit = min(all_orbits, key=lambda o: abs(o["jacobi"] - orbit_jacobi))
            else:
                raise HTTPException(status_code=400, detail="Specify either index or jacobi")

            x, y, z = plotter._propagate_orbit(orbit, num_points=1000)
            if not plot_req.dimensionless:
                x *= interpreter.lunit
                y *= interpreter.lunit
                z *= interpreter.lunit

            ax.plot(x, y, z, label="Orbit", alpha=0.8)
            ax.scatter([x[0]], [y[0]], [z[0]], s=50, c="red", marker="o", label="Initial")
            plotter._plot_libration_point(ax, plot_req.dimensionless)
            plotter._autoscale_axes(ax, x, y, z)

            unit = "LU" if plot_req.dimensionless else "km"
            ax.set_xlabel(f"x ({unit})")
            ax.set_ylabel(f"y ({unit})")
            ax.set_zlabel(f"z ({unit})")
            ax.view_init(elev=plot_req.elev, azim=plot_req.azim)
            ax.set_title(f"{interpreter.system_name} - Orbit")

            plt.tight_layout()
            plt.savefig(tmpfile.name)
            plt.close(fig)
            return FileResponse(tmpfile.name, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Plot error: {str(e)}")

@app.post("/plot/family")
def plot_family(plot_req: OrbitFamilyPlotRequest):
    try:
        interpreter = CR3BPResultInterpreter(plot_req.result_bundle["result"])
        plotter = CR3BPPlotter(interpreter)

        key = plot_req.key
        val_range = plot_req.val_range
        max_display = plot_req.max_display

        raw_orbits = [
            dict(zip(interpreter.fields, map(float, row)))
            for row in interpreter.data
        ]
        if val_range:
            raw_orbits = [o for o in raw_orbits if val_range[0] <= o[key] <= val_range[1]]

        raw_orbits.sort(key=lambda o: o[key])
        selected = raw_orbits[::max(1, len(raw_orbits) // max_display)]

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
            plt.ioff()
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            cmap = plt.get_cmap("viridis")
            norm = Normalize(min(o[key] for o in selected), max(o[key] for o in selected))

            for orbit in selected:
                x, y, z = plotter._propagate_orbit(orbit)
                if not plot_req.dimensionless:
                    x *= interpreter.lunit
                    y *= interpreter.lunit
                    z *= interpreter.lunit
                ax.plot(x, y, z, color=cmap(norm(orbit[key])))

            plotter._plot_libration_point(ax, plot_req.dimensionless)
            plotter._autoscale_axes(ax, x, y, z)

            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax.view_init(elev=plot_req.elev, azim=plot_req.azim)
            ax.set_title(f"{interpreter.system_name} - Family by {key.title()}")

            sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            plt.colorbar(sm, ax=ax, label=key.title())

            plt.tight_layout()
            plt.savefig(tmpfile.name)
            plt.close(fig)
            return FileResponse(tmpfile.name, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Family plot error: {str(e)}")
