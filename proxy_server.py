from fastapi import FastAPI, Query, Body, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, Tuple
import tempfile
from functools import lru_cache
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
    key: Optional[str] = None  # "jacobi", "period", or "stability"
    value: Optional[float] = None
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

@lru_cache(maxsize=500)
def cached_nasa_query(**params):
    raw_api = CR3BPOrbitAPI(use_proxy=False)
    return raw_api.query(**params)

@app.post("/orbits/query")
def query_orbits(req: QueryRequest):
    try:
        api = CR3BPOrbitAPI(use_proxy=False)
        api.query = cached_nasa_query

        builder = CR3BPQueryBuilder(api)
        result_bundle = builder.fetch_with_filters(
            sys=req.sys,
            family=req.family,
            libr=req.libr,
            branch=req.branch,
            jacobi_override=(req.jacobimin, req.jacobimax) if req.jacobimin is not None and req.jacobimax is not None else None,
            period_override=(req.periodmin, req.periodmax) if req.periodmin is not None and req.periodmax is not None else None,
            stability_override=(req.stabmin, req.stabmax) if req.stabmin is not None and req.stabmax is not None else None,
            periodunits=req.periodunits
        )
        return result_bundle
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/orbits/info")
def get_orbit_metadata(result_bundle: Dict[str, Any] = Body(...)):
    try:
        interpreter = CR3BPResultInterpreter(result_bundle["result"])
        return {
            "system_info": {
                "name": interpreter.system_name,
                "lunit": interpreter.lunit,
                "tunit": interpreter.tunit,
                "mass_ratio": interpreter.mass_ratio,
                "libration_points": interpreter.libration_points,
            },
            "limits": interpreter.response.get("limits", {}),
            "count": interpreter.response.get("count", 0),
            "sample_orbits": interpreter.orbits[:5]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/orbits/select")
def select_orbit(req: OrbitSelectRequest):
    try:
        result = req.result_bundle["result"]
        interpreter = CR3BPResultInterpreter(result)
        return interpreter.select_orbit_by_index(req.index, dimensionless=req.dimensionless)
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
            system_info=interpreter.system,
            query_info=req.result_bundle["filters"],
            data=interpreter.response["data"]
        )
        file_path = exporter.to_csv()
        return {"export_path": file_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/export/cfg")
def export_cfg(req: ExportRequest):
    try:
        interpreter = CR3BPResultInterpreter(req.result_bundle["result"])
        exporter = CR3BPExporter(
            fields=interpreter.fields,
            system_info=interpreter.system,
            query_info=req.result_bundle["filters"],
            data=interpreter.response["data"]
        )
        file_path = exporter.to_cfg(index=req.index)
        return {"export_path": file_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/export/trajectory")
def export_trajectory(req: ExportRequest):
    try:
        interpreter = CR3BPResultInterpreter(req.result_bundle["result"])
        exporter = CR3BPExporter(
            fields=interpreter.fields,
            system_info=interpreter.system,
            query_info=req.result_bundle["filters"],
            data=interpreter.response["data"]
        )
        file_path = exporter.export_propagated_orbit_csv(index=req.index)
        return {"export_path": file_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/plot/single")
def plot_single_orbit(plot_req: OrbitPlotRequest):
    try:
        interpreter = CR3BPResultInterpreter(plot_req.result_bundle["result"])
        plotter = CR3BPPlotter(interpreter)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
            if plot_req.index is not None:
                orbit = dict(zip(interpreter.fields, map(float, interpreter.data[plot_req.index])))
                plotter.plot_orbit_by_property(
                    key="jacobi",
                    value=orbit["jacobi"],
                    dimensionless=plot_req.dimensionless,
                    save_path=tmpfile.name,
                    azim=plot_req.azim,
                    elev=plot_req.elev
                )
            elif plot_req.key and plot_req.value is not None:
                plotter.plot_orbit_by_property(
                    key=plot_req.key,
                    value=plot_req.value,
                    dimensionless=plot_req.dimensionless,
                    save_path=tmpfile.name,
                    azim=plot_req.azim,
                    elev=plot_req.elev
                )
            else:
                raise HTTPException(status_code=400, detail="Specify index or (key and value)")

        return FileResponse(tmpfile.name, media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Plot error: {str(e)}")


@app.post("/plot/family")
def plot_family(plot_req: OrbitFamilyPlotRequest):
    try:
        interpreter = CR3BPResultInterpreter(plot_req.result_bundle["result"])
        plotter = CR3BPPlotter(interpreter)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
            file_path = plotter.plot_family_by_range(
                key=plot_req.key,
                val_range=plot_req.val_range,
                dimensionless=plot_req.dimensionless,
                max_display=plot_req.max_display,
                azim=plot_req.azim,
                elev=plot_req.elev,
                save_path=tmpfile.name
            )
            return FileResponse(file_path, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Family plot error: {str(e)}")

