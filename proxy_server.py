from fastapi import FastAPI, Query, Body, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field, field_validator, model_validator
from enum import Enum
from typing import Optional, Dict, Any, Tuple
import re
import tempfile
from functools import lru_cache

from core import (
    CR3BPOrbitAPI,
    CR3BPQueryBuilder,
    CR3BPResultInterpreter,
    CR3BPExporter,
    CR3BPPlotter
)

app = FastAPI(title="CR3BP Periodic Orbits Assistant")

@app.get("/")
def root():
    return {
        "message": "Welcome to the CR3BP Periodic Orbits API!",
        "docs": "Visit /docs for the interactive API documentation.",
        "openapi": "Visit /openapi.json for the OpenAPI schema."
    }

# === ENUMS ===
class Family(str, Enum):
    halo = "halo"
    vertical = "vertical"
    axial = "axial"
    lyapunov = "lyapunov"
    longp = "longp"
    short = "short"
    butterfly = "butterfly"
    dragonfly = "dragonfly"
    resonant = "resonant"
    dro = "dro"
    dpo = "dpo"
    lpo = "lpo"

class PeriodUnits(str, Enum):
    s = "s"
    h = "h"
    d = "d"
    TU = "TU"

# ==== Models ====
class BaseQueryRequest(BaseModel):
    sys: str
    family: Family
    libr: Optional[int] = None
    branch: Optional[str] = None
    periodunits: PeriodUnits = PeriodUnits.TU

    @field_validator("sys")
    def sys_must_be_primary_secondary(cls, v):
        if not re.fullmatch(r"[a-z]+-[a-z]+", v):
            raise ValueError("sys must be formatted as 'primary-secondary', lowercase.")
        return v

    @field_validator("libr")
    def libr_must_be_in_range(cls, v):
        if v is not None and not (1 <= v <= 5):
            raise ValueError("libr must be an integer between 1 and 5.")
        return v

class FilteredQueryRequest(BaseQueryRequest):
    jacobimin: Optional[float] = None
    jacobimax: Optional[float] = None
    periodmin: Optional[float] = None
    periodmax: Optional[float] = None
    stabmin: Optional[float] = None
    stabmax: Optional[float] = None

    @field_validator("periodmin", "periodmax", "stabmin", "stabmax")
    def must_be_positive(cls, v):
        if v is not None and v < 0:
            raise ValueError("Values must be positive.")
        return v

    @model_validator(mode="after")
    def check_family_requires_libr_and_branch(self):
        family = self.family
        libr = self.libr
        branch = self.branch

        libr_required = {"lyapunov", "halo", "vertical", "axial", "longp", "short"}
        if family in libr_required and libr is None:
            raise ValueError(f"Family '{family}' requires a libration point (1–5).")

        branch_rules = {
            "halo": ["N", "S"],
            "dragonfly": ["N", "S"],
            "butterfly": ["N", "S"],
            "lpo": ["E", "W"],
            "resonant": "integer"
        }

        if family in branch_rules:
            if branch is None:
                raise ValueError(f"Family '{family}' requires a branch.")
            allowed = branch_rules[family]
            if allowed == "integer":
                if not branch.isdigit():
                    raise ValueError(f"Family '{family}' requires an integer branch like '12' for 1:2.")
            elif branch not in allowed:
                raise ValueError(f"Family '{family}' requires branch in {allowed}.")

        return self

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

# ==== Helper ====
# @lru_cache(maxsize=500)
# def cached_nasa_query(**params):
#     raw_api = CR3BPOrbitAPI(use_proxy=False)
#     return raw_api.query(**params)

def query_nasa_direct(**params):
    raw_api = CR3BPOrbitAPI(use_proxy=False)
    return raw_api.query(**params)


# === INFO ENDPOINT ===
@app.post("/orbits/info", summary="Retrieve orbit family metadata")
def get_family_info(req: BaseQueryRequest):
    try:
        api = CR3BPOrbitAPI(use_proxy=False)
        builder = CR3BPQueryBuilder(api)
        result = builder.query_raw_info(
            sys=req.sys,
            family=req.family,
            libr=req.libr,
            branch=req.branch,
            periodunits=req.periodunits
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# FILTERED QUERY
@app.post("/orbits/filter",  summary="Filter periodic orbits based on constraints")  # MODIFIED: Return raw NASA filtered results
def get_filtered_family(req: FilteredQueryRequest):
    """
     Returns filtered periodic orbits based on system, family, and optional constraints such as
     Jacobi constant, period, and stability index. Useful for selecting subsets of orbits for
     analysis, visualization, or mission planning.

     Fields:
     - `sys`: Primary-secondary system (e.g., "earth-moon")
     - `family`: Orbit family (e.g., halo, vertical)
     - Optional filters: `jacobimin`, `jacobimax`, `periodmin`, `periodmax`, `stabmin`, `stabmax`
     - `periodunits`: Time units (TU, s, h, d)
     """
    try:
        params = {k: v for k, v in req.model_dump().items() if v is not None}  # MODIFIED
        result = query_nasa_direct(**params)  # MODIFIED
        return result  # MODIFIED
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# SELECT ORBIT
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

# ==== Endpoints ====
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