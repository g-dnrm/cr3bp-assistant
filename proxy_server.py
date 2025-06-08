from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import httpx

app = FastAPI()

NASA_API_URL = "https://ssd-api.jpl.nasa.gov/periodic_orbits.api"

@app.get("/query_orbits")
async def query_orbits(request: Request):
    try:
        # Forward all query parameters to NASA API
        async with httpx.AsyncClient() as client:
            response = await client.get(NASA_API_URL, params=dict(request.query_params))
        return JSONResponse(content=response.json(), status_code=response.status_code)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
