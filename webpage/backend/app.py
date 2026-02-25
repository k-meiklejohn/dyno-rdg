from fastapi import FastAPI
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from render import render_svg
from schema import PARAM_SCHEMA, ROW_SCHEMA


app = FastAPI()

@app.get("/schema")
def schema():
    return {"params": PARAM_SCHEMA, "row": ROW_SCHEMA}

@app.post("/render")
async def render(data: dict):
    svg = render_svg(data["rows"], data["params"])
    return Response(svg, media_type="image/svg+xml")

# mount frontend LAST
app.mount("/", StaticFiles(directory="../frontend", html=True), name="frontend")


@app.get("/schema")
def schema():
    return {"params":PARAM_SCHEMA,"row":ROW_SCHEMA}

@app.post("/render")
async def render(data: dict):
    svg = render_svg(data["params"], data["rows"])
    return Response(svg, media_type="image/svg+xml")
