"""FastAPI service for the FEMTO-RUL two-stage system (causal detector -> stage-2 RUL).

Run from the repository root:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

Then open http://127.0.0.1:8000/ for the demo page, /docs for the API.
"""
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException
from fastapi import Path as PathParam
from fastapi.responses import HTMLResponse

from api.schemas import (DemoBearing, HealthResponse, PredictRequest, PredictResponse,
                         ReplayResponse)
from api.service import RULService

service = RULService()
INDEX_HTML = (Path(__file__).resolve().parent / "static" / "index.html").read_text(encoding="utf-8")
# bearing names for the Swagger dropdown, read from the bundle folder (no model loading at import)
DEMO_NAMES = sorted(p.stem for p in (service.bundle_dir / "demo").glob("*.json"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # The bundle is loaded ONCE at startup, never per request.
    service.load()
    service.warm_up()
    yield


app = FastAPI(
    title="FEMTO-RUL Two-Stage API",
    description=(
        "Remaining useful life (RUL) of FEMTO-ST / PRONOSTIA bearings in two stages. A causal, "
        "self-calibrating detector raises the alarm; only under an active alarm does a linear "
        "quantile regression, trained on degradation curves only, estimate the RUL. Before the "
        "alarm, rul_seconds is deliberately null.\n\n"
        "**Quickest look:** `GET /demo/replay/{bearing}` replays one of the 11 official test "
        "bearings through the same code as `/predict`; the page at `/` draws it.\n\n"
        "**Streaming:** `POST /predict` is stateless. Each call sends the latest raw snapshot plus "
        "the `feature_history` returned by the previous call."
    ),
    version="0.2.0",
    lifespan=lifespan,
)


def _require_loaded() -> None:
    if not service.loaded:
        raise HTTPException(status_code=503, detail="Service not ready: bundle not loaded")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index() -> str:
    return INDEX_HTML


@app.get("/health", response_model=HealthResponse, tags=["service"])
def health() -> HealthResponse:
    if not service.loaded:
        return HealthResponse(status="error", artifacts_loaded=False, detail="bundle not loaded")
    return HealthResponse(status="ok", artifacts_loaded=True, model_version=service.model_version,
                          dataset=service.bundle.dataset.name)


@app.get("/model", tags=["service"], summary="What is served: versions, parameters, evaluation at export")
def model_card() -> dict:
    _require_loaded()
    return service.bundle.model_card()


@app.post("/predict", response_model=PredictResponse, tags=["streaming"])
def predict(req: PredictRequest) -> PredictResponse:
    _require_loaded()
    try:
        result = service.step(snapshot_h=req.snapshot_h, snapshot_v=req.snapshot_v,
                              operating_condition=req.operating_condition,
                              feature_history=req.feature_history)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return PredictResponse(**result, model_version=service.model_version, dataset=service.bundle.dataset.name)


@app.get("/demo/bearings", response_model=list[DemoBearing], tags=["demo"])
def demo_bearings() -> list[DemoBearing]:
    _require_loaded()
    return [DemoBearing(**service.bundle.demo_info(n)) for n in service.bundle.demo_bearings]


@app.get("/demo/replay/{bearing}", response_model=ReplayResponse, tags=["demo"],
         summary="Replay a full official test bearing through the deployed two-stage system")
def demo_replay(
    bearing: Annotated[str, PathParam(
        description="One of the official test bearings, e.g. Bearing1_3.",
        openapi_examples={n: {"value": n} for n in DEMO_NAMES},
    )],
) -> ReplayResponse:
    _require_loaded()
    try:
        return ReplayResponse(**service.bundle.replay(bearing))
    except KeyError:
        raise HTTPException(status_code=404,
                            detail=f"unknown bearing {bearing!r}; available: {service.bundle.demo_bearings}")
