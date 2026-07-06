"""FastAPI per il sistema a due stadi FEMTO-RUL (rilevatore -> regressione).

Avvio dal root del repo:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from api.schemas import HealthResponse, PredictRequest, PredictResponse
from api.service import RULService

service = RULService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Gli artefatti si caricano UNA volta all'avvio, mai per-richiesta.
    service.load()
    yield


app = FastAPI(
    title="FEMTO-RUL Two-Stage API",
    description=(
        "RUL estimation su cuscinetti FEMTO/PRONOSTIA con sistema a due stadi: "
        "il rilevatore di onset (detect_fpt_slope) fa scattare l'allarme, e solo "
        "da quel momento la regressione RUL viene esposta. Pre-allarme, "
        "rul_seconds e' deliberatamente null: la stima non sarebbe attendibile."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    if not service.loaded:
        return HealthResponse(status="error", artifacts_loaded=False, detail="artefatti non caricati")
    return HealthResponse(status="ok", artifacts_loaded=True, model_version=service.model_version)


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    if not service.loaded:
        raise HTTPException(status_code=503, detail="Servizio non pronto: artefatti non caricati")
    try:
        result = service.step(
            snapshot_h=req.snapshot_h,
            snapshot_v=req.snapshot_v,
            condition=req.operating_condition,
            feature_history=req.feature_history,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return PredictResponse(**result, model_version=service.model_version)
