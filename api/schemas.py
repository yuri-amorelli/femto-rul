"""Contratto dell'API a due stadi (rilevatore -> regressione RUL).

Pattern stateless con stato a carico del client: il server non memorizza
nulla tra le chiamate; la feature_history viaggia nella risposta e il
client la rimanda alla chiamata successiva.
"""
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

SNAPSHOT_LEN = 2560  # campioni per snapshot FEMTO (25.6 kHz x 0.1 s)


class PredictRequest(BaseModel):
    operating_condition: Literal[1, 2, 3] = Field(
        description="Condizione operativa dichiarata dal client (regime carico/velocita')."
    )
    snapshot_h: list[float] = Field(
        description=f"Segnale grezzo canale orizzontale, esattamente {SNAPSHOT_LEN} campioni."
    )
    snapshot_v: list[float] = Field(
        description=f"Segnale grezzo canale verticale, esattamente {SNAPSHOT_LEN} campioni."
    )
    feature_history: list[list[float]] = Field(
        default_factory=list,
        description=(
            "Storia delle feature NON scalate restituita dalla risposta precedente. "
            "Lista vuota al primo snapshot del cuscinetto."
        ),
    )

    @field_validator("snapshot_h", "snapshot_v")
    @classmethod
    def _check_snapshot_len(cls, v: list[float]) -> list[float]:
        if len(v) != SNAPSHOT_LEN:
            raise ValueError(f"lo snapshot deve contenere esattamente {SNAPSHOT_LEN} campioni, ricevuti {len(v)}")
        return v

    @field_validator("feature_history")
    @classmethod
    def _check_history_rows(cls, v: list[list[float]]) -> list[list[float]]:
        if v:
            widths = {len(row) for row in v}
            if len(widths) != 1:
                raise ValueError("feature_history contiene righe di larghezza diversa")
        return v


class PredictResponse(BaseModel):
    alarm_state: Literal["calibrating", "monitoring", "warning", "alarm"] = Field(
        description="calibrating: soglia in apprendimento sulla fase iniziale; "
                    "monitoring: sano; warning: allarmi passati revocati (degradazione "
                    "non-terminale); alarm: allarme attivo, RUL attendibile."
    )
    alarm: bool = Field(description="True se e solo se alarm_state == 'alarm'.")
    alarm_onset_index: Optional[int] = Field(
        default=None, description="Snapshot di inizio dell'allarme attivo, se presente."
    )
    revoked_alarms: int = Field(default=0, description="Allarmi scattati e poi revocati.")
    rul_seconds: Optional[float] = Field(
        default=None, description="RUL stimata in secondi. Presente solo in stato 'alarm'."
    )
    snapshots_seen: int = Field(description="Snapshot nella traiettoria dopo questa chiamata.")
    feature_history: list[list[float]] = Field(
        description="Storia feature aggiornata: il client la rimanda alla prossima chiamata."
    )
    model_version: str = Field(description="Identificativo artefatti per riproducibilita'.")


class HealthResponse(BaseModel):
    status: Literal["ok", "error"]
    artifacts_loaded: bool
    model_version: Optional[str] = None
    detail: Optional[str] = None

class PredictRequest(BaseModel):
    operating_condition: Literal[1, 2, 3]
    snapshot_h: list[float]          # 2560 campioni, canale orizzontale
    snapshot_v: list[float]          # 2560 campioni, canale verticale
    feature_history: list[list[float]] = []   # dalla risposta precedente, [] al primo snapshot
