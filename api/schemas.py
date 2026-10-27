"""API contract of the two-stage service (causal detector -> stage-2 RUL).

Stateless, with the state held by the client: the server stores nothing between
calls; feature_history travels in the response and the client sends it back
with the next snapshot.

Compatibility rule: fields are added, never changed or removed. Dataset-specific
checks (snapshot length, operating conditions, history width) are made by the
service against the loaded bundle and answered with 422.
"""
from typing import Literal, Optional

from pydantic import BaseModel, Field

State = Literal["calibrating", "monitoring", "warning", "alarm"]


class PredictRequest(BaseModel):
    operating_condition: int = Field(
        description="Operating condition declared by the client (FEMTO-ST: 1, 2 or 3). "
                    "Not an input of the current model; validated and kept for monitoring by condition.")
    snapshot_h: list[float] = Field(
        description="Raw horizontal-channel snapshot (FEMTO-ST: exactly 2560 samples).")
    snapshot_v: list[float] = Field(
        description="Raw vertical-channel snapshot (FEMTO-ST: exactly 2560 samples).")
    feature_history: list[list[float]] = Field(
        default_factory=list,
        description="feature_history returned by the previous response, unchanged. "
                    "Empty list at the first snapshot of a bearing.")


class PredictResponse(BaseModel):
    alarm_state: State = Field(
        description="calibrating: the bearing's own healthy threshold is still being learnt; "
                    "monitoring: healthy; warning: an earlier alarm was revoked (non-terminal "
                    "degradation); alarm: alarm active, RUL estimated.")
    alarm: bool = Field(description="True if and only if alarm_state == 'alarm'.")
    alarm_onset_index: Optional[int] = Field(
        default=None, description="Snapshot where the active alarm started, if any.")
    revoked_alarms: int = Field(default=0, description="Alarms raised and later revoked.")
    rul_seconds: Optional[float] = Field(
        default=None,
        description="Estimated remaining useful life [s], only under an active alarm. A conservative "
                    "quantile (see rul_quantile), not a mean: in the evaluation it is usually early.")
    rul_quantile: Optional[float] = Field(
        default=None, description="Quantile estimated by the stage-2 model (0.3 for the current model).")
    snapshots_seen: int = Field(description="Snapshots in the trajectory after this call.")
    feature_history: list[list[float]] = Field(
        description="Updated history: send it back with the next call.")
    model_version: str = Field(description="Bundle version, for reproducibility.")
    dataset: Optional[str] = Field(default=None, description="Dataset the served bundle was built for.")


class HealthResponse(BaseModel):
    status: Literal["ok", "error"]
    artifacts_loaded: bool
    model_version: Optional[str] = None
    dataset: Optional[str] = None
    detail: Optional[str] = None


class DemoBearing(BaseModel):
    bearing: str
    operating_condition: int
    n_snapshots: int
    official_truncation_index: int = Field(
        description="Last snapshot given in the official Test_set: the point where the challenge asked for the RUL.")
    official_actual_rul_s: Optional[float] = None
    note: Optional[str] = None


class StateSegment(BaseModel):
    state: State
    start: int
    end: int = Field(description="Inclusive.")


class ReplaySummary(BaseModel):
    alarm_at_end: bool
    final_onset_index: Optional[int] = None
    final_onset_pct_of_life: Optional[float] = None
    warning_before_end_s: Optional[float] = None
    first_alarm_pct_of_life: Optional[float] = None
    revoked_alarms: int
    curve_snapshots: int = Field(description="Snapshots of the final alarm, where stage 2 is evaluated.")
    curve_mae_s: Optional[float] = None
    curve_late_fraction: Optional[float] = Field(
        default=None, description="Share of curve predictions above the true RUL (late, the costly side).")
    exposed_mae_s: Optional[float] = Field(
        default=None, description="Error over every snapshot where a RUL was returned, revoked alarms included.")
    state_at_truncation: State
    rul_at_truncation_s: Optional[float] = None


class ReplayResponse(BaseModel):
    bearing: str
    operating_condition: int
    snapshot_period_s: float
    n_snapshots: int
    official_truncation_index: int
    official_actual_rul_s: Optional[float] = None
    note: Optional[str] = None
    summary: ReplaySummary
    segments: list[StateSegment]
    h_rms: list[float] = Field(description="Horizontal RMS per snapshot, the detector's input.")
    rul_pred_s: list[Optional[float]] = Field(description="Served RUL per snapshot (null without an active alarm).")
    rul_true_s: list[float] = Field(description="Seconds to the end of the recording.")
    model_version: str
