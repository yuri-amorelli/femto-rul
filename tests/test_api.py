"""HTTP contract of the API, through FastAPI's test client (startup and shutdown included)."""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import app
from src import config


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:          # runs the lifespan: the bundle is loaded once
        yield c


def snapshot(seed: int = 0) -> dict:
    s = np.random.default_rng(seed).standard_normal((config.SNAPSHOT_LEN, 2))
    return {"snapshot_h": s[:, 0].tolist(), "snapshot_v": s[:, 1].tolist()}


def test_health_reports_the_bundle_version(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["artifacts_loaded"] is True
    assert body["model_version"].startswith("femto-stage2-q0.30-") and body["dataset"] == "femto"


def test_demo_page_is_served(client):
    r = client.get("/")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    assert "Bearing degradation monitor" in r.text and "demo/replay/" in r.text


def test_demo_bearings(client):
    bearings = {b["bearing"]: b for b in client.get("/demo/bearings").json()}
    assert len(bearings) == 11
    assert bearings["Bearing2_7"]["n_snapshots"] == 230
    assert bearings["Bearing1_3"]["official_truncation_index"] == 1801


def test_replay_of_a_test_bearing(client):
    r = client.get("/demo/replay/Bearing3_3")
    assert r.status_code == 200
    body = r.json()
    n = body["n_snapshots"]
    assert len(body["h_rms"]) == len(body["rul_pred_s"]) == len(body["rul_true_s"]) == n
    assert body["segments"][0]["start"] == 0 and body["segments"][-1]["end"] == n - 1
    assert body["summary"]["alarm_at_end"] is True and body["segments"][-1]["state"] == "alarm"
    assert body["model_version"] == client.get("/health").json()["model_version"]


def test_unknown_bearing_is_404(client):
    r = client.get("/demo/replay/Bearing9_9")
    assert r.status_code == 404 and "available" in r.json()["detail"]


def test_predict_streams_with_client_held_history(client):
    r = client.post("/predict", json={"operating_condition": 1, **snapshot(1), "feature_history": []})
    assert r.status_code == 200
    first = r.json()
    assert first["alarm_state"] == "calibrating" and first["rul_seconds"] is None
    assert first["snapshots_seen"] == 1 and len(first["feature_history"][0]) == 4
    r = client.post("/predict", json={"operating_condition": 1, **snapshot(2),
                                      "feature_history": first["feature_history"]})
    assert r.status_code == 200 and r.json()["snapshots_seen"] == 2


@pytest.mark.parametrize("change, message", [
    ({"snapshot_h": [0.0] * 100}, "exactly 2560 samples"),
    ({"operating_condition": 4}, "operating_condition"),
    ({"feature_history": [[0.0] * 34]}, "another model version"),
])
def test_predict_rejects_malformed_requests(client, change, message):
    payload = {"operating_condition": 1, **snapshot(), "feature_history": []} | change
    r = client.post("/predict", json=payload)
    assert r.status_code == 422 and message in r.json()["detail"]


def test_predict_rejects_a_missing_field(client):
    r = client.post("/predict", json={"operating_condition": 1})
    assert r.status_code == 422


def test_model_card(client):
    card = client.get("/model").json()
    assert card["model_version"] == client.get("/health").json()["model_version"]
    assert card["stage2"]["quantile"] == 0.3
    assert card["evaluation_at_export"]["curve_bearings"] == 9


def test_swagger_offers_the_test_bearings_as_examples(client):
    spec = client.get("/openapi.json").json()
    params = spec["paths"]["/demo/replay/{bearing}"]["get"]["parameters"]
    assert "Bearing1_3" in params[0]["examples"]
