import json

import pytest
import requests

from utils import ServerPreset


@pytest.fixture
def server():
    server = ServerPreset.tinyllama2()
    server.n_gpu_layer = 0
    yield server
    server.stop()


def test_disabled(server):
    server.start()
    assert server.make_request("GET", "/densex/requests").status_code == 404
    response = server.make_request("POST", "/completion", data={"prompt": "hello", "n_predict": 1})
    assert "X-DenseX-Request-Id" not in response.headers


def test_summary_and_invalid_json_are_private(server):
    server.server_telemetry = True
    server.start()
    prompt = "distinctive_prompt_secret_47821"
    response = server.make_request("POST", "/completion", data={"prompt": prompt, "n_predict": 4})
    assert response.status_code == 200
    request_id = int(response.headers["X-DenseX-Request-Id"])
    invalid = requests.post(server.make_url("/completion"), data='{"secret_94872":', timeout=30)
    assert invalid.status_code == 400
    invalid_id = int(invalid.headers["X-DenseX-Request-Id"])

    snapshot = server.make_request("GET", "/densex/requests").body
    records = {row["id"]: row for row in snapshot["requests"]}
    record = records[request_id]
    assert record["status"] == "completed"
    assert record["output_tokens"] == response.body["timings"]["predicted_n"]
    assert record["task_count"] == 1
    assert record["queue_ms"] is not None and record["processing_ms"] is not None
    assert records[invalid_id]["status"] == "error"
    assert records[invalid_id]["finish_reason"] == "invalid_request"
    payload = json.dumps(snapshot)
    assert prompt not in payload
    assert "secret_94872" not in payload
    content = response.body["content"]
    if len(content) > 16:
        assert content not in payload


def test_retention_reports_eviction(server):
    server.server_telemetry = True
    server.start()
    for _ in range(130):
        response = requests.post(server.make_url("/completion"), data="{", timeout=30)
        assert response.status_code == 400
    snapshot = server.make_request("GET", "/densex/requests").body
    assert len(snapshot["requests"]) == 128
    assert len(snapshot["events"]) == 128
    assert snapshot["requests_evicted"] == 2
    assert snapshot["events_evicted"] >= 2
