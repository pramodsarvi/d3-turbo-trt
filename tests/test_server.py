from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

from d3trt.server import build_server


def _get(url: str):
    with urllib.request.urlopen(url, timeout=10) as response:
        return response.status, json.loads(response.read())


def _post(url: str, body):
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def test_reference_http_contract(engine):
    server = build_server(engine, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    base = f"http://{host}:{port}"
    try:
        status, health = _get(base + "/health")
        assert status == 200
        assert health["status"] == "ready"
        assert health["max_input_tokens"] is not None
        assert "text" in health["modalities"]
        assert health["authentication"] is False

        status, models = _get(base + "/v1/models")
        assert status == 200
        assert models["models"][0]["name"] == engine.frontend.d3.model_name

        body = {
            "state": "The HTTP route is healthy.",
            "questions": {
                "route": {
                    "type": "choice",
                    "instructions": "Pick a team.",
                    "criteria": {"sre": "reliability", "web": "frontend"},
                }
            },
        }
        status, reply = _post(base + "/v1/systemone", body)
        assert status == 200
        assert "answers" in reply and "usage" in reply
        assert reply["answers"]["route"]["type"] == "choice"
        assert set(reply["answers"]["route"]["probabilities"]) == {"sre", "web"}

        status, error = _post(base + "/v1/systemone", {"questions": {}})
        assert status == 422
        assert "detail" in error

        status, error = _post(
            base + "/v1/systemone",
            {"questions": {"bad": {"type": "choice", "instructions": "no criteria"}}},
        )
        assert status == 422
        assert "invalid_questions" in error["detail"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)
