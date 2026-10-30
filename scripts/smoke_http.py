#!/usr/bin/env python3
"""Black-box check of a running d3-trt server: /health, /v1/models, /v1/systemone, 422s.

    python scripts/smoke_http.py --base http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request


def call(url: str, body=None, timeout: float = 600):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--wait", type=float, default=600, help="seconds to wait for /health")
    ap.add_argument("--pid", type=int, help="abort early if this server process exits")
    args = ap.parse_args()

    deadline = time.time() + args.wait
    while True:
        try:
            status, health = call(args.base + "/health", timeout=5)
            if status == 200 and health.get("status") == "ready":
                break
        except Exception:
            pass
        if args.pid:
            try:
                os.kill(args.pid, 0)
            except OSError:
                print("FAIL: server process exited (see server.log)")
                return 1
        if time.time() > deadline:
            print("FAIL: server not ready")
            return 1
        time.sleep(2)

    failures = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        print(f"{'PASS' if ok else 'FAIL'}: {name} {detail}")
        if not ok:
            failures.append(name)

    status, models = call(args.base + "/v1/models")
    check("/v1/models", status == 200 and bool(models.get("models")))

    body = {
        "state": "The checkout page returns HTTP 500 after the last deploy.",
        "questions": {
            "route": {
                "type": "choice",
                "instructions": "Which team should handle this?",
                "criteria": {"sre": "reliability and incidents", "web": "frontend UI"},
            },
            "urgent": {"type": "noul", "instructions": "Is this urgent?"},
        },
    }
    t0 = time.time()
    status, reply = call(args.base + "/v1/systemone", body)
    ms = (time.time() - t0) * 1000
    ok = status == 200 and set(reply.get("answers", {})) == {"route", "urgent"}
    check("/v1/systemone", ok, f"({ms:.0f} ms)")
    if ok:
        probs = reply["answers"]["route"]["probabilities"]
        check("probabilities sum to 1", abs(sum(probs.values()) - 1) < 1e-3, json.dumps(probs))
        t0 = time.time()
        call(args.base + "/v1/systemone", body)
        print(f"INFO: repeat (warm state cache) {(time.time() - t0) * 1000:.0f} ms")

    status, _ = call(args.base + "/v1/systemone", {"questions": {}})
    check("422 on empty questions", status == 422)
    status, _ = call(args.base + "/v1/systemone", {"questions": {"b": {"type": "choice", "instructions": "x"}}})
    check("422 on choice without criteria", status == 422)

    print("ALL PASSED" if not failures else f"FAILED: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
