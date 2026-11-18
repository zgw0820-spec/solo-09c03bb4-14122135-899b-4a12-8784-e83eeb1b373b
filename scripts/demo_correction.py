#!/usr/bin/env python3
"""Label-correction demo: fix a mis-labelled sample while training is running.

Run against Docker Compose (the delivered setup)::

    docker compose up -d --build
    python3 scripts/demo_correction.py

Run against local processes (no Docker available)::

    python3 scripts/demo_correction.py --local

Steps:
1. submit a job and wait until the batch containing sample ``--record`` has
   been consumed more than once (it is in an early epoch);
2. read the job to see the stored label;
3. ``POST /jobs/{id}/corrections`` with the old and new label; the job rewinds
   to the breakpoint before that sample's first batch and discards later ones;
4. wait for training to finish;
5. submit the same records with the corrected label from scratch and assert the
   final weights, momentum and loss are bit-identical.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = os.environ.get("API_BASE", "http://127.0.0.1:8000")


def call(method: str, path: str, body: dict | None = None, expect: int = 200):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(
        f"{BASE}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode()
        raise SystemExit(f"{method} {path} -> HTTP {exc.code}: {detail}") from exc


def wait_for(predicate, timeout: float, interval: float = 0.2, message: str = ""):
    deadline = time.time() + timeout
    while time.time() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(interval)
    raise SystemExit(f"timed out waiting for {message}")


def _state_at_least(job_id: str, step: int) -> dict | None:
    state = call("GET", f"/jobs/{job_id}")
    return state if state["completed_steps"] >= step else None


def _state_if_completed(job_id: str) -> dict | None:
    state = call("GET", f"/jobs/{job_id}")
    return state if state["status"] == "completed" else None


def sample_records(count: int = 40, seed: int = 3) -> dict:
    import random

    rng = random.Random(seed)
    features, labels = [], []
    for index in range(count):
        vibration = rng.gauss(0.2, 1.0)
        current = rng.gauss(1.0, 1.0)
        temperature = rng.gauss(0.5, 1.0)
        label = 1 if vibration + current + 0.3 * temperature > 2.1 else 0
        if index % 11 == 0:
            label = 1
        features.append(
            [
                round(vibration, 4),
                round(current, 4),
                round(temperature, 4),
                round(abs(vibration - current), 4),
            ]
        )
        labels.append(label)
    return {"features": features, "labels": labels, "epochs": 3}


def show(job: dict) -> None:
    print(
        f"  status={job['status']:<9} steps={job['completed_steps']}/{job['total_steps']}"
        f" progress={job['progress']:.0%} loss={job['loss']:.6f}"
        f" weights={[round(w, 6) for w in job['weights']]}"
    )


def step_1_submit(data: dict) -> dict:
    print("\n[1] submitting 40 records, 3 epochs (8 per batch -> 5 batches/epoch, 15 steps)")
    job = call("POST", "/jobs", data)
    print(f"    job_id={job['job_id']}")
    show(job)
    return job


def step_2_watch(job_id: str, record: int, first_affected: int) -> dict:
    # Wait until we are at least one full epoch past the sample's first batch:
    # this proves the old label was actually used for training, so the demo
    # exercises the rollback path rather than a pre-training relabel.
    target = first_affected + 5
    print(f"\n[2] waiting until step >= {target} (sample {record} first used at step {first_affected})")
    state = wait_for(
        lambda: _state_at_least(job_id, target),
        timeout=180,
        message="training progress",
    )
    show(state)
    return state


def step_3_correct(job_id: str, record: int, old_label: int, new_label: int) -> dict:
    print(
        f"\n[3] correcting sample {record}: label {old_label} -> {new_label}"
        f" via POST /jobs/{job_id}/corrections"
    )
    result = call(
        "POST",
        f"/jobs/{job_id}/corrections",
        {"record_index": record, "old_label": old_label, "new_label": new_label},
    )
    print(
        f"    first affected step={result['first_affected_step']}"
        f" rolled back from step {result['rolled_back_from_step']}"
        f" to {result['resumed_from_step']}"
        f" ({result['deleted_checkpoints']} checkpoints discarded)"
    )
    after = call("GET", f"/jobs/{job_id}")
    show(after)
    steps = [c["step"] for c in call("GET", f"/jobs/{job_id}/checkpoints")]
    print(f"    retained checkpoint steps: {steps}")
    return result


def step_4_finish(job_id: str) -> dict:
    print("\n[4] waiting for workers to continue from the retained breakpoint")
    final = wait_for(
        lambda: _state_if_completed(job_id),
        timeout=180,
        message="completion after correction",
    )
    show(final)
    checkpoints = call("GET", f"/jobs/{job_id}/checkpoints")
    steps = [c["step"] for c in checkpoints]
    assert steps == list(range(1, final["total_steps"] + 1)), steps
    print(f"    checkpoint steps strictly 1..{final['total_steps']}")
    return final


def step_5_reference(final: dict, corrected: dict) -> None:
    print("\n[5] equivalence: training the corrected data from scratch")
    job = call("POST", "/jobs", corrected)
    reference = wait_for(
        lambda: _state_if_completed(job["job_id"]),
        timeout=180,
        message="from-scratch reference completion",
    )
    same_weights = reference["final_weights"] == final["final_weights"]
    same_momentum = reference["momentum_velocity"] == final["momentum_velocity"]
    same_loss = reference["loss"] == final["loss"]
    print(f"    from-scratch weights = {reference['final_weights']}")
    print(f"    weights identical   : {same_weights}")
    print(f"    momentum identical  : {same_momentum}")
    print(f"    loss identical      : {same_loss} ({reference['loss']!r})")
    if not (same_weights and same_momentum and same_loss):
        raise SystemExit("FAIL: corrected run differs from from-scratch training")
    print("    PASS: correcting in place is bit-identical to retraining from scratch")


def local_up() -> None:
    data = ROOT / "data"
    data.mkdir(exist_ok=True)
    env = dict(os.environ)
    env.update(
        TRAINING_DATA_DIR=str(data),
        TRAINING_LEASE_SECONDS="3",
        TRAINING_HEARTBEAT_INTERVAL="0.8",
        TRAINING_BATCH_DELAY_SECONDS="0.5",
        PYTHONPATH=str(ROOT),
        PYTHONUNBUFFERED="1",
    )
    logs = data / "logs"
    logs.mkdir(exist_ok=True)
    api = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--port", "8000"],
        cwd=ROOT,
        env=env,
        stdout=open(logs / "api.log", "w"),
        stderr=subprocess.STDOUT,
    )
    (ROOT / ".demo-api.pid").write_text(str(api.pid))
    for worker in ("worker-1", "worker-2"):
        proc = subprocess.Popen(
            [sys.executable, "-m", "app.worker", "--worker-id", worker],
            cwd=ROOT,
            env=env,
            stdout=open(logs / f"{worker}.log", "w"),
            stderr=subprocess.STDOUT,
        )
        (ROOT / f".demo-{worker}.pid").write_text(str(proc.pid))
    wait_for(_healthy, timeout=30, message="local API")


def _healthy() -> bool:
    try:
        return call("GET", "/health")["status"] == "ok"
    except (urllib.error.URLError, KeyError, TimeoutError):
        return False


def local_down() -> None:
    for name in ("worker-1", "worker-2"):
        pid_file = ROOT / f".demo-{name}.pid"
        if pid_file.exists():
            try:
                os.kill(int(pid_file.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass
            pid_file.unlink()
    api_pid = ROOT / ".demo-api.pid"
    if api_pid.exists():
        try:
            os.kill(int(api_pid.read_text()), signal.SIGKILL)
        except ProcessLookupError:
            pass
        api_pid.unlink()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true", help="start API and workers locally")
    parser.add_argument(
        "--record",
        type=int,
        default=23,
        help="zero-based sample index to correct (default 23 -> batch 3, so the demo keeps earlier breakpoints)",
    )
    args = parser.parse_args()

    if args.local:
        local_up()
    else:
        wait_for(_healthy, timeout=30, message="API (is docker compose up?)")

    try:
        data = sample_records()
        record = args.record
        first_affected = record // 8 + 1
        job = step_1_submit(data)
        before = step_2_watch(job["job_id"], record, first_affected)
        old_label = before["labels"][record]
        new_label = 1 - old_label
        step_3_correct(job["job_id"], record, old_label, new_label)
        final = step_4_finish(job["job_id"])
        corrected = dict(data)
        corrected["labels"] = list(data["labels"])
        corrected["labels"][record] = new_label
        step_5_reference(final, corrected)
    finally:
        if args.local:
            local_down()
    return 0


if __name__ == "__main__":
    sys.exit(main())
