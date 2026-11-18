"""Process-level label-correction test: real API and two worker processes.

Flow:
1. submit a job and let worker A train past the batch that contains the sample;
2. correct the label through the HTTP API (the same call Swagger makes);
3. kill worker A; worker B takes over and resumes from the retained breakpoint;
4. compare the final weights/loss/momentum with an uninterrupted from-scratch
   run on the corrected labels.

Also checks that a correction arrives while worker A is mid-run and that the
job's progress, breakpoints and losses stay mutually consistent.
"""
from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
LEASE_SECONDS = 2.0
HEARTBEAT_INTERVAL = 0.3
BATCH_DELAY = 0.25


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def env_for(db: Path, **extra: str) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "TRAINING_DB_PATH": str(db),
            "TRAINING_LEASE_SECONDS": str(LEASE_SECONDS),
            "TRAINING_HEARTBEAT_INTERVAL": str(HEARTBEAT_INTERVAL),
            "TRAINING_POLL_INTERVAL": "0.05",
            "TRAINING_BATCH_DELAY_SECONDS": str(BATCH_DELAY),
            "PYTHONUNBUFFERED": "1",
        }
    )
    env.update(extra)
    return env


class Server:
    def __init__(self, db: Path, port: int) -> None:
        self.db = db
        self.port = port
        self.proc: subprocess.Popen | None = None

    def __enter__(self) -> "Server":
        self.proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "app.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(self.port),
                "--log-level",
                "warning",
            ],
            cwd=ROOT,
            env=env_for(self.db),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        deadline = time.time() + 30
        while time.time() < deadline:
            try:
                self.get("/health")
                return self
            except OSError:
                time.sleep(0.1)
        raise RuntimeError("API did not start")

    def __exit__(self, *exc) -> None:
        if self.proc is not None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=5)

    def get(self, path: str):
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}{path}", timeout=5) as response:
            return json.loads(response.read())

    def post(self, path: str, body: dict):
        data = json.dumps(body).encode()
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())


def start_worker(db: Path, worker_id: str, **env: str) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-m", "app.worker", "--worker-id", worker_id],
        cwd=ROOT,
        env=env_for(db, **env),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def wait_for(predicate, timeout: float, interval: float = 0.05, message: str = "condition"):
    deadline = time.time() + timeout
    while time.time() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(interval)
    raise AssertionError(f"timed out waiting for {message}")


def wait_completed(server: Server, job_id: str, timeout: float = 120):
    return wait_for(
        lambda: (state := server.get(f"/jobs/{job_id}"))["status"] == "completed" and state,
        timeout=timeout,
        message="job completion",
    )


def stop_worker(proc: subprocess.Popen, hard: bool = False) -> None:
    if proc.poll() is not None:
        return
    proc.kill() if hard else proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def dataset(records: int = 40, seed: int = 31):
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(records, 4)).tolist()
    labels = [1 if row[0] + row[1] > 0 else 0 for row in features]
    return {"features": features, "labels": labels, "epochs": 3}, features, labels


def correct(server: Server, job_id: str, index: int, old: int, new: int):
    status, body = server.post(
        f"/jobs/{job_id}/corrections",
        {"record_index": index, "old_label": old, "new_label": new},
    )
    assert status == 200, body
    return body


def assert_consistent(server: Server, job_id: str) -> None:
    """Progress, latest checkpoint and losses must agree at a quiescent point."""
    job = server.get(f"/jobs/{job_id}")
    checkpoints = server.get(f"/jobs/{job_id}/checkpoints")
    if job["completed_steps"]:
        latest = checkpoints[-1]
        assert latest["step"] == job["completed_steps"] == job["latest_checkpoint_step"]
        assert latest["weights"] == job["weights"]
        assert latest["momentum_velocity"] == job["momentum_velocity"]
        assert latest["loss"] == pytest.approx(job["loss"], abs=1e-12)
    assert job["loss"] == pytest.approx(job["evaluation_loss"], abs=1e-9)


def reference(server_factory, data: dict) -> dict:
    db, port = server_factory()
    with Server(db, port) as server:
        _, job = server.post("/jobs", data)
        worker = start_worker(db, "solo")
        try:
            return wait_completed(server, job["job_id"])
        finally:
            stop_worker(worker)


def test_correction_survives_takeover_and_matches_scratch(tmp_path, tmp_path_factory):
    data, _, labels = dataset()
    # Correct sample 9: it belongs to batch 2, so anything after epoch-1 step 1
    # was computed with the old label.
    index = 9
    corrected = list(labels)
    corrected[index] = 1 - corrected[index]
    corrected_data = dict(data, labels=corrected)

    expected = reference(lambda: (tmp_path_factory.mktemp("ref") / "t.db", free_port()), corrected_data)

    db = tmp_path / "training.db"
    with Server(db, free_port()) as server:
        _, job = server.post("/jobs", data)
        job_id = job["job_id"]
        worker_a = start_worker(db, "worker-a")
        try:
            wait_for(
                lambda: server.get(f"/jobs/{job_id}")["completed_steps"] >= 4,
                timeout=60,
                message="worker A progress",
            )
            before = server.get(f"/jobs/{job_id}")
            result = correct(server, job_id, index, labels[index], corrected[index])
            assert result["first_affected_step"] == 2
            assert result["rolled_back_from_step"] == before["completed_steps"]
            assert result["resumed_from_step"] == 1

            rewound = server.get(f"/jobs/{job_id}")
            assert rewound["completed_steps"] == 1
            assert rewound["labels"][index] == corrected[index]
            steps = [c["step"] for c in server.get(f"/jobs/{job_id}/checkpoints")]
            assert steps == [1]
            assert_consistent(server, job_id)
        finally:
            stop_worker(worker_a, hard=True)

        # worker B takes over after the lease expires and replays with the
        # corrected label.
        worker_b = start_worker(db, "worker-b")
        try:
            final = wait_completed(server, job_id)
        finally:
            stop_worker(worker_b)

        assert_consistent(server, job_id)
        assert final["labels"] == corrected
        assert final["final_weights"] == expected["final_weights"]
        assert final["momentum_velocity"] == expected["momentum_velocity"]
        assert final["loss"] == expected["loss"]

        checkpoints = server.get(f"/jobs/{job_id}/checkpoints")
        assert [c["step"] for c in checkpoints] == list(range(1, final["total_steps"] + 1))
        assert {c["worker_id"] for c in checkpoints} == {"worker-a", "worker-b"}
        assert max(c["step"] for c in checkpoints if c["worker_id"] == "worker-a") == 1

        events = server.get(f"/jobs/{job_id}/events")
        corrected_events = [e for e in events if e["event"] == "label_corrected"]
        assert len(corrected_events) == 1
        assert corrected_events[0]["detail"]["resumed_from_step"] == 1
        history = server.get(f"/jobs/{job_id}/corrections")
        assert len(history) == 1 and history[0]["new_label"] == corrected[index]


def test_correction_is_rejected_after_completion(tmp_path):
    data, _, labels = dataset(records=16, seed=41)
    db = tmp_path / "training.db"
    with Server(db, free_port()) as server:
        _, job = server.post("/jobs", data)
        worker = start_worker(db, "solo")
        try:
            final = wait_completed(server, job["job_id"])
        finally:
            stop_worker(worker)
        status, body = server.post(
            f"/jobs/{job['job_id']}/corrections",
            {"record_index": 0, "old_label": labels[0], "new_label": 1 - labels[0]},
        )
        assert status == 409
        assert "completed" in body["detail"]
        after = server.get(f"/jobs/{job['job_id']}")
        assert after["final_weights"] == final["final_weights"]
        assert after["status"] == "completed"


def test_other_jobs_keep_training_normally(tmp_path):
    data_a, _, labels_a = dataset(records=24, seed=51)
    data_b, _, labels_b = dataset(records=24, seed=52)
    expected = reference(lambda: (tmp_path / "ref.db", free_port()), data_b)
    db = tmp_path / "training.db"
    with Server(db, free_port()) as server:
        _, job_a = server.post("/jobs", data_a)
        _, job_b = server.post("/jobs", data_b)
        workers = [start_worker(db, f"w{i}") for i in (1, 2)]
        try:
            wait_for(
                lambda: server.get(f"/jobs/{job_a['job_id']}")["completed_steps"] >= 2,
                timeout=60,
                message="job A progress",
            )
            correct(server, job_a["job_id"], 7, labels_a[7], 1 - labels_a[7])
            final_a = wait_completed(server, job_a["job_id"])
            final_b = wait_completed(server, job_b["job_id"])
        finally:
            for worker in workers:
                stop_worker(worker)
    assert final_a["status"] == "completed" and final_a["completed_steps"] == final_a["total_steps"]
    assert final_b["final_weights"] == expected["final_weights"]
    assert final_b["loss"] == expected["loss"]
    assert final_b["labels"] == labels_b
