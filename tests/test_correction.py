"""Label-correction tests.

The contract under test: correcting a sample in a queued/running job must make
the final weights identical to training from scratch on the corrected data,
while reusing the breakpoint before the sample's first batch and discarding the
breakpoints that were computed with the old label.
"""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import lease as lm
from app import store
from app.config import Settings
from app.db import init_db
from app.features import normalise
from app.lease import LeaseLost
from app.model import dataset_loss, sgd_step
from app.worker import Worker


def make_settings(tmp_path, **overrides) -> Settings:
    base = dict(
        db_path=tmp_path / "training.db",
        lease_seconds=0.6,
        heartbeat_interval=0.1,
        poll_interval=0.01,
        batch_delay_seconds=0.0,
    )
    base.update(overrides)
    return Settings(**base)


@pytest.fixture()
def settings(tmp_path):
    s = make_settings(tmp_path)
    s.validate()
    init_db(s.db_path)
    return s


def sample(records: int = 24, seed: int = 0):
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(records, 3)).tolist()
    labels = [int(i) % 2 for i in range(records)]
    return features, labels


def reference_train(features, labels, epochs, lr=0.1, momentum=0.9, batch_size=8):
    design, targets = normalise(features, labels)
    weights = np.zeros(design.shape[1])
    velocity = np.zeros(design.shape[1])
    batches = []
    for _ in range(epochs):
        for start in range(0, design.shape[0], batch_size):
            weights, velocity, loss = sgd_step(
                weights,
                velocity,
                design[start : start + batch_size],
                targets[start : start + batch_size],
                lr,
                momentum,
            )
            batches.append((weights.copy(), velocity.copy(), loss))
    return weights, velocity, batches


def owned_job(settings, features, labels, epochs, worker="w"):
    job = store.create_job(settings.db_path, features, labels, epochs)
    lm.register_worker(settings.db_path, worker)
    lease = lm.acquire(settings.db_path, job["job_id"], worker, settings.lease_seconds)
    return job, lease


def drain(settings, job_id, lease):
    while True:
        result = store.advance_job(settings.db_path, lease)
        if result["status"] == "completed":
            return result


def assert_matches_reference(settings, job_id, features, labels, epochs):
    final = store.get_job(settings.db_path, job_id)
    weights, velocity, batches = reference_train(features, labels, epochs)
    assert final["status"] == "completed"
    assert final["completed_steps"] == final["total_steps"] == len(batches)
    assert final["final_weights"] == [float(v) for v in weights]
    assert final["momentum_velocity"] == [float(v) for v in velocity]
    assert final["labels"] == labels
    checkpoints = store.list_checkpoints(settings.db_path, job_id)
    assert [c["step"] for c in checkpoints] == list(range(1, len(batches) + 1))
    assert [c["batch_loss"] for c in checkpoints] == pytest.approx(
        [loss for _, _, loss in batches], abs=0.0
    )
    return final, checkpoints


def test_correction_before_sample_enters_training_matches_scratch(settings):
    features, labels = sample(24, seed=1)
    job, lease = owned_job(settings, features, labels, 3)
    # Record 20 is in batch 3 (0-based batch 2); commit only two steps.
    for _ in range(2):
        store.advance_job(settings.db_path, lease)
    result = store.correct_label(settings.db_path, job["job_id"], 20, labels[20], 1 - labels[20])
    assert result["first_affected_step"] == 3
    assert result["rolled_back_from_step"] == 2
    assert result["resumed_from_step"] == 2  # nothing to discard

    corrected = list(labels)
    corrected[20] = 1 - labels[20]
    # The retained breakpoints' full-dataset losses must describe the new label.
    design, targets = normalise(features, corrected)
    for checkpoint in store.list_checkpoints(settings.db_path, job["job_id"]):
        assert checkpoint["loss"] == pytest.approx(
            dataset_loss(np.asarray(checkpoint["weights"]), design, targets), abs=1e-12
        )

    drain(settings, job["job_id"], lease)
    assert_matches_reference(settings, job["job_id"], features, corrected, 3)


def test_correction_after_sample_entered_training_rolls_back(settings):
    features, labels = sample(24, seed=2)
    job, lease = owned_job(settings, features, labels, 3)
    # Record 9 is in batch 2; run through step 7 (more than one epoch).
    for _ in range(7):
        store.advance_job(settings.db_path, lease)
    before = store.list_checkpoints(settings.db_path, job["job_id"])
    result = store.correct_label(settings.db_path, job["job_id"], 9, labels[9], 1 - labels[9])
    assert result["first_affected_step"] == 2
    assert result["rolled_back_from_step"] == 7
    assert result["resumed_from_step"] == 1
    assert result["deleted_checkpoints"] == 6

    after = store.list_checkpoints(settings.db_path, job["job_id"])
    assert [c["step"] for c in after] == [1]
    # The retained breakpoint's weights, momentum and batch loss are reused...
    assert after[0]["weights"] == before[0]["weights"]
    assert after[0]["momentum_velocity"] == before[0]["momentum_velocity"]
    assert after[0]["batch_loss"] == before[0]["batch_loss"]
    # ...but its full-dataset loss now reflects the corrected label.
    corrected = list(labels)
    corrected[9] = 1 - labels[9]
    design, targets = normalise(features, corrected)
    assert after[0]["loss"] == pytest.approx(
        dataset_loss(np.asarray(after[0]["weights"]), design, targets), abs=1e-12
    )
    current = store.get_job(settings.db_path, job["job_id"])
    assert current["completed_steps"] == current["latest_checkpoint_step"] == 1
    assert current["loss"] == pytest.approx(after[0]["loss"], abs=1e-12)

    drain(settings, job["job_id"], lease)
    assert_matches_reference(settings, job["job_id"], features, corrected, 3)


def test_correction_in_a_fully_consumed_epoch_still_matches_scratch(settings):
    """Correcting after epoch 1 rewinds to the pre-sample breakpoint of epoch 1.

    Steps in later epochs saw the old label too, so every one of them must be
    discarded - not merely the current epoch's remainder.
    """
    features, labels = sample(16, seed=3)  # 2 batches/epoch
    job, lease = owned_job(settings, features, labels, 4)
    for _ in range(6):
        store.advance_job(settings.db_path, lease)
    result = store.correct_label(settings.db_path, job["job_id"], 0, labels[0], 1 - labels[0])
    assert result["first_affected_step"] == 1
    assert result["resumed_from_step"] == 0
    assert store.list_checkpoints(settings.db_path, job["job_id"]) == []
    assert store.get_job(settings.db_path, job["job_id"])["completed_steps"] == 0

    corrected = list(labels)
    corrected[0] = 1 - labels[0]
    drain(settings, job["job_id"], lease)
    assert_matches_reference(settings, job["job_id"], features, corrected, 4)


def test_two_corrections_and_reverting_a_label(settings):
    features, labels = sample(20, seed=4)
    job, lease = owned_job(settings, features, labels, 3)
    for _ in range(4):
        store.advance_job(settings.db_path, lease)
    corrected = list(labels)
    store.correct_label(settings.db_path, job["job_id"], 12, corrected[12], 1 - corrected[12])
    corrected[12] = 1 - corrected[12]
    store.correct_label(settings.db_path, job["job_id"], 12, corrected[12], labels[12])
    corrected[12] = labels[12]
    history = store.list_corrections(settings.db_path, job["job_id"])
    assert [(c["old_label"], c["new_label"]) for c in history] == [
        (labels[12], 1 - labels[12]),
        (1 - labels[12], labels[12]),
    ]

    drain(settings, job["job_id"], lease)
    assert_matches_reference(settings, job["job_id"], features, corrected, 3)


def test_correction_rejects_completed_jobs(settings):
    features, labels = sample(8, seed=5)
    job, lease = owned_job(settings, features, labels, 2)
    drain(settings, job["job_id"], lease)
    with pytest.raises(store.CorrectionRejected):
        store.correct_label(settings.db_path, job["job_id"], 0, labels[0], 1 - labels[0])
    unchanged = store.get_job(settings.db_path, job["job_id"])
    assert unchanged["status"] == "completed"
    assert unchanged["labels"] == labels


def test_correction_validation(settings):
    features, labels = sample(10, seed=6)
    job, _ = owned_job(settings, features, labels, 2)
    with pytest.raises(store.InvalidCorrection):
        store.correct_label(settings.db_path, job["job_id"], 10, labels[0], 1)
    with pytest.raises(store.InvalidCorrection):
        store.correct_label(settings.db_path, job["job_id"], -1, labels[0], 1)
    with pytest.raises(store.InvalidCorrection):
        store.correct_label(settings.db_path, job["job_id"], 0, labels[0], 2)
    with pytest.raises(store.LabelMismatch):
        store.correct_label(settings.db_path, job["job_id"], 0, 1 - labels[0], labels[0])
    with pytest.raises(store.InvalidCorrection):
        store.correct_label(settings.db_path, job["job_id"], 0, labels[0], labels[0])
    with pytest.raises(store.JobNotFound):
        store.correct_label(settings.db_path, "missing", 0, 0, 1)


def test_correction_while_a_worker_keeps_training_matches_scratch(tmp_path):
    """Correct mid-run from another connection while the worker keeps going."""
    settings = make_settings(tmp_path, batch_delay_seconds=0.02)
    init_db(settings.db_path)
    features, labels = sample(40, seed=7)
    job = store.create_job(settings.db_path, features, labels, 3)
    worker = Worker(settings, "w1")
    thread = threading.Thread(target=worker.run, daemon=True)
    thread.start()
    try:
        deadline = time.time() + 20
        while time.time() < deadline:
            current = store.get_job(settings.db_path, job["job_id"])
            if current["completed_steps"] >= 3:
                break
            time.sleep(0.005)
        # Record 13 is in batch 2, which has already been consumed more than
        # once by now; the correction rewinds and the live worker must carry on
        # with the corrected label instead of replaying its old calculation.
        result = store.correct_label(
            settings.db_path, job["job_id"], 13, labels[13], 1 - labels[13]
        )
        assert result["resumed_from_step"] < result["rolled_back_from_step"]
        deadline = time.time() + 30
        while time.time() < deadline:
            if store.get_job(settings.db_path, job["job_id"])["status"] == "completed":
                break
            time.sleep(0.01)
    finally:
        worker.request_stop()
        thread.join(timeout=10)

    corrected = list(labels)
    corrected[13] = 1 - labels[13]
    assert_matches_reference(settings, job["job_id"], features, corrected, 3)


def test_correction_then_crash_resumes_and_matches_scratch(settings):
    features, labels = sample(24, seed=8)
    job, lease = owned_job(settings, features, labels, 3)
    for _ in range(6):
        store.advance_job(settings.db_path, lease)
    store.correct_label(settings.db_path, job["job_id"], 17, labels[17], 1 - labels[17])
    # The worker dies right after the correction; its lease expires.
    lm.release(settings.db_path, lease)
    time.sleep(settings.lease_seconds + 0.05)
    resumed = lm.acquire(settings.db_path, job["job_id"], "w2", settings.lease_seconds)
    assert resumed is not None
    drain(settings, job["job_id"], resumed)

    corrected = list(labels)
    corrected[17] = 1 - labels[17]
    assert_matches_reference(settings, job["job_id"], features, corrected, 3)


def test_stale_worker_cannot_commit_old_label_after_correction(settings):
    """A worker holding a superseded lease cannot write its stale result."""
    features, labels = sample(24, seed=9)
    job, old = owned_job(settings, features, labels, 3, worker="old")
    store.advance_job(settings.db_path, old)
    time.sleep(settings.lease_seconds + 0.05)
    lm.register_worker(settings.db_path, "new")
    new = lm.acquire(settings.db_path, job["job_id"], "new", settings.lease_seconds)
    assert new is not None and new.epoch > old.epoch
    store.correct_label(settings.db_path, job["job_id"], 20, labels[20], 1 - labels[20])
    snapshot = store.get_job(settings.db_path, job["job_id"])
    with pytest.raises(LeaseLost):
        store.advance_job(settings.db_path, old)
    after = store.get_job(settings.db_path, job["job_id"])
    assert after["completed_steps"] == snapshot["completed_steps"]
    assert after["weights"] == snapshot["weights"]
    assert after["labels"][20] == 1 - labels[20]
    drain(settings, job["job_id"], new)

    corrected = list(labels)
    corrected[20] = 1 - labels[20]
    assert_matches_reference(settings, job["job_id"], features, corrected, 3)


def test_api_correction_endpoints(tmp_path, monkeypatch):
    db = tmp_path / "api.db"
    monkeypatch.setenv("TRAINING_DB_PATH", str(db))
    monkeypatch.setenv("TRAINING_LEASE_SECONDS", "0.6")
    monkeypatch.setenv("TRAINING_HEARTBEAT_INTERVAL", "0.1")

    import app.main as main_module

    monkeypatch.setattr(main_module, "settings", Settings.from_env())
    with TestClient(main_module.app) as client:
        features, labels = sample(16, seed=10)
        response = client.post(
            "/jobs", json={"features": features, "labels": labels, "epochs": 2}
        )
        assert response.status_code == 201
        job_id = response.json()["job_id"]

        corrected = client.post(
            f"/jobs/{job_id}/corrections",
            json={"record_index": 5, "old_label": labels[5], "new_label": 1 - labels[5]},
        )
        assert corrected.status_code == 200
        body = corrected.json()
        assert body["old_label"] == labels[5]
        assert body["new_label"] == 1 - labels[5]
        assert body["resumed_from_step"] == 0

        current = client.get(f"/jobs/{job_id}").json()
        assert current["labels"][5] == 1 - labels[5]
        assert current["status"] == "queued"

        history = client.get(f"/jobs/{job_id}/corrections").json()
        assert len(history) == 1 and history[0]["record_index"] == 5
        events = client.get(f"/jobs/{job_id}/events").json()
        assert any(event["event"] == "label_corrected" for event in events)

        stale = client.post(
            f"/jobs/{job_id}/corrections",
            json={"record_index": 6, "old_label": 1 - labels[6], "new_label": labels[6]},
        )
        assert stale.status_code == 409
        invalid = client.post(
            f"/jobs/{job_id}/corrections",
            json={"record_index": 99, "old_label": 0, "new_label": 1},
        )
        assert invalid.status_code == 422
        assert client.post(
            "/jobs/missing/corrections",
            json={"record_index": 0, "old_label": 0, "new_label": 1},
        ).status_code == 404

        schema = client.get("/openapi.json").json()
        assert "/jobs/{job_id}/corrections" in schema["paths"]


def test_correction_does_not_touch_other_jobs(settings):
    features_a, labels_a = sample(16, seed=11)
    features_b, labels_b = sample(16, seed=12)
    job_a, lease_a = owned_job(settings, features_a, labels_a, 2, worker="wa")
    job_b, lease_b = owned_job(settings, features_b, labels_b, 2, worker="wb")
    store.advance_job(settings.db_path, lease_a)
    store.advance_job(settings.db_path, lease_b)
    before_b = store.get_job(settings.db_path, job_b["job_id"])

    store.correct_label(settings.db_path, job_a["job_id"], 0, labels_a[0], 1 - labels_a[0])
    after_b = store.get_job(settings.db_path, job_b["job_id"])
    assert after_b == before_b

    drain(settings, job_b["job_id"], lease_b)
    assert_matches_reference(settings, job_b["job_id"], features_b, labels_b, 2)


def test_interleaved_commits_and_corrections_never_keep_old_label_effect(settings):
    """Race batch commits against corrections from several threads.

    Whatever the interleaving, every committed step after the retained
    breakpoint must be reproducible with the final (corrected) labels, and no
    stale old-label computation may survive as a later checkpoint.
    """
    features, labels = sample(24, seed=13)
    job, lease = owned_job(settings, features, labels, 3)
    target_index = 3  # first batch
    corrected = list(labels)
    corrected[target_index] = 1 - corrected[target_index]
    errors: list[BaseException] = []
    stop = threading.Event()

    def train() -> None:
        try:
            while not stop.is_set():
                store.advance_job(settings.db_path, lease)
                time.sleep(0.002)
        except BaseException as exc:  # noqa: BLE001 - collected for the assertion
            errors.append(exc)

    def corrector() -> None:
        try:
            for _ in range(4):
                current = store.get_job(settings.db_path, job["job_id"])["labels"][target_index]
                store.correct_label(
                    settings.db_path, job["job_id"], target_index, current, 1 - current
                )
                time.sleep(0.004)
        except store.CorrectionRejected:
            pass
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    trainer = threading.Thread(target=train)
    correctors = [threading.Thread(target=corrector) for _ in range(2)]
    trainer.start()
    for thread in correctors:
        thread.start()
    for thread in correctors:
        thread.join()
    stop.set()
    trainer.join()

    assert not errors
    # The final label is whatever the last correction wrote; finish or rewind
    # to a clean state and replay the remainder under that label.
    final_label = store.get_job(settings.db_path, job["job_id"])["labels"][target_index]
    corrected = list(labels)
    corrected[target_index] = final_label
    if store.get_job(settings.db_path, job["job_id"])["status"] != "completed":
        # The original worker still owns the lease; let it finish from the
        # consistent prefix the race left behind.
        drain(settings, job["job_id"], lease)
    final, checkpoints = assert_matches_reference(
        settings, job["job_id"], features, corrected, 3
    )
    assert final["labels"] == corrected
    # No checkpoint may carry a full-dataset loss measured against the old label.
    design, targets = normalise(features, corrected)
    for checkpoint in checkpoints:
        assert checkpoint["loss"] == pytest.approx(
            dataset_loss(np.asarray(checkpoint["weights"]), design, targets), abs=1e-12
        )
