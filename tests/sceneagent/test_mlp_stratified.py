"""Stratified sampling, masked training step, and the section 3.4 vote.

No Gemini calls and no downloaded meshes. Tensors in the training test are fake.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from sceneagent.mlp.apply import vote_loaded_prior
from sceneagent.mlp.prior import PhysicsPrior
from sceneagent.mlp.stratum import (
    guess_material,
    is_near_duplicate,
    labels_should_stop,
    order_bucket,
)
from sceneagent.mlp.train_prior import (
    CHECKPOINT,
    SPLAT_CHECKPOINT,
    _Adam,
    _fit,
    _predict_row,
    _validation_count,
    holdout_ids,
    loss_and_gradients,
    train_and_report,
    train_fitted_and_report,
    validation_and_test_ids,
)
from sceneagent.taxonomy import MATERIAL_CLASSES


def test_guess_and_near_duplicate():
    assert guess_material("tomato soup can") == "metal"
    assert guess_material("cracker box") == "paper and cardboard"
    assert guess_material("throw pillow") == "fabric"
    assert guess_material("candle") == "unknown"
    assert is_near_duplicate("Blue Mug", ["Red Mug"])
    assert not is_near_duplicate("tomato soup can", ["master chef can"])


def test_order_bucket_keeps_a_small_and_a_large_object():
    names = {
        "plastic": ("bottle", "helmet", "crate", "bin"),
        "wood": ("table", "desk", "shelf", "cabinet"),
    }
    items = []
    for material in ("plastic", "wood"):
        for index, extent in enumerate((0.05, 0.12, 0.4, 1.5)):
            items.append({
                "id": f"{material}-{index}",
                "name": f"{material} {names[material][index]}",
                "material": material,
                "extent": extent,
            })
    picked = []
    for material in ("plastic", "wood"):
        group = [row for row in items if row["material"] == material]
        picked.extend(order_bucket(group, limit=4))
    for material in ("plastic", "wood"):
        extents = [row["extent"] for row in picked if row["material"] == material]
        assert min(extents) == pytest.approx(0.05)
        assert max(extents) == pytest.approx(1.5)
        assert len(extents) == 4


def test_labels_stop_when_unknown_is_common_or_a_source_collapses():
    mixed = [
        {"status": "ok", "source": "ycb", "name": f"object {index}", "material": MATERIAL_CLASSES[index % 5]}
        for index in range(20)
    ]
    assert labels_should_stop(mixed) is None
    unknown = [
        {"status": "ok", "source": "gso", "name": f"object {index}", "material": "unknown" if index < 8 else "plastic"}
        for index in range(20)
    ]
    assert "unknown" in labels_should_stop(unknown)
    collapsed = [
        {"status": "ok", "source": "abo", "name": f"chair {index}", "material": "plastic"}
        for index in range(12)
    ]
    assert "abo" in labels_should_stop(collapsed)


def test_identity_holdout_includes_the_three_known_objects():
    ids = [
        "002_master_chef_can",
        "Weisshai_Great_White_Shark",
        "B079TXJCR5",
        "003_cracker_box",
        "other_object",
    ]
    held = holdout_ids(ids, rate=0.0)
    assert {"002_master_chef_can", "Weisshai_Great_White_Shark", "B079TXJCR5"} <= held
    train = [obj_id for obj_id in ids if obj_id not in held]
    assert set(train).isdisjoint(held)
    assert "003_cracker_box" in train
    validation, test = validation_and_test_ids(held)
    assert len(validation) == _validation_count(len(held)) == 1
    assert set(validation) | set(test) == set(held)
    assert set(validation).isdisjoint(test)
    assert set(train).isdisjoint(validation)
    assert set(train).isdisjoint(test)
    assert "003_cracker_box" not in validation
    assert "003_cracker_box" not in test


def test_validation_is_thirty_percent_of_the_holdout_in_sha256_order():
    held = {f"id-{index:04d}" for index in range(50)}
    held.update(("002_master_chef_can", "Weisshai_Great_White_Shark", "B079TXJCR5"))
    validation, test = validation_and_test_ids(held)
    ordered = sorted(
        held,
        key=lambda obj_id: (hashlib.sha256(obj_id.encode("utf-8")).digest(), obj_id),
    )
    # 53 * 0.3 = 15.9, half-up to 16. Python round-half-even is not used.
    assert _validation_count(len(ordered)) == 16
    assert _validation_count(15) == 5
    assert _validation_count(5) == 2
    assert _validation_count(1) == 0
    assert _validation_count(462) == 139
    assert validation == ordered[:16]
    assert test == ordered[16:]
    assert validation != sorted(held)[:16]
    assert set(validation).isdisjoint(test)
    again_validation, again_test = validation_and_test_ids(list(reversed(tuple(sorted(held)))))
    assert again_validation == validation
    assert again_test == test
    absent_validation, absent_test = validation_and_test_ids(["only-object"])
    assert absent_validation == []
    assert absent_test == ["only-object"]
    assert "002_master_chef_can" not in absent_validation
    assert "002_master_chef_can" not in absent_test


def test_saved_462_holdout_uses_139_validation_ids_not_the_first_30():
    """The backed-up 462-id holdout is split in sha256 order, half up.

    3 * 462 = 1386, and 1386 / 10 leaves remainder 6, so half-up yields 139
    validation ids and 323 test ids. The first 30 lexicographic ids are not
    the validation slice. Train membership stays the complement: 3736 - 462.
    """

    report_path = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "mlp"
        / "train_report_splat_holdout462.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    held = report["holdout_ids"]
    assert report["n_holdout"] == len(held) == len(set(held)) == 462
    assert report["n_fitted"] - 462 == report["n_train"] == 3274
    validation, test = validation_and_test_ids(held)
    assert _validation_count(462) == 139
    assert len(validation) == 139
    assert len(test) == 323
    assert validation[0] == "B07F3T4RSD"
    assert validation[-1] == "B07BMTNH22"
    assert test[0] == "B071J4SDZZ"
    lexical = sorted(set(held))
    assert validation != lexical[:30]
    assert validation != lexical[:139]
    assert set(validation).isdisjoint(test)
    assert set(validation) | set(test) == set(held)
    for known in ("002_master_chef_can", "Weisshai_Great_White_Shark", "B079TXJCR5"):
        assert known in report["known_ids_absent"]
        assert known not in held
        assert known not in validation
        assert known not in test


def test_train_ids_stay_out_of_the_validation_and_test_slices():
    ids = [f"obj-{index:04d}" for index in range(400)]
    ids.extend(("002_master_chef_can", "Weisshai_Great_White_Shark", "B079TXJCR5"))
    held = holdout_ids(list(reversed(ids)))
    validation, test = validation_and_test_ids(held)
    train = [obj_id for obj_id in ids if obj_id not in held]
    assert len(validation) == _validation_count(len(held))
    assert len(validation) + len(test) == len(held)
    assert len(train) == len(ids) - len(held)
    assert set(train).isdisjoint(validation)
    assert set(train).isdisjoint(test)
    assert set(validation) | set(test) == set(held)
    for known in ("002_master_chef_can", "Weisshai_Great_White_Shark", "B079TXJCR5"):
        assert known in validation or known in test
        assert known not in train


def _zero_grads(prior):
    return {
        name: np.zeros_like(getattr(prior, name))
        for name in ("w1", "b1", "w2", "b2", "w_out", "b_out")
    }


def test_validation_loss_selects_the_checkpoint_and_can_stop(tmp_path, monkeypatch):
    prior = PhysicsPrior.initialize(0)
    train_rows = [{"id": "train-a", "prior_input": "unused"}]
    val_rows = [{"id": "val-a", "prior_input": "unused"}]
    seen = []
    marker = {"n": 0}

    def fake_features(row):
        seen.append(row["id"])
        return np.zeros((2, 912), dtype=np.float64)

    def fake_loss_and_gradients(model, features, label):
        seen.append(label["id"])
        marker["n"] += 1
        model.w1[0, 0] = float(marker["n"])
        return 1.0, _zero_grads(model)

    schedule = [8, 7, 6, 5, 4, 3, 2, 1, 1.5, 1.5, 1.5, 1.5, 9]
    snapshots = []

    def fake_score(model, rows):
        assert [row["id"] for row in rows] == ["val-a"]
        snapshots.append(np.array(model.w1, copy=True))
        return {"loss": schedule[len(snapshots) - 1], "material_accuracy": 0.0}

    monkeypatch.setattr("sceneagent.mlp.train_prior._features", fake_features)
    monkeypatch.setattr("sceneagent.mlp.train_prior.loss_and_gradients", fake_loss_and_gradients)
    monkeypatch.setattr("sceneagent.mlp.train_prior.score_split", fake_score)
    checkpoint = tmp_path / "physics_prior_splat.npz"
    fit = _fit(
        prior,
        train_rows,
        epochs=16,
        lr=1e-3,
        seed=0,
        checkpoint=checkpoint,
        val_rows=val_rows,
    )
    assert fit["checkpoint_epoch"] == 8
    assert len(fit["train_loss"]) == 12
    assert fit["val_loss"] == schedule[:12]
    assert set(seen) == {"train-a"}
    assert len(snapshots) == 12
    assert not np.array_equal(snapshots[7], snapshots[-1])
    assert np.array_equal(prior.w1, snapshots[7])
    loaded = PhysicsPrior.load(checkpoint)
    assert np.array_equal(loaded.w1, snapshots[7])


def test_triangle_fit_keeps_the_last_epoch_and_stops_on_train_loss(tmp_path, monkeypatch):
    prior = PhysicsPrior.initialize(0)
    train_rows = [{"id": "train-a", "prior_input": "unused"}]
    marker = {"n": 0}
    scheduled = iter([5, 4, 3, 2, 1, 0.5, 0.4, 0.3, 0.9, 0.9, 0.9, 0.9, 0.1])

    def fake_features(row):
        return np.zeros((2, 912), dtype=np.float64)

    def fake_loss_and_gradients(model, features, label):
        marker["n"] += 1
        model.w1[0, 0] = float(marker["n"])
        return next(scheduled), _zero_grads(model)

    def fake_score(model, rows):
        raise AssertionError("triangle fit must not score a validation split")

    monkeypatch.setattr("sceneagent.mlp.train_prior._features", fake_features)
    monkeypatch.setattr("sceneagent.mlp.train_prior.loss_and_gradients", fake_loss_and_gradients)
    monkeypatch.setattr("sceneagent.mlp.train_prior.score_split", fake_score)
    fit = _fit(
        prior,
        train_rows,
        epochs=16,
        lr=1e-3,
        seed=0,
        checkpoint=tmp_path / "physics_prior.npz",
    )
    assert fit["val_loss"] is None
    assert fit["checkpoint_epoch"] == 12
    assert len(fit["train_loss"]) == 12
    assert marker["n"] == 12
    assert prior.w1[0, 0] > 11.0
    loaded = PhysicsPrior.load(tmp_path / "physics_prior.npz")
    assert loaded.w1[0, 0] > 11.0


def test_training_step_ignores_slots_28_through_32_and_reduces_loss():
    prior = PhysicsPrior.initialize(0)
    features = np.random.default_rng(0).normal(scale=0.02, size=(6, 912))
    label = {
        "material": "plastic",
        "body_type": "dynamic rigid",
        "collision": "box",
        "log_mass": -0.4,
        "log_density": 6.1,
        "rigidity": 0.8,
        "deformability": 0.2,
    }
    before, grads = loss_and_gradients(prior, features, label)
    assert np.all(grads["b_out"][28:] == 0.0)
    assert np.all(grads["w_out"][:, 28:] == 0.0)
    slot = 24
    epsilon = 1e-5
    prior.b_out[slot] += epsilon
    plus, _plus_grads = loss_and_gradients(prior, features, label)
    prior.b_out[slot] -= 2 * epsilon
    minus, _minus_grads = loss_and_gradients(prior, features, label)
    prior.b_out[slot] += epsilon
    numerical = (plus - minus) / (2 * epsilon)
    assert grads["b_out"][slot] == pytest.approx(numerical, rel=1e-4, abs=1e-5)
    optimizer = _Adam(
        [prior.w1, prior.b1, prior.w2, prior.b2, prior.w_out, prior.b_out],
        lr=0.05,
        decay=0.0,
    )
    optimizer.step([grads[name] for name in ("w1", "b1", "w2", "b2", "w_out", "b_out")])
    after, _after_grads = loss_and_gradients(prior, features, label)
    assert after < before


def test_vote_writes_log_mass_once_and_scales_unknown(tmp_path):
    prior = PhysicsPrior.initialize(1)
    prior.b1[:] = 0.0
    prior.b2[:] = 0.0
    prior.b_out[:] = 0.0
    prior.b_out[0] = 6.0
    prior.b_out[24] = -0.7
    prior.b_out[25] = 6.2
    prior.b_out[26] = 2.0
    prior.b_out[27] = -2.0
    path = tmp_path / "physics_prior.npz"
    prior.save(path)
    features = np.zeros((3, 912))
    mask = np.array([True, False, True])
    voted = vote_loaded_prior(
        features,
        mask,
        vlm={"source": "vlm", "class": "plastic", "weight": 1.0},
        keyword={"source": "keyword", "class": "plastic", "weight": 1.0},
        checkpoint=path,
        name="can",
        prior=PhysicsPrior.load(path),
    )
    assert voted["proposal"]["source"] == "physics_prior"
    assert voted["proposal"]["class"] == "plastic"
    assert voted["record"]["mass_kg"] is None
    assert voted["record"]["density_kg_m3"] is None
    assert voted["record"]["log_mass"] == pytest.approx(-0.7)
    assert voted["record"]["log_density"] == pytest.approx(6.2)
    assert voted["winner"] == "plastic"

    prior.b_out[:] = 0.0
    prior.b_out[9] = 8.0
    prior.b_out[24] = 1.5
    prior.b_out[25] = 2.5
    unknown = vote_loaded_prior(
        features,
        mask,
        vlm={"source": "vlm", "class": "plastic", "weight": 0.3},
        keyword={"source": "keyword", "class": "unknown", "weight": 1.0},
        checkpoint=path,
        name="mystery",
        prior=prior,
    )
    assert unknown["proposal"]["source"] == "physics_prior"
    assert unknown["proposal"]["class"] == "unknown"
    assert unknown["winner"] == "unknown"
    assert unknown["scores"]["unknown"] == pytest.approx(0.4 * unknown["proposal"]["weight"] + 0.4)
    assert unknown["record"]["log_mass"] == pytest.approx(1.5)
    assert unknown["record"]["mass_kg"] is None


def test_fitted_vote_names_the_splat_checkpoint_not_the_triangle_file():
    import ast
    import inspect

    def checkpoint_name(function):
        tree = ast.parse(inspect.getsource(function))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "_predict_row"
        ]
        assert len(calls) == 1
        assert len(calls[0].args) == 3
        assert isinstance(calls[0].args[2], ast.Name)
        return calls[0].args[2].id

    assert checkpoint_name(train_fitted_and_report) == "SPLAT_CHECKPOINT"
    assert checkpoint_name(train_and_report) == "CHECKPOINT"


def test_predict_row_forwards_the_splat_checkpoint(tmp_path, monkeypatch):
    features = tmp_path / "row.npy"
    np.save(features, np.zeros((2, 912), dtype=np.float64))
    seen = {}

    def fake_vote(array, mask, *, vlm, keyword, checkpoint, name, prior=None):
        seen["checkpoint"] = checkpoint
        return {
            "proposal": {"source": "physics_prior", "class": "plastic", "weight": 0.5},
            "decoded": {
                "material": "plastic",
                "body_type": "rigid",
                "collision": "box",
                "log_mass": -1.0,
                "log_density": 6.0,
                "rigidity": 0.9,
                "deformability": 0.1,
            },
            "winner": "plastic",
            "scores": {"plastic": 1.5},
            "record": {"body_type": "rigid"},
        }

    monkeypatch.setattr("sceneagent.mlp.train_prior.vote_loaded_prior", fake_vote)
    row = {
        "id": "obj",
        "source": "abo",
        "name": "cup",
        "material": "plastic",
        "body_type": "rigid",
        "collision": "box",
        "prior_input": str(features),
    }
    predicted = _predict_row(object(), row, SPLAT_CHECKPOINT)
    assert seen["checkpoint"] == SPLAT_CHECKPOINT
    assert seen["checkpoint"] != CHECKPOINT
    assert seen["checkpoint"].name == "physics_prior_splat.npz"
    assert predicted["proposal"]["source"] == "physics_prior"
    assert predicted["vote_winner"] == "plastic"
