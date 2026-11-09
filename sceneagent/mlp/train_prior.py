"""Train the width-32 physics prior on VLM labels.

The training table is the pilot labels plus the stratified labels. Each row
is one object. The MLP runs on every in-mask Gaussian, the 33 outputs are
averaged, and the loss is applied to that mean. Whole objects are held out,
including the can, the shark, and the pillow. The fitted-splat trainer
then splits that holdout into validation and test. The saved npz is our
checkpoint for Harvard section 3.4. It is not Harvard's prior.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

from sceneagent.mlp.apply import vote_loaded_prior
from sceneagent.mlp.prior import (
    OUTPUT_DIM,
    SLOT_BODY,
    SLOT_COLLISION,
    SLOT_DEFORMABILITY,
    SLOT_LOG_DENSITY,
    SLOT_LOG_MASS,
    SLOT_MATERIAL,
    SLOT_RIGIDITY,
    PhysicsPrior,
    masked_prior_loss,
)
from sceneagent.mlp.stratum import labels_should_stop
from sceneagent.taxonomy import (
    BODY_TYPES,
    COLLISION_PRIMITIVES,
    MATERIAL_CLASSES,
    keyword_material,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "data" / "mlp"
PILOT_LABELS = ROOT / "pilot_labels.jsonl"
STRATIFIED_LABELS = ROOT / "stratified_labels.jsonl"
CHECKPOINT = ROOT / "physics_prior.npz"
REPORT_PATH = ROOT / "train_report.json"
# Fitted-splat run. This file is not the epoch-2 triangle checkpoint.
SPLAT_CHECKPOINT = ROOT / "physics_prior_splat.npz"
SPLAT_REPORT = ROOT / "train_report_splat.json"
FITTED_MARKER_NAME = "fitted_splat.json"
KNOWN_IDS = (
    "002_master_chef_can",
    "Weisshai_Great_White_Shark",
    "B079TXJCR5",
)
VLM_WEIGHT = 1.0


_CHUNK = 4096


def loss_and_gradients(prior, features, label):
    """Loss of the in-mask mean, and gradients of the six weight arrays.

    Slots 28–32 have zero gradient. The numeric heads match
    ``masked_prior_loss``: squared error on the log slots, and squared error
    between the sigmoid outputs and the rigidity and deformability targets.
    Rows are applied in chunks so a 40k-Gaussian mesh is not one allocation.
    The chunked mean and the summed gradient match the full-array formulas.
    """

    if getattr(features, "ndim", None) != 2 or features.shape[1] != 912 or features.shape[0] < 1:
        raise ValueError(f"training row must be (N, 912), got {getattr(features, 'shape', None)}")
    count = int(features.shape[0])
    total = np.zeros(OUTPUT_DIM, dtype=np.float64)
    for start in range(0, count, _CHUNK):
        chunk = np.asarray(features[start:start + _CHUNK], dtype=np.float64)
        if chunk.ndim != 2 or chunk.shape[1] != 912 or not np.isfinite(chunk).all():
            raise ValueError("training features are not finite")
        hidden_1 = np.tanh(chunk @ prior.w1 + prior.b1)
        hidden_2 = np.tanh(hidden_1 @ prior.w2 + prior.b2)
        total += (hidden_2 @ prior.w_out + prior.b_out).sum(axis=0)
    mean = total / count
    loss = masked_prior_loss(
        mean,
        material=label["material"],
        body=label["body_type"],
        collision=label["collision"],
        log_mass=label["log_mass"],
        log_density=label["log_density"],
        rigidity=label["rigidity"],
        deformability=label["deformability"],
    )
    grad_mean = _mean_gradient(mean, label)
    per_row_grad = grad_mean / count
    grad_w1 = np.zeros_like(prior.w1)
    grad_b1 = np.zeros_like(prior.b1)
    grad_w2 = np.zeros_like(prior.w2)
    grad_b2 = np.zeros_like(prior.b2)
    grad_w_out = np.zeros_like(prior.w_out)
    grad_b_out = np.zeros_like(prior.b_out)
    for start in range(0, count, _CHUNK):
        chunk = np.asarray(features[start:start + _CHUNK], dtype=np.float64)
        hidden_1 = np.tanh(chunk @ prior.w1 + prior.b1)
        hidden_2 = np.tanh(hidden_1 @ prior.w2 + prior.b2)
        per_row = np.broadcast_to(per_row_grad, (chunk.shape[0], OUTPUT_DIM)).copy()
        grad_w_out += hidden_2.T @ per_row
        grad_b_out += per_row.sum(axis=0)
        grad_hidden_2 = per_row @ prior.w_out.T
        grad_z2 = grad_hidden_2 * (1.0 - hidden_2 * hidden_2)
        grad_w2 += hidden_1.T @ grad_z2
        grad_b2 += grad_z2.sum(axis=0)
        grad_hidden_1 = grad_z2 @ prior.w2.T
        grad_z1 = grad_hidden_1 * (1.0 - hidden_1 * hidden_1)
        grad_w1 += chunk.T @ grad_z1
        grad_b1 += grad_z1.sum(axis=0)
    if not np.isfinite(grad_w_out[:, 28:]).all() or np.any(grad_w_out[:, 28:]) or np.any(grad_b_out[28:]):
        raise ValueError("slots 28-32 received a gradient")
    grads = {
        "w1": grad_w1,
        "b1": grad_b1,
        "w2": grad_w2,
        "b2": grad_b2,
        "w_out": grad_w_out,
        "b_out": grad_b_out,
    }
    return float(loss), grads


def holdout_ids(ids, rate=0.12):
    """Hold out whole objects. The three known ids are always in the holdout.

    The extra holdout is a stable hash of the object id, so the split does
    not depend on row order. An id is never in both the train set and this
    holdout. The fitted trainer splits this pool into validation and test.
    The triangle trainer still scores the whole holdout.
    """

    present = list(dict.fromkeys(ids))
    held = {obj_id for obj_id in present if obj_id in KNOWN_IDS}
    rest = [obj_id for obj_id in present if obj_id not in held]
    for obj_id in rest:
        digest = hashlib.sha256(obj_id.encode("utf-8")).digest()
        if digest[0] / 256.0 < rate:
            held.add(obj_id)
    if len(held) >= len(present) and rest:
        held.discard(rest[0])
    if not held.issuperset(set(KNOWN_IDS) & set(present)):
        raise RuntimeError("known objects were not held out")
    return held


def _validation_count(n_holdout):
    """How many holdout ids become validation: 30%, rounded half up.

    Half-up, not Python's round-half-even. ``(3 * n) / 10`` uses integers
    so a binary ``0.3 * n`` cannot fall on the wrong side of a tie. A
    remainder of 5 or more rounds away from zero (15 ids become 5, not 4).
    A remainder below 5 rounds down.
    """

    if n_holdout < 0:
        raise ValueError(f"holdout count is negative: {n_holdout}")
    quotient, remainder = divmod(3 * int(n_holdout), 10)
    if remainder >= 5:
        return quotient + 1
    return quotient


def validation_and_test_ids(held_ids):
    """Split one existing holdout into validation and test.

    Sort by sha256(id) bytes, then by the id string. The first 30%, rounded
    half up, is validation. The rest is test. There is no shuffle. Train ids
    are not an argument. An id absent from ``held_ids`` is not added,
    including the can, the shark, and the pillow.
    """

    ordered = sorted(
        list(dict.fromkeys(held_ids)),
        key=lambda obj_id: (hashlib.sha256(obj_id.encode("utf-8")).digest(), obj_id),
    )
    n_validation = _validation_count(len(ordered))
    return ordered[:n_validation], ordered[n_validation:]


def train_and_report(epochs=30, lr=1e-3, seed=0):
    """Fit the prior, save the npz, and write ``train_report.json``."""

    rows = load_training_rows()
    reason = labels_should_stop(rows)
    if reason:
        raise SystemExit(f"stopped before training: {reason}")
    by_source = _count_sources(rows)
    if any(by_source.get(source, 0) < 20 for source in ("ycb", "gso", "abo")):
        raise SystemExit(f"stopped before training: a source has fewer than 20 labels ({by_source})")
    held = holdout_ids([row["id"] for row in rows])
    train_rows = [row for row in rows if row["id"] not in held]
    hold_rows = [row for row in rows if row["id"] in held]
    if len(train_rows) < 10 or len(hold_rows) < 3:
        raise SystemExit(
            f"stopped before training: train {len(train_rows)} holdout {len(hold_rows)}"
        )
    prior = PhysicsPrior.initialize(seed)
    fit = _fit(prior, train_rows, epochs=epochs, lr=lr, seed=seed)
    history = fit["train_loss"]
    prior.save(CHECKPOINT)
    loaded = PhysicsPrior.load(CHECKPOINT)
    train_score = score_split(loaded, train_rows)
    hold_score = score_split(loaded, hold_rows)
    predictions = [_predict_row(loaded, row, CHECKPOINT) for row in hold_rows]
    known = [row for row in predictions if row["id"] in KNOWN_IDS]
    known.sort(key=lambda row: KNOWN_IDS.index(row["id"]))
    report = {
        "checkpoint": str(CHECKPOINT),
        "weights": "ours",
        "section": "3.4",
        "n_train": len(train_rows),
        "n_holdout": len(hold_rows),
        "sources": by_source,
        "epochs_ran": len(history),
        "train_loss_last": history[-1] if history else None,
        "train": train_score,
        "holdout": hold_score,
        "holdout_predictions": predictions,
        "known": known,
        "w_out_shape": [32, 33],
        "n_materials": 10,
        "n_body": 6,
        "n_collision": 8,
        "n_numeric": 9,
        "materials": list(MATERIAL_CLASSES),
        "body_types": list(BODY_TYPES),
        "collision_primitives": list(COLLISION_PRIMITIVES),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"train material {train_score['material_accuracy']:.3f} "
        f"holdout material {hold_score['material_accuracy']:.3f} "
        f"wrote {CHECKPOINT}",
        flush=True,
    )
    return report


def load_training_rows():
    """Pilot labels first, then stratified labels. One row per object id."""

    found = {}
    for path in (PILOT_LABELS, STRATIFIED_LABELS):
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("status") != "ok" or record.get("id") in found:
                continue
            prior_path = Path(record.get("prior_input") or "")
            if not prior_path.is_file():
                continue
            shape = tuple(int(value) for value in np.load(prior_path, mmap_mode="r").shape)
            if len(shape) != 2 or shape[1] != 912 or shape[0] < 1:
                continue
            found[record["id"]] = {
                "id": record["id"],
                "source": record["source"],
                "name": record.get("name") or record["id"],
                "material": record["material"],
                "body_type": record["body_type"],
                "collision": record["collision"],
                "log_mass": float(record["log_mass"]),
                "log_density": float(record["log_density"]),
                "rigidity": float(record["rigidity"]),
                "deformability": float(record["deformability"]),
                "prior_input": str(prior_path),
                "extent_m": record.get("extent_m"),
            }
    return list(found.values())


def score_split(prior, rows):
    """Accuracy of the three heads and mean absolute error of the four numbers."""

    if not rows:
        raise ValueError("score split is empty")
    totals = {name: 0.0 for name in (
        "loss", "material", "body", "collision", "log_mass", "log_density", "rigidity", "deformability",
    )}
    for row in rows:
        features = _features(row)
        mask = np.ones(features.shape[0], dtype=bool)
        mean = prior.in_mask_mean(features, mask)
        totals["loss"] += masked_prior_loss(
            mean,
            material=row["material"],
            body=row["body_type"],
            collision=row["collision"],
            log_mass=row["log_mass"],
            log_density=row["log_density"],
            rigidity=row["rigidity"],
            deformability=row["deformability"],
        )
        decoded = _decode_indices(mean)
        totals["material"] += float(decoded["material"] == row["material"])
        totals["body"] += float(decoded["body_type"] == row["body_type"])
        totals["collision"] += float(decoded["collision"] == row["collision"])
        totals["log_mass"] += abs(decoded["log_mass"] - row["log_mass"])
        totals["log_density"] += abs(decoded["log_density"] - row["log_density"])
        totals["rigidity"] += abs(decoded["rigidity"] - row["rigidity"])
        totals["deformability"] += abs(decoded["deformability"] - row["deformability"])
    count = float(len(rows))
    return {
        "n": len(rows),
        "loss": totals["loss"] / count,
        "material_accuracy": totals["material"] / count,
        "body_accuracy": totals["body"] / count,
        "collision_accuracy": totals["collision"] / count,
        "log_mass_mae": totals["log_mass"] / count,
        "log_density_mae": totals["log_density"] / count,
        "rigidity_mae": totals["rigidity"] / count,
        "deformability_mae": totals["deformability"] / count,
    }


def _fit(prior, rows, epochs, lr, seed, checkpoint=None, val_rows=None):
    """Adam on ``rows`` only.

    Without ``val_rows``, early stopping reads the train loss and the file
    keeps the last epoch, which is the triangle-row trainer. With
    ``val_rows``, the same patience rule reads the validation loss, and the
    saved weights are the earliest epoch with the lowest validation loss.
    The test set is not an argument and is not read here.
    """

    if checkpoint is None:
        checkpoint = CHECKPOINT
    params = [prior.w1, prior.b1, prior.w2, prior.b2, prior.w_out, prior.b_out]
    adam = _Adam(params, lr=lr)
    rng = np.random.default_rng(seed)
    history = []
    val_history = []
    best_val = math.inf
    best_weights = None
    best_epoch = None
    selecting = val_rows is not None
    if selecting and len(val_rows) < 1:
        raise ValueError("validation split is empty")
    for epoch in range(epochs):
        order = rng.permutation(len(rows))
        total = 0.0
        for index in order:
            row = rows[int(index)]
            loss, grads = loss_and_gradients(prior, _features(row), row)
            if not math.isfinite(loss):
                raise RuntimeError(f"loss became non-finite on {row['id']}")
            total += loss
            adam.step([grads[name] for name in ("w1", "b1", "w2", "b2", "w_out", "b_out")])
        mean_loss = total / len(rows)
        history.append(mean_loss)
        if selecting:
            val_score = score_split(prior, val_rows)
            val_loss = float(val_score["loss"])
            if not math.isfinite(val_loss):
                raise RuntimeError(f"validation loss became non-finite at epoch {epoch + 1}")
            val_history.append(val_loss)
            if val_loss < best_val:
                best_val = val_loss
                best_weights = [np.array(param, copy=True) for param in params]
                best_epoch = epoch + 1
            print(
                f"epoch {epoch + 1} train_loss {mean_loss:.4f} "
                f"val_loss {val_loss:.4f} val_material {val_score['material_accuracy']:.3f}",
                flush=True,
            )
            stop_on = val_history
        else:
            prior.save(checkpoint)
            print(f"epoch {epoch + 1} train_loss {mean_loss:.4f}", flush=True)
            stop_on = history
        # Threshold 0.995. From epoch 12, stop when the best of the last 4
        # losses is not at least 0.5% below the best loss before that window.
        # The comparison is strict, so a drop of exactly 0.5% does not stop.
        # Fitted runs pass validation losses; the triangle run passes train loss.
        if epoch + 1 >= 12 and min(stop_on[-4:]) > min(stop_on[:-4]) * 0.995:
            break
    if selecting:
        for param, saved in zip(params, best_weights):
            param[...] = saved
        prior.save(checkpoint)
    return {
        "train_loss": history,
        "val_loss": val_history if selecting else None,
        "checkpoint_epoch": best_epoch if selecting else (len(history) or None),
    }


def _predict_row(prior, row, checkpoint):
    """Vote one object with the checkpoint this prior was loaded from.

    The triangle trainer passes ``physics_prior.npz``. The fitted-splat
    trainer passes ``physics_prior_splat.npz``. ``vote_loaded_prior`` reads
    that path when ``prior`` is omitted.
    """

    features = _features(row)
    mask = np.ones(features.shape[0], dtype=bool)
    voted = vote_loaded_prior(
        features,
        mask,
        vlm={"source": "vlm", "class": row["material"], "weight": VLM_WEIGHT},
        keyword=keyword_material(row["name"]),
        checkpoint=checkpoint,
        name=row["name"],
        prior=prior,
    )
    decoded = voted["decoded"]
    record = voted["record"]
    return {
        "id": row["id"],
        "source": row["source"],
        "name": row["name"],
        "label_material": row["material"],
        "label_body_type": row["body_type"],
        "label_collision": row["collision"],
        "material": decoded["material"],
        "body_type": decoded["body_type"],
        "collision": decoded["collision"],
        "log_mass": decoded["log_mass"],
        "log_density": decoded["log_density"],
        "rigidity": decoded["rigidity"],
        "deformability": decoded["deformability"],
        "resolved_body_type": record["body_type"],
        "proposal": voted["proposal"],
        "vote_winner": voted["winner"],
        "scores": voted["scores"],
    }


def _features(row):
    array = np.load(row["prior_input"], mmap_mode="r")
    if array.ndim != 2 or array.shape[1] != 912:
        raise ValueError(f"{row['id']} prior_input shape is {array.shape}")
    return array


def _decode_indices(mean):
    from sceneagent.mlp.apply import decode_mean

    return decode_mean(mean)


def _mean_gradient(mean, label):
    grad = np.zeros(OUTPUT_DIM, dtype=np.float64)
    grad[SLOT_MATERIAL] = _softmax(mean[SLOT_MATERIAL])
    grad[SLOT_MATERIAL][MATERIAL_CLASSES.index(label["material"])] -= 1.0
    grad[SLOT_BODY] = _softmax(mean[SLOT_BODY])
    grad[SLOT_BODY][BODY_TYPES.index(label["body_type"])] -= 1.0
    grad[SLOT_COLLISION] = _softmax(mean[SLOT_COLLISION])
    grad[SLOT_COLLISION][COLLISION_PRIMITIVES.index(label["collision"])] -= 1.0
    grad[SLOT_LOG_MASS] = 2.0 * (float(mean[SLOT_LOG_MASS]) - float(label["log_mass"]))
    grad[SLOT_LOG_DENSITY] = 2.0 * (float(mean[SLOT_LOG_DENSITY]) - float(label["log_density"]))
    for slot, key in ((SLOT_RIGIDITY, "rigidity"), (SLOT_DEFORMABILITY, "deformability")):
        value = _sigmoid(float(mean[slot]))
        grad[slot] = 2.0 * (value - float(label[key])) * value * (1.0 - value)
    return grad


def _softmax(logits):
    shifted = np.asarray(logits, dtype=np.float64)
    shifted = shifted - np.max(shifted)
    exponential = np.exp(shifted)
    return exponential / exponential.sum()


def _sigmoid(value):
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _count_sources(rows):
    counts = {"ycb": 0, "gso": 0, "abo": 0}
    for row in rows:
        if row["source"] in counts:
            counts[row["source"]] += 1
    return counts


class _Adam:
    def __init__(self, params, lr, beta1=0.9, beta2=0.999, eps=1e-8, decay=1e-6, clip=5.0):
        self.params = params
        self.moment = [np.zeros_like(param) for param in params]
        self.velocity = [np.zeros_like(param) for param in params]
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.decay = decay
        self.clip = clip
        self.step_index = 0

    def step(self, grads):
        clipped = _clip(grads, self.clip)
        self.step_index += 1
        bias_1 = 1.0 - self.beta1 ** self.step_index
        bias_2 = 1.0 - self.beta2 ** self.step_index
        for param, moment, velocity, grad in zip(self.params, self.moment, self.velocity, clipped):
            grad = np.asarray(grad, dtype=np.float64)
            if self.decay:
                grad = grad + self.decay * param
            moment *= self.beta1
            moment += (1.0 - self.beta1) * grad
            velocity *= self.beta2
            velocity += (1.0 - self.beta2) * (grad * grad)
            param -= self.lr * (moment / bias_1) / (np.sqrt(velocity / bias_2) + self.eps)


def _clip(grads, clip):
    norm = math.sqrt(sum(float(np.square(grad).sum()) for grad in grads))
    if not math.isfinite(norm):
        raise RuntimeError("gradient is not finite")
    if norm <= clip or norm == 0.0:
        return grads
    scale = clip / norm
    return [grad * scale for grad in grads]


def load_fitted_training_rows(root=None):
    """Status=ok stratified labels whose ``prior_input.npy`` came from a fitted splat.

    A triangle stand-in, including one whose opacity is the constant 0.99,
    has no fitted marker and is not a training row. Pilot-only ids are not
    read. The label file's path is ignored when it does not point at the
    fitted array; the object directory is ``root/source/id``.
    """

    root = Path(root) if root is not None else ROOT
    labels_path = root / "stratified_labels.jsonl"
    if not labels_path.is_file():
        raise FileNotFoundError(f"missing stratified labels at {labels_path}")
    found = {}
    with labels_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("status") != "ok" or record.get("id") in found:
                continue
            source = record.get("source")
            obj_id = record.get("id")
            if source not in {"ycb", "gso", "abo"} or not obj_id:
                continue
            directory = root / source / obj_id
            prior_path = directory / "prior_input.npy"
            marker_path = directory / FITTED_MARKER_NAME
            if not prior_path.is_file() or not marker_path.is_file():
                continue
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            if marker.get("kind") != "fitted" or marker.get("opacity_constant_099"):
                continue
            if float(marker.get("opacity_std") or 0.0) < 1e-3:
                continue
            shape = tuple(int(value) for value in np.load(prior_path, mmap_mode="r").shape)
            if len(shape) != 2 or shape[1] != 912 or shape[0] < 1:
                continue
            for key in ("material", "body_type", "collision", "log_mass", "log_density", "rigidity", "deformability"):
                if record.get(key) is None:
                    break
            else:
                found[obj_id] = {
                    "id": obj_id,
                    "source": source,
                    "name": record.get("name") or obj_id,
                    "status": "ok",
                    "material": record["material"],
                    "body_type": record["body_type"],
                    "collision": record["collision"],
                    "log_mass": float(record["log_mass"]),
                    "log_density": float(record["log_density"]),
                    "rigidity": float(record["rigidity"]),
                    "deformability": float(record["deformability"]),
                    "prior_input": str(prior_path),
                    "extent_m": record.get("extent_m"),
                }
    return list(found.values())


def train_fitted_and_report(epochs=30, lr=1e-3, seed=0):
    """Train on fitted-splat rows and write ``physics_prior_splat.npz``.

    Does not read or write ``physics_prior.npz``. Slots 28–32 stay masked.
    Train ids are the fitted ids ``holdout_ids`` does not select (rate 0.12,
    plus the can, the shark, and the pillow when those ids are present).
    That holdout is sorted by sha256(id) then the id string. The first 30%,
    rounded half up, is validation. The rest is test. Early stopping and the
    saved checkpoint use validation loss. The factor is 0.995. The test set
    is scored once after that checkpoint is chosen.
    """

    rows = load_fitted_training_rows()
    if len(rows) < 12:
        raise SystemExit(f"stopped before training: only {len(rows)} fitted rows")
    reason = labels_should_stop(rows)
    if reason:
        raise SystemExit(f"stopped before training: {reason}")
    by_source = _count_sources(rows)
    held = holdout_ids([row["id"] for row in rows])
    validation_ids, test_ids = validation_and_test_ids(held)
    by_id = {row["id"]: row for row in rows}
    missing = [obj_id for obj_id in validation_ids + test_ids if obj_id not in by_id]
    if missing:
        raise SystemExit(f"stopped before training: holdout ids without rows {missing[:5]}")
    validation_set = set(validation_ids)
    test_set = set(test_ids)
    train_rows = [row for row in rows if row["id"] not in held]
    val_rows = [by_id[obj_id] for obj_id in validation_ids]
    test_rows = [by_id[obj_id] for obj_id in test_ids]
    if {row["id"] for row in train_rows} & (validation_set | test_set):
        raise RuntimeError("train ids moved into validation or test")
    if (validation_set & test_set) or ((validation_set | test_set) != set(held)):
        raise SystemExit("stopped before training: validation and test do not partition the holdout")
    if len(train_rows) < 10 or len(val_rows) < 1 or len(test_rows) < 1:
        raise SystemExit(
            f"stopped before training: train {len(train_rows)} "
            f"validation {len(val_rows)} test {len(test_rows)}"
        )
    if SPLAT_CHECKPOINT.resolve() == CHECKPOINT.resolve():
        raise SystemExit("refusing to write the fitted prior over physics_prior.npz")
    print(
        f"split n_train {len(train_rows)} n_validation {len(val_rows)} n_test {len(test_rows)}",
        flush=True,
    )
    print(
        "early_stopping validation_loss threshold 0.995 window 4 min_epoch 12",
        flush=True,
    )
    prior = PhysicsPrior.initialize(seed)
    fit = _fit(
        prior,
        train_rows,
        epochs=epochs,
        lr=lr,
        seed=seed,
        checkpoint=SPLAT_CHECKPOINT,
        val_rows=val_rows,
    )
    history = fit["train_loss"]
    prior.save(SPLAT_CHECKPOINT)
    loaded = PhysicsPrior.load(SPLAT_CHECKPOINT)
    train_score = score_split(loaded, train_rows)
    val_score = score_split(loaded, val_rows)
    test_score = score_split(loaded, test_rows)
    predictions = [_predict_row(loaded, row, SPLAT_CHECKPOINT) for row in test_rows]
    known = [row for row in predictions if row["id"] in KNOWN_IDS]
    known.sort(key=lambda row: KNOWN_IDS.index(row["id"]))
    present = {row["id"] for row in rows}
    report = {
        "checkpoint": str(SPLAT_CHECKPOINT),
        "weights": "ours",
        "section": "3.4",
        "fitted_splat": True,
        "gemini_called": False,
        "acquire_called": False,
        "n_fitted": len(rows),
        "n_train": len(train_rows),
        "n_validation": len(val_rows),
        "n_test": len(test_rows),
        "n_holdout": len(held),
        "validation_fraction": 0.3,
        "validation_rounding": "half-up",
        "holdout_order": "sha256(id), id",
        "seed": seed,
        "early_stop_metric": "validation_loss",
        "early_stop_threshold": 0.995,
        "early_stop_window": 4,
        "early_stop_min_epoch": 12,
        "checkpoint_rule": "earliest epoch with the lowest validation loss",
        "sources": by_source,
        "epochs_ran": len(history),
        "checkpoint_epoch": fit["checkpoint_epoch"],
        "train_loss_first": history[0] if history else None,
        "train_loss_last": history[-1] if history else None,
        "train_loss_by_epoch": history,
        "val_loss_by_epoch": fit["val_loss"],
        "val_loss_best": min(fit["val_loss"]) if fit["val_loss"] else None,
        "train": train_score,
        "validation": val_score,
        "test": test_score,
        "train_ids": [row["id"] for row in train_rows],
        "validation_ids": validation_ids,
        "test_ids": test_ids,
        "known_ids_in_validation": [obj_id for obj_id in KNOWN_IDS if obj_id in validation_set],
        "known_ids_in_test": [obj_id for obj_id in KNOWN_IDS if obj_id in test_set],
        "known_ids_absent": [obj_id for obj_id in KNOWN_IDS if obj_id not in present],
        "test_predictions": predictions,
        "known": known,
        "w_out_shape": [32, 33],
        "n_materials": 10,
        "n_body": 6,
        "n_collision": 8,
        "n_numeric": 9,
        "masked_slots": [28, 29, 30, 31, 32],
        "materials": list(MATERIAL_CLASSES),
        "body_types": list(BODY_TYPES),
        "collision_primitives": list(COLLISION_PRIMITIVES),
    }
    convert_report = ROOT / "fitted_convert_report.json"
    if convert_report.is_file():
        report["conversion"] = json.loads(convert_report.read_text(encoding="utf-8"))
    SPLAT_REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"fitted epochs {len(history)} checkpoint_epoch {fit['checkpoint_epoch']} "
        f"validation {_format_metrics(val_score)}",
        flush=True,
    )
    print(f"test {_format_metrics(test_score)} wrote {SPLAT_CHECKPOINT}", flush=True)
    return report


def _format_metrics(score):
    return (
        f"material {score['material_accuracy']:.6f} "
        f"body {score['body_accuracy']:.6f} "
        f"collision {score['collision_accuracy']:.6f} "
        f"log_mass_mae {score['log_mass_mae']:.6f} "
        f"log_density_mae {score['log_density_mae']:.6f} "
        f"rigidity_mae {score['rigidity_mae']:.6f} "
        f"deformability_mae {score['deformability_mae']:.6f}"
    )


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "fitted":
        train_fitted_and_report()
        return
    train_and_report()


if __name__ == "__main__":
    main()
