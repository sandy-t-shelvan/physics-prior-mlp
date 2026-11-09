# Physics-prior MLP

A per-Gaussian MLP predicts physical attributes from a fitted 3D Gaussian splat. The preprint-style note is [`paper/physics_prior_mlp.tex`](paper/physics_prior_mlp.tex). It is not an arXiv submission. See [`paper/README.md`](paper/README.md). `pdflatex` was not on this machine, so the PDF was not built.

The trained checkpoint is [`data/mlp/physics_prior_splat.npz`](data/mlp/physics_prior_splat.npz): the epoch-10 weights from the 14-epoch fitted-splat run. The numbers in the paper are the `train`, `validation`, and `test` objects in [`data/mlp/train_report_splat.json`](data/mlp/train_report_splat.json).

An earlier 30-epoch holdout-only run, without a validation split, is kept separately as `data/mlp/physics_prior_splat_holdout462.npz` and `data/mlp/train_report_splat_holdout462.json`. It is not the checkpoint the paper reports.

Labels used as the original targets are `data/mlp/stratified_labels.4090.jsonl` (4427 ok rows). `data/mlp/stratified_labels.remaining.jsonl` (1796 ok rows) was not used to train this checkpoint.

Meshes and per-Gaussian `prior_input.npy` files are not in this repository.

## Score the checkpoint

Print the stored scores with no feature files:

```python
import json
from pathlib import Path

report = json.loads(Path("data/mlp/train_report_splat.json").read_text(encoding="utf-8"))
print(report["epochs_ran"], report["checkpoint_epoch"], report["n_fitted"])
print(report["train"])
print(report["validation"])
print(report["test"])
```

To recompute those scores, put the repository root on `PYTHONPATH` and call `score_split`. Each row needs `material`, `body_type`, `collision`, `log_mass`, `log_density`, `rigidity`, `deformability`, and `prior_input` pointing at a local `(N, 912)` array. Those arrays are not shipped here. `score_split` runs the MLP on every Gaussian and averages the 33 outputs. It does not average the 912 inputs, and it does not train.

```python
from sceneagent.mlp.prior import PhysicsPrior
from sceneagent.mlp.train_prior import score_split

prior = PhysicsPrior.load("data/mlp/physics_prior_splat.npz")
print(score_split(prior, rows))
```

Loading and scoring need NumPy. The mesh and view code also imports Pillow and Trimesh. The contract, view, and training tests are `tests/sceneagent/test_mlp_contract.py`, `tests/sceneagent/test_mlp_views.py`, and `tests/sceneagent/test_mlp_stratified.py`.
