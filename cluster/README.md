# Running on the Phoenix cluster (HUJI SLURM)

CPU-only jobs (LightGBM / XGBoost) — **never request a GPU**.

## 0. Fill in the placeholders
Open `run_train.slurm` and `run_eval.slurm` and confirm the `<PLACEHOLDER>` SBATCH lines
for your account. Defaults are taken from this user's existing lab scripts:

| Option | Default | Alternative |
|---|---|---|
| `--account` | `nirf` | `killable-cs` (for the `killable` partition) |
| `--partition` | `short` (≤ 3 h) | `killable` (long, preemptible) |
| `--cpus-per-task` | `8` | match your allocation |
| `--mem` | `32G` | raise for full-year enriched data |

Use `short` for the Jan–Feb `train_set.csv` (trains in minutes). Switch to
`killable` + `killable-cs` and a longer `--time` for the large full-year enriched data.

## 1. One-time setup
```bash
ssh <user>@phoenix.cs.huji.ac.il
cd <path>/HUJI-IML-Hackathon
git pull
bash cluster/env_setup.sh          # creates the 'bikedemand' conda env (or a venv)
```

## 2. Stage the data (NOT in git)
`dataset/train_set.csv` is gitignored. Copy it onto Phoenix once:
```bash
scp train_set.csv <user>@phoenix.cs.huji.ac.il:<path>/HUJI-IML-Hackathon/dataset/
```

## 3. Train + evaluate a submission
```bash
sbatch cluster/run_train.slurm submissions/challenge_1_ensamble
sbatch cluster/run_eval.slurm  submissions/challenge_1_ensamble
```
Read results from `cluster/logs/<job>-<id>.out` (and `.err`). `eval_local.py` prints the
per-city + blended MAE table.

## Notes
- Threads: the scripts export `OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK` so LightGBM/XGBoost
  match the allocation; pass `--n-jobs $SLURM_CPUS_PER_TASK` is handled for eval.
- **Big data downloads** (`tools/fetch_*.py`, `enrichment/build_*.py`) hit the internet.
  Phoenix compute nodes may be firewalled — run those on an interactive/login node that
  has internet per cluster policy, not inside a batch job.
- Nothing in the model code depends on SLURM; the `.slurm` files are just wrappers, so
  `python train.py` / `python tools/eval_local.py <folder>` still work locally.
