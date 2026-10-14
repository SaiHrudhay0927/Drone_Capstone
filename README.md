# Drone Capstone — DQN, DDQN and NEWDQN

A modular Python project based on the supplied Colab implementation and Guo, Huang & Huang, **A Deep Q-Network-based Algorithm for Obstacle Avoidance and Target Tracking for Drones**, IEEE SMC 2023, DOI 10.1109/SMC53992.2023.10393953.

**Start here:** the completed experiment is in `results/faculty_run/RESULTS.md`. Saved models and raw logs are included. This is a documented reproduction attempt, not the authors' code or a claim of identical paper results. Read `docs/PAPER_ALIGNMENT.md` before presenting it.

## Training output matching the reference examples

Open **`Training_Results.ipynb`** to see preloaded outputs in this order: DQN log + curve, DDQN log + curve, NEWDQN log + curve, then all three curves on one graph. The displayed run uses seed 42, 1,000 episodes and a 30-episode rolling mean. No training is needed to view the saved outputs. Opening/executing notebooks may require notebook support in your editor; PNGs below need no extensions.

- Individual log-and-curve images: `results/faculty_run/<algorithm>/seed_42/training_report.png`.
- Individual curve-only images: `results/faculty_run/<algorithm>/seed_42/success_curve.png`.
- Final graph: `results/faculty_run/training_reports/combined_success_seed_42.png`.
- Seed-averaged graph: `results/faculty_run/training_reports/combined_success_all_seeds.png`.
- Full ordered report for every seed: `results/faculty_run/training_reports/TRAINING_REPORTS.md`.

Future `train.py` / `compare.py` runs print episode, average reward, last-50 success rate and epsilon every 50 episodes, and generate the individual plots automatically. `compare.py` also generates the final combined graphs. Epsilon is the exploration probability used during the logged episode.

To regenerate these plots from existing CSVs:

```powershell
.\.venv\Scripts\python.exe reporting.py --results results/faculty_run
```

The reference-style charts use a rolling window of 30; the original aggregate `training_comparison.png` continues to use 50. No interpolation or invented curves are used. NEWDQN is blue, DQN orange and DDQN grey on the new combined graphs, matching your reference colors.

## Open in Antigravity on Windows

1. Extract the entire ZIP. Keep all subfolders together.
2. Open the extracted `drone_capstone` folder in Antigravity using its folder-opening action.
3. Open a terminal in that folder. Use Python 3.11 or 3.12, then run:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

If Python 3.12 is not installed, replace `py -3.12` with the command for your installed Python. Directly invoking the virtual environment avoids PowerShell activation-policy problems. If prompted for a Python interpreter in the editor, select `.venv\Scripts\python.exe`.

## Show the trained model immediately

No retraining needed. This generates a GIF/PNG and opens the live Matplotlib animation:

```powershell
.\.venv\Scripts\python.exe demo.py --checkpoint results/faculty_run/newdqn/seed_42/model.pt --map paper8 --show
```

Replace `newdqn` with `dqn` or `ddqn` to show the other algorithms. Change `--map paper8` to `--map paper10` to demonstrate transfer to an unseen map. Use `--scenario-seed 100001` for another scenario. The default scenario is fixed in advance, not chosen for success; both successful and failed runs are valid demonstrations.

Prepared animations for all three algorithms on both maps are in `results/demos/`. Open the GIF files directly for a demonstration without Python installed.

## Run all three algorithms

```powershell
.\.venv\Scripts\python.exe compare.py --output results/local_comparison
```

Default: 1,000 training episodes × 3 seeds × 3 algorithms; 200 held-out greedy evaluation episodes per map per model. CPU only; a GPU is not needed. Use a new output directory for each experiment; existing results are protected from accidental overwrite.

Quick pipeline check (not a convergence experiment):

```powershell
.\.venv\Scripts\python.exe compare.py --episodes 20 --eval-episodes 10 --seeds 42 --output results/local_smoke
```

## Train algorithms separately

```powershell
.\.venv\Scripts\python.exe train.py --algorithm dqn --output results/local_dqn
.\.venv\Scripts\python.exe train.py --algorithm ddqn --output results/local_ddqn
.\.venv\Scripts\python.exe train.py --algorithm newdqn --output results/local_newdqn
```

Each command imports its own algorithm module and the shared environment, map, reward, network and pipeline. The individual modules should not be executed directly.

Evaluate an existing model without learning:

```powershell
.\.venv\Scripts\python.exe evaluate.py --checkpoint results/faculty_run/ddqn/seed_42/model.pt --episodes 200 --output results/local_eval
```

## File guide

| File | Purpose |
|---|---|
| `agents/dqn.py` | DQN target: max over target-network Q values |
| `agents/ddqn.py` | Online action selection, target-network evaluation |
| `agents/newdqn.py` | DQN target, multi-pool replay and adaptive exploration |
| `agents/base.py` | Shared optimizer, action selection and learning mechanics |
| `environment/maps.py` | Separate 8×8/10×10 paper and Colab obstacle layouts |
| `environment/grid_env.py` | Movement, target motion, observations, capture and termination |
| `environment/rewards.py` | Distance, direction, step and terminal rewards |
| `models/q_network.py` | Shared 6–64–64–4 neural network |
| `memory/replay.py` | Uniform, success, failure and temporary experience pools |
| `configs/default.json` | Paper-inspired experiment parameters |
| `configs/colab_comparison.json` | Colab map/reward/curriculum comparison preset |
| `training.py`, `train.py` | Shared training pipeline and single-algorithm entry point |
| `evaluation.py`, `evaluate.py` | Held-out evaluation and checkpoint entry point |
| `compare.py`, `plotting.py` | Run all algorithms and plot measured comparisons |
| `demo.py` | Checkpoint loading, simulation, GIF and trajectory rendering |
| `tests/test_core.py` | Target-rule, replay, exploration and environment checks |
| `docs/PAPER_ALIGNMENT.md` | Assumptions, deviations and source mapping |
| `docs/FACULTY_GUIDE.md` | Presentation order and explanations |
| `reference/` | Original uploaded code as text and original paper |

`compare.py → training.py → algorithm agent → environment / rewards / map`, with transitions stored in `memory/replay.py` and Q values produced by `models/q_network.py`. All algorithms use the same shared task and network to isolate algorithm differences.

## Outputs

Every `results/<experiment>/<algorithm>/seed_<n>/` contains:

- `model.pt`: online and target network weights, optimizer state, config and metadata.
- `training.csv`: per-episode reward, success, steps, epsilon, loss and OBE-reset count.
- `evaluation_<map>.csv`: every held-out evaluation outcome.
- `trajectory_<map>.json`: the first evaluation trajectory (not cherry-picked).
- `replay.csv`: final replay contents; a replay-buffer snapshot, **not a complete dataset of all transitions**.
- `summary.json`, `config.json`: measured metrics, software versions and settings.

Top-level comparison files contain seed means and population standard deviations. Model checkpoints support inference; exact interrupted-training resume is not implemented because replay and RNG states are not checkpointed.

For macOS/Linux, use `python3 -m venv .venv`, then `.venv/bin/python` instead of `.\.venv\Scripts\python.exe`. On Colab, upload/extract the ZIP, change into its folder, install requirements, and run `!python compare.py --output results/colab_run`. There are no mandatory Google Colab imports.
