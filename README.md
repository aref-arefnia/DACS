# DACS — Deadline-Aware Adaptive Client Selection

Implementation of the **DACS (Deadline-Aware Adaptive Client Selection)** algorithm for client selection in Federated Learning (FL), designed as a drop-in replacement for the well-known **Oort** selector in the **[FedScale](https://github.com/SymbioticLab/FedScale)** framework.

This code is a direct implementation of Algorithm 1 (`ClientSelection(C, k, ε)`) from the following paper:

> **DACS: Deadline-Aware Adaptive Client Selection in Federated Learning**
> Aref Arefnia and Abdolah Chalechale
> Department of Computer Engineering, Razi University, Kermanshah, Iran
> Springer *Cluster Computing*, 2026 — DOI: [10.1007/s10586-026-06236-0](https://doi.org/10.1007/s10586-026-06236-0)

---

## 1. Core Idea

Like Oort, DACS uses an **exploitation/exploration** strategy to select a subset of clients in each federated training round. The key difference is that, in addition to a client's **statistical utility** (data quality), DACS also factors in a **deadline-aware** term called **Temporal Proximity (TP)** in the scoring function. This favors clients whose completion time is close to a round's "preferred deadline" — penalizing both stragglers and clients that finish far too fast (i.e., underutilized capacity).

Key formulas:

| Symbol | Meaning | Formula |
|---|---|---|
| `U_stat(i)` | Client's statistical utility (Eq. 6) | The client's recorded reward (derived from its loss/gradient in that round) |
| `TP(i)` | Temporal proximity to the preferred deadline (Eq. 7) | `exp(-(t_i - T)² / (2·σ²))` |
| `Utility(i)` | Final client score (Eq. 9) | `U_stat(i) × TP(i)` |
| `T` | Round's preferred deadline (Eq. 11) | `(3·T_min + T_max) / 4` over the durations of the exploited set |

### Algorithm steps (`ClientSelection(C, k, ε)`) mapped to the code

| Pseudocode step | Method/section in `dacs.py` |
|---|---|
| Step 1 — Compute `Utility(i)` for explored clients `E` | Loop over `E` in `getTopK`, via `_utility()` |
| Step 2 — Exploitation; `W ← Top (1-ε)·k` from `E` | `exploitLen` and deterministic top-k selection from `utility` |
| Step 3 — Determine preferred deadline `T` (Eq. 11) | Computing `t_min`, `t_max`, and `self.round_prefer_duration` |
| Step 4 — Exploration; add `ε·k` new clients from `C \ E` | `exploreLen` and top-k selection from `explore_utility` |
| Step 5 — `E ← E ∪ W` | `self.explored.update(pickedClients)` |

---

## 2. Installation & Integration with FedScale

DACS mirrors the shape of FedScale's Oort module exactly (`create_training_selector`, `create_testing_selector`, identical method signatures), so swapping it in is straightforward.

### 2.1. Copy the file into the FedScale tree

In the FedScale repository, the Oort module typically lives at:

```
FedScale/thirdparty/oort/oort.py
```

Place `dacs.py` in a matching location:

```bash
mkdir -p FedScale/thirdparty/dacs
cp dacs.py FedScale/thirdparty/dacs/dacs.py
touch FedScale/thirdparty/dacs/__init__.py
```

> Note: the file depends on `.utils.lp` (a helper module for LP-based selection, used only by `_testing_selector`). Copy it over as well from next to the original Oort file:
> ```bash
> cp -r FedScale/thirdparty/oort/utils FedScale/thirdparty/dacs/utils
> ```

### 2.2. Wire it into `client_manager`

In FedScale, the selector is chosen based on `args.sample_mode` (in `fedscale/cloud/client_manager.py`). The exact import location can vary slightly between FedScale versions; grep for it to find the right spot:

```bash
grep -rn "oort" FedScale/fedscale/cloud/client_manager.py
```

You should find something like:

```python
if args.sample_mode == "oort":
    from thirdparty.oort.oort import create_training_selector
    self.ucb_sampler = create_training_selector(args)
```

Add a matching branch for DACS:

```python
elif args.sample_mode == "dacs":
    from thirdparty.dacs.dacs import create_training_selector
    self.ucb_sampler = create_training_selector(args)
```

All methods FedScale calls on `self.ucb_sampler` (`register_client`, `select_participant`, `update_client_util`, `update_duration`, `getAllMetrics`, `get_median_reward`, `get_client_reward`) are implemented in `dacs.py` — no other changes to `client_manager.py` are needed.

### 2.3. Required job config (YAML)

In your FedScale job config (e.g., `benchmark/configs/.../conf.yml`), add/change:

```yaml
job_conf:
  - sample_mode: dacs        # instead of oort
  - exploration_factor: 0.2  # eps in Algorithm 1 (exploration's share of k)
  - sigma: 150                # sigma in Eq. 7 (Gaussian bandwidth of temporal proximity)
```

Other parameters required by the original Oort implementation (`exploration_decay`, `exploration_min`, `cut_off_util`, `round_threshold`, `blacklist_rounds`, ...) are not used in this simplified version and can be removed from the config.

---

## 3. Standalone Usage (outside FedScale)

```python
from dacs import create_training_selector

# args must have at least these two fields:
#   exploration_factor -> epsilon (0 < eps < 1)
#   sigma (or sigma0)  -> Gaussian bandwidth of TP, defaults to 150 if omitted
selector = create_training_selector(args)

# The first time a client is seen:
selector.register_client(client_id=client_id, feedbacks={
    "reward": init_reward,      # initial utility estimate (e.g., based on data size)
    "duration": est_duration,   # estimated round completion time for this client
})

# At the start of each round, select k clients from the available pool:
selected_clients = selector.select_participant(
    num_of_clients=k,
    feasible_clients=set(available_client_ids),
)

# After a client finishes training that round, record its real feedback:
selector.update_client_util(client_id, feedbacks={
    "reward": real_reward,       # e.g., average loss/gradient norm for that round
    "duration": real_duration,   # actual completion time
    "time_stamp": current_round,
    "status": True,              # False means the client is no longer available
})
```

### API summary

| Method | Purpose |
|---|---|
| `create_training_selector(args)` | Create a training selector instance (DACS) |
| `register_client(client_id, feedbacks)` | Register a new client with an initial reward/duration estimate |
| `select_participant(num_of_clients, feasible_clients)` | Run Algorithm 1 and return the list of clients selected for the current round |
| `update_client_util(client_id, feedbacks)` | Update a client's real reward/duration after training |
| `update_duration(client_id, duration)` | Update a client's completion time only, without changing its reward |
| `get_client_reward(armId)` | Get a client's full profile |
| `get_median_reward()` | Average reward across all registered clients |
| `getAllMetrics()` | Full dictionary of all clients (`totalArms`) |

---

## 4. Testing

A self-contained simulation (`tests/test_dacs.py`) exercises the selector against synthetic clients — each with a fixed "true" utility and completion time observed with noise — over several rounds, and checks that Algorithm 1's invariants hold (exactly `k` clients returned once enough are available, no duplicates, only feasible clients chosen, and the explored set never shrinks). It needs only the Python standard library:

```bash
python3 tests/test_dacs.py
```

Expected output ends with:

```
All assertions passed.
```

---

## 5. Citation

If you use this implementation, please cite the original paper:

```bibtex
@article{arefnia2026dacs,
  title   = {DACS: Deadline-Aware Adaptive Client Selection in Federated Learning},
  author  = {Arefnia, Aref and Chalechale, Abdolah},
  journal = {Cluster Computing},
  year    = {2026},
  doi     = {10.1007/s10586-026-06236-0},
  url     = {https://doi.org/10.1007/s10586-026-06236-0}
}
```

This implementation is built on the code structure of **[Oort](https://github.com/SymbioticLab/Oort)** (the client selection framework from the FedScale ecosystem) and is designed for use with **[FedScale](https://github.com/SymbioticLab/FedScale)**.
