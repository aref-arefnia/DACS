"""Standalone simulation that exercises the DACS training selector.

Runs multiple simulated federated-learning rounds against a population of
synthetic clients (each with a fixed "true" statistical utility and
completion time, observed with noise) and checks that `_training_selector`
behaves as Algorithm 1 specifies: it returns the right number of clients,
never repeats a client within a round, only draws from feasible clients,
and its explored set only grows over time.
"""

import os
import random
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dacs.dacs import create_training_selector

NUM_CLIENTS = 30
NUM_ROUNDS = 15
CLIENTS_PER_ROUND = 8


def make_synthetic_clients(num_clients, seed=42):
    rng = random.Random(seed)
    clients = {}
    for client_id in range(1, num_clients + 1):
        clients[client_id] = {
            "true_reward": rng.uniform(1.0, 100.0),
            "true_duration": rng.uniform(10.0, 300.0),
        }
    return clients


def observe(true_value, rng, noise_frac=0.1):
    noise = rng.uniform(-noise_frac, noise_frac) * true_value
    return max(1e-3, true_value + noise)


def run_simulation():
    rng = random.Random(7)
    args = SimpleNamespace(exploration_factor=0.2, sigma=80)
    selector = create_training_selector(args)

    clients = make_synthetic_clients(NUM_CLIENTS)
    for client_id, profile in clients.items():
        selector.register_client(str(client_id), feedbacks={
            "reward": observe(profile["true_reward"], rng),
            "duration": observe(profile["true_duration"], rng),
        })

    feasible = set(clients.keys())
    seen_explored_sizes = []

    for round_num in range(1, NUM_ROUNDS + 1):
        picked = selector.select_participant(CLIENTS_PER_ROUND, feasible_clients=feasible)

        assert len(picked) <= CLIENTS_PER_ROUND, (
            f"round {round_num}: selected {len(picked)} > k={CLIENTS_PER_ROUND}"
        )
        assert len(picked) == len(set(picked)), f"round {round_num}: duplicate client in selection"
        assert all(int(c) in feasible for c in picked), f"round {round_num}: infeasible client selected"

        for client_id in picked:
            profile = clients[int(client_id)]
            selector.update_client_util(client_id, feedbacks={
                "reward": observe(profile["true_reward"], rng),
                "duration": observe(profile["true_duration"], rng),
                "time_stamp": selector.training_round,
                "status": True,
            })

        seen_explored_sizes.append(len(selector.explored))

        print(
            f"round {round_num:2d}: picked={picked}, "
            f"explored_so_far={len(selector.explored)}/{NUM_CLIENTS}, "
            f"T={selector.round_prefer_duration:.1f}"
        )

    assert seen_explored_sizes == sorted(seen_explored_sizes), "explored set must never shrink"
    assert seen_explored_sizes[-1] > 0, "selector never explored any client"

    print("\nAll assertions passed.")
    print(f"Final explored clients: {seen_explored_sizes[-1]}/{NUM_CLIENTS}")
    print(f"Median reward across all registered clients: {selector.get_median_reward():.2f}")


if __name__ == "__main__":
    run_simulation()
