"""Stub for Oort's LP-based testing-selector helper.

`_testing_selector.select_by_category()` (inherited, unrelated to the DACS
training-selector) calls `run_select_by_category`, which in FedScale solves a
mixed-integer/LP problem to pick clients matching a requested per-category
sample distribution. That solver is not part of this repository.

The DACS training selector (`_training_selector` / `create_training_selector`)
does not use this module at all. This stub exists only so `dacs.py` can be
imported standalone without pulling in FedScale's full LP solver. If you need
`_testing_selector.select_by_category()`, replace this file with the real
implementation from FedScale's `thirdparty/oort/utils/lp.py`.
"""


def run_select_by_category(request_list, data_distribution, client_info, max_num_clients, model_size, greedy_heuristic):
    raise NotImplementedError(
        "run_select_by_category is not included in this repository; "
        "copy it from FedScale's thirdparty/oort/utils/lp.py to use "
        "_testing_selector.select_by_category()."
    )
