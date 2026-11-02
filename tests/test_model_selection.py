"""Choosing a reticulation count from a sweep: logL, AIC, BIC, elbow."""

import math

from phynetpy.ModelSelection import SweepResult, SweepRow, reticulation_sweep


def _result(log_liks, data_size=100, params=3):
    rows = []
    previous = None
    for k, value in enumerate(log_liks):
        n_params = params * k
        rows.append(SweepRow(
            k=k, best_log_lik=value, all_log_liks=[value], n_params=n_params,
            aic=2.0 * n_params - 2.0 * value,
            bic=n_params * math.log(data_size) - 2.0 * value,
            delta_log_lik=None if previous is None else value - previous,
        ))
        previous = value
    return SweepResult(rows=rows, data_size=data_size, params_per_reticulation=params, base_params=0)


def test_each_criterion_picks_its_own_k():
    sweep = _result([-100.0, -50.0, -49.0])
    assert sweep.best_by("logL") == 2
    # Gain 50 pays for 3 parameters; gain 1 does not.
    assert sweep.best_by("aic") == 1
    assert sweep.best_by("bic") == 1
    # The second step is under 10% of the first, so the elbow is at k=1.
    assert sweep.best_by("elbow") == 1


def test_a_gain_below_the_penalty_keeps_the_tree():
    sweep = _result([-1000.0, -998.0])
    assert sweep.best_by("aic") == 0           # gain 2 < 3 parameters
    assert sweep.best_by("bic") == 0           # gain 2 < 1.5 ln(100) = 6.9
    assert sweep.best_by("logL") == 1


def test_the_elbow_needs_two_steps_to_see_a_bend():
    # With only k = 0, 1 the single step is also the largest step, so the
    # elbow returns 1 for any positive gain, however small.
    assert _result([-1000.0, -999.0]).best_by("elbow") == 1
    # A flat curve (no gain) returns the first k.
    assert _result([-1000.0, -1000.0]).best_by("elbow") == 0
    # With a third k the bend is visible: [50, 1] bends at 1, [50, 45] does not.
    assert _result([-1000.0, -950.0, -949.0]).best_by("elbow") == 1
    assert _result([-1000.0, -950.0, -905.0]).best_by("elbow") == 2


def test_reticulation_sweep_keeps_the_best_seed_and_the_deltas():
    calls = []

    def search_fn(k, seed):
        calls.append((k, seed))
        return {0: -100.0, 1: -60.0, 2: -59.0}[k] - (0.5 if seed == 1 else 0.0)

    sweep = reticulation_sweep(search_fn, [0, 1, 2], seeds=(0, 1), data_size=10, progress=False)
    assert calls == [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]
    assert [row.best_log_lik for row in sweep.rows] == [-100.0, -60.0, -59.0]
    assert [row.delta_log_lik for row in sweep.rows] == [None, 40.0, 1.0]
    assert sweep.best_by("elbow") == 1
