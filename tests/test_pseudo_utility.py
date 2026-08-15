import numpy as np
import pytest
import json
from pathlib import Path

from src.pseudo_utility import fit_global_parameters, transform


def test_global_scenarios_are_finite_nonnegative_and_weakly_order_preserving():
    scores = np.array([-3.0, -1.0, -1.0, 0.0, 5.0])
    parameters = fit_global_parameters(scores)
    for scenario in ("global_shift_q90_scale", "global_robust_softplus"):
        values = transform(scenario, scores, parameters)
        assert np.all(np.isfinite(values))
        assert np.all(values >= 0.0)
        assert np.all(np.diff(values) >= 0.0)
        assert values[1] == values[2]


def test_midrank_percentile_uses_exact_ties_and_constant_half_fallback():
    scores = np.array([2.0, 1.0, 2.0, 5.0])
    values = transform("within_user_midrank_percentile", scores)
    assert values == pytest.approx([0.5, 0.125, 0.5, 0.875])
    assert transform("within_user_midrank_percentile", np.ones(3)) == pytest.approx(
        np.full(3, 0.5)
    )


def test_positive_part_standardization_is_row_specific_and_constant_safe():
    scores = np.array([[1.0, 2.0, 3.0], [7.0, 7.0, 7.0]])
    values = transform("positive_part_user_standardization", scores)
    assert values.shape == scores.shape
    assert values[0, 0] == 0.0
    assert values[0, 2] > 0.0
    np.testing.assert_array_equal(values[1], np.zeros(3))


def test_scenarios_reject_nonfinite_inputs_and_unknown_ids():
    with pytest.raises(ValueError, match="finite"):
        fit_global_parameters([1.0, np.nan])
    with pytest.raises(ValueError, match="unknown"):
        transform("money", [1.0])


def test_frozen_scenario_ids_are_all_implemented():
    path = (
        Path(__file__).parents[1]
        / "configs"
        / "cycles"
        / "s1-v2-20260814"
        / "pseudo_utility_scenarios.json"
    )
    config = json.loads(path.read_text(encoding="utf-8"))
    parameters = fit_global_parameters([-1.0, 0.0, 2.0])
    for scenario in config["scenarios"]:
        values = transform(scenario["scenario_id"], [-1.0, 0.0, 2.0], parameters)
        assert np.all(np.isfinite(values))
        assert np.all(values >= 0.0)
