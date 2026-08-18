# -*- coding:utf-8 -*-
"""
Regression test pinning the R1 fix for Recyclers.__init__.

ABM_CE_PV_Model.recycling_process_change() (called before agents are
created) overwrites self.model.original_recycling_cost for the frelp/asu/
hybrid recycling_process scenarios (e.g. to [0.068, 0.068, 0.068]). Recyclers
must read original_recycling_cost from self.model.original_recycling_cost,
NOT from self.model.config.cost.original_recycling_cost, or that override is
silently skipped and frelp/asu/hybrid parity breaks.

This test uses a lightweight MagicMock model (no full ABM_CE_PV_Model / real
data files required) and patches numpy.random.triangular as a spy so it can
assert exactly which triple of values reached the triangular draw, without
depending on the rest of Recyclers.__init__ succeeding.
"""

from unittest.mock import MagicMock, patch

from ABM_CE_PV_RecyclerAgents import Recyclers

# Distinctive, non-overlapping sentinel triples: MODEL_TRIPLE mimics a
# frelp/asu/hybrid override (~0.068 range); CONFIG_TRIPLE mimics the
# untouched CostConfig.original_recycling_cost default (~400 range).
MODEL_TRIPLE = [0.068, 0.070, 0.069]
CONFIG_TRIPLE = [400.0 - 1e-6, 400.0 + 1e-6, 400.0]


def _make_fake_model() -> MagicMock:
    """Build a minimal fake model exposing only what Recyclers.__init__ touches."""
    model = MagicMock()
    model.original_recycling_cost = MODEL_TRIPLE
    model.config.cost.original_recycling_cost = CONFIG_TRIPLE
    model.config.eol.init_eol_rate = {"recycle": 0.1}
    model.config.cost.recycling_learning_shape_factor = -0.39
    model.original_eol_baseline_volume = 1000.0
    model.repairability = 0.55
    model.num_consumers = 20
    model.recycler_facilities = ["RecyclerA"]
    model.hazardous_waste_regulation_enabled = False
    return model


def test_recycler_draws_original_recycling_cost_from_model_not_config() -> None:
    """
    Recyclers.__init__ must draw from self.model.original_recycling_cost.

    Pins the R1 fix: if the source is ever reverted to
    self.model.config.cost.original_recycling_cost, this test fails because
    np.random.triangular would be called with CONFIG_TRIPLE instead of
    MODEL_TRIPLE, silently breaking frelp/asu/hybrid recycling_process parity.
    """
    model = _make_fake_model()

    with patch("numpy.random.triangular", return_value=0.075) as mock_triangular:
        Recyclers(20, model, MagicMock())

    mock_triangular.assert_called_once_with(
        MODEL_TRIPLE[0],
        MODEL_TRIPLE[2],
        MODEL_TRIPLE[1],
    )
