"""Shared pytest configuration.

Test-speed cap: the suite's wall time is dominated by a handful of tests that fit real CatBoost/LightGBM
models. Their purpose is plumbing / correctness / serialization — NOT predictive quality — so we cap the
boosting tree-count for the whole test session (production defaults in ``adapters.boosting`` are untouched;
this only rebinds the module globals the fit path reads). One boosting fit drops ~10x, with no change to
what any test asserts. Remove this to profile against production tree-counts.
"""

from __future__ import annotations

import honestml.adapters.boosting as _boosting

# production: 300 (fixed) / 1000 (early-stopping ceiling, ADR-0080). Tests need a model that fits and
# round-trips, not one that converges, so a small count is equivalent for them and far cheaper.
_boosting._N_ESTIMATORS = 30
_boosting._N_ESTIMATORS_ES = 60
