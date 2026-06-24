# Byte-frozen native artifacts

Committed artifacts with native model bodies (xgb `.ubj`, cat `.cbm`, lgbm `.txt`) + recorded
expected predictions. `test_native_serialization::test_byte_frozen_fixture_loads` reloads them at
runtime: a boosting-library bump that breaks native `load_model` (or shifts predictions) **fails CI**
— a same-version round-trip (or pickle) could not detect that.

Regenerate (then commit + update the versions below):

```
uv run python tests/fixtures/native_artifacts/generate.py
```

**Generated under (recorded versions):** xgboost 3.2.0 · catboost 1.2.10 · lightgbm 4.6.0 ·
scikit-learn 1.7.2 · numpy 2.2.6 · 2026-06-12.
