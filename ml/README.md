# Modeling

This folder owns model training, evaluation, and prediction code. The `etl/`
folder prepares canonical games, polls, Elo state, and feature snapshots;
`supabase/migrations/` owns the database schema; and `artifacts/` stores local
exports and trained model files.

Run training from the repository root with:

```bash
python -m ml.train --root artifacts/history --output artifacts/models
```

The current experiment compares persistence with Ridge and histogram gradient
boosting. XGBoost or CatBoost can be added after this time ordered baseline is
established.
