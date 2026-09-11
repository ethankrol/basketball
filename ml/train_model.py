"""Train chronological AP-poll models from historical backfill artifacts.

The script intentionally keeps the last two seasons out of model selection and
reports them as an untouched final test. It trains numeric regressors only; team
names/IDs are excluded to avoid memorizing historical poll reputation.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import pandas as pd

from etl.features import FEATURE_VERSION, content_hash


EXCLUDE = {"run_id", "season", "target_week", "cutoff_date_exclusive", "previous_poll_date",
           "feature_version", "team_id", "team_name"}
TARGET = "actual_score"


def load_rows(root, start=2003, end=2026):
    rows = []
    for path in sorted(root.glob("*/runs/*/*/features.json")):
        try:
            season = int(path.parts[-5])
        except (ValueError, IndexError):
            continue
        if not start <= season <= end:
            continue
        directory = path.parent
        report = json.loads((directory / "report.json").read_text())
        predictions = { (p["run_id"], p["team_id"]): p
                        for p in json.loads((directory / "predictions.json").read_text()) }
        for row in json.loads(path.read_text()):
            p = predictions.get((row["run_id"], row["team_id"]))
            if p is None or report["feature_version"] != FEATURE_VERSION:
                raise ValueError(f"Missing/mismatched prediction for {path}")
            rows.append({**row, TARGET: p[TARGET], "actual_rank": p["actual_rank"]})
    if not rows:
        raise ValueError("No feature artifacts found")
    data = pd.DataFrame(rows)
    data = data.drop_duplicates(["season", "target_week", "team_id"], keep="last")
    return data.sort_values(["season", "target_week", "team_id"]).reset_index(drop=True)


def metrics(frame, prediction):
    values = frame[["season", "target_week", "team_id", TARGET]].copy()
    values["prediction"] = np.clip(prediction, 0, 1)
    scores = []
    errors = []
    for _, poll in values.groupby(["season", "target_week"], sort=True):
        actual = set(poll.nlargest(25, TARGET).team_id)
        predicted = set(poll.nlargest(25, "prediction").team_id)
        scores.append(len(actual & predicted))
        errors.extend((poll.prediction - poll[TARGET]).to_numpy())
    err = np.asarray(errors)
    return {"polls": len(scores), "mean_top25_overlap": float(np.mean(scores)),
            "mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean(err ** 2)))}


def train(data, test_seasons=2, seed=42):
    seasons = sorted(data.season.unique())
    if len(seasons) <= test_seasons + 1:
        raise ValueError("Need more seasons for chronological train/test")
    test = seasons[-test_seasons:]
    validation = seasons[-test_seasons-1:-test_seasons]
    train = [s for s in seasons if s not in set(test + validation)]
    features = sorted(set(data.columns) - EXCLUDE - {TARGET, "actual_rank"})
    X = data[features].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    # Fit imputation on training rows only. Explicit missing indicators retain
    # the difference between no games and a missing source statistic.
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy="median", add_indicator=True)
    train_mask = data.season.isin(train)
    X_train = imputer.fit_transform(X.loc[train_mask])
    X_val = imputer.transform(X.loc[data.season.isin(validation)])
    X_test = imputer.transform(X.loc[data.season.isin(test)])
    y = data[TARGET].to_numpy()
    prior = data["previous_normal_points"].to_numpy()
    y_delta = y - prior

    from sklearn.linear_model import Ridge
    from sklearn.ensemble import HistGradientBoostingRegressor
    models = {
        "ridge_score": (Ridge(alpha=10.0), y[train_mask]),
        "hgb_score": (HistGradientBoostingRegressor(max_iter=300, max_leaf_nodes=15,
                                    learning_rate=.04, l2_regularization=5,
                                    random_state=seed), y[train_mask]),
        "hgb_delta": (HistGradientBoostingRegressor(max_iter=300, max_leaf_nodes=15,
                                    learning_rate=.04, l2_regularization=5,
                                    random_state=seed), y_delta[train_mask]),
    }
    report = {"feature_version": FEATURE_VERSION, "train_seasons": train,
              "validation_seasons": validation, "test_seasons": test, "features": features,
              "models": {}, "target": TARGET, "seed": seed}
    fitted = {}
    for name, (model, target) in models.items():
        model.fit(X_train, target)
        val_pred = model.predict(X_val)
        if name == "hgb_delta": val_pred += prior[data.season.isin(validation)]
        val_frame = data[data.season.isin(validation)]
        test_pred = model.predict(X_test)
        if name == "hgb_delta": test_pred += prior[data.season.isin(test)]
        report["models"][name] = {"validation": metrics(val_frame, val_pred),
                                  "test": metrics(data[data.season.isin(test)], test_pred)}
        fitted[name] = model
    # Select on validation overlap, with MAE as tie-breaker, then refit selected
    # model on train + validation before the untouched test is reported.
    selected = max(report["models"], key=lambda n: (report["models"][n]["validation"]["mean_top25_overlap"],
                                                     -report["models"][n]["validation"]["mae"]))
    refit_mask = data.season.isin(train + validation)
    X_refit = imputer.fit_transform(X.loc[refit_mask])
    X_test = imputer.transform(X.loc[data.season.isin(test)])
    target = y[refit_mask] - prior[refit_mask] if selected == "hgb_delta" else y[refit_mask]
    final_model = models[selected][0].__class__(**models[selected][0].get_params())
    final_model.fit(X_refit, target)
    report["selected_model"] = selected
    report["test_selected_after_refit"] = metrics(data[data.season.isin(test)],
        final_model.predict(X_test) + (prior[data.season.isin(test)] if selected == "hgb_delta" else 0))
    report["baseline_persistence_test"] = metrics(data[data.season.isin(test)], prior[data.season.isin(test)])
    return final_model, imputer, features, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("artifacts/history"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/model"))
    parser.add_argument("--start", type=int, default=2003)
    parser.add_argument("--end", type=int, default=2026)
    args = parser.parse_args()
    data = load_rows(args.root, args.start, args.end)
    model, imputer, features, report = train(data)
    args.output.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump({"model": model, "imputer": imputer, "features": features,
                 "target": report["target"], "model_name": report["selected_model"]},
                args.output / "model.joblib")
    (args.output / "features.json").write_text(json.dumps(features, indent=2) + "\n")
    # Pandas/numpy scalars can appear in metric values; normalize them before
    # hashing and writing the report so the artifact is portable across Python
    # environments.
    report = json.loads(json.dumps(report, default=lambda value: value.item()
                                   if hasattr(value, "item") else str(value)))
    report["artifact_hash"] = content_hash(report)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
