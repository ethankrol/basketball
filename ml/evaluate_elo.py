"""Compare Elo configurations using chronological rolling validation.

This intentionally rebuilds local histories for each configuration and never
publishes them. The final two seasons remain the untouched test set.
"""

import argparse
import json
from pathlib import Path
import shutil

from etl.backfill_history import run_history
from ml.train_model import load_rows, train


def evaluate(source_root, output_root, configs, history_config, start=2001, end=2026):
    output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for config in configs:
        name = f"k{config['k']}-{'mov' if config['mov'] else 'nomov'}"
        root = output_root / name / "history"
        for season in range(start, end + 1):
            destination = root / str(season)
            destination.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_root / str(season) / "source.json", destination / "source.json")
        manifest = run_history(root, start, end, history_config,
                               carryover=config.get("carryover", .75), elo_k=config["k"],
                               margin_of_victory=config["mov"])
        data = load_rows(root, start, end)
        _, _, _, _, _, report = train(data)
        rolling = report["rolling_validation_summary"][report["selected_model"]]
        summaries.append({"name": name, "k": config["k"], "margin_of_victory": config["mov"],
                          "rolling_mean_rank_error": rolling["top25_mean_absolute_rank_error"],
                          "rolling_exact_rank_accuracy": rolling["top25_weekly_exact_rank_accuracy"],
                          "test": report["test_selected_after_refit"], "selected_model": report["selected_model"],
                          "manifest": str(output_root / name / f"manifest_{start}_{end}.json")})
        (output_root / name / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        (output_root / name / f"manifest_{start}_{end}.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output_root / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("artifacts/history"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/elo_experiments"))
    parser.add_argument("--config", type=Path, default=Path("etl/configs/history.json"))
    args = parser.parse_args()
    configs = [{"k": k, "mov": mov} for k in (20, 25, 30, 40) for mov in (False, True)]
    print(json.dumps(evaluate(args.source_root, args.output_root, configs,
                              json.loads(args.config.read_text())), indent=2))


if __name__ == "__main__":
    main()
