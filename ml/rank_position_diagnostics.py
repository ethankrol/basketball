"""Per-position diagnostics for the nested chronological poll evaluation."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.train_model import load_rows
from ml.walk_forward_experiments import CONFIGS, feature_sets, fit_predict


UNRANKED = 26


def prediction_records(frame, prediction):
    """Return one record for every official Top-25 team in each poll.

    A team outside the predicted Top 25 is assigned the confusion-matrix bucket
    26, while position-level errors retain its ordinal rank among all teams in
    the prediction pool. Retention therefore remains distinct from within-one.
    """
    values = frame[["season", "target_week", "team_id", "actual_rank"]].copy()
    # Match the core metrics exactly: vote-share predictions are constrained to
    # the valid [0, 1] range before ties are broken deterministically by team ID.
    values["prediction"] = np.clip(np.asarray(prediction), 0, 1)
    records = []
    for (season, week), poll in values.groupby(["season", "target_week"], sort=True):
        ordered = poll.sort_values(["prediction", "team_id"], ascending=[False, True])
        ranks = {team_id: rank for rank, team_id in enumerate(ordered.team_id, 1)}
        for row in poll[poll.actual_rank.between(1, 25)].itertuples(index=False):
            raw_rank = ranks[row.team_id]
            predicted_rank = raw_rank if raw_rank <= 25 else UNRANKED
            records.append({"season": int(season), "week": int(week),
                            "poll_id": f"{int(season)}-{int(week)}",
                            "team_id": row.team_id, "actual_rank": int(row.actual_rank),
                            "predicted_rank": int(predicted_rank),
                            "raw_predicted_rank": int(raw_rank),
                            "retained": bool(raw_rank <= 25),
                            "rank_error": int(raw_rank - row.actual_rank)})
    return records


def bootstrap_rate_ci(values, poll_ids=None, samples=2000, seed=42):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return [None, None]
    if poll_ids is not None:
        frame = pd.DataFrame({"value": values, "poll_id": poll_ids})
        # Every rank region contributes the same number of rows per poll, so a
        # mean within poll followed by resampling polls is a cluster bootstrap.
        values = frame.groupby("poll_id", sort=False).value.mean().to_numpy()
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    estimates = values[indices].mean(axis=1)
    return [float(value) for value in np.quantile(estimates, [.025, .975])]


def summarize_rows(rows, bootstrap_samples=2000):
    errors = np.asarray([row["rank_error"] for row in rows], dtype=float)
    absolute = np.abs(errors)
    exact = absolute == 0
    within_one = absolute <= 1
    within_two = absolute <= 2
    retained = np.asarray([row["retained"] for row in rows], dtype=float)
    return {"exact_accuracy": float(exact.mean()),
            "exact_accuracy_95pct_ci": bootstrap_rate_ci(
                exact, [row["poll_id"] for row in rows], bootstrap_samples),
            "within_one_accuracy": float(within_one.mean()),
            "within_two_accuracy": float(within_two.mean()),
            "mean_absolute_rank_error": float(absolute.mean()),
            "median_absolute_rank_error": float(np.median(absolute)),
            "mean_signed_error": float(errors.mean()),
            "top25_retention": float(retained.mean()),
            "observations": int(len(rows))}


def aggregate(records, bootstrap_samples=2000):
    per_rank = []
    for rank in range(1, 26):
        rows = [row for row in records if row["actual_rank"] == rank]
        per_rank.append({"actual_rank": rank, **summarize_rows(rows, bootstrap_samples)})
    for row in per_rank:
        low, high = max(1, row["actual_rank"] - 2), min(25, row["actual_rank"] + 2)
        nearby = [record for record in records if low <= record["actual_rank"] <= high]
        row["smoothed_exact_accuracy"] = float(np.mean(
            [record["rank_error"] == 0 for record in nearby]))

    regions = []
    for name, low, high in (("1-5", 1, 5), ("6-10", 6, 10), ("11-15", 11, 15),
                            ("16-20", 16, 20), ("21-25", 21, 25)):
        rows = [row for row in records if low <= row["actual_rank"] <= high]
        regions.append({"rank_region": name, **summarize_rows(rows, bootstrap_samples)})

    matrix = [[0 for _ in range(26)] for _ in range(25)]
    for row in records:
        matrix[row["actual_rank"] - 1][row["predicted_rank"] - 1] += 1
    normalized = [[count / sum(line) if sum(line) else 0 for count in line] for line in matrix]
    return {"overall": summarize_rows(records, bootstrap_samples),
            "rank_regions": regions, "per_position": per_rank,
            "confusion_matrix": {"actual_ranks": list(range(1, 26)),
                                 "predicted_ranks": list(range(1, 26)) + ["unranked"],
                                 "counts": matrix, "row_normalized": normalized}}


def _svg_start(width, height, title, description):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
            f'<title id="title">{title}</title><desc id="desc">{description}</desc>',
            '<rect width="100%" height="100%" fill="#fbfaf7"/>']


def write_accuracy_svg(per_rank, path, title="Accuracy by actual rank",
                       subtitle="Nested poll-level walk-forward evaluation, 2015–2024; shaded band is the 95% bootstrap interval for exact accuracy"):
    width, height = 1120, 650
    left, right, top, bottom = 85, 35, 75, 85
    plot_w, plot_h = width - left - right, height - top - bottom
    x = lambda rank: left + (rank - 1) * plot_w / 24
    y = lambda value: top + (1 - value) * plot_h
    svg = _svg_start(width, height, "Accuracy by actual AP poll rank",
                     "Exact, within-one, within-two, smoothed exact, and Top-25 retention rates.")
    svg += [f'<text x="{left}" y="38" font-family="sans-serif" font-size="25" font-weight="700" fill="#172554">{title}</text>',
            f'<text x="{left}" y="61" font-family="sans-serif" font-size="13" fill="#475569">{subtitle}</text>']
    for value in np.linspace(0, 1, 6):
        py = y(value)
        svg += [f'<line x1="{left}" y1="{py:.1f}" x2="{width-right}" y2="{py:.1f}" stroke="#dbe2ea"/>',
                f'<text x="{left-12}" y="{py+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="12" fill="#64748b">{value:.0%}</text>']
    points_low = " ".join(f"{x(row['actual_rank']):.1f},{y(row['exact_accuracy_95pct_ci'][0]):.1f}" for row in per_rank)
    points_high = " ".join(f"{x(row['actual_rank']):.1f},{y(row['exact_accuracy_95pct_ci'][1]):.1f}" for row in reversed(per_rank))
    svg.append(f'<polygon points="{points_low} {points_high}" fill="#2563eb" opacity="0.14"/>')
    series = [("exact_accuracy", "Exact", "#1d4ed8", ""),
              ("smoothed_exact_accuracy", "Smoothed exact (±2 ranks)", "#7c3aed", "6 4"),
              ("within_one_accuracy", "Within one", "#059669", ""),
              ("within_two_accuracy", "Within two", "#d97706", ""),
              ("top25_retention", "Top-25 retention", "#be123c", "5 4")]
    for field, label, color, dash in series:
        points = " ".join(f"{x(row['actual_rank']):.1f},{y(row[field]):.1f}" for row in per_rank)
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        svg.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.7"{dash_attr}/>' )
    for rank in range(1, 26):
        px = x(rank)
        svg += [f'<line x1="{px:.1f}" y1="{height-bottom}" x2="{px:.1f}" y2="{height-bottom+5}" stroke="#64748b"/>',
                f'<text x="{px:.1f}" y="{height-bottom+22}" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#475569">{rank}</text>']
    svg += [f'<text x="{left+plot_w/2}" y="{height-24}" text-anchor="middle" font-family="sans-serif" font-size="14" fill="#334155">Actual rank</text>',
            f'<text x="22" y="{top+plot_h/2}" transform="rotate(-90 22 {top+plot_h/2})" text-anchor="middle" font-family="sans-serif" font-size="14" fill="#334155">Accuracy / retention</text>']
    legend_x, legend_y = left + 15, top + 17
    for index, (_, label, color, dash) in enumerate(series):
        lx = legend_x + (index % 3) * 300
        ly = legend_y + (index // 3) * 25
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        svg += [f'<line x1="{lx}" y1="{ly}" x2="{lx+28}" y2="{ly}" stroke="{color}" stroke-width="3"{dash_attr}/>',
                f'<text x="{lx+36}" y="{ly+4}" font-family="sans-serif" font-size="12" fill="#334155">{label}</text>']
    svg.append('</svg>')
    path.write_text("\n".join(svg) + "\n")


def write_error_svg(per_rank, path):
    width, height = 1120, 560
    left, right, top, bottom = 85, 35, 70, 80
    plot_w, plot_h = width - left - right, height - top - bottom
    max_value = max(3.0, max(row["mean_absolute_rank_error"] for row in per_rank) * 1.1,
                    max(abs(row["mean_signed_error"]) for row in per_rank) * 1.1)
    x = lambda rank: left + (rank - 1) * plot_w / 24
    y = lambda value: top + (max_value - value) * plot_h / (2 * max_value)
    zero = y(0)
    svg = _svg_start(width, height, "Rank error by actual AP poll rank",
                     "Mean absolute and signed error using ordinal rank among all predicted teams.")
    svg += [f'<text x="{left}" y="36" font-family="sans-serif" font-size="25" font-weight="700" fill="#172554">Rank error by actual rank</text>',
            f'<text x="{left}" y="58" font-family="sans-serif" font-size="13" fill="#475569">Signed error = predicted rank − actual rank; positive means ranked too low</text>',
            f'<line x1="{left}" y1="{zero:.1f}" x2="{width-right}" y2="{zero:.1f}" stroke="#64748b" stroke-width="1.4"/>']
    for row in per_rank:
        px = x(row["actual_rank"])
        bar_y = y(row["mean_absolute_rank_error"])
        svg += [f'<rect x="{px-8:.1f}" y="{bar_y:.1f}" width="16" height="{zero-bar_y:.1f}" fill="#93c5fd" opacity="0.85"/>',
                f'<text x="{px:.1f}" y="{height-bottom+21}" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#475569">{row["actual_rank"]}</text>']
    points = " ".join(f"{x(row['actual_rank']):.1f},{y(row['mean_signed_error']):.1f}" for row in per_rank)
    svg.append(f'<polyline points="{points}" fill="none" stroke="#be123c" stroke-width="3"/>')
    for row in per_rank:
        svg.append(f'<circle cx="{x(row["actual_rank"]):.1f}" cy="{y(row["mean_signed_error"]):.1f}" r="3" fill="#be123c"/>')
    for value in np.linspace(-max_value, max_value, 7):
        py = y(value)
        svg.append(f'<text x="{left-12}" y="{py+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="12" fill="#64748b">{value:.1f}</text>')
    svg += [f'<text x="{left+plot_w/2}" y="{height-22}" text-anchor="middle" font-family="sans-serif" font-size="14" fill="#334155">Actual rank</text>',
            f'<rect x="{left+15}" y="{top+10}" width="18" height="12" fill="#93c5fd"/><text x="{left+42}" y="{top+21}" font-family="sans-serif" font-size="12" fill="#334155">Mean absolute error</text>',
            f'<line x1="{left+200}" y1="{top+16}" x2="{left+228}" y2="{top+16}" stroke="#be123c" stroke-width="3"/><text x="{left+237}" y="{top+21}" font-family="sans-serif" font-size="12" fill="#334155">Mean signed error</text>', '</svg>']
    path.write_text("\n".join(svg) + "\n")


def write_confusion_svg(confusion, path):
    matrix = np.asarray(confusion["row_normalized"])
    cell, left, top = 31, 95, 85
    width, height = left + 26 * cell + 75, top + 25 * cell + 90
    svg = _svg_start(width, height, "Actual versus predicted rank confusion matrix",
                     "Row-normalized 25 by 26 confusion matrix including an unranked prediction column.")
    svg += [f'<text x="{left}" y="34" font-family="sans-serif" font-size="24" font-weight="700" fill="#172554">Rank confusion matrix</text>',
            f'<text x="{left}" y="57" font-family="sans-serif" font-size="13" fill="#475569">Cell color is the share of teams at each actual rank; U means predicted outside the Top 25</text>']
    for row in range(25):
        for col in range(26):
            value = matrix[row, col]
            intensity = min(1.0, value / .55)
            red = round(239 - 209 * intensity)
            green = round(246 - 117 * intensity)
            blue = round(255 - 55 * intensity)
            svg.append(f'<rect x="{left+col*cell}" y="{top+row*cell}" width="{cell-1}" height="{cell-1}" fill="rgb({red},{green},{blue})"><title>Actual {row+1}, predicted {col+1 if col < 25 else "unranked"}: {value:.1%}</title></rect>')
        svg.append(f'<text x="{left-10}" y="{top+row*cell+20}" text-anchor="end" font-family="sans-serif" font-size="10" fill="#475569">{row+1}</text>')
    for col in range(26):
        label = str(col + 1) if col < 25 else "U"
        svg.append(f'<text x="{left+col*cell+15}" y="{top+25*cell+18}" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#475569">{label}</text>')
    svg += [f'<text x="{left+13*cell}" y="{height-20}" text-anchor="middle" font-family="sans-serif" font-size="14" fill="#334155">Predicted rank</text>',
            f'<text x="24" y="{top+12.5*cell}" transform="rotate(-90 24 {top+12.5*cell})" text-anchor="middle" font-family="sans-serif" font-size="14" fill="#334155">Actual rank</text>', '</svg>']
    path.write_text("\n".join(svg) + "\n")


def write_csv(rows, path):
    fields = ["actual_rank", "exact_accuracy", "exact_accuracy_95pct_ci_low",
              "exact_accuracy_95pct_ci_high", "smoothed_exact_accuracy",
              "within_one_accuracy", "within_two_accuracy", "mean_absolute_rank_error",
              "median_absolute_rank_error", "mean_signed_error", "top25_retention", "observations"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            output = dict(row)
            output["exact_accuracy_95pct_ci_low"], output["exact_accuracy_95pct_ci_high"] = output.pop("exact_accuracy_95pct_ci")
            writer.writerow({field: output[field] for field in fields})


def run(data, nested_report, bootstrap_samples=2000):
    experiment = data[data.season <= 2024].copy()
    sets = feature_sets(experiment)
    selected = {int(row["season"]): row["selected_candidate"]
                for row in nested_report["outer_results"]}
    records = []
    for season in range(2015, 2025):
        config = CONFIGS[selected[season]]
        for week in sorted(experiment.loc[experiment.season.eq(season), "target_week"].unique()):
            train_mask = (experiment.season.isin(range(season - 4, season)) |
                          (experiment.season.eq(season) & experiment.target_week.lt(week)))
            validation_mask = experiment.season.eq(season) & experiment.target_week.eq(week)
            prediction, _ = fit_predict(experiment, sets[config["features"]], config,
                                        train_mask, validation_mask)
            records.extend(prediction_records(experiment.loc[validation_mask], prediction))
    report = aggregate(records, bootstrap_samples)
    report["protocol"] = {"evaluation_seasons": [2015, 2024], "polls": len(set(
        row["poll_id"] for row in records)), "selected_model_per_season": selected,
        "excluded_holdout_seasons": [2025, 2026], "bootstrap_samples": bootstrap_samples,
        "bootstrap_unit": "poll",
        "rank_error_convention": "ordinal predicted rank among all teams in the poll prediction pool",
        "confusion_unranked_convention": "predicted ranks outside the Top 25 are grouped into column 26",
        "signed_error_definition": "predicted rank - actual rank",
        "positive_signed_error_means": "model ranked the team too low"}
    report["protocol"]["rank_tie_note"] = (
        "Official poll ties can produce more than one observation at a rank and can skip ranks.")
    report["prediction_records"] = records
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("artifacts/history"))
    parser.add_argument("--nested-report", type=Path,
                        default=Path("artifacts/model_experiments/walk_forward_nested.json"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("artifacts/model_experiments/rank_position"))
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()
    report = run(load_rows(args.root, 2003, 2026),
                 json.loads(args.nested_report.read_text()), args.bootstrap_samples)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "rank_position_diagnostics.json").write_text(json.dumps(report, indent=2) + "\n")
    write_csv(report["per_position"], args.output_dir / "rank_position_metrics.csv")
    write_accuracy_svg(report["per_position"], args.output_dir / "rank_accuracy_by_position.svg")
    write_error_svg(report["per_position"], args.output_dir / "rank_error_by_position.svg")
    write_confusion_svg(report["confusion_matrix"], args.output_dir / "rank_confusion_matrix.svg")
    print(json.dumps({key: report[key] for key in ("protocol", "overall", "rank_regions")}, indent=2))


if __name__ == "__main__":
    main()
