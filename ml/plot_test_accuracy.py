"""Plot held-out test accuracy over time and by actual rank."""

import argparse
import json
from pathlib import Path

import numpy as np

from ml.rank_position_diagnostics import aggregate, write_accuracy_svg


def write_time_svg(rows, path):
    width, height = 1200, 650
    left, right, top, bottom = 80, 35, 70, 90
    plot_w, plot_h = width - left - right, height - top - bottom
    x = lambda i: left + i * plot_w / max(1, len(rows) - 1)
    y = lambda value: top + (1 - value) * plot_h
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
           '<title>Held-out accuracy throughout the test period</title>',
           '<desc>Weekly exact, within-one, within-two, and Top-25 overlap accuracy for the untouched test seasons.</desc>',
           '<rect width="100%" height="100%" fill="#fbfaf7"/>',
           f'<text x="{left}" y="36" font-family="sans-serif" font-size="25" font-weight="700" fill="#172554">Held-out accuracy over time</text>',
           f'<text x="{left}" y="58" font-family="sans-serif" font-size="13" fill="#475569">Each point is one test poll; the vertical marker separates test seasons</text>']
    for value in np.linspace(0, 1, 6):
        py = y(value)
        svg += [f'<line x1="{left}" y1="{py:.1f}" x2="{width-right}" y2="{py:.1f}" stroke="#dbe2ea"/>',
                f'<text x="{left-12}" y="{py+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="12" fill="#64748b">{value:.0%}</text>']
    if len(rows) > 1:
        for index in range(1, len(rows)):
            if rows[index]["season"] != rows[index-1]["season"]:
                px = (x(index) + x(index-1)) / 2
                svg.append(f'<line x1="{px:.1f}" y1="{top}" x2="{px:.1f}" y2="{height-bottom}" stroke="#94a3b8" stroke-dasharray="5 4"/>')
    series = [("top25_exact_rank_accuracy", "Exact", "#1d4ed8"),
              ("top25_within_one_rank_accuracy", "Within one", "#059669"),
              ("top25_within_two_rank_accuracy", "Within two", "#d97706"),
              ("overlap_rate", "Top-25 overlap", "#be123c")]
    for field, label, color in series:
        points = " ".join(f"{x(i):.1f},{y(row[field]):.1f}" for i, row in enumerate(rows))
        svg.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.7"/>')
        for i, row in enumerate(rows):
            svg.append(f'<circle cx="{x(i):.1f}" cy="{y(row[field]):.1f}" r="2.5" fill="{color}"/>')
    for i, row in enumerate(rows):
        px = x(i)
        svg += [f'<line x1="{px:.1f}" y1="{height-bottom}" x2="{px:.1f}" y2="{height-bottom+5}" stroke="#64748b"/>',
                f'<text x="{px:.1f}" y="{height-bottom+20}" transform="rotate(-45 {px:.1f} {height-bottom+20})" text-anchor="end" font-family="sans-serif" font-size="10" fill="#475569">{row["season"]} W{row["week"]}</text>']
    svg += [f'<text x="{left+plot_w/2}" y="{height-20}" text-anchor="middle" font-family="sans-serif" font-size="14" fill="#334155">Test poll</text>',
            f'<text x="22" y="{top+plot_h/2}" transform="rotate(-90 22 {top+plot_h/2})" text-anchor="middle" font-family="sans-serif" font-size="14" fill="#334155">Accuracy / overlap</text>']
    for index, (_, label, color) in enumerate(series):
        lx = left + 20 + index * 245
        ly = top + 18
        svg += [f'<line x1="{lx}" y1="{ly}" x2="{lx+28}" y2="{ly}" stroke="{color}" stroke-width="3"/>',
                f'<text x="{lx+36}" y="{ly+4}" font-family="sans-serif" font-size="12" fill="#334155">{label}</text>']
    svg.append('</svg>')
    path.write_text("\n".join(svg) + "\n")


def run(report_path, output_dir):
    report = json.loads(report_path.read_text())
    rows = []
    for row in report["test_poll_results"]:
        rows.append({**row, "overlap_rate": row["mean_top25_overlap"] / 25})
    rows.sort(key=lambda row: (row["season"], row["week"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "test_accuracy_by_poll.json").write_text(json.dumps(rows, indent=2) + "\n")
    write_time_svg(rows, output_dir / "test_accuracy_over_time.svg")
    rank_report = aggregate(report["test_position_records"], bootstrap_samples=2000)
    (output_dir / "test_rank_position_diagnostics.json").write_text(json.dumps(rank_report, indent=2) + "\n")
    write_accuracy_svg(rank_report["per_position"], output_dir / "test_accuracy_by_rank.svg",
                       title="Held-out accuracy by actual rank",
                       subtitle="Untouched test seasons 2025–2026; shaded band is a poll-bootstrap 95% interval")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=Path("artifacts/models/report.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/model_experiments/test_plots"))
    args = parser.parse_args()
    run(args.report, args.output_dir)


if __name__ == "__main__":
    main()
