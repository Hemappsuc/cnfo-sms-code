import argparse
import csv
from pathlib import Path

from common import (
    GROUPS, INSERTIONS, LABELS, OBFUSCATIONS, ROOT, check_records, digest,
    load_data, parse_label, read_json, read_records, valid_record, write_json,
)


def metrics(rows):
    confusion = {label: {p: 0 for p in (*LABELS, "INVALID")} for label in LABELS}
    for row in rows:
        pred = row.get("pred_label")
        confusion[row["label"]][pred if pred in LABELS else "INVALID"] += 1
    per_class = {}
    for label in LABELS:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in LABELS if other != label)
        fn = sum(confusion[label].values()) - tp
        per_class[label] = {
            "support": tp + fn, "tp": tp, "fp": fp, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        }
    n = len(rows)
    invalid = sum(confusion[label]["INVALID"] for label in LABELS)
    return {
        "n": n, "accuracy": sum(confusion[label][label] for label in LABELS) / n if n else 0.0,
        "macro_f1": sum(c["f1"] for c in per_class.values()) / 8,
        "invalid": invalid, "invalid_rate": invalid / n if n else 0.0,
        "per_class": per_class, "confusion": confusion,
    }


def audit_cell(directory, datasets, allow_partial=False):
    manifest = read_json(directory / "manifest.json")
    if (manifest["data_sha256"] != {g: digest(rows) for g, rows in datasets.items()}
            or manifest["counts"] != {g: len(rows) for g, rows in datasets.items()}):
        raise ValueError(f"Dataset/selection differs from this run: {directory}")
    language = manifest["language"]
    grouped, missing, attempts = {}, {}, {}
    for group, samples in datasets.items():
        records = read_records(directory / f"{group}.jsonl")
        check_records(records, samples)
        missing[group] = len(samples) - len(records)
        if missing[group] and not allow_partial:
            raise ValueError(f"Missing {missing[group]} records in {directory}/{group}; resume inference first")
        grouped[group] = []
        attempts[group] = sum(r.get("attempt", 1) for r in records.values())
        for sample in samples:
            record = records.get(sample["id"])
            if record and record.get("parse_ok") != valid_record(record, language):
                raise ValueError(f"Inconsistent parse status: {directory}/{group}/{sample['id']}")
            pred = parse_label(record.get("answer"), language) if record and valid_record(record, language) else None
            grouped[group].append({**sample, "pred_label": pred})
    scenarios = {group: metrics(grouped[group] + grouped["normal"])
                 for group in GROUPS if group != "normal"}
    baseline = scenarios["fraud"]["macro_f1"]
    average7 = sum(scenarios[g]["macro_f1"] for g in OBFUSCATIONS) / 7
    average_ins = sum(scenarios[g]["macro_f1"] for g in INSERTIONS) / 2
    fraud_labels = LABELS[1:]
    missed = {group: sum(r["pred_label"] not in fraud_labels for r in grouped[group]) / len(grouped[group])
              for group in scenarios}
    false_positive = sum(r["pred_label"] in fraud_labels for r in grouped["normal"]) / len(grouped["normal"])
    summary = {
        "backend": manifest["backend"], "model": manifest["model"], "language": language,
        "complete": not any(missing.values()), "subset": manifest["limit"] is not None,
        "Base": baseline, **{g: scenarios[g]["macro_f1"] for g in (*OBFUSCATIONS, *INSERTIONS)},
        "Avg7": average7, "Delta7": baseline - average7,
        "AvgIns": average_ins, "DeltaIns": baseline - average_ins,
        "mean_missed_fraud_7": sum(missed[g] for g in OBFUSCATIONS) / 7,
        "false_positive_rate": false_positive,
        "invalid_unique": sum(r["pred_label"] is None for rows in grouped.values() for r in rows),
    }
    report = {
        "summary": summary, "missing": missing, "attempts": attempts,
        "eight_class_scenarios": scenarios,
        "baseline_relative_loss": {g: baseline - result["macro_f1"] for g, result in scenarios.items()},
        "missed_fraud_rate": missed, "normal_only": metrics(grouped["normal"]),
    }
    return report


def main():
    parser = argparse.ArgumentParser(description="Compute the paper's eight-class metrics and screening errors")
    parser.add_argument("--data", type=Path, default=ROOT / "dataset")
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    datasets = load_data(args.data, args.limit)
    manifests = sorted(args.results.glob("*/*/*/manifest.json"))
    if not manifests:
        parser.error("No results found")
    reports = [(p.parent, audit_cell(p.parent, datasets, args.allow_partial)) for p in manifests]
    summaries = []
    for directory, report in reports:
        write_json(directory / "metrics.json", report)
        summary = report["summary"]
        summaries.append(summary)
        print(f"{summary['model']}/{summary['language']} Base={summary['Base']:.4f} "
              f"Avg7={summary['Avg7']:.4f} invalid={summary['invalid_unique']}")
    write_json(args.results / "summary.json", summaries)
    with (args.results / "summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)


if __name__ == "__main__":
    main()
