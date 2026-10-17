import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SIZES = ("0.6B", "1.7B", "4B", "8B", "14B", "32B")
LABELS = (
    "无", "虚假网络投资理财", "虚假购物服务", "虚假信用服务",
    "冒充电商物流客服", "冒充领导、熟人", "冒充公检法及政府机关", "网络婚恋、交友",
)
TOKENS = (
    "none", "fake_investment", "fake_shopping", "fake_credit",
    "impersonate_ecommerce", "impersonate_acquaintance", "impersonate_authority", "online_dating",
)
GROUPS = {
    "fraud": "01_基线数据_无扰动/基线数据_fraud.json",
    "normal": "01_基线数据_无扰动/基线数据_normal.json",
    "02_Emoji": "02_Emoji/Emoji.json",
    "03_VSim": "03_VSim/VSim.json",
    "04_Split": "04_Split/Split.json",
    "05_Trad": "05_Trad/Trad.json",
    "06_PY_Init": "06_PY_Init/PY_Init.json",
    "07_Homo": "07_Homo/Homo.json",
    "08_Shuff": "08_Shuff/Shuff.json",
    "09_不可见字符插入": "09_不可见字符插入/不可见字符插入.json",
    "10_特殊效果字符插入": "10_特殊效果字符插入/特殊效果字符插入.json",
}
OBFUSCATIONS = tuple(GROUPS)[2:9]
INSERTIONS = tuple(GROUPS)[9:]
SOURCE_FIELDS = ("id", "text", "label", "binary", "perturbation")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(value):
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def load_data(directory, limit=None):
    directory = Path(directory)
    datasets = {}
    for group, relative in GROUPS.items():
        found = [p for p in directory.rglob(Path(relative).name)
                 if p.as_posix().endswith("/" + relative)]
        if len(found) != 1:
            raise ValueError(f"Expected one {relative} under {directory}; found {len(found)}")
        rows = read_json(found[0])
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"Expected a nonempty JSON array: {found[0]}")
        kept, seen = [], set()
        for row in rows:
            if not isinstance(row, dict) or not set(SOURCE_FIELDS) <= row.keys():
                raise ValueError(f"Malformed sample in {group}")
            sid = row["id"]
            if type(sid) is not int or sid in seen:
                raise ValueError(f"Duplicate or non-integer ID in {group}: {sid}")
            seen.add(sid)
            if row["label"] == "其他":
                continue
            if (row["label"] not in LABELS or not isinstance(row["text"], str)
                    or not row["text"].strip()):
                raise ValueError(f"Invalid text or label in {group}/{sid}")
            if ((group == "normal") != (row["label"] == LABELS[0])
                    or row["binary"] != ("normal" if group == "normal" else "fraud")):
                raise ValueError(f"Invalid class in {group}/{sid}")
            kept.append(row)
        if not kept:
            raise ValueError(f"No included samples in {group}")
        datasets[group] = sorted(kept, key=lambda r: r["id"])
    baseline = {r["id"]: r for r in datasets["fraud"]}
    for group in (*OBFUSCATIONS, *INSERTIONS):
        if {r["id"] for r in datasets[group]} != set(baseline):
            raise ValueError(f"Unpaired sample IDs: {group}")
        for row in datasets[group]:
            original = baseline[row["id"]]
            if row["label"] != original["label"] or row.get("original_text") != original["text"]:
                raise ValueError(f"Unpaired label or original_text: {group}/{row['id']}")
    return {g: rows[:limit] if limit else rows for g, rows in datasets.items()}


def load_prompt(language):
    return (ROOT / "scripts" / "prompts" / f"{language}.txt").read_text(
        encoding="utf-8").rstrip("\r\n")


def render_prompt(template, text):
    if template.count("{text}") != 1:
        raise ValueError("Prompt must contain exactly one {text} placeholder")
    return template.replace("{text}", text)


def parse_label(answer, language):
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result

    try:
        obj = json.loads(answer, object_pairs_hook=unique_keys)
    except (TypeError, ValueError):
        return None
    if not isinstance(obj, dict) or set(obj) != {"fraud_type"}:
        return None
    allowed = LABELS if language == "cn" else TOKENS
    value = obj["fraud_type"]
    return LABELS[allowed.index(value)] if isinstance(value, str) and value in allowed else None


def valid_record(record, language):
    label = parse_label(record.get("answer"), language)
    return (record.get("finish_reason") in ("stop", "eos") and label is not None
            and record.get("parse_ok") is True and record.get("pred_label") == label)


def read_records(path):
    path = Path(path)
    latest = {}
    if not path.exists():
        return latest
    with path.open("rb+") as handle:
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            if not line.endswith(b"\n"):
                handle.truncate(offset)
                break
            record = json.loads(line)
            latest[record["id"]] = record
    return latest


def check_records(records, samples):
    source = {r["id"]: r for r in samples}
    for sid, record in records.items():
        if sid not in source or any(record.get(k) != source[sid][k] for k in SOURCE_FIELDS):
            raise ValueError(f"Result/data mismatch for ID {sid}")


def prepare_cell(args, engine, language, datasets):
    prompt = load_prompt(language)
    directory = args.results / engine.backend / engine.name / language
    manifest = {
        "schema_version": 1, "backend": engine.backend, "model": engine.name,
        "language": language, "seed": args.seed, "temperature": 0,
        "batch_size": 1, "concurrency": 1, "max_new_tokens": args.max_new_tokens,
        "prompt": prompt, "prompt_sha256": digest(prompt),
        "data_sha256": {g: digest(rows) for g, rows in datasets.items()},
        "counts": {g: len(rows) for g, rows in datasets.items()},
        "limit": args.limit, "parser": "exact_json_label_v1", "engine": engine.settings,
    }
    path = directory / "manifest.json"
    if path.exists():
        if read_json(path) != manifest:
            raise ValueError(f"Experiment settings changed; use a new results directory: {directory}")
    else:
        write_json(path, manifest)
    return directory, prompt


def run_cell(args, engine, language, datasets, retry=False):
    directory, template = prepare_cell(args, engine, language, datasets)
    remaining = 0
    for group, samples in datasets.items():
        path = directory / f"{group}.jsonl"
        records = read_records(path)
        check_records(records, samples)
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            for sample in samples:
                previous = records.get(sample["id"])
                if previous and (not retry or valid_record(previous, language)):
                    continue
                response = engine.infer(render_prompt(template, sample["text"]))
                pred = parse_label(response.get("answer"), language)
                ok = pred is not None and response["finish_reason"] in ("stop", "eos")
                record = {
                    **{k: sample[k] for k in SOURCE_FIELDS}, **response,
                    "pred_label": pred if ok else None, "parse_ok": ok,
                    "attempt": previous.get("attempt", 0) + 1 if previous else 1,
                }
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                records[sample["id"]] = record
                print(f"{engine.name}/{language}/{group} id={sample['id']} valid={ok}", flush=True)
        remaining += sum(not valid_record(records[r["id"]], language) for r in samples)
    return remaining


def pending_count(args, backend, model, datasets):
    count = 0
    for language in args.languages:
        directory = args.results / backend / model / language
        path = directory / "manifest.json"
        if path.exists():
            manifest = read_json(path)
            expected = {
                "backend": backend, "model": model, "language": language,
                "seed": args.seed, "max_new_tokens": args.max_new_tokens,
                "prompt_sha256": digest(load_prompt(language)), "limit": args.limit,
                "data_sha256": {g: digest(rows) for g, rows in datasets.items()},
            }
            if any(manifest.get(k) != v for k, v in expected.items()):
                raise ValueError(f"Experiment settings changed: {directory}")
        for group, rows in datasets.items():
            records = read_records(directory / f"{group}.jsonl")
            check_records(records, rows)
            count += sum(r["id"] not in records or not valid_record(records[r["id"]], language)
                         for r in rows)
    return count


def add_arguments(parser):
    parser.add_argument("--data", type=Path, default=ROOT / "dataset")
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--languages", nargs="+", choices=("cn", "en"), default=["cn", "en"])
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--validate-only", action="store_true")


def validate_args(args):
    if args.max_new_tokens < 1 or (args.limit is not None and args.limit < 1):
        raise ValueError("Token limit and sample limit must be positive")
    if not 0 <= args.seed < 2**32:
        raise ValueError("Seed must be in [0, 2**32)")
    args.languages = list(dict.fromkeys(args.languages))
    datasets = load_data(args.data, args.limit)
    for language in args.languages:
        render_prompt(load_prompt(language), "")
    print("Samples per group:", {g: len(rows) for g, rows in datasets.items()}, flush=True)
    return datasets
