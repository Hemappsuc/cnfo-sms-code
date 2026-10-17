import argparse
import time

from common import add_arguments, pending_count, run_cell, validate_args
from run_api import ApiEngine, add_api_arguments, load_models
from run_local import LocalEngine, add_local_arguments


def main():
    parser = argparse.ArgumentParser(description="Repeat missing and invalid outputs until all parse successfully")
    add_arguments(parser)
    add_api_arguments(parser)
    add_local_arguments(parser)
    parser.add_argument("--backend", choices=("local", "api"), required=True)
    parser.add_argument("--api-models", nargs="+")
    parser.add_argument("--max-rounds", type=int, default=0, help="0 means unlimited")
    parser.add_argument("--retry-delay", type=float, default=5)
    args = parser.parse_args()
    if args.max_rounds < 0 or args.retry_delay < 0 or args.timeout <= 0:
        parser.error("Invalid retry or timeout setting")
    datasets = validate_args(args)
    configs = load_models(args.api_config) if args.backend == "api" else {}
    names = list(dict.fromkeys(args.api_models or configs)) if configs else [f"Qwen3-{s}" for s in args.models]
    if configs and set(names) - configs.keys():
        parser.error("Unknown API model ID; see api_models.json")
    if args.validate_only:
        return
    unresolved = 0
    for name in names:
        if not pending_count(args, args.backend, name, datasets):
            print(f"{name}: all records are valid", flush=True)
            continue
        engine = (ApiEngine(name, configs[name], args) if configs
                  else LocalEngine(name.removeprefix("Qwen3-"), args))
        try:
            round_number = 0
            while True:
                round_number += 1
                remaining = sum(run_cell(args, engine, language, datasets, retry=True)
                                for language in args.languages)
                print(f"{name}: round={round_number} invalid={remaining}", flush=True)
                if not remaining:
                    break
                if args.max_rounds and round_number >= args.max_rounds:
                    unresolved += remaining
                    break
                time.sleep(args.retry_delay)
        finally:
            engine.close()
    if unresolved:
        raise SystemExit(f"{unresolved} invalid outputs remain; all attempts were saved")


if __name__ == "__main__":
    main()
