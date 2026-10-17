import argparse
import importlib.metadata
import os
import re
from pathlib import Path
from urllib.parse import urlsplit

from common import ROOT, add_arguments, read_json, run_cell, validate_args


def load_models(path):
    configs = read_json(path)
    for name, config in configs.items():
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
            raise ValueError(f"Invalid model ID: {name}")
        if config["token_parameter"] not in ("max_tokens", "max_completion_tokens"):
            raise ValueError(f"Invalid token parameter for {name}")
        if set(config["request"]) - {"extra_body", "reasoning_effort"}:
            raise ValueError(f"Unsupported request overrides for {name}")
        extra = config["request"].get("extra_body", {})
        if set(extra) - {"thinking", "enable_thinking"}:
            raise ValueError(f"Unsupported thinking options for {name}")
        if extra.get("enable_thinking") is False or extra.get("thinking", {}).get("type") == "disabled":
            raise ValueError(f"Thinking must remain enabled for {name}")
    return configs


class ApiEngine:
    backend = "api"

    def __init__(self, name, config, args):
        from openai import OpenAI

        self.name, self.config, self.args = name, config, args
        base_url = os.getenv(config["base_url_env"], config["base_url"]).rstrip("/")
        parsed = urlsplit(base_url)
        if (parsed.scheme != "https" or not parsed.hostname or parsed.username
                or parsed.password or parsed.query or parsed.fragment):
            raise ValueError("API base URL must use HTTPS without embedded credentials")
        key = os.getenv(config["api_key_env"], "").strip()
        if not key:
            raise ValueError(f"Set {config['api_key_env']} before running {name}")
        self.client = OpenAI(api_key=key, base_url=base_url, timeout=args.timeout, max_retries=0)
        self.settings = {
            "thinking": True, "base_url": base_url,
            "request": self.options("<prompt>"),
            "allowed_returned_models": config.get("allowed_returned_models", [name]),
            "openai_version": importlib.metadata.version("openai"),
        }

    def options(self, prompt):
        return {
            "model": self.name, "messages": [{"role": "user", "content": prompt}],
            "temperature": 0, "stream": True,
            self.config["token_parameter"]: self.args.max_new_tokens,
            **self.config["request"],
        }

    def infer(self, prompt):
        from httpx import TransportError
        from openai import APIConnectionError, APIStatusError

        content, reasoning, returned = [], [], set()
        finish, usage, response_id = None, None, None
        try:
            with self.client.chat.completions.create(**self.options(prompt)) as stream:
                for chunk in stream:
                    response_id = chunk.id or response_id
                    if chunk.model:
                        returned.add(chunk.model)
                    if chunk.usage:
                        usage = chunk.usage.model_dump()
                    for choice in chunk.choices:
                        if choice.index != 0:
                            raise RuntimeError("Provider returned more than one completion")
                        delta = choice.delta
                        if delta.content:
                            content.append(delta.content)
                        thought = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
                        if thought:
                            reasoning.append(thought)
                        if choice.finish_reason:
                            finish = choice.finish_reason
        except APIStatusError as exc:
            if exc.status_code not in (408, 409, 429) and exc.status_code < 500:
                raise RuntimeError(
                    f"{self.name}: HTTP {exc.status_code}; check model availability and support for "
                    "thinking with temperature=0. Request settings were not changed."
                ) from None
            return {"raw_output": "".join(content), "answer": "", "reasoning": "".join(reasoning),
                    "finish_reason": "api_error", "error": f"HTTP {exc.status_code}"}
        except (APIConnectionError, TransportError):
            return {"raw_output": "".join(content), "answer": "", "reasoning": "".join(reasoning),
                    "finish_reason": "api_error", "error": "APIConnectionError"}
        if not returned or not returned <= set(self.config.get("allowed_returned_models", [self.name])):
            raise RuntimeError(f"Unexpected returned model for {self.name}: {sorted(returned)}")
        raw = "".join(content)
        if "</think>" in raw:
            thought, answer = raw.rsplit("</think>", 1)
            reasoning.append(thought.removeprefix("<think>"))
        else:
            answer = "" if "<think>" in raw else raw
        return {
            "raw_output": raw, "answer": answer.strip(), "reasoning": "".join(reasoning),
            "finish_reason": finish or "incomplete_stream", "usage": usage,
            "returned_models": sorted(returned), "response_id": response_id,
            "thinking_requested": True, "reasoning_observed": bool(reasoning),
        }

    def close(self):
        self.client.close()


def add_api_arguments(parser):
    parser.add_argument("--api-config", type=Path, default=ROOT / "scripts" / "api_models.json")
    parser.add_argument("--timeout", type=float, default=180)


def main():
    parser = argparse.ArgumentParser(description="Run the eight API models sequentially with thinking enabled")
    add_arguments(parser)
    add_api_arguments(parser)
    parser.add_argument("--models", nargs="+")
    args = parser.parse_args()
    datasets = validate_args(args)
    configs = load_models(args.api_config)
    names = list(dict.fromkeys(args.models or configs))
    if set(names) - configs.keys():
        parser.error("Unknown model ID in --models; see api_models.json")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.validate_only:
        return
    for name in names:
        engine = ApiEngine(name, configs[name], args)
        try:
            for language in args.languages:
                run_cell(args, engine, language, datasets)
        finally:
            engine.close()


if __name__ == "__main__":
    main()
