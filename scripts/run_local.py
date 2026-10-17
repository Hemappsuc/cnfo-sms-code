import argparse
import gc
import importlib.metadata
import os
from pathlib import Path

from common import ROOT, SIZES, add_arguments, digest, read_json, run_cell, validate_args


class LocalEngine:
    backend = "local"

    def __init__(self, size, args):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, set_seed

        set_seed(args.seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if not torch.cuda.is_available() and args.device == "auto":
            raise RuntimeError("CUDA is unavailable; use --device cpu --dtype float32 for a CPU run")
        self.name = f"Qwen3-{size}"
        directory = args.model_root / self.name
        config = read_json(directory / "config.json")
        if config.get("quantization_config"):
            raise ValueError("Quantized weights are not allowed")
        download_path = directory / "download_manifest.json"
        download = read_json(download_path) if download_path.exists() else None
        if download is not None:
            if not download.get("complete") or download.get("repo_id") != f"Qwen/{self.name}":
                raise ValueError(f"Incomplete or mismatched model download: {directory}")
            for filename, info in download["files"].items():
                path = directory / filename
                if not path.is_file() or path.stat().st_size != info["bytes"]:
                    raise ValueError(f"Model file is missing or incomplete: {path}")
        self.tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            directory, local_files_only=True, trust_remote_code=False,
            torch_dtype=getattr(torch, args.dtype), device_map=args.device if args.device == "auto"
            else {"": args.device}, use_safetensors=True, attn_implementation="eager",
        ).eval()
        eos = self.model.generation_config.eos_token_id or self.tokenizer.eos_token_id
        self.eos = set(eos if isinstance(eos, list) else [eos])
        self.generation = GenerationConfig(
            do_sample=False, num_beams=1, max_new_tokens=args.max_new_tokens,
            temperature=None, top_p=None, top_k=None, repetition_penalty=1.0,
            eos_token_id=eos, pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        self.torch, self.args = torch, args
        self.settings = {
            "thinking": False, "decode": "greedy", "dtype": args.dtype,
            "quantization": None, "attention": "eager", "device": args.device,
            "model_config_sha256": digest(config),
            "download": download,
            "versions": {name: importlib.metadata.version(name)
                         for name in ("torch", "transformers", "accelerate")},
        }

    def infer(self, prompt):
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )
        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        width = inputs["input_ids"].shape[-1]
        if width + self.args.max_new_tokens > self.model.config.max_position_embeddings:
            raise ValueError("Input and output budget exceed context length; input was not truncated")
        inputs = inputs.to(self.model.get_input_embeddings().weight.device)
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs, generation_config=self.generation, use_model_defaults=False,
                do_sample=False, num_beams=1,
            )[0, width:].tolist()
        raw = self.tokenizer.decode(output, skip_special_tokens=True)
        return {
            "raw_output": raw, "answer": raw.strip(), "reasoning": "",
            "finish_reason": "eos" if output and output[-1] in self.eos else "length",
            "input_tokens": width, "output_tokens": len(output),
        }

    def close(self):
        del self.model
        del self.tokenizer
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


def add_local_arguments(parser):
    parser.add_argument("--models", nargs="+", choices=SIZES, default=list(SIZES))
    parser.add_argument("--model-root", type=Path, default=ROOT / "models")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")


def main():
    parser = argparse.ArgumentParser(description="Run all six dense Qwen3 models sequentially")
    add_arguments(parser)
    add_local_arguments(parser)
    args = parser.parse_args()
    datasets = validate_args(args)
    if args.validate_only:
        return
    for size in dict.fromkeys(args.models):
        engine = LocalEngine(size, args)
        try:
            for language in args.languages:
                run_cell(args, engine, language, datasets)
        finally:
            engine.close()


if __name__ == "__main__":
    main()
