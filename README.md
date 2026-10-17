# CNFO-SMS Experiments

Minimal code for Chinese SMS fraud classification, API model comparison, and dense Qwen3 scaling experiments.

```text
dataset/                       input data, initially empty
results/                       predictions and metrics, initially empty
models/                        downloaded weights, initially empty
model_download/
  download_models.py
scripts/
  run_local.py
  run_api.py
  retry_invalid.py
  evaluate.py
  common.py
  api_models.json
  prompts/
    cn.txt
    en.txt
requirements.txt
README.md
```

## Setup

Use Python 3.10 or later. Install a CUDA build of PyTorch suitable for the GPU, then install the dependencies:

```bash
python -m pip install -r requirements.txt
```

The API runner and evaluator do not require local model weights. An API-only environment needs only `openai==1.109.1`; evaluation uses the Python standard library.

Git does not preserve empty directories. After cloning, create them with:

```bash
python -c "from pathlib import Path; [Path(p).mkdir(exist_ok=True) for p in ('dataset', 'results', 'models')]"
```

The supplied directory has these empty folders already. Data, generated results, weights, and local credentials are excluded by `.gitignore`.

## Data

Extract the original compressed-text dataset into `dataset/`, retaining its group folders. A nested dataset root is also accepted.

```text
dataset/
  01_基线数据_无扰动/
    基线数据_fraud.json
    基线数据_normal.json
  02_Emoji/Emoji.json
  03_VSim/VSim.json
  04_Split/Split.json
  05_Trad/Trad.json
  06_PY_Init/PY_Init.json
  07_Homo/Homo.json
  08_Shuff/Shuff.json
  09_不可见字符插入/不可见字符插入.json
  10_特殊效果字符插入/特殊效果字符插入.json
```

Each file is a JSON array. Records contain `id`, `text`, `label`, `binary`, and `perturbation`. Perturbed records also contain `original_text`, matching the unperturbed fraud message. The original text, Unicode characters, and IDs are preserved. Records labelled `其他` are excluded consistently. The full dataset contains 6,917 included fraud messages per condition and 2,056 legitimate messages. `PY_Init` retains the original filename and denotes full Pinyin substitution without tones.

Validate all groups and prompts without downloading weights or sending API requests:

```bash
python scripts/run_local.py --validate-only
python scripts/run_api.py --validate-only
```

## Inference settings

| Setting | Local Qwen3 | API models |
|---|---|---|
| Execution | One model at a time; one SMS per call | One model at a time; one SMS per request |
| Prompt | One user message; no system message or examples | Same |
| Languages | Chinese and English instructions; Chinese SMS inputs | Same |
| Temperature | 0, implemented as greedy decoding (`do_sample=False`, `num_beams=1`) | Explicit `temperature=0` request |
| Output budget | 8,192 new tokens | Same; includes reasoning when counted by the provider |
| Thinking | Disabled, retaining the local greedy experiment setting | Requested on, following the current manuscript |
| Precision | BF16 by default; no quantization | Provider managed |
| Seed | 42 | Recorded as run metadata; not sent to providers |

Transformers does not use a temperature during greedy decoding, so its sampling temperature is left unset. This implements the temperature-zero policy without passing an invalid sampling temperature. There is no batching, concurrent request pool, or quantization option. Models are released before loading the next model.

Both prompt files transcribe the paper's prompt figure; JSON examples use executable ASCII quotation marks, and figure line wrapping is removed. The Chinese instruction preserves the figure's wording, including `Fraud_type只能以下其一`. Only `{text}` is substituted. The older scripts used different instruction wording. This package follows the current manuscript and the requested settings; it does not include historical predictions or claim that revised settings reproduce their exact scores. The common 8,192-token output budget is an explicit setting of this release; the manuscript does not specify a token cap. Use the same `--max-new-tokens` value for both runners and all retries if choosing another cap.

API thinking controls are provider specific. The requested controls and endpoint are stored in every run manifest. Reasoning returned by a provider is saved separately from the final answer; absence of visible reasoning does not establish that internal thinking is disabled. Some providers reject or ignore temperature zero in thinking mode, including restrictions documented for [Claude](https://platform.claude.com/docs/en/docs/build-with-claude/extended-thinking) and [DeepSeek](https://api-docs.deepseek.com/guides/thinking_mode/). An accepted API request therefore does not establish deterministic decoding. The runner never silently changes temperature, thinking, or the model ID to recover from a rejection. Use a compatible endpoint and document its effective settings before treating a run as paper evidence.

The manuscript specifies API thinking mode but does not specify reasoning effort, a thinking-token budget, or a strict JSON parser. The `medium` effort values, Claude's 1,024-token thinking budget, and the exact-JSON parser are implementation choices in this package. They are not verified historical experimental settings. Local thinking is disabled based on the existing greedy-run configuration; the manuscript does not explicitly specify the local thinking switch. Temperature zero, single-record execution, and no local quantization follow the accompanying code-release instructions. Matching the reported numerical tables additionally requires the historical prompt, parser, model revisions, provider behavior, and generation settings to match.

## Download models

```bash
python model_download/download_models.py
```

This downloads the original dense `Qwen/Qwen3-{0.6B,1.7B,4B,8B,14B,32B}` repositories to `models/Qwen3-<size>/`. It resolves each requested revision to a commit and records file hashes. Interrupted downloads resume against the same commit. A subset can be downloaded with `--models 0.6B 1.7B`. See the [official Qwen3 model card](https://huggingface.co/Qwen/Qwen3-0.6B).

## Local scaling experiment

```bash
python scripts/run_local.py
```

All six sizes run sequentially with both prompts. The paper used a 96 GB GPU; sufficient GPU memory and host memory are required for the selected unquantized model. Optional selections:

```bash
python scripts/run_local.py --models 0.6B 1.7B --languages cn
python scripts/run_local.py --models 0.6B --device cpu --dtype float32
```

To check a small selection, use a separate results root:

```bash
python scripts/run_local.py --models 0.6B --limit 2 --results results/smoke
```

## API comparison

`scripts/api_models.json` contains the eight evaluated models, their API identifiers, endpoint defaults, credential environment variable names, and thinking controls. `display_name` records the manuscript name where it differs from the historical API identifier. Endpoint availability and relay parameter support must be checked with the chosen provider; model IDs are never substituted automatically.

Set credentials in the terminal. For example, in PowerShell:

```powershell
$env:RIVO_API_KEY = "your-key"
$env:MIMO_API_KEY = "your-key"
$env:DASHSCOPE_API_KEY = "your-key"
$env:DEEPSEEK_API_KEY = "your-key"
python scripts/run_api.py
```

On Linux, use `export RIVO_API_KEY="your-key"` and the corresponding commands for the other keys. Optional endpoint overrides are `RIVO_BASE_URL`, `MIMO_BASE_URL`, `DASHSCOPE_BASE_URL`, and `DEEPSEEK_BASE_URL`. API keys are read from the environment and are not written to results. `.env` files are not loaded automatically.

Run a subset using exact API model IDs:

```bash
python scripts/run_api.py --models qwen3.8-flash glm-5.2 --languages cn en
```

## Resume and retry invalid records

Both runners append and flush one JSONL record after each SMS. Repeat the original command to resume missing records. Existing outputs, including invalid ones, are retained. Each model and language has a manifest; incompatible data, prompts, or inference settings are rejected within the same output directory. Use one writer per experiment directory, and stop inference before evaluating it.

Retry missing records and invalid final answers until all outputs parse successfully:

```bash
python scripts/retry_invalid.py --backend local
python scripts/retry_invalid.py --backend api
```

Selections and a finite retry limit are also available:

```bash
python scripts/retry_invalid.py --backend local --models 0.6B --max-rounds 3
python scripts/retry_invalid.py --backend api --api-models qwen3.8-flash --max-rounds 3
```

Pass the original `--data`, `--results`, `--languages`, `--limit`, `--dtype`, `--device`, `--seed`, and `--max-new-tokens` options where applicable. Every attempt remains in the JSONL log; evaluation selects the latest attempt for each ID. A valid record requires a completed answer containing exactly one JSON object with one `fraud_type` field and a permitted label. A valid but incorrect classification is never retried. Length-limited outputs, refusals, malformed JSON, interrupted streams, and recoverable API failures remain invalid until retried. Credential errors, unsupported requests, unknown models, and local memory failures stop execution.

By default `--max-rounds 0` means unlimited retries with a five-second pause between rounds. Greedy decoding may repeat the same invalid answer indefinitely; use Ctrl+C to stop or set a finite limit. The retry loop does not change the prompt, token budget, or decoding settings. Any record still invalid at evaluation time is counted as an error.

## Metrics

```bash
python scripts/evaluate.py
```

For a limited check, use matching options:

```bash
python scripts/evaluate.py --limit 2 --results results/smoke
```

For each condition, the evaluator combines its fraud predictions with the same legitimate predictions, then computes eight-class Macro-F1. Base uses unperturbed fraud messages. `Avg7` averages the seven obfuscations only; `AvgIns` averages the two insertion conditions. `Delta7` and `DeltaIns` equal Base minus the corresponding average. All calculations use unrounded scores. Invalid outputs count as false negatives for the true class and are excluded from the eight-class macro average as a separate category. Missed fraud includes both legitimate predictions and invalid outputs for fraudulent messages; the false-positive rate counts legitimate messages assigned a valid fraud category.

Incomplete runs are rejected by default. `--allow-partial` permits diagnostics and counts missing outputs as invalid; such summaries are marked incomplete. Per-class metrics and confusion matrices are saved in each cell's `metrics.json`. Combined summaries are saved to `results/summary.json` and `results/summary.csv`.

```text
results/
  local/Qwen3-0.6B/cn/
    manifest.json
    fraud.jsonl
    normal.jsonl
    02_Emoji.jsonl
    ...
    metrics.json
  api/qwen3.8-flash/en/
    ...
  summary.json
  summary.csv
```
