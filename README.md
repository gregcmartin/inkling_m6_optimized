# Inkling 2 Mini on Apple Silicon

Serve the public Inkling 2 Mini checkpoint with oMLX using a 32K context, medium reasoning, SSD-backed prompt-prefix reuse, and the upstream fused Inkling Metal kernels.

The quick start uses the unmodified `mlx-community/Inkling-2-Mini-oQ4e-mtp` base checkpoint. It is a 4-bit-class mixed-precision MLX build with a bundled DSpark draft. The plain `mlx-community/Inkling-2-Mini-4bit` checkpoint is also available for standard MLX affine 4-bit weights. The bundled-draft build is the one used by these scripts and exercises oMLX Lightning MTP. Its model card reports an effective average of 4.915 bits per weight for the language and vision weights. [oQ4e + DSpark checkpoint](https://huggingface.co/mlx-community/Inkling-2-Mini-oQ4e-mtp) · [Standard MLX 4-bit checkpoint](https://huggingface.co/mlx-community/Inkling-2-Mini-4bit)

## Quick start

Use a native Apple Silicon terminal on macOS 15 or later. For this profile, start with 32 GB of unified memory and allow at least 25 GB of disk space for the checkpoint, runtime, and initial cache.

```sh
brew install uv
git clone https://github.com/gregcmartin/inkling_m6_optimized.git
cd inkling_m6_optimized
./setup.sh
./serve.sh
```

`setup.sh` creates a private Python 3.11 environment, installs the pinned oMLX 0.8.3 wheel, stages the backport in a private package overlay, seeds the profile, and downloads the pinned public checkpoint (about 17 GB). A repeat setup keeps any edits to the existing profile. `serve.sh` listens on `127.0.0.1:8000`. Set `INKLING_PORT` before starting to use another port. The default data directory is `~/.inkling-m6-optimized`, or set `INKLING_HOME` in both terminals to use a different location.

## Verify and benchmark

Run these from a second terminal in the checkout.

```sh
runtime="${INKLING_HOME:-$HOME/.inkling-m6-optimized}/venv"
"$runtime/bin/python" benchmark.py --suite prefix
"$runtime/bin/python" benchmark.py --suite speed | tee benchmark-results.jsonl
```

Use `--url` to point the benchmark at a different port. The prefix test warms the model, sends a fresh ~8.4K-token system message, and sends it again with a different user question. Expect `cached_tokens: 0` on the first request and roughly 8K on the second. The speed suite uses fresh prompts at roughly 512, 8,192, and 30,720 input tokens and reports first-token latency, first-answer latency, decode throughput, and completion status. It uses the same 1,536-token output budget for each case. `finish: length` indicates a token cap. Python output passing the suite means it parsed successfully.

To get cache hits in an app, keep the shared system instructions and reference text byte-for-byte stable at the beginning of the request, then append the changing question. Put timestamps, request-specific metadata, and other variable text after the shared prefix. Check `usage.prompt_tokens_details.cached_tokens` for the actual number reused. The first token may contain reasoning. `first_answer_s` measures when answer text starts.

For the public checkpoint on an M5 Max, the included prefix probe reused 8,192 tokens and cut client first-token time from 5.35 s to 0.79 s. Both requests returned their requested code. Its three poem cases passed at all tested context lengths. The weather cases hit their 1,536-token budget.

### Compare with stock oMLX

Stop the server and restart it with `INKLING_ENGINE=stock ./serve.sh`. Re-run the same benchmark, then restart with `./serve.sh` for the fused build. Both modes use the same weights and profile. Compare the cold rows for the kernel effect and the reused row for the prefix-cache effect. On the M5 Max with the public checkpoint, one cold probe took 6.94 s on stock and 5.35 s with the fused build.

On a 32 GB M6 running the same profile with a separately fine-tuned oQ4e checkpoint, the matched 8.4K kernel probe reduced server first-token time from 13.51 s to 12.30 s. In the separate 8K poem test, reusing 8,198 prompt tokens cut client first-token time from 10.19 s to 0.55 s. These are single-run reference measurements. Use the included scripts to measure the public base checkpoint on your Mac.

## Settings and runtime

| Setting | Value |
| --- | --- |
| Context window | 32,768 tokens |
| Model output limit | 4,096 tokens by default |
| Reasoning | Medium via chat template |
| KV cache | Native BF16, with SSD prefix reuse |
| Speculative decode | Lightning MTP with the checkpoint's DSpark draft |
| Memory guard | Balanced, with up to two concurrent requests |

The seeded files are in `config/`. Change the installed `profile/model_settings.json` for model defaults and `profile/settings.json` for server and cache settings, then restart. The context must accommodate the prompt and generated response together.

The fused kernels combine operations such as attention mixing and normalization into fewer Metal dispatches. The patch backports the kernel change and its GPU threadgroup fallback to the tagged 0.8.3 wheel. It is applied only to the private environment by setup. [oMLX 0.8.3](https://github.com/jundot/omlx/releases/tag/v0.8.3) · [Fused kernels](https://github.com/jundot/omlx/pull/5557) · [GPU fallback](https://github.com/jundot/omlx/pull/5563)
