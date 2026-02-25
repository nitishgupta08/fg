# FunctionGemma ONNX Converter (UV)

This folder contains a UV-managed flow to export `google/functiongemma-270m-it` to ONNX and run a smoke test.

## 1) Environment setup

```bash
cd /Users/nitishgupta/Developer/functiongemma/converter
uv sync
```

Create and populate your Hugging Face token in `converter/.env`:

```bash
cd /Users/nitishgupta/Developer/functiongemma/converter
# edit .env and set HF_TOKEN=hf_xxx
```

Optional CUDA runtime install (Linux/Windows):

```bash
cd /Users/nitishgupta/Developer/functiongemma/converter
uv remove onnxruntime
uv add onnxruntime-gpu
uv sync
```

## 2) Export model to ONNX

Default command (tries CUDA, falls back to CPU; FP16 is used only when CUDA is available, and opset is auto-detected before export):

```bash
cd /Users/nitishgupta/Developer/functiongemma/converter
uv run python convert_to_onnx.py
```

Explicit CPU export:

```bash
cd /Users/nitishgupta/Developer/functiongemma/converter
uv run python convert_to_onnx.py --device cpu
```

Explicit opset override:

```bash
cd /Users/nitishgupta/Developer/functiongemma/converter
uv run python convert_to_onnx.py --opset 19
```

Custom output path:

```bash
cd /Users/nitishgupta/Developer/functiongemma/converter
uv run python convert_to_onnx.py --output-dir artifacts/custom/functiongemma-onnx
```

## 3) Smoke test exported ONNX model

```bash
cd /Users/nitishgupta/Developer/functiongemma/converter
uv run python smoke_test_onnx.py --model-dir artifacts/google/functiongemma-270m-it/onnx-cpu-fp32
```

Custom prompt:

```bash
cd /Users/nitishgupta/Developer/functiongemma/converter
uv run python smoke_test_onnx.py \
  --model-dir artifacts/google/functiongemma-270m-it/onnx-cpu-fp32 \
  --prompt "Set living room AC to 22C and turn on fan."
```

## Output location

ONNX artifacts are written to:

`/Users/nitishgupta/Developer/functiongemma/converter/artifacts/<model-id-path>/onnx-<device>-<precision>`

Examples:
- Hugging Face model-id `google/functiongemma-270m-it`, CPU fallback:
  `.../artifacts/google/functiongemma-270m-it/onnx-cpu-fp32`
- Hugging Face model-id `google/functiongemma-270m-it`, CUDA export:
  `.../artifacts/google/functiongemma-270m-it/onnx-cuda-fp16`
- Local model path `/models/my-model`:
  `.../artifacts/local/my-model/onnx-cpu-fp32` (or `onnx-cuda-fp16`)

## Troubleshooting

- If CUDA is unavailable, export runs on CPU and uses FP32.
- If model download fails, confirm `HF_TOKEN` is set in `converter/.env` and your account has access to `google/functiongemma-270m-it`.
- If opset auto-detection fails, the converter falls back to opset 18.
- If you hit runtime incompatibility with the selected opset, retry with a manual override:

```bash
cd /Users/nitishgupta/Developer/functiongemma/converter
uv run python convert_to_onnx.py --opset 16
```
