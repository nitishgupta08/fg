#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import onnx
import torch
from dotenv import load_dotenv
from optimum.exporters.onnx import main_export
from optimum.exporters.onnx.convert import _get_submodels_and_onnx_configs
from optimum.exporters.tasks import TasksManager
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export FunctionGemma to ONNX.")
    parser.add_argument(
        "--model-id",
        default="google/gemma-3-270m-it",
        help="Hugging Face model ID or local model path.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where ONNX artifacts are saved. If omitted, an auto name is used.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=None,
        help="ONNX opset version. If omitted, the recommended model-specific opset is auto-detected.",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda",
        help="Export device. Falls back to CPU when CUDA is unavailable.",
    )
    parser.add_argument(
        "--no-trust-remote-code",
        action="store_true",
        help="Disable trust_remote_code for model/tokenizer loading.",
    )
    return parser.parse_args()


def resolve_device(device: str) -> str:
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU export.")
        return "cpu"
    return device


def validate_onnx_files(output_dir: Path) -> None:
    onnx_files = sorted(output_dir.rglob("*.onnx"))
    if not onnx_files:
        raise FileNotFoundError(f"No .onnx files found under: {output_dir}")

    for onnx_file in onnx_files:
        onnx.checker.check_model(str(onnx_file), full_check=False)
        print(f"Validated ONNX graph: {onnx_file}")


def load_env_auth(script_dir: Path) -> None:
    dotenv_path = script_dir / ".env"
    load_dotenv(dotenv_path=dotenv_path)

    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token


def sanitize_segment(segment: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", segment.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "unknown"


def is_local_model_path(model_id: str) -> bool:
    expanded = Path(model_id).expanduser()
    if expanded.exists():
        return True
    return model_id.startswith(("/", "./", "../", "~"))


def model_id_to_artifact_parts(model_id: str) -> list[str]:
    if is_local_model_path(model_id):
        local_base = Path(model_id).expanduser().resolve().name
        return ["local", sanitize_segment(local_base)]

    normalized = model_id.strip().strip("/")
    if not normalized:
        return ["unknown-model"]

    return [sanitize_segment(part) for part in normalized.split("/")]


def resolve_output_dir(
    script_dir: Path, output_dir: str | None, model_id: str, device: str, use_fp16: bool
) -> Path:
    if output_dir:
        return Path(output_dir).resolve()

    precision = "fp16" if use_fp16 else "fp32"
    mode_dir = f"onnx-{device}-{precision}"
    auto_dir = (
        script_dir
        / "artifacts"
        / Path(*model_id_to_artifact_parts(model_id))
        / mode_dir
    )
    return auto_dir.resolve()


def detect_recommended_opset(
    model_id: str,
    task: str,
    trust_remote_code: bool,
    device: str,
) -> int:
    model = TasksManager.get_model_from_task(
        task,
        model_id,
        trust_remote_code=trust_remote_code,
        framework="pt",
        device=device,
    )
    dtype = str(getattr(model, "dtype", "float32"))
    if "bfloat16" in dtype:
        float_dtype = "bf16"
    elif "float16" in dtype:
        float_dtype = "fp16"
    else:
        float_dtype = "fp32"

    onnx_config, _ = _get_submodels_and_onnx_configs(
        model=model,
        task=task,
        monolith=False,
        custom_onnx_configs={},
        custom_architecture=False,
        float_dtype=float_dtype,
        fn_get_submodels=None,
        preprocessors=[],
        _variant="default",
        library_name="transformers",
        model_kwargs=None,
    )
    return int(onnx_config.DEFAULT_ONNX_OPSET)


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    load_env_auth(script_dir)

    task = "text-generation-with-past"
    trust_remote_code = not args.no_trust_remote_code
    device = resolve_device(args.device)
    use_fp16 = device == "cuda"
    output_dir = resolve_output_dir(
        script_dir=script_dir,
        output_dir=args.output_dir,
        model_id=args.model_id,
        device=device,
        use_fp16=use_fp16,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if not use_fp16:
        print("Exporting in FP32 because CPU export is selected.")

    if args.opset is None:
        try:
            recommended_opset = detect_recommended_opset(
                model_id=args.model_id,
                task=task,
                trust_remote_code=trust_remote_code,
                device=device,
            )
            print(f"Auto-detected recommended opset: {recommended_opset}")
            effective_opset = recommended_opset
        except Exception as exc:
            effective_opset = 18
            print(
                "Could not auto-detect recommended opset. "
                f"Falling back to safe default opset={effective_opset}. Error: {exc}"
            )
    else:
        effective_opset = args.opset
        print(f"Using user-specified opset: {effective_opset}")

    print(
        "Starting export with "
        f"model={args.model_id}, output={output_dir}, device={device}, "
        f"opset={effective_opset}, fp16={use_fp16}, trust_remote_code={trust_remote_code}"
    )

    main_export(
        model_name_or_path=args.model_id,
        output=output_dir,
        task=task,
        opset=effective_opset,
        device=device,
        trust_remote_code=trust_remote_code,
        fp16=use_fp16,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=trust_remote_code,
    )
    tokenizer.save_pretrained(output_dir)

    validate_onnx_files(output_dir)
    print(f"Export complete. Artifacts saved to: {output_dir}")


if __name__ == "__main__":
    main()
