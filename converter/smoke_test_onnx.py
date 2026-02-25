#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

import onnxruntime as ort
import torch
from dotenv import load_dotenv
from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test an exported ONNX model.")
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Path to exported ONNX model directory.",
    )
    parser.add_argument(
        "--prompt",
        default="Turn on kitchen lights and set brightness to 40%.",
        help="Prompt used for generation.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=16,
        help="Maximum number of generated tokens.",
    )
    return parser.parse_args()


def pick_provider() -> str:
    providers = ort.get_available_providers()
    if "CUDAExecutionProvider" in providers:
        return "CUDAExecutionProvider"
    return "CPUExecutionProvider"


def load_env_auth(script_dir: Path) -> None:
    dotenv_path = script_dir / ".env"
    load_dotenv(dotenv_path=dotenv_path)

    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token


def main() -> None:
    args = parse_args()
    load_env_auth(Path(__file__).resolve().parent)

    model_dir = Path(args.model_dir).resolve()
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")

    provider = pick_provider()
    print(f"Using ONNX Runtime provider: {provider}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = ORTModelForCausalLM.from_pretrained(
        model_dir,
        provider=provider,
    )

    inputs = tokenizer(args.prompt, return_tensors="pt")
    input_len = int(inputs["input_ids"].shape[-1])

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )

    output_len = int(outputs.shape[-1])
    if output_len <= input_len:
        raise RuntimeError("Smoke test failed: model did not generate new tokens.")

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if not text.strip():
        raise RuntimeError("Smoke test failed: decoded output was empty.")

    print("Smoke test passed.")
    print("Generated text:")
    print(text)


if __name__ == "__main__":
    main()
