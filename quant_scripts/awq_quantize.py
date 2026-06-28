#!/usr/bin/env python3
"""
AWQ 4-bit quantization for Qwen3.5-9B using llmcompressor.

autoawq does not support the qwen3_5 hybrid architecture (DeltaNet + full
attention). This script uses llmcompressor's AWQModifier instead, which is
model-agnostic (operates on nn.Linear weights regardless of attention type)
and produces a standard compressed-tensors / safetensors checkpoint.

Expected output size for Qwen3.5-9B:
  AWQ W4G128  ~5.0 GB

Usage (from repo root):
    python quant_scripts/awq_quantize.py
    python quant_scripts/awq_quantize.py --model-path ./models/Qwen__Qwen3.5-9B
    python quant_scripts/awq_quantize.py --group-size 64

Runtime: ~1-2 h on RTX 3090.  Needs ~18 GB VRAM to load 9B in bf16.

NOTE: The script applies a Python 3.14 compatibility patch at startup that
      fixes a pydantic/annotationlib conflict in llmcompressor. The patch is
      safe and does not affect any runtime model weights.
"""

# ---- Python 3.14 compatibility fix (must be first) -------------------------
# In Python 3.14, annotationlib puts vars(owner) into eval() locals when
# evaluating forward-ref annotations. pydantic's BaseModel subclasses have a
# dict() method, which shadows the builtin dict type and breaks annotations of
# the form dict[str, Any]. We restore the builtin before every evaluation.
import annotationlib as _annlib
import builtins as _b

_orig_fwd_eval = _annlib.ForwardRef.evaluate


def _patched_fwd_eval(self, *, globals=None, locals=None, **kw):
    if locals and "dict" in locals and not isinstance(locals["dict"], type):
        locals = dict(locals)
        locals["dict"] = _b.dict
    return _orig_fwd_eval(self, globals=globals, locals=locals, **kw)


_annlib.ForwardRef.evaluate = _patched_fwd_eval
# ---------------------------------------------------------------------------

import argparse
import glob
import os
import sys


def _resolve_model_path(args_path):
    if args_path and os.path.isdir(args_path):
        return args_path
    try:
        import yaml
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root, "config.yaml")) as f:
            cfg = yaml.safe_load(f)
        repo = cfg["model"]["repo"]
        cache = cfg["model"]["cache_dir"].lstrip("./")
        path = os.path.join(root, cache, repo.replace("/", "__"))
        if os.path.isdir(path):
            return path
        print(f"[warn] auto-detected path does not exist: {path}")
    except Exception as e:
        print(f"[warn] could not read config.yaml: {e}")
    if args_path:
        print(f"[error] path not found: {args_path}")
    else:
        print("[error] could not auto-detect model path — pass --model-path")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--group-size", type=int, default=128,
                        help="AWQ group size (128 = standard; 64 = slightly better quality)")
    parser.add_argument("--calib-seqlen", type=int, default=2048)
    parser.add_argument("--calib-nsamples", type=int, default=128)
    parser.add_argument("--no-ignore-lm-head", action="store_true",
                        help="Also quantize lm_head (NOT recommended — QuantStar uses int8 lm_head at runtime)")
    args = parser.parse_args()

    model_path = _resolve_model_path(args.model_path)
    base = os.path.basename(model_path.rstrip("/"))
    output_dir = args.output_dir or os.path.join(
        os.path.dirname(model_path),
        f"{base}-AWQ-W4G{args.group_size}"
    )

    print(f"Model:    {model_path}")
    print(f"Output:   {output_dir}")
    print(f"Group:    {args.group_size}")
    print(f"Calib:    {args.calib_nsamples} × {args.calib_seqlen} tokens")
    print()

    try:
        from llmcompressor import oneshot
        from llmcompressor.modifiers.transform.awq import AWQModifier
        from llmcompressor.modifiers.quantization import QuantizationModifier
        from compressed_tensors.quantization import (
            QuantizationScheme, QuantizationArgs,
            QuantizationType, QuantizationStrategy,
        )
    except ImportError as e:
        print(f"ERROR: llmcompressor not installed.\n  {e}")
        print("Run: pip install llmcompressor")
        sys.exit(1)

    ignore = [] if args.no_ignore_lm_head else ["lm_head"]

    # AWQ = activation-aware weight scaling pass + int4 packing.
    # AWQModifier computes per-channel scales from calibration activations;
    # QuantizationModifier applies them and packs weights to int4.
    recipe = [
        AWQModifier(duo_scaling="both"),
        QuantizationModifier(
            config_groups={
                "default": QuantizationScheme(
                    targets=["Linear"],
                    weights=QuantizationArgs(
                        num_bits=4,
                        type=QuantizationType.INT,
                        strategy=QuantizationStrategy.GROUP,
                        group_size=args.group_size,
                        symmetric=False,
                    ),
                )
            },
            ignore=ignore,
        ),
    ]

    print(f"Running AWQ W4G{args.group_size} …")
    print("Expected time: ~1-2 h on RTX 3090\n")

    os.makedirs(output_dir, exist_ok=True)

    oneshot(
        model=model_path,
        recipe=recipe,
        dataset="wikitext",
        dataset_config_name="wikitext-2-raw-v1",
        num_calibration_samples=args.calib_nsamples,
        max_seq_length=args.calib_seqlen,
        trust_remote_code_model=True,
        output_dir=output_dir,
    )

    shards = glob.glob(os.path.join(output_dir, "*.safetensors"))
    total_gb = sum(os.path.getsize(f) for f in shards) / 1e9
    print(f"\nDone.  {total_gb:.2f} GB across {len(shards)} shard(s).")
    print(f"\nTo use: set config.yaml model.repo to the output path.")
    print("Also update load_and_quantize_model() in quantize.py to load")
    print("without BitsAndBytesConfig when the model is pre-quantized.")


if __name__ == "__main__":
    main()
