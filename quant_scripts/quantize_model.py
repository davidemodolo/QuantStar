#!/usr/bin/env python3
"""
Offline AWQ W4G128 quantization for Qwen3.5-9B.

Loads the base bf16 model, runs calibration, saves a ~5 GB W4G128 checkpoint.
Point config.yaml model.repo at the output dir and run the server as usual
(no BitsAndBytes needed — transformers auto-detects the compressed-tensors format).

Usage:
    python quant_scripts/quantize_model.py
    python quant_scripts/quantize_model.py --bits 3        # W3G128 ~4 GB
    python quant_scripts/quantize_model.py --model-path ./models/Qwen__Qwen3.5-9B
    python quant_scripts/quantize_model.py --output-dir ./models/my-quantized

Requires:  pip install llmcompressor
Runtime:   ~1-2 h on RTX 3090, needs ~18 GB VRAM.
"""

# ── Python 3.14 compatibility fixes ──────────────────────────────────────────
# Fix 1 — pydantic/annotationlib:
#   pydantic's BaseModel has a dict() method; Python 3.14's annotationlib puts
#   vars(owner) into eval() locals, shadowing the builtin dict type and breaking
#   dict[str, Any] annotations.
import annotationlib as _annlib
import builtins as _b

_orig_fwd_eval = _annlib.ForwardRef.evaluate


def _fix_fwd_eval(self, *, globals=None, locals=None, **kw):
    if locals and "dict" in locals and not isinstance(locals["dict"], type):
        locals = dict(locals)
        locals["dict"] = _b.dict
    return _orig_fwd_eval(self, globals=globals, locals=locals, **kw)


_annlib.ForwardRef.evaluate = _fix_fwd_eval

# Fix 2 — argparse._check_help:
#   Python 3.14 added _check_help which validates help strings with %-formatting.
#   llmcompressor's dataset_arguments.py contains "train[:50%]" in a help string
#   which argparse misreads as a broken format specifier.
import argparse as _argparse
_argparse.ArgumentParser._check_help = lambda self, action: None
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import glob
import logging
import os
import sys
import time

logging.basicConfig(
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("quantize")


# ─── helpers ─────────────────────────────────────────────────────────────────

def _resolve_model_path(args_path: str | None) -> str:
    if args_path and os.path.isdir(args_path):
        return os.path.abspath(args_path)
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
        log.warning("auto-detected path does not exist: %s", path)
    except Exception as e:
        log.warning("could not read config.yaml: %s", e)
    if args_path:
        log.error("path not found: %s", args_path)
    else:
        log.error("could not auto-detect model path — pass --model-path")
    sys.exit(1)


def _sizeof_dir(path: str) -> float:
    total = 0
    for f in glob.glob(os.path.join(path, "**", "*"), recursive=True):
        if os.path.isfile(f):
            total += os.path.getsize(f)
    return total / 1e9


def _build_recipe(bits: int, group_size: int, ignore_lm_head: bool):
    from llmcompressor.modifiers.transform.awq import AWQModifier
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from compressed_tensors.quantization import (
        QuantizationScheme, QuantizationArgs,
        QuantizationType, QuantizationStrategy,
    )

    ignore = ["lm_head"] if ignore_lm_head else []
    scheme = QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(
            num_bits=bits,
            type=QuantizationType.INT,
            strategy=QuantizationStrategy.GROUP,
            group_size=group_size,
            symmetric=False,
        ),
    )
    return [
        AWQModifier(duo_scaling="both"),
        QuantizationModifier(
            config_groups={"default": scheme},
            ignore=ignore,
        ),
    ]


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Quantize Qwen3.5-9B to W4/W3 int4")
    parser.add_argument("--model-path", default=None,
                        help="Path to bf16 base model (auto-detected from config.yaml if omitted)")
    parser.add_argument("--output-dir", default=None,
                        help="Where to save the quantized model (default: <model>-AWQ-W<bits>G<group>)")
    parser.add_argument("--bits", type=int, default=4, choices=[3, 4],
                        help="Weight bit-width: 4 = ~5 GB, 3 = ~4 GB (default: 4)")
    parser.add_argument("--group-size", type=int, default=128,
                        help="Quantization group size (default: 128)")
    parser.add_argument("--calib-samples", type=int, default=128,
                        help="Number of calibration samples (default: 128)")
    parser.add_argument("--calib-seqlen", type=int, default=2048,
                        help="Calibration sequence length (default: 2048)")
    args = parser.parse_args()

    # ── step 1: resolve paths ────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("QuantStar offline quantization")
    log.info("=" * 60)

    model_path = _resolve_model_path(args.model_path)
    base = os.path.basename(model_path.rstrip("/"))
    output_dir = args.output_dir or os.path.join(
        os.path.dirname(model_path),
        f"{base}-AWQ-W{args.bits}G{args.group_size}",
    )

    log.info("Source model : %s", model_path)
    log.info("Output dir   : %s", output_dir)
    log.info("Scheme       : W%dG%d (AWQ, asymmetric)", args.bits, args.group_size)
    log.info("Calibration  : %d samples × %d tokens", args.calib_samples, args.calib_seqlen)
    log.info("")

    # ── step 2: import llmcompressor ─────────────────────────────────────────
    log.info("[1/4] Importing llmcompressor …")
    t0 = time.time()
    try:
        from llmcompressor import oneshot
    except ImportError as e:
        log.error("llmcompressor not installed: %s", e)
        log.error("Run: pip install llmcompressor")
        sys.exit(1)
    log.info("      done in %.1f s", time.time() - t0)

    # ── step 3: build recipe ─────────────────────────────────────────────────
    log.info("[2/4] Building AWQ W%dG%d recipe …", args.bits, args.group_size)
    recipe = _build_recipe(args.bits, args.group_size, ignore_lm_head=True)
    log.info("      %d modifiers: %s",
             len(recipe), ", ".join(type(m).__name__ for m in recipe))

    # ── step 4: quantize ─────────────────────────────────────────────────────
    log.info("[3/4] Loading model + running calibration …")
    log.info("      (this step takes ~1-2 h on an RTX 3090 — grab a coffee)")
    os.makedirs(output_dir, exist_ok=True)
    t1 = time.time()

    oneshot(
        model=model_path,
        recipe=recipe,
        dataset="wikitext",
        dataset_config_name="wikitext-2-raw-v1",
        num_calibration_samples=args.calib_samples,
        max_seq_length=args.calib_seqlen,
        trust_remote_code_model=True,
        output_dir=output_dir,
        # Keep the DeltaNet CUDA kernels out of FX tracing
        tracing_ignore=[
            "_update_causal_mask", "create_causal_mask", "_update_mamba_mask",
            "make_causal_mask", "get_causal_mask", "mask_interface", "mask_function",
            "_prepare_4d_causal_attention_mask", "_prepare_fsmt_decoder_inputs",
            "_prepare_4d_causal_attention_mask_with_cache_position",
            "_update_linear_attn_mask", "project_per_layer_inputs",
            "chunk_gated_delta_rule", "fused_recurrent_gated_delta_rule",
            "causal_conv1d_fn", "causal_conv1d_update",
            "torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule",
            "torch_causal_conv1d_update",
        ],
    )

    elapsed = time.time() - t1
    log.info("      quantization done in %.0f min", elapsed / 60)

    # ── step 5: report ───────────────────────────────────────────────────────
    log.info("[4/4] Verifying output …")
    shards = glob.glob(os.path.join(output_dir, "*.safetensors"))
    total_gb = _sizeof_dir(output_dir)
    log.info("      %d shard(s), %.2f GB total", len(shards), total_gb)
    log.info("")
    log.info("=" * 60)
    log.info("Done!  Quantized model saved to:")
    log.info("  %s", output_dir)
    log.info("")
    log.info("Next steps:")
    log.info("  1. In config.yaml set:  model.repo: \"%s\"", output_dir)
    log.info("  2. Run the server normally — no BNB needed, the model")
    log.info("     auto-loads in W%d format via compressed-tensors.", args.bits)
    log.info("  3. The int8 lm_head and int4 KV cache in quantize.py")
    log.info("     still apply on top of the pre-quantized weights.")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
