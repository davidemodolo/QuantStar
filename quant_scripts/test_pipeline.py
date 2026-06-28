#!/usr/bin/env python3
"""
Quick pre-flight checks before running quantize_model.py.

Tests every step that could fail in the real 1-2h run WITHOUT actually running
the full quantization.  The --full flag adds a real forward-pass test (~60s,
needs GPU + 18 GB VRAM) to catch DeltaNet kernel issues early.

Usage:
    python quant_scripts/test_pipeline.py          # fast checks, no GPU needed
    python quant_scripts/test_pipeline.py --full   # also loads model + forward pass
"""

# ── Python 3.14 fixes (same as quantize_model.py) ────────────────────────────
import annotationlib as _annlib
import builtins as _b

_orig_fwd_eval = _annlib.ForwardRef.evaluate


def _fix_fwd_eval(self, *, globals=None, locals=None, **kw):
    if locals and "dict" in locals and not isinstance(locals["dict"], type):
        locals = dict(locals)
        locals["dict"] = _b.dict
    return _orig_fwd_eval(self, globals=globals, locals=locals, **kw)


_annlib.ForwardRef.evaluate = _fix_fwd_eval

import argparse as _argparse
_argparse.ArgumentParser._check_help = lambda self, action: None
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import os
import sys
import time
import traceback

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

PASS = "✓"
FAIL = "✗"
SKIP = "–"


def step(name: str):
    print(f"\n{'─'*60}")
    print(f"  {name}")
    print(f"{'─'*60}")


def ok(msg: str = ""):
    print(f"  {PASS}  {msg}" if msg else f"  {PASS}")


def fail(msg: str, exc: Exception | None = None):
    print(f"  {FAIL}  {msg}")
    if exc:
        traceback.print_exc()
    return False


def _resolve_model_path() -> str | None:
    try:
        import yaml
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root, "config.yaml")) as f:
            cfg = yaml.safe_load(f)
        repo = cfg["model"]["repo"]
        cache = cfg["model"]["cache_dir"].lstrip("./")
        path = os.path.join(root, cache, repo.replace("/", "__"))
        return path if os.path.isdir(path) else None
    except Exception:
        return None


# ─── individual checks ───────────────────────────────────────────────────────

def check_imports() -> bool:
    step("Check 1 / 5  — imports")
    checks = [
        ("torch",                    "import torch"),
        ("transformers",             "import transformers"),
        ("llmcompressor",            "import llmcompressor"),
        ("llmcompressor.oneshot",    "from llmcompressor import oneshot"),
        ("AWQModifier",              "from llmcompressor.modifiers.transform.awq import AWQModifier"),
        ("QuantizationModifier",     "from llmcompressor.modifiers.quantization import QuantizationModifier"),
        ("compressed_tensors",       "from compressed_tensors.quantization import QuantizationScheme, QuantizationArgs, QuantizationType, QuantizationStrategy"),
    ]
    all_ok = True
    items = tqdm(checks, desc="  imports") if tqdm else checks
    for label, stmt in items:
        try:
            exec(stmt, {})
            ok(label)
        except Exception as e:
            fail(f"{label}: {e}")
            all_ok = False
    return all_ok


def check_config() -> bool:
    step("Check 2 / 5  — config and model path")
    try:
        import yaml
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cfg_path = os.path.join(root, "config.yaml")
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        ok(f"config.yaml loaded: {cfg_path}")
    except Exception as e:
        return fail("config.yaml not found or invalid", e)

    model_path = _resolve_model_path()
    if model_path is None:
        return fail(f"model path does not exist (check config.yaml model.repo / cache_dir)")
    ok(f"model path: {model_path}")

    # Check expected files
    for fname in ["config.json", "tokenizer.json"]:
        p = os.path.join(model_path, fname)
        if os.path.isfile(p):
            ok(f"  {fname}")
        else:
            fail(f"  {fname} not found — model download may be incomplete")

    import glob
    shards = glob.glob(os.path.join(model_path, "*.safetensors"))
    if shards:
        total_gb = sum(os.path.getsize(f) for f in shards) / 1e9
        ok(f"  {len(shards)} safetensors shard(s), {total_gb:.1f} GB")
    else:
        fail("  no .safetensors found — model download may be incomplete")

    return True


def check_argparse_compat() -> bool:
    step("Check 3 / 6  — oneshot argument parsing  (Python 3.14 compat)")
    try:
        from transformers import HfArgumentParser
        from llmcompressor.args.dataset_arguments import DatasetArguments
        from llmcompressor.args.model_arguments import ModelArguments
        # oneshot() does exactly this internally; Python 3.14 _check_help would
        # raise ValueError on "train[:50%]" help strings without our patch
        parser = HfArgumentParser([ModelArguments, DatasetArguments])
        ok("HfArgumentParser instantiation OK (argparse _check_help patch working)")
        return True
    except Exception as e:
        return fail("HfArgumentParser instantiation failed — argparse patch may not be working", e)


def check_recipe() -> bool:
    step("Check 4 / 6  — recipe instantiation")
    try:
        from llmcompressor.modifiers.transform.awq import AWQModifier
        from llmcompressor.modifiers.quantization import QuantizationModifier
        from compressed_tensors.quantization import (
            QuantizationScheme, QuantizationArgs,
            QuantizationType, QuantizationStrategy,
        )
        ok("imports fine")

        for bits in (4, 3):
            scheme = QuantizationScheme(
                targets=["Linear"],
                weights=QuantizationArgs(
                    num_bits=bits, type=QuantizationType.INT,
                    strategy=QuantizationStrategy.GROUP,
                    group_size=128, symmetric=False,
                ),
            )
            recipe = [
                AWQModifier(duo_scaling="both"),
                QuantizationModifier(
                    config_groups={"default": scheme},
                    ignore=["lm_head"],
                ),
            ]
            ok(f"W{bits}G128 recipe: {[type(m).__name__ for m in recipe]}")
        return True
    except Exception as e:
        return fail("recipe instantiation failed", e)


def check_gpu() -> bool:
    step("Check 5 / 6  — GPU memory")
    try:
        import torch
        if not torch.cuda.is_available():
            print(f"  {SKIP}  no CUDA — quantization will run on CPU (very slow!)")
            return True
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            free, total = torch.cuda.mem_get_info(i)
            free_gb = free / 1e9
            total_gb = total / 1e9
            ok(f"GPU {i}: {props.name}  {total_gb:.0f} GB total  {free_gb:.1f} GB free")
            if total_gb < 16:
                print(f"       ⚠  less than 16 GB — may OOM loading the bf16 model (~18 GB)")
        return True
    except Exception as e:
        return fail("GPU check failed", e)


def check_forward_pass(model_path: str) -> bool:
    step("Check 6 / 6  — model load + forward pass  (--full, ~60 s)")
    try:
        import torch
        from transformers import AutoTokenizer, AutoConfig

        t0 = time.time()
        print("  Loading tokenizer …")
        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        ok(f"tokenizer loaded in {time.time()-t0:.1f}s")

        print("  Loading config …")
        cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        ok(f"model_type={cfg.model_type}, arch={cfg.architectures}")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  Loading model in bf16 on {device} (may take 30-60 s) …")
        t1 = time.time()

        arch = getattr(cfg, "architectures", [])
        if "Qwen3_5ForConditionalGeneration" in arch:
            from transformers import AutoModelForImageTextToText
            model = AutoModelForImageTextToText.from_pretrained(
                model_path, dtype=torch.bfloat16, device_map=device,
                trust_remote_code=True,
            )
        else:
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(
                model_path, dtype=torch.bfloat16, device_map=device,
                trust_remote_code=True,
            )
        ok(f"model loaded in {time.time()-t1:.1f}s  ({type(model).__name__})")

        print("  Running forward pass (4 tokens) …")
        t2 = time.time()
        ids = tok("Hello world", return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            out = model(input_ids=ids)
        ok(f"forward pass OK in {time.time()-t2:.1f}s  logits shape={tuple(out.logits.shape)}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return True

    except Exception as e:
        return fail("forward pass failed", e)


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true",
                        help="Also load the model and run a forward pass (~60s, needs GPU)")
    parser.add_argument("--model-path", default=None)
    args = parser.parse_args()

    print()
    print("QuantStar quantization pre-flight checks")
    print("=" * 60)

    results = []
    results.append(("imports",        check_imports()))
    results.append(("config",         check_config()))
    results.append(("argparse-compat", check_argparse_compat()))
    results.append(("recipe",         check_recipe()))
    results.append(("gpu",            check_gpu()))

    if args.full:
        model_path = args.model_path or _resolve_model_path()
        if model_path:
            results.append(("forward-pass", check_forward_pass(model_path)))
        else:
            print(f"\n  {SKIP}  --full skipped: model path not found")
    else:
        print(f"\n  {SKIP}  forward-pass skipped (run with --full to test model load + inference)")

    # summary
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    all_ok = True
    for name, passed in results:
        icon = PASS if passed else FAIL
        print(f"  {icon}  {name}")
        if not passed:
            all_ok = False

    print()
    if all_ok:
        print("All checks passed.  You can now run:")
        print("  python quant_scripts/quantize_model.py")
    else:
        print("Some checks failed — fix the issues above before running quantize_model.py.")
    print()

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
