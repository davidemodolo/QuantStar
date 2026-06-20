# QuenStar v2

**Qwen3.6-27B quantized inference in 24 GB VRAM.**

- 4-bit weight quantization (bitsandbytes NF4)
- 4-bit KV cache quantization (quanto backend, patched for DeltaNet)
- OpenAI-compatible streaming API server
- Interactive CLI chat (Rich)

## Architecture

```
quenstar/
├── __init__.py     # package version
├── __main__.py     # CLI entry point (download/serve/chat/info)
├── config.py       # dataclass config, YAML + env var loading
├── download.py     # HuggingFace Hub model download
├── quantize.py     # model load + bitsandbytes 4-bit NF4 + quantized KV cache
├── engine.py       # InferenceEngine: generation, tokenization, VRAM monitor
├── server.py       # FastAPI OpenAI-compatible API server
└── cli.py          # Rich-based interactive chat
```

Qwen3.6-27B has a hybrid architecture of 48 linear_attention (DeltaNet) layers + 16 full_attention layers. The QuantizedCache is monkey-patched to dispatch each layer to the correct cache type (quanto-quantized attention or quanto-quantized linear attention with recurrent states).

## Quickstart

```bash
./run.sh download    # download Qwen3.6-27B (52 GB, one-time)
./run.sh chat        # interactive CLI
./run.sh serve       # OpenAI API on 127.0.0.1:9898
./run.sh info        # show configuration
./run.sh init        # register in OpenCode config
```

Or use Python directly:

```bash
python -m quenstar download
python -m quenstar serve
python -m quenstar chat
python -m quenstar info
```

## Configuration

Edit `config.yaml` or use environment variables (`QUENSTAR_*` prefix):

### config.yaml

```yaml
model:
  repo: "Qwen/Qwen3.6-27B"
  cache_dir: "./models"
  torch_dtype: "bfloat16"
  attn_implementation: "sdpa"    # or "flash_attention_2" / "eager"

quantization:
  weight_bits: 4                 # bitsandbytes NF4
  kv_cache_bits: 4               # quanto 4-bit KV cache

inference:
  max_context: 262144            # full 262k context window
  max_new_tokens: 65536
  temperature: 0.7
  top_p: 0.8
  top_k: 20
  presence_penalty: 1.5

server:
  host: "127.0.0.1"
  port: 9898

logging:
  level: "INFO"
```

### Environment variables

| Variable | Overrides |
|----------|-----------|
| `QUENSTAR_MODEL_REPO` | `model.repo` |
| `QUENSTAR_MODEL_CACHE` | `model.cache_dir` |
| `QUENSTAR_WEIGHT_BITS` | `quantization.weight_bits` |
| `QUENSTAR_KV_BITS` | `quantization.kv_cache_bits` |
| `QUENSTAR_MAX_CONTEXT` | `inference.max_context` |
| `QUENSTAR_HOST` | `server.host` |
| `QUENSTAR_PORT` | `server.port` |
| `QUENSTAR_LOG_LEVEL` | `logging.level` |

## API

OpenAI-compatible endpoints served by FastAPI on `127.0.0.1:9898`:

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness check |
| `GET` | `/health/vram` | GPU memory stats (`allocated_gb`, `total_gb`, `reserved_gb`, `free_gb`) |
| `GET` | `/v1/models` | List models |
| `GET` | `/v1/models/{id}` | Get single model |
| `POST` | `/v1/chat/completions` | Chat completion (streaming SSE + non-streaming) |

### Streaming response format

When `stream: true`, each SSE chunk follows the OpenAI delta format:

- `delta.content` — regular response text
- `delta.reasoning_content` / `delta.reasoning_text` — model thinking (between `<think>`/`</think>` tags)
- `delta.tool_calls` — incremental tool call deltas (parsed from `<tool_call>` XML)

The server strips `<think>`/`</think>` tags from content and emits them as `reasoning_content` deltas. For small tasks (e.g. title generation), thinking is auto-disabled to avoid leaking raw tags.

### Tool calling

When `tools` are provided, the model uses `<tool_call>` XML syntax. The server parses this incrementally in streaming mode, emitting `tool_calls` deltas with function name and arguments. In non-streaming mode, the full `<tool_call>` XML is parsed and returned as a structured `tool_calls` array.

## Memory Budget

| Component | Size |
|-----------|------|
| Weights (4-bit NF4) | ~16.5 GB |
| KV cache per 1k tokens | ~16 KB (4-bit quantized) |
| Full 262k context KV cache | ~4.3 GB |
| Overhead (CUDA runtime) | ~1-2 GB |
| **Total at 262k context** | **~22 GB** |

| Context | KV cache | Total VRAM |
|---------|----------|------------|
| 32k | 0.5 GB | 18.0 GB |
| 64k | 1.1 GB | 18.6 GB |
| 128k | 2.1 GB | 19.6 GB |
| 262k | 4.3 GB | 21.8 GB |

## Dependencies

**Runtime:** torch, transformers, accelerate, bitsandbytes, quanto, huggingface_hub, fastapi, uvicorn, sse-starlette, pyyaml, rich, tqdm

**Build:** setuptools

See `pyproject.toml` for version constraints. `run.sh` handles the full setup (venv creation, CUDA 12.6 pip index, dependency installation, transformers patching).

## Testing

```bash
./test_quenstar.sh
```

End-to-end bash test suite that:
1. Starts the server and waits for readiness
2. Tests single non-streaming request
3. Tests streaming (SSE) request
4. Tests 3 concurrent requests (stress test)
5. Verifies no `<think>` tag leak into content deltas
6. Checks server log for thread crashes, CUDA OOM, and triton autotune errors
