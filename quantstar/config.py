from __future__ import annotations

import os
from dataclasses import dataclass, field

import yaml


@dataclass
class ModelConfig:
    repo: str = "Qwen/Qwen3.6-27B"
    cache_dir: str = "./models"
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "sdpa"


@dataclass
class InferenceConfig:
    max_new_tokens: int = 65536
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 9898


@dataclass
class LoggingConfig:
    level: str = "INFO"


@dataclass
class QuantStarConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def load_config(path: str = "config.yaml") -> QuantStarConfig:
    cfg = QuantStarConfig()

    if os.path.exists(path):
        with open(path) as f:
            raw = yaml.safe_load(f) or {}

        if "model" in raw:
            for k, v in raw["model"].items():
                if hasattr(cfg.model, k):
                    setattr(cfg.model, k, v)
        if "inference" in raw:
            for k, v in raw["inference"].items():
                if hasattr(cfg.inference, k):
                    setattr(cfg.inference, k, v)
        if "server" in raw:
            for k, v in raw["server"].items():
                if hasattr(cfg.server, k):
                    setattr(cfg.server, k, v)
        if "logging" in raw:
            for k, v in raw["logging"].items():
                if hasattr(cfg.logging, k):
                    setattr(cfg.logging, k, v)

    for key, value in os.environ.items():
        if key == "QUANTSTAR_MODEL_REPO":
            cfg.model.repo = value
        elif key == "QUANTSTAR_MODEL_CACHE":
            cfg.model.cache_dir = value
        elif key == "QUANTSTAR_HOST":
            cfg.server.host = value
        elif key == "QUANTSTAR_PORT":
            cfg.server.port = int(value)
        elif key == "QUANTSTAR_LOG_LEVEL":
            cfg.logging.level = value

    return cfg
