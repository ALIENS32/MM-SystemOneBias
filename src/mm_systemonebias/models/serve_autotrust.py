"""Launch the official AutoTrust System One service from a downloaded checkpoint."""

from __future__ import annotations
import argparse
import os
import subprocess
import sys
from pathlib import Path
import yaml
from .resources import require_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--port", type=int)
    parser.add_argument("--visible-devices")
    parser.add_argument("--tensor-parallel-size", type=int)
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--max-num-seqs", type=int)
    parser.add_argument("--gpu-memory-utilization", type=float)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    path = require_model(config)
    port = args.port or config.get("port", 8000)
    visible = args.visible_devices or os.environ.get("CUDA_VISIBLE_DEVICES")
    environment = os.environ.copy()
    if visible:
        environment["CUDA_VISIBLE_DEVICES"] = visible
    command = [
        sys.executable,
        str(path / "serve_decide.py"),
        "--model",
        str(path),
        "--served-model-name",
        config["source"],
        "--enable-lora",
        "--max-lora-rank",
        "32",
        "--lora-modules",
        f"jev-decision={path / 'adapter_vllm'}",
        "--logprobs-mode",
        "processed_logprobs",
        "--max-model-len",
        str(args.max_model_len or config.get("max_model_len", 32768)),
        "--enable-prefix-caching",
        "--mamba-cache-mode",
        "align",
        "--limit-mm-per-prompt",
        '{"image":8}',
        "--max-num-seqs",
        str(args.max_num_seqs or config.get("max_num_seqs", 16)),
        "--trust-request-chat-template",
        "--tensor-parallel-size",
        str(args.tensor_parallel_size or config.get("tensor_parallel_size", 2)),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization or config.get("gpu_memory_utilization", 0.9)),
        "--port",
        str(port),
    ]
    print(f"status: serving\nmodel: {config['id']}\nport: {port}\nartifacts: {path}", flush=True)
    subprocess.run(command, env=environment, check=True)


if __name__ == "__main__":
    main()
