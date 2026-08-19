#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["huggingface-hub>=0.34"]
# ///
"""Preflight check for LLM training.

Reports hardware, auth, and toolchain state so a run can be routed to local
GPU vs Hugging Face Jobs before any GPU time is spent.

    uv run preflight.py
    uv run preflight.py --json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys

OK, WARN, BAD = "[ok]", "[!]", "[x]"


def sh(cmd: list[str], timeout: int = 15) -> str | None:
    """Run a command, returning stripped stdout or None if it fails."""
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def detect_gpus() -> dict:
    """Detect NVIDIA GPUs via nvidia-smi, falling back to Apple MPS."""
    info: dict = {"backend": "cpu", "devices": [], "total_vram_gb": 0.0, "free_vram_gb": 0.0}

    csv = sh([
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.free,driver_version",
        "--format=csv,noheader,nounits",
    ])
    if csv:
        info["backend"] = "cuda"
        for line in csv.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            try:
                vram, free = float(parts[1]) / 1024.0, float(parts[2]) / 1024.0
            except ValueError:
                continue
            info["devices"].append(
                {"name": parts[0], "vram_gb": round(vram, 1), "free_gb": round(free, 1)}
            )
            info["total_vram_gb"] += vram
            info["free_vram_gb"] += free
        info["total_vram_gb"] = round(info["total_vram_gb"], 1)
        info["free_vram_gb"] = round(info["free_vram_gb"], 1)
        if info["devices"]:
            info["driver"] = parts[3] if len(parts) > 3 else None
        return info

    if platform.system() == "Darwin" and platform.machine() == "arm64":
        info["backend"] = "mps"
        mem = sh(["sysctl", "-n", "hw.memsize"])
        if mem and mem.isdigit():
            # MPS shares system memory; usable share is well below the total.
            info["total_vram_gb"] = round(int(mem) / 1024**3, 1)
            info["devices"].append(
                {"name": "Apple Silicon (unified memory)", "vram_gb": info["total_vram_gb"]}
            )
    return info


def ambient_versions() -> dict:
    """Versions in the ambient interpreter (informational; uv scripts self-resolve)."""
    probe = (
        "import json;d={}\n"
        "for m in ['torch','transformers','trl','peft','datasets','accelerate',"
        "'deepspeed','vllm','flash_attn','bitsandbytes','twinkle_eval']:\n"
        "    try:\n"
        "        d[m]=__import__(m).__version__\n"
        "    except Exception:\n"
        "        pass\n"
        "print(json.dumps(d))"
    )
    for exe in (shutil.which("python3"), shutil.which("python")):
        if not exe:
            continue
        out = sh([exe, "-c", probe], timeout=60)
        if out:
            try:
                return json.loads(out)
            except json.JSONDecodeError:
                pass
    return {}


def hf_auth() -> dict:
    try:
        from huggingface_hub import whoami
    except ImportError:
        return {"logged_in": False, "error": "huggingface_hub unavailable"}
    try:
        me = whoami()
        return {
            "logged_in": True,
            "user": me.get("name"),
            "orgs": [o.get("name") for o in me.get("orgs", []) if isinstance(o, dict)],
        }
    except Exception as exc:  # noqa: BLE001 - any auth failure means "not usable"
        return {"logged_in": False, "error": type(exc).__name__}


def collect() -> dict:
    gpus = detect_gpus()
    total, used, free = shutil.disk_usage(os.getcwd())
    return {
        "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "python": sys.version.split()[0],
        "uv": (sh(["uv", "--version"]) or "not found"),
        "hf_cli": (sh(["hf", "version"]) or sh(["hf", "--version"]) or "not found"),
        "gpu": gpus,
        "hf_auth": hf_auth(),
        "packages": ambient_versions(),
        "disk_free_gb": round(free / 1024**3, 1),
        "env": {
            k: ("<set>" if "TOKEN" in k else os.environ[k])
            for k in ("HF_TOKEN", "HF_HOME", "HF_HUB_CACHE", "CUDA_VISIBLE_DEVICES")
            if k in os.environ
        },
    }


def advise(d: dict) -> list[str]:
    """Turn raw facts into routing decisions."""
    out = []
    gpu, n = d["gpu"], len(d["gpu"]["devices"])
    vram = gpu["total_vram_gb"]

    if gpu["backend"] == "cuda":
        free = gpu.get("free_vram_gb", vram)
        out.append(f"{OK} CUDA: {n} device(s), {free} GB free of {vram} GB total VRAM")
        # Another process holding VRAM is invisible in the total, and sizing a
        # run against the total is then an OOM waiting to happen.
        if vram - free > 1.0:
            out.append(
                f"{WARN} {round(vram - free, 1)} GB already in use by another process. "
                f"Size the run against {free} GB: plan_memory.py --vram {free}"
            )
        if n > 1:
            out.append(
                f"{OK} Multi-GPU available -> use accelerate/DeepSpeed ZeRO-3 "
                f"(see references/hardware.md)"
            )
        if vram < 24:
            out.append(f"{WARN} <24 GB total: prefer QLoRA, or offload to HF Jobs")
    elif gpu["backend"] == "mps":
        out.append(f"{WARN} Apple Silicon (MPS), {vram} GB unified memory")
        out.append(
            f"{WARN} No CUDA: bitsandbytes/flash-attn/DeepSpeed unavailable. "
            f"Use --smoke-test only; route real runs to a CUDA box or HF Jobs."
        )
    else:
        out.append(f"{BAD} No GPU detected -> HF Jobs is the only practical target")

    auth = d["hf_auth"]
    if auth.get("logged_in"):
        orgs = f" (orgs: {', '.join(auth['orgs'])})" if auth.get("orgs") else ""
        out.append(f"{OK} Hugging Face: logged in as {auth['user']}{orgs}")
    else:
        out.append(f"{BAD} Not logged in to Hugging Face -> run `hf auth login`")
        out.append(f"{WARN} Gated datasets and push_to_hub will fail without it")

    if d["uv"] == "not found":
        # uv builds the script environments; it is not needed to run inside one.
        # Invoking a script through its own interpreter is a documented pattern,
        # and uv is often absent from that PATH - not an error.
        out.append(f"{WARN} uv not on PATH - fine inside an existing script env; "
                   f"needed to create one (https://docs.astral.sh/uv/)")
    else:
        out.append(f"{OK} uv {d['uv']}")

    if d["hf_cli"] == "not found":
        out.append(f"{WARN} `hf` CLI not on PATH -> only needed for HF Jobs and gated logins")

    if d["disk_free_gb"] < 100:
        out.append(
            f"{WARN} Only {d['disk_free_gb']} GB free. Model + dataset caches are large; "
            f"set HF_HOME to a bigger volume."
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Preflight check for LLM training")
    ap.add_argument("--json", action="store_true", help="emit raw JSON only")
    args = ap.parse_args()

    data = collect()
    if args.json:
        print(json.dumps(data, indent=2, ensure_ascii=False))
        return 0

    print("=" * 68)
    print("  hsun-trainer preflight")
    print("=" * 68)
    print(f"platform    : {data['platform']}")
    print(f"python      : {data['python']}")
    print(f"disk free   : {data['disk_free_gb']} GB")
    for dev in data["gpu"]["devices"]:
        free = f", {dev['free_gb']} GB free" if "free_gb" in dev else ""
        print(f"device      : {dev['name']} ({dev['vram_gb']} GB{free})")
    if data["packages"]:
        print("ambient pkgs: " + ", ".join(f"{k}=={v}" for k, v in sorted(data["packages"].items())))
    if data["env"]:
        print("env         : " + ", ".join(f"{k}={v}" for k, v in data["env"].items()))

    print("-" * 68)
    for line in advise(data):
        print(line)
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
