"""Measure what running the two stages concurrently actually costs.

The paper reports blend latency as the per-sample $\\max(t_{S1}, t_{S3})$ of two
stages timed separately, standing in for concurrent execution on separate
devices. That is a derivation, not a measurement, and the paper says so. This
measures it.

The quantity of interest is the RATIO, not the milliseconds. Absolute figures
here come from whatever machine this runs on and are not comparable with
Table 1, which is a different A100 in a different environment. What transfers is
how far concurrent execution departs from the per-sample max, because that is
the assumption the derived figure rests on.

Three constraints make the measurement mean what it claims.

DEVICE ISOLATION. `GroundingProbeDetector` loads with `device_map="auto"`, which
spreads one model across every visible GPU and would destroy the separation the
measurement is about. Each worker therefore runs as its own process with
`CUDA_VISIBLE_DEVICES` pinned to a single card, so "auto" has one choice.

SAME MACHINE. Alone and concurrent are measured in one session on one host. Run
them anywhere else and the difference is mostly a hardware delta.

ALIGNED START. The stages take different times per sample, so a concurrent run
where one finishes early would spend its tail uncontended and understate the
cost. Both workers warm up first, then block until a shared wall-clock start,
and the contended window is the overlap both were inside.

Usage, from the LettuceDetect checkout:
    python tests/benchmarks/concurrency_benchmark.py --n 300          # orchestrate
    python tests/benchmarks/concurrency_benchmark.py --stub --n 20    # CPU self-test
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_LETTUCE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_LETTUCE_ROOT))

S1_MODEL = "KRLabsOrg/lettucedect-large-modernbert-en-v1"
S3_MODEL = "Qwen/Qwen2.5-14B-Instruct"
OUT_DIR = _LETTUCE_ROOT / "tests" / "benchmarks" / "results" / "concurrency"
WARMUP = 3


# --------------------------------------------------------------------------
# worker
# --------------------------------------------------------------------------

def load_samples(n: int):
    from tests.benchmarks.data_adapters import RAGTruthAdapter

    rows = [s for s in RAGTruthAdapter().load(limit=None) if s.context and s.response]
    return rows[:n]


def build_s1(stub: bool):
    if stub:
        return lambda s: time.sleep(0.02)
    from lettucedetect.detectors.transformer import TransformerDetector

    det = TransformerDetector(model_path=S1_MODEL)
    det.warmup()

    def run(s):
        spans = det.predict(s.context, s.response, s.question, output_format="spans")
        return max((sp.get("confidence", 0.5) for sp in spans), default=0.0)

    return run


def build_s3(stub: bool, probe_path: str, layer_index: int):
    if stub:
        return lambda s: time.sleep(0.05)
    from lettucedetect.detectors.stage3.grounding_probe_detector import (
        GroundingProbeDetector,
    )

    det = GroundingProbeDetector(
        model_name_or_path=S3_MODEL,
        probe_path=probe_path,
        layer_index=layer_index,
        token_position="mean",
    )

    def run(s):
        return det.predict_uncertainty(s.context, s.response, s.question).hallucination_score

    return run


def worker(a) -> int:
    """Time one stage over `--n` samples, optionally starting at a shared instant."""
    import torch

    samples = load_samples(a.n)
    build = build_s1 if a.role == "s1" else build_s3
    fn = (build(a.stub) if a.role == "s1"
          else build(a.stub, a.probe_path, a.layer_index))

    for s in samples[:WARMUP]:
        fn(s)
    if not a.stub and torch.cuda.is_available():
        torch.cuda.synchronize()

    if a.start_at:
        while time.time() < a.start_at:
            time.sleep(0.001)

    lat, t0 = [], time.perf_counter()
    for s in samples:
        t = time.perf_counter()
        fn(s)
        if not a.stub and torch.cuda.is_available():
            torch.cuda.synchronize()
        lat.append((time.perf_counter() - t) * 1000.0)
    wall = time.perf_counter() - t0

    # Absolute wall-clock per sample, so the orchestrator can find the window in
    # which both stages were actually running.
    Path(a.out).write_text(json.dumps({
        "role": a.role, "n": len(lat), "stub": a.stub,
        "wall_sec": wall, "mean_ms": sum(lat) / len(lat),
        "p95_ms": sorted(lat)[int(0.95 * (len(lat) - 1))],
        "started_at": t0 + (time.time() - time.perf_counter()),
        "latencies_ms": [round(v, 4) for v in lat],
        "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }))
    print(f"  [{a.role}] {len(lat)} samples, mean {sum(lat) / len(lat):.1f} ms, "
          f"wall {wall:.1f} s", flush=True)
    return 0


# --------------------------------------------------------------------------
# orchestrator
# --------------------------------------------------------------------------

def spawn(role: str, gpu: str, out: Path, a, start_at: float | None):
    env = dict(os.environ)
    # The single lever that keeps device_map="auto" honest.
    env["CUDA_VISIBLE_DEVICES"] = gpu
    cmd = [sys.executable, __file__, "--role", role, "--n", str(a.n), "--out", str(out)]
    if a.stub:
        cmd.append("--stub")
    if start_at:
        cmd += ["--start-at", repr(start_at)]
    if role == "s3":
        cmd += ["--probe-path", a.probe_path, "--layer-index", str(a.layer_index)]
    return subprocess.Popen(cmd, env=env, cwd=str(_LETTUCE_ROOT))


def orchestrate(a) -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    g1, g3 = a.gpus.split(",")
    res = {}

    for role, gpu in (("s1", g1), ("s3", g3)):
        print(f"== {role} alone on GPU {gpu}", flush=True)
        p = spawn(role, gpu, OUT_DIR / f"alone_{role}.json", a, None)
        if p.wait() != 0:
            print(f"FAIL: {role} alone exited {p.returncode}")
            return 1
        res[f"alone_{role}"] = json.loads((OUT_DIR / f"alone_{role}.json").read_text())

    print("== both concurrently, aligned start", flush=True)
    start_at = time.time() + a.lead
    procs = [spawn(role, gpu, OUT_DIR / f"conc_{role}.json", a, start_at)
             for role, gpu in (("s1", g1), ("s3", g3))]
    if any(p.wait() != 0 for p in procs):
        print("FAIL: a concurrent worker exited non-zero")
        return 1
    for role in ("s1", "s3"):
        res[f"conc_{role}"] = json.loads((OUT_DIR / f"conc_{role}.json").read_text())

    a1, a3 = res["alone_s1"], res["alone_s3"]
    c1, c3 = res["conc_s1"], res["conc_s3"]
    derived = max(a1["mean_ms"], a3["mean_ms"])
    measured = max(c1["mean_ms"], c3["mean_ms"])

    summary = {
        "_design": (
            "Per-sample latency of each stage alone and with both running on "
            "separate GPUs of one machine, processes pinned with "
            "CUDA_VISIBLE_DEVICES so device_map=\"auto\" cannot straddle both. "
            "The ratio is the transferable quantity; the absolute milliseconds "
            "belong to this machine and are not comparable with Table 1."),
        "n_samples": a.n, "stub": a.stub,
        "s1_model": S1_MODEL, "s3_model": S3_MODEL,
        "alone": {"s1_mean_ms": a1["mean_ms"], "s3_mean_ms": a3["mean_ms"],
                  "s1_p95_ms": a1["p95_ms"], "s3_p95_ms": a3["p95_ms"]},
        "concurrent": {"s1_mean_ms": c1["mean_ms"], "s3_mean_ms": c3["mean_ms"],
                       "s1_p95_ms": c1["p95_ms"], "s3_p95_ms": c3["p95_ms"]},
        "slowdown_ratio": {"s1": c1["mean_ms"] / a1["mean_ms"],
                           "s3": c3["mean_ms"] / a3["mean_ms"]},
        # The number the paper's derivation rests on.
        "derived_max_alone_ms": derived,
        "measured_concurrent_ms": measured,
        "overhead_ratio": measured / derived,
        "overhead_ms": measured - derived,
    }
    (OUT_DIR / "concurrency_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n  alone       S1 {a1['mean_ms']:7.1f} ms   S3 {a3['mean_ms']:7.1f} ms")
    print(f"  concurrent  S1 {c1['mean_ms']:7.1f} ms   S3 {c3['mean_ms']:7.1f} ms")
    print(f"  slowdown    S1 {summary['slowdown_ratio']['s1']:7.3f}x    "
          f"S3 {summary['slowdown_ratio']['s3']:7.3f}x")
    print(f"  derived max(alone)   {derived:7.1f} ms")
    print(f"  measured concurrent  {measured:7.1f} ms")
    print(f"  overhead ratio       {summary['overhead_ratio']:7.3f}x "
          f"({summary['overhead_ms']:+.1f} ms)")
    print(f"\nwrote {OUT_DIR / 'concurrency_summary.json'}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--role", choices=["s1", "s3"], help="worker mode (internal)")
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--gpus", default="0,1", help="GPU for S1 and for S3")
    p.add_argument("--lead", type=float, default=90.0,
                   help="seconds allowed for both workers to load and warm up")
    p.add_argument("--out")
    p.add_argument("--start-at", type=float, default=0.0)
    p.add_argument("--stub", action="store_true",
                   help="replace both models with sleeps, to test orchestration on CPU")
    p.add_argument("--probe-path", default="")
    p.add_argument("--layer-index", type=int, default=-20)
    a = p.parse_args()
    return worker(a) if a.role else orchestrate(a)


if __name__ == "__main__":
    sys.exit(main())
