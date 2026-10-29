"""Evaluate selected generator subsets in Grounding Probes Table 2.

Uses released detectors and an existing local RAGTruth test split. No answer
generation, probe training, score caching, or automatic error fallback occurs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import logging
import math
import os
from pathlib import Path
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone

# Allow `python scripts/reproduce_table2.py` from an uninstalled checkout too.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path("/mnt/data/wenzelin")
GENERATORS = ("llama-2-7b-chat", "llama-2-13b-chat")
SUPPORTED_GENERATORS = (*GENERATORS, "mistral-7B-instruct")
TASKS = ("Summary", "Data2txt", "QA")
ALPHA = 0.45
THRESHOLD = 0.5
LOG = logging.getLogger("reproduce_table2")


def probability(value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"Invalid hallucination probability: {value}")
    return value


def token_score(tokens: list[dict]) -> float:
    if not tokens:
        raise ValueError("LettuceDetect returned no answer tokens")
    # Keep continuous probabilities, including tokens predicted as supported.
    return max(probability(token["prob"]) for token in tokens)


def paper_token_score(tokens: list[dict]) -> float:
    """Match the paper benchmark: zero if no hallucinated token/span exists."""
    token_score(tokens)  # Validate every probability, even on supported tokens.
    return max((probability(t["prob"]) for t in tokens if t["pred"] == 1), default=0.0)


def blend_score(lettuce: float, probe: float) -> float:
    return ALPHA * probability(lettuce) + (1 - ALPHA) * probability(probe)


def hallucination_label(item: dict) -> int:
    labels = item["hallucination_labels_processed"] or {}
    result = int(bool(labels.get("evident_conflict") or labels.get("baseless_info")))
    # Independently check the response-level annotation against the raw spans
    # carried by the pinned processed dataset.
    raw_spans = json.loads(item["hallucination_labels"])
    if result != int(bool(raw_spans)):
        raise ValueError(f"Processed label disagrees with raw spans: {item['id']}")
    return result


def select_samples(test, smoke: bool, generators=GENERATORS) -> list[dict]:
    if not generators or len(set(generators)) != len(generators):
        raise ValueError("Specify at least one generator, without duplicates")
    if set(generators) - set(SUPPORTED_GENERATORS):
        raise ValueError(f"Unsupported generators: {generators}")
    samples = []
    ids = set()
    for index, item in enumerate(test):
        if item["model"] not in generators:
            continue
        sample_id = str(item["id"])
        if sample_id in ids:
            raise ValueError(f"Duplicate response ID: {sample_id}")
        ids.add(sample_id)
        for key in ("query", "context", "output"):
            if not isinstance(item[key], str):
                raise ValueError(f"{sample_id}: {key} must be a string")
        if not item["context"].strip() or not item["output"].strip():
            raise ValueError(f"{sample_id}: empty context or response")
        samples.append(dict(item, sample_id=sample_id, test_index=index,
                            label=hallucination_label(item)))
    counts = Counter((s["model"], s["task_type"]) for s in samples)
    expected = Counter({(g, t): 150 for g in generators for t in TASKS})
    if counts != expected:
        raise ValueError(f"Expected 150 responses per generator/task; got {dict(counts)}")
    if smoke:
        seen = set()
        chosen = []
        for sample in samples:
            key = (sample["model"], sample["task_type"])
            if key not in seen:
                chosen.append(sample)
                seen.add(key)
        return chosen
    return samples


def summarize(rows: list[dict]) -> dict:
    from sklearn.metrics import f1_score, roc_auc_score

    result = {}
    for generator in dict.fromkeys(r["generator"] for r in rows):
        group = [r for r in rows if r["generator"] == generator]
        labels = [r["label"] for r in group]
        methods = {}
        for method in ("lettucedetect", "probe", "blend"):
            scores = [probability(r[method]) for r in group]
            methods[method] = {
                "auroc": float(roc_auc_score(labels, scores)) if len(set(labels)) == 2 else None,
                "f1_at_0_5": float(f1_score(
                    labels, [int(s >= THRESHOLD) for s in scores], zero_division=0
                )),
            }
        result[generator] = {
            "n": len(group), "n_hallucinated": sum(labels),
            "tasks": dict(Counter(r["task_type"] for r in group)),
            "methods": methods,
        }
    return result


def load_inputs(args):
    import numpy as np
    from datasets import load_from_disk
    from lettucedetect.detectors.stage3.probes.grounding_probe import GroundingProbe

    ds = load_from_disk(str(args.data_dir))
    if len(ds["train"]) != 15090 or len(ds["test"]) != 2700:
        raise ValueError("Expected official RAGTruth train/test sizes 15090/2700")
    samples = select_samples(ds["test"], args.smoke, args.generators)
    for model_dir in (args.observer, args.lettuce_model):
        if not (model_dir / "config.json").is_file():
            raise FileNotFoundError(model_dir / "config.json")
    probe = GroundingProbe.load(args.probe)
    required = {"family": "qwen", "model_size": "14b", "layer_index": -20,
                "token_position": "mean", "pca_components": 512}
    if any(probe.metadata.get(k) != v for k, v in required.items()):
        raise ValueError(f"Unexpected probe metadata: {probe.metadata}")
    prediction = probe.predict_proba(np.zeros((1, 5120), dtype=np.float32))
    probability(prediction[0])
    return ds, samples, probe.metadata


def evaluate(args, samples: list[dict]) -> list[dict]:
    import gc
    import torch
    from lettucedetect.detectors.transformer import TransformerDetector
    from lettucedetect.detectors.stage3.grounding_probe_detector import GroundingProbeDetector

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one CUDA GPU using CUDA_VISIBLE_DEVICES")
    torch.manual_seed(0)
    LOG.info("GPU: %s; visible device: %s", torch.cuda.get_device_name(0),
             os.environ.get("CUDA_VISIBLE_DEVICES", "unset"))
    rows = [dict(sample_id=s["sample_id"], test_index=s["test_index"],
                 generator=s["model"], task_type=s["task_type"], label=s["label"])
            for s in samples]
    # Run separately so the two models need not reside in GPU memory together.
    LOG.info("Loading LettuceDetect-large (released FP32 weights)")
    detector = TransformerDetector(str(args.lettuce_model), max_length=4096, device="cuda:0")
    for i, (sample, row) in enumerate(zip(samples, rows), 1):
        LOG.info("LettuceDetect %d/%d id=%s", i, len(samples), row["sample_id"])
        tokens = detector.predict(
            context=[sample["context"]], answer=sample["output"],
            question=sample["query"], output_format="tokens",
        )
        row["lettucedetect_raw"] = token_score(tokens)
        row["lettucedetect"] = paper_token_score(tokens)
    del detector
    gc.collect()
    torch.cuda.empty_cache()

    LOG.info("Loading Qwen2.5-14B-Instruct observer in FP16")
    detector = GroundingProbeDetector(
        model_name_or_path=str(args.observer), probe_path=str(args.probe),
        layer_index=-20, token_position="mean", threshold=THRESHOLD,
    )
    # The official loader uses device_map=auto; reject CPU/disk offload.
    if any(p.device.type != "cuda" or p.dtype != torch.float16
           for p in detector._model.parameters()):
        raise RuntimeError("Observer must fit on the selected GPU entirely in FP16")
    detector._model.eval()
    for i, (sample, row) in enumerate(zip(samples, rows), 1):
        LOG.info("Probe %d/%d id=%s", i, len(samples), row["sample_id"])
        result = detector.predict_uncertainty(
            context=[sample["context"]], answer=sample["output"], question=sample["query"],
        )
        row["probe"] = probability(result.hallucination_score)
        row["blend"] = blend_score(row["lettucedetect"], row["probe"])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "datasets/RAGTruth-processed")
    parser.add_argument("--observer", type=Path, default=ROOT / "models/Qwen2.5-14B-Instruct")
    parser.add_argument("--lettuce-model", type=Path,
                        default=ROOT / "models/lettucedect-large-modernbert-en-v1")
    parser.add_argument("--probe", type=Path, default=ROOT / (
        "models/lettucedetect-grounding-probes/probe_14b_qwen_pca512.joblib"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--generators", nargs="+", choices=SUPPORTED_GENERATORS,
                        default=list(GENERATORS),
                        help="Generator subsets to evaluate (default: the two Llama-2 models)")
    parser.add_argument("--smoke", action="store_true", help="One sample per selected generator/task")
    parser.add_argument("--preflight", action="store_true", help="CPU data/probe checks only; no outputs")
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        stream=sys.stdout)
    ds, samples, metadata = load_inputs(args)
    LOG.info("Validated official test split; selected %d responses: %s", len(samples),
             dict(Counter(s["model"] for s in samples)))
    if args.preflight:
        LOG.info("CPU preflight passed; no large models loaded and no GPU inference performed")
        return

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output = args.output_dir or Path(__file__).resolve().parents[1] / "outputs" / (
        f"table2-{'smoke' if args.smoke else 'full'}-{stamp}")
    output.mkdir(parents=True, exist_ok=False)
    handler = logging.FileHandler(output / "run.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(handler)
    versions = {p: importlib.metadata.version(p) for p in (
        "torch", "transformers", "scikit-learn", "numpy", "datasets", "accelerate", "joblib")}
    repo = Path(__file__).resolve().parents[1]
    config = {
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "output_dir": str(output.resolve()), "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], cwd=repo, text=True),
        "versions": versions, "test_fingerprint": ds["test"]._fingerprint,
        "probe_sha256": hashlib.sha256(args.probe.read_bytes()).hexdigest(),
        "probe_metadata": metadata, "alpha_lettucedetect": ALPHA, "threshold": THRESHOLD,
        "observer_dtype": "float16", "observer_layer": -20, "pooling": "response_mean",
        "lettucedetect_dtype": "float32", "max_length": 4096,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "n_samples": len(samples), "generators": args.generators,
        "lettucedetect_score_rule": "max probability over predicted hallucinated tokens; zero if none",
        "augmentations": [], "strategy": "blend",
    }
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    LOG.info("Output directory: %s", output.resolve())
    try:
        rows = evaluate(args, samples)
        metrics = summarize(rows)
        with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        LOG.info("Completed %d responses\n%s", len(rows), json.dumps(metrics, indent=2))
        if args.smoke:
            LOG.info("Smoke metrics are diagnostic only")
        else:
            LOG.info("Full %d-response evaluation complete", len(rows))
    except Exception:
        LOG.exception("Evaluation failed; no fallback, no resume, and no completed metrics")
        raise


if __name__ == "__main__":
    main()
