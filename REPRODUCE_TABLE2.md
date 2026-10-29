# Grounding Probes Table 2: Llama-2 and Mistral generators

This entrypoint evaluates the existing RAGTruth test responses, not new Llama
generations. It uses all 450 responses per generator (150 per task), for 900 in
total by default. Use `--generators mistral-7B-instruct` to evaluate only the
450 Mistral responses. The frozen observer is **Qwen2.5-14B-Instruct** for every
generator subset; no Mistral weights are needed.

## Environment and protocol

Run from `/mnt/data/wenzelin/GroundingProbes` on iie01, using the
`grounding-probes` Conda environment. The released probe requires scikit-learn
**1.8.0**, the version with which it was serialized.

The pinned `wandb/RAGTruth-processed` revision is
`eb4f4b9d1b68eb7092d3e1a61c0cd82d9808737b`. Its existing local `save_to_disk`
copy is used, with the official train/test split preserved. Labels are checked
against the raw span annotations carried by that copy.

LettuceDetect-large uses the official token detector in FP32, with response-level
score equal to the maximum probability over tokens predicted as hallucinated,
or zero when no hallucinated tokens/spans are predicted. This matches the
paper's benchmark score handling. The unthresholded token maximum is also
saved as `lettucedetect_raw` for inspection, but is not used in paper metrics.
The probe uses the official observer loader/extractor in FP16, layer -20,
response-token mean pooling, and the released PCA-512 probe. Both retain the
official input templates, 4096-token handling, and inference implementation.
The blend is `0.45 * lettucedetect + 0.55 * probe`; F1 uses threshold 0.5.
There are no lexical/model2vec augmentations, cascade routing, or error fallback.

The two detectors run in separate phases on one GPU to reduce peak residency.
This runner reports accuracy, not the paper's concurrent latency measurements.
It does not train, regenerate answers, cache scores, or resume interrupted runs.

## CPU checks

```bash
cd /mnt/data/wenzelin/GroundingProbes
source /mnt/data/wenzelin/anaconda3/etc/profile.d/conda.sh
conda activate grounding-probes
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
python -m unittest discover -s tests/unit -p test_reproduce_table2_pytest.py -v
python scripts/reproduce_table2.py --preflight
```

## GPU smoke test: 6 responses

Inspect `nvidia-smi` first. Replace `1` below with a GPU you can use; reserve
approximately 40 GB or more free memory for the FP16 observer and activations.
Only the selected card is exposed to the official automatic device mapper.

```bash
cd /mnt/data/wenzelin/GroundingProbes
source /mnt/data/wenzelin/anaconda3/etc/profile.d/conda.sh
conda activate grounding-probes
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
python -u scripts/reproduce_table2.py --smoke \
  --data-dir /mnt/data/wenzelin/datasets/RAGTruth-processed \
  --observer /mnt/data/wenzelin/models/Qwen2.5-14B-Instruct \
  --lettuce-model /mnt/data/wenzelin/models/lettucedect-large-modernbert-en-v1 \
  --probe /mnt/data/wenzelin/models/lettucedetect-grounding-probes/probe_14b_qwen_pca512.joblib
```

Smoke mode chooses the first test response in each generator/task cell, six in
total. Its metrics are diagnostic only; AUROC is null for a single-class group.

## Full evaluation: 900 responses

After smoke inference succeeds, run in the same environment with the same GPU:

```bash
python -u scripts/reproduce_table2.py \
  --data-dir /mnt/data/wenzelin/datasets/RAGTruth-processed \
  --observer /mnt/data/wenzelin/models/Qwen2.5-14B-Instruct \
  --lettuce-model /mnt/data/wenzelin/models/lettucedect-large-modernbert-en-v1 \
  --probe /mnt/data/wenzelin/models/lettucedetect-grounding-probes/probe_14b_qwen_pca512.joblib
```

## Mistral only: 450 responses

In the same activated environment, with a single available GPU exposed:

```bash
# CPU check only: must select 450 Mistral responses.
python scripts/reproduce_table2.py --generators mistral-7B-instruct --preflight

# Optional smoke inference: 3 responses, one per task.
python -u scripts/reproduce_table2.py --generators mistral-7B-instruct --smoke

# Full Mistral inference: 450 responses, no Llama-2 inference.
python -u scripts/reproduce_table2.py --generators mistral-7B-instruct
```

To select all three generators explicitly, pass
`--generators mistral-7B-instruct llama-2-7b-chat llama-2-13b-chat` (1350 responses).
Every selected generator must have 150 official test responses per task.
Summaries contain only the selected generators.

Earlier runs from commit `4ee09a5` used the unthresholded token maximum as the
primary LettuceDetect score. Their original files remain unchanged. Those
scores can be analyzed separately with values below 0.5 set to zero to match
the paper, as already verified for the two Llama-2 subsets. New runs use the
paper-compatible score directly and record the rule in `config.json`.

All evaluation loads are offline. Network operations such as Git fetch/push on
iie servers require `source /mnt/data/wenzelin/opt/bin/mix-lan.sh` first.

## Outputs and comparison

Each run creates a new `outputs/table2-smoke-<timestamp>/` or
`outputs/table2-full-<timestamp>/` directory and prints its absolute location.
Use `--output-dir /absolute/new/directory` to choose another **new** directory.
Existing output directories are rejected rather than overwritten.

- `run.log`: progress and any inference error, including the active sample ID.
- `config.json`: paths, package versions, Git state, data fingerprint, probe
  hash, and effective protocol settings.
- `predictions.csv`: response ID, test position, generator, task, label, and
  paper-compatible LettuceDetect, probe, and blend scores, plus the raw
  LettuceDetect maximum.
- `metrics.json`: sample/task/label counts and the three methods' AUROC and
  F1@0.5, reported separately for each selected generator.

Prediction and metric files are written after successful inference; a failed
run leaves its config/log and exits nonzero. There is no automatic recovery or
silent skipping. GPU inference has not been validated by the CPU checks.

Paper Table 2 reference AUROC values:

| Generator | LettuceDetect-large | Grounding Probe | Blend |
|---|---:|---:|---:|
| Mistral-7B-Instruct | 0.882 | 0.886 | 0.931 |
| Llama-2-7B-chat | 0.820 | 0.842 | 0.875 |
| Llama-2-13B-chat | 0.885 | 0.916 | 0.934 |

These are comparison targets, not enforced success thresholds. Do not tune on
the test set to match the published numbers.
