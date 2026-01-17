# Distillation Phase 1 (8k) — distill datasets (offline)

This directory documents how we build the **Phase 1 (8k)** distillation datasets (offline, teacher-forced DistillKit style).

Outputs are written to `/data/distillation_1/...` inside the container.

## Canonical Phase‑1 training dataset (what to train on)

The canonical, interleaved Phase‑1 dataset directory is:

- `/data/distillation_1/phase1_mix_8k_665m`

This is built by merging the component datasets below with:

```
scripts/run_phase1_merge_8k_665m.sh
```

Do not point training at the older `/data/distillation_1/phase1_mix_8k` output (it was an earlier merge run).

## What was built
- Output format: DistillKit-ready parquet shards (`train/`, `validation/`), with:
  - `input_ids` (Harmony tokens)
  - `labels` masked to assistant tokens (`-100` elsewhere)
  - `spans.assistant_token_start` + per-example assistant span lengths
- Tokenizer: Harmony (`o200k_harmony`, `HARMONY_GPT_OSS`)
- Reasoning level: **high** (baked into the Harmony system header for every example)

## Split / limits
- Train/val split: **99/1** (deterministic by per-example id hash)
- Max seq len: **8192**
- Packing: **not packed** here (pack at training time)

## Datasets in Phase 1

### 1) PleIAs/SYNTH (primary)
- Source: `/data/raw_0/pleias_synth` (PleIAs/SYNTH parquet shards; ok if this is a symlink to a host-mounted dataset, e.g. `/datasets/PleIAs_synth`)
- Shard list used: `synth_phase1_shards.txt` (kept for provenance)
- Output (Phase‑1 budgeted): `/data/distillation_1/synth_distill_phase1_high_550m`

Sampling / filters:
- English fraction target: **0.75**
- Memorization cap: **0.70** (by exercise label)
- Global downsample: **~0.20** (targets ~550M total tokens from the shard subset)
- Drops:
  - empty answers
  - “failed after maximum attempts”
  - constraint/result mismatches (math/MCQ)

Exact counts and distributions are recorded in the output `manifest.json`.

Note:
- `scripts/build_synth_distill.py` is deterministic and will keep **all** rows that pass filters unless you set `--global-keep < 1.0`.
- `scripts/merge_distill_datasets.py --weight-by ...` only affects the **interleaving schedule**; it does **not** downsample inputs.

Recreate the shard subset dir (if needed)
```
rm -rf /data/synth_phase1
mkdir -p /data/synth_phase1
while IFS= read -r p; do
  b="$(basename "$p")"
  if [[ "$p" == /* ]]; then
    src="$p"
  else
    # `synth_phase1_shards.txt` is a list of filenames like `synth_010.parquet`.
    # Resolve relative entries under `/data/pleias_synth` (usually a symlink to `/datasets/PleIAs_synth`).
    src="/data/pleias_synth/$p"
  fi
  ln -sf "$src" "/data/synth_phase1/$b"
done < data/distillation_1/synth_phase1_shards.txt
```

Build command (reference)
```
PYARROW_NUM_THREADS=20 OMP_NUM_THREADS=20 \
python3 scripts/build_synth_distill.py \
  --input-dir /data/synth_phase1 \
  --output-dir /data/distillation_1/synth_distill_phase1_high_550m \
  --max-seq-len 8192 \
  --batch-size 4096 \
  --english-frac 0.75 \
  --memorization-cap 0.70 \
  --global-keep 0.20 \
  --reasoning-level high
```

### 2) Code — nvidia/OpenCodeInstruct
- Source: `/datasets/nvidia__OpenCodeInstruct/data` (parquet shards)
- Output: `/data/distillation_1/code_opencodeinstruct_55m`

Notes:
- Prompt = `input`
- Final = `output`
- Keeps only `average_test_score == 1.0` (perfect tests)
- Records the randomized shard order in `input_shards_order.txt` under the output dir

Build command (reference)
```
PYARROW_NUM_THREADS=20 OMP_NUM_THREADS=20 python3 scripts/build_opencode_distill.py \
  --input-dir /datasets/nvidia__OpenCodeInstruct/data \
  --output-dir /data/distillation_1/code_opencodeinstruct_55m \
  --max-seq-len 8192 \
  --batch-size 4096 \
  --target-total-tokens 55000000 \
  --reasoning-level high \
  --shuffle-shards
```

### 3) Math — nvidia/Nemotron-Math-v2 (low)
- Source: `/datasets/nvidia__Nemotron-Math-v2/data/low.jsonl`
- Output: `/data/distillation_1/math_nemotron_v2_low_55m`

Notes:
- Uses dataset-provided `assistant.reasoning_content` as Harmony `analysis` and `assistant.content` as `final`.
- Drops tool rows for Phase 1 and keeps only examples that fit `max_seq_len=8192`.

Build command (reference)
```
PYARROW_NUM_THREADS=20 OMP_NUM_THREADS=20 python3 scripts/build_nemotron_math_distill.py \
  --input /datasets/nvidia__Nemotron-Math-v2/data/low.jsonl \
  --output-dir /data/distillation_1/math_nemotron_v2_low_55m \
  --max-seq-len 8192 \
  --target-total-tokens 55000000 \
  --reasoning-level high \
  --drop-tool-rows
```

### 4) Helpfulness / assistant style — OpenAssistant/oasst2
This is the lightweight “soft skills” slice. The builder extracts prompt→assistant pairs from OASST2.

Build command (reference)
```
PYARROW_NUM_THREADS=20 OMP_NUM_THREADS=20 python3 scripts/build_oasst2_distill.py \
  --input-dir /data/raw_0/OpenAssistant__oasst2 \
  --output-dir /data/distillation_1/helpfulness_oasst2_15m \
  --max-seq-len 8192 \
  --target-total-tokens 15000000 \
  --english-frac 0.90 \
  --reasoning-level high
```

Note:
- The dataset is finite; `--target-total-tokens` is a cap, not a guarantee. The builder will stop early if it exhausts the source.

## Optional: single interleaved Phase‑1 dataset directory

If you want training to consume a single directory (instead of 4 roots), you must **rewrite**
the datasets into a canonical schema because the per-dataset `source` struct fields differ.

This is what `scripts/merge_distill_datasets.py` does: it reads each dataset’s shards and
writes a new dataset with a canonical `source` struct (and a `meta_json` field to preserve provenance).

Example command:
```
PYARROW_NUM_THREADS=20 OMP_NUM_THREADS=20 python3 scripts/merge_distill_datasets.py \
  --input synth=/data/distillation_1/synth_distill_phase1_high_550m \
  --input code=/data/distillation_1/code_opencodeinstruct_55m \
  --input math=/data/distillation_1/math_nemotron_v2_low_55m \
  --input chat=/data/distillation_1/helpfulness_oasst2_15m \
  --output-dir /data/distillation_1/phase1_mix_8k_665m \
  --weight-by assistant_tokens \
  --seed 1337
```

Helper script (same as above + stats + samples):
```
scripts/run_phase1_merge_8k_665m.sh
```

After that, you can point training at `/data/distillation_1/phase1_mix_8k_665m` and ignore the individual roots.
