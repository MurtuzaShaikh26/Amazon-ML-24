# Amazon ML Challenge 2024 — Entity Value Extraction

Extract measurement values from product images. Given a product image and an
entity name (`item_weight`, `width`, `voltage`, …), predict a string of the form
`"<number> <unit>"` — `"34 gram"`, `"12.5 centimetre"`, `"2.56 ounce"`.

A research repo, not a submission script: every run writes its config, metrics,
per-entity breakdown and error analysis into `results/`, and appends a row to
`results/leaderboard.csv`. The metrics are the deliverable.

---

## The task

| Column | Meaning |
|---|---|
| `index` | unique sample ID |
| `image_link` | public Amazon CDN image URL |
| `group_id` | product category code |
| `entity_name` | what to extract, e.g. `item_weight` |
| `entity_value` | target, e.g. `"34 gram"` (**absent in `test.csv`**) |

Allowed units per entity are enumerated in the dataset's `src/constants.py`.
**A prediction using any other unit is invalid.** Submission format is a
two-column CSV (`index`, `prediction`); predict the empty string when no value
is found.

### What the data actually looks like

Run `python scripts/run_local.py eda` to regenerate; tables land in
`results/eda/`. The findings that shaped the code:

| Finding | Value | Consequence |
|---|---|---|
| Number format | 92.01% are `str(float(x))` | `number_format: float` |
| Empty labels | **0.00%** | blanking is never optimal |
| Ranges | `[100.0, 240.0] volt`, 1.24% | `range_rule: bracket` |
| `entity_name` skew | **31.5x** | class-weighted loss |
| `train.csv` `index` column | **absent** | synthesised from row position |
| Units outside `constants.py` | 1.64% *of the labels* | `reject_invalid_units` costs ~0.2% |
| Exponent output (`2.2e2 kilogram`) | explicitly **invalid** | never emitted; exponent *input* is converted |

### Metric: F1 with exact string match

| prediction | ground truth | outcome |
|---|---|---|
| non-empty, **exactly equal** | non-empty | TP |
| non-empty, differs | non-empty | FP |
| non-empty | empty | FP |
| empty | non-empty | FN |
| empty | empty | TN |

`precision = TP/(TP+FP)`, `recall = TP/(TP+FN)`, `F1 = 2PR/(P+R)`.

Exact match means **formatting matters as much as the number**. Post-processing
is a first-class part of the solution, not a tidy-up step — which is why every
run reports F1 **both raw and post-processed**, so the value of normalisation is
a measured number rather than an assumption.

> **The formatting convention was measured, not guessed.** The EDA over all
> 263,859 labels found that **92.01% are exactly `str(float(x))`** — `500.0
> gram`, not `500 gram`. Six of the eight entities are 100% float-style.
> Feeding the ground-truth labels back through normalisation, the initial
> "strip the `.0`" guess reproduced only **35.17%** of them; the measured
> convention reproduces **91.80%**. That single fix is the largest score lever
> in the repo, and it came from the EDA rather than the model. See
> [`NOTES.md`](NOTES.md) → Run 000.

---

## ⚠️ The frozen evaluation split

**`test.csv` has no `entity_value`.** It is the blind leaderboard set and cannot
be used for local evaluation. All splits come from `train.csv`:

* **5,000 rows are held out as a frozen evaluation set.** Generated once,
  written to [`results/splits/split_seed42.json`](results/splits/), committed,
  and **never changed**. Every run — this one and every future one — evaluates
  on exactly those rows.
* **10,000 rows** are sampled from the remainder for training in run 001.

On every subsequent run the split is **loaded and verified**: it is regenerated
and compared against the committed file, and any disagreement raises
`SplitMismatch` rather than overwriting it.

> **Do not "fix" a split mismatch by deleting the JSON.** A moving eval set
> turns the leaderboard into noise, and the damage is invisible until someone
> tries to trust it. If verification fails, `train.csv` changed or the
> stratification logic was edited — investigate that instead.

Stratification key: `entity_name` × whether `entity_value` is empty, crossed
with a coarse log-magnitude bin where strata stay large enough. Merging never
crosses an entity or emptiness boundary, so entity proportions match across the
full file, the eval split and the training subset to within 1 percentage point
(asserted, and logged as a comparison table).

---

## Repository layout

```
configs/           base.yaml + one YAML per run (extends: inheritance)
src/amlc24/
  paths.py         environment detection — the ONLY place paths are resolved
  config.py        YAML inheritance, deep merge, dot access, config_hash
  data/            load, eda, splits, images, dataset (+ collator)
  prompts/         versioned prompt templates (prompt_v1, …)
  models/          Qwen2-VL loading, quantisation, LoRA
  train/           Trainer wiring, completion-only loss, class weighting
  inference/       batched greedy generation
  postprocess/     unit vocabulary + normalisation rules
  metrics/         competition F1: micro + macro, per-entity, per-group,
                   per-unit, error analysis
  results/         per-run artefacts, leaderboard, submission writer
  pipeline/        run_eda, run_finetune  ← notebooks call only these
kaggle_notebook/   thin orchestrators (no logic)
scripts/           download_images.py, run_local.py
results/           leaderboard.csv, splits/, runs/, eda/   (committed)
tests/             311 tests
```

### Design rules

* **No logic in notebooks.** They locate the repo, call a pipeline function, and
  display results.
* **No hardcoded paths outside `paths.py`.**
* **`logging`, never `print`, inside `src/`.**
* **Per-entity (class-wise) and per-group (category-wise) F1 in every run's
  output, alongside both micro and macro F1.**
* **`results/` is committed** — only adapter binaries under
  `results/runs/*/checkpoints/` are ignored.

---

## Running it

### Local (CPU is enough for everything except training)

```bash
pip install -r requirements.txt

# put train.csv / test.csv (and the dataset's src/constants.py) under ./data/
# note: train.csv has no `index` column; it is synthesised from row position

python -m pytest tests/ -q          # 311 tests, ~9s
python scripts/run_local.py eda     # profile train.csv -> results/eda/
python scripts/run_local.py splits  # create/verify the frozen 5k split
python scripts/download_images.py   # ~15k images, resized, resumable
```

`run_local.py train` runs the full pipeline but needs a GPU.
`--zero-shot` evaluates the base model without fine-tuning (the baseline the
fine-tune has to beat); `--smoke` runs a tiny end-to-end plumbing check.

### Kaggle

The competition archive mounts as
`/kaggle/input/<slug>/student_resource 3/dataset/train.csv` — note the extra
nesting and the space in the folder name. `paths.py` finds it with a bounded
recursive search (pruning image directories), so no slug or layout is
hardcoded. If `train_csv_found` is `False` in the first cell's output, the
dataset simply is not attached — use **+ Add Input** in the sidebar.

1. Push this repo to GitHub.
2. Create a notebook, **Accelerator → GPU T4 ×2** (see the T4 note below).
3. Attach the competition dataset, and the resized-image dataset if you made one.
4. Upload the repo as a Dataset, or let the notebook `git clone` it (internet is
   enabled). The bootstrap cell finds `src/` either way — no slug is hardcoded,
   and it `git pull`s an existing clone so a re-run picks up the latest code.
5. Run [`kaggle_notebook/run000_eda.ipynb`](kaggle_notebook/) first (CPU, minutes),
   then [`run001_qwen2vl_8bit_10k.ipynb`](kaggle_notebook/).

---

## Image preprocessing and the Kaggle upload workflow

`scripts/download_images.py` downloads **only** the ~15k images the frozen split
references — not the full dataset — using 32 threads with retry and per-URL
failure logging. Images are **resized on download** so the longest side is
448 px and saved as JPEG q90.

448 px is chosen to match `max_pixels = 256 × 28 × 28`: the processor would
downscale anything larger anyway, so storing full-resolution images wastes disk
and CPU decode time for no gain. The resize cuts the image set roughly 10–20×.

Downloads skip files that already exist, so the script is resumable, and a dead
CDN URL skips that row with a warning rather than crashing the run.

> **Upload the resized folder to Kaggle as a private Dataset and reuse it.**
> Kaggle sessions are ephemeral — anything not saved as a dataset output is lost
> when the session ends. Re-downloading 15k images at the start of every run
> burns ~20 minutes of the 12-hour budget for nothing. `paths.py` detects a
> mounted image dataset automatically and the download step becomes a no-op.

---

## Model and training

**Qwen2-VL-7B-Instruct, 8-bit QLoRA.** The memory budget on a 16 GB T4 is tight
(~13–14 GB expected), so several settings are load-bearing rather than
incidental:

| Setting | Value | Why |
|---|---|---|
| `quantization.bits` | `8` | Config flag. **Flip to `4` if a run OOMs — that is the only change needed.** |
| `processor.max_pixels_tokens` | `256` | **The knob that decides whether this fits.** Qwen2-VL's dynamic resolution emits thousands of visual tokens at its defaults. |
| `attn_implementation` | `sdpa` | T4 is Turing: **no flash-attention-2**. |
| `fp16` | `true` | T4 has **no bf16**. |
| batch / accumulation | `1` / `8` | Effective batch 8 within VRAM. |
| `gradient_checkpointing` | `true` | Trades compute for activation memory. |
| `optim` | `paged_adamw_8bit` | Paged states survive fragmentation spikes. |
| LoRA | `r=16, α=32, dropout=0.05` on `q_proj,k_proj,v_proj,o_proj` | **Language model only** — the vision tower is frozen. |
| schedule | 3 epochs, cosine, `warmup_ratio=0.03`, `lr=2e-4` | |
| `train.class_weights` | `sqrt_inverse` | 31.5x entity imbalance (see below). |

### Class-weighted loss

`entity_name` is skewed 31.5x (`item_weight` 38.95% vs
`maximum_weight_recommendation` 1.24%), so unweighted training spends ~72% of
its gradient on weight-and-dimension rows. `train.class_weights` applies a
per-sample loss multiplier, normalised to mean 1.0 so the scheme changes the
balance between classes without changing the effective learning rate.

`sqrt_inverse` (default) caps the spread at ~5.6x. Full `inverse` would give the
rarest class a ~31x multiplier — and at batch size 1 that multiplier lands on an
entire step's gradient, which under fp16 risks loss spikes. Disable with
`train.class_weights.enabled: false` to run the unweighted ablation.

Because the metric is micro-averaged and `item_weight` dominates it, **macro F1
is the number that reveals whether weighting worked.** Both are reported every
run and both are on the leaderboard.

**Loss is computed on completion tokens only.** Prompt tokens, padding, and
vision placeholders are masked to `-100`. Training on the prompt wastes capacity
and degrades output formatting — and formatting is the metric.

Expected on a T4: **~7–9 h** for 10k × 3 epochs plus 5k-row evaluation, inside
the 12-hour session limit but not by much.

---

## Adding a new run

1. **Write a config** in `configs/`:

   ```yaml
   extends: base.yaml
   run_id: run002_qwen2vl_4bit_25k
   description: >
     Does 4-bit quantisation with 2.5x the data beat 8-bit at 10k?

   quantization: {bits: 4}
   data: {train_size: 25000}
   ```

   Only state what differs; everything else is inherited. The resolved config is
   hashed into `config_hash` on the leaderboard row.

2. **Copy a notebook**, change the `CONFIG` constant, and run it.

3. **Record the result** in `NOTES.md` — hypothesis, what happened, what you
   learned.

The eval split does not change, so the new number is directly comparable to
every previous one. That is the entire point.

### Things worth varying

Prompt templates are named functions selected by `prompt.template`, so a prompt
change is a config edit (and a new `config_hash`), not an untracked code edit.
Every post-processing rule is individually toggleable under `postprocess:`, so
each one's contribution can be measured by ablation.

---

## Results

`results/leaderboard.csv` carries one row per run:

```
run_id, timestamp, description, model, quant_bits, lora_r, n_train, n_eval,
epochs, lr, max_pixels, f1_raw, f1_post, precision, recall, train_seconds,
config_hash, notes
```

Each `results/runs/{run_id}/` holds `config.yaml`, `metrics.json`,
`predictions_eval.csv`, `f1_by_entity.csv` (class-wise), `f1_by_group.csv`
(category-wise), `f1_by_unit.csv`, `error_analysis.csv`, `log.txt` and
`env.json`.

**Class-wise and category-wise evaluation appear in every run.**
`f1_by_entity.csv` breaks the score down by `entity_name` with a
`f1_delta_from_postprocess` column showing which entities normalisation
rescued; `f1_by_group.csv` does the same by `group_id` (product category),
pooling groups with fewer than 20 eval rows so the long tail does not become
noise.

`error_analysis.csv` is the most actionable of these: it ranks the most frequent
`(predicted, actual)` mismatch pairs per entity and flags the ones that differ
*only* in the unit — those are missing aliases, and free score.

## Submissions

`results/submission.py` writes the two-column `index,prediction` file and
validates it against the official rules before writing — the same checks as the
dataset's `src/sanity.py`, plus one it does not make: **the row count must equal
`test.csv` exactly**, or the file is not evaluated at all. Test indices the
model did not produce are filled with an empty prediction, which costs one false
negative rather than the whole submission.

```python
from amlc24.results.submission import write_submission
write_submission(predictions, "submission.csv", test_df=test_df)
```

See [`NOTES.md`](NOTES.md) for the research journal.
