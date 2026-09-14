# Research journal

One section per run. Write the hypothesis **before** the run, fill in the result
after, and keep "What I learned" honest — a run that disproves the hypothesis is
worth more than one that vaguely confirms it.

---

## Scaffolding decisions

Choices made where the spec was silent. Recorded so they can be revisited rather
than rediscovered.

### Data and splits

* **Stratum merging never crosses an entity or emptiness boundary.** The spec
  said to merge strata under 10 members but not *into what*. Merging into a
  cross-entity bucket would break the entity-proportion assertion the splitter
  makes, and folding empty-valued rows in with real values would misrepresent
  the empty rate — and empty rows are a scored population (they drive FN/TN).
  So an undersized `(entity, emptiness, magnitude)` cell drops its magnitude
  bin, and anything still thin is absorbed into the largest stratum sharing its
  `(entity, emptiness)` prefix. If a whole `(entity, emptiness)` class is small,
  it stays small and is logged. Thin stratum: harmless. Drifting entity mix:
  invalidates every cross-run comparison.
* **Largest-remainder allocation** when sampling per stratum, so the total is
  exactly 5,000/10,000 while each stratum stays within one row of proportional.
* **`split_seed42.json` also stores the `pool`** (all non-eval indices), so a
  future run can draw a larger training subset without touching the eval set.
* **`save_split` refuses to overwrite.** Freezing is enforced in code, not by
  convention.
* **Parquet cache for the CSVs**, keyed on source mtime + size, under
  `results/cache/` (git-ignored).

### Images

* **448 px longest side, JPEG q90.** Chosen to match `max_pixels = 256×28×28`:
  the processor downscales anything larger anyway, so bigger files buy nothing
  and cost disk plus CPU decode on every step.
* **sha1-of-URL filenames.** Stable across machines and sessions, and free of
  characters that upset Windows.
* **Missing images yield a grey 64×64 placeholder** rather than an exception,
  and `drop_missing_images: true` removes those rows before training. Losing a
  handful of rows is survivable; losing a 9-hour run to one dead CDN URL is not.
* Writes go to a `.tmp` file then rename, so an interrupted download cannot
  leave a truncated JPEG that a resumed run would happily skip.

### Prompts

* **The format example uses a unit valid for the entity being asked about.**
  A single hardcoded `"34 gram"` example would put a weight example inside a
  voltage prompt — confusing the instruction and nudging the model toward an
  invalid unit. (Caught by a test.)
* **Three templates ship**: `prompt_v1` (baseline), `prompt_v2_terse`,
  `prompt_v3_system`. Variants exist so prompt changes are config-level
  ablations with their own `config_hash`, not untracked edits.
* An explicit end-of-turn token is appended to the training target, so the model
  learns to stop rather than running to `max_new_tokens` and trailing junk.

### Post-processing

* **Blank rather than guess.** An unparseable or invalid output becomes `""`.
  Empty costs one FN; a wrong non-empty answer costs an FP *and* precision.
* **`range_rule` now defaults to `bracket`**, not `blank`. The spec said
  `blank`, but the EDA found zero empty labels, which makes blanking provably
  losing (an FN and an FP cost the same), and found that the labels' own range
  notation is `[100.0, 240.0] volt`. `blank`/`max`/`min` remain available; how
  often the rule fires is counted and logged.
* **Ranges are detected before single values**, otherwise `"10 to 20 gram"`
  parses as a confident `"10 gram"`.
* **Number formatting**: the scaffolding guessed "integers without `.0`".
  **The EDA disproved this** — 92.01% of labels are `str(float(x))`, so the
  default is now `number_format: float`. See Run 000 below. This is exactly why
  the audit exists rather than an assumption.
* **`constants.py` is parsed with `ast.literal_eval`, never imported**, so a
  tampered dataset file cannot execute code. A built-in fallback copy is used
  when the file is absent, and any drift between the two is logged as a warning.
* Alias collisions are guarded: `"oz"` stays `ounce` and cannot be captured by
  `fluid ounce`.

### Training

* **`remove_unused_columns=False`** is essential — the default silently drops
  `pixel_values` and `image_grid_thw`, and training then runs on text alone and
  scores near zero with no error.
* **LoRA targets are resolved to fully-qualified module paths** excluding
  `visual.*`. PEFT matches on name suffix, and `q_proj` exists in both towers,
  so a bare target list would adapt the vision encoder we intend to freeze.
* **Adapter-only checkpoints.** A merged 7B checkpoint is ~15 GB against
  Kaggle's 20 GB working limit; the adapter is ~40 MB. Best checkpoint by
  eval loss.
* **Eval loss during training uses 10% of the eval set** (`eval_loss_fraction`)
  — it only picks the best checkpoint. Final scoring always uses all 5,000 rows.
* **`dataloader_num_workers=0`.** PIL images through the collator must be
  pickled to workers; on Kaggle that costs more than the parallel decode saves.
* `bf16` is hardcoded `False` in `TrainingArguments`, not just left to config —
  it is never correct on Turing.

### Inference

* **Left padding is forced during generation** and restored afterwards. With
  right padding, short sequences continue from pad tokens and emit garbage.
* Batch size 4 by default: after training, a T4 has little headroom for the
  KV cache on top of ~256 visual tokens per image.

### Infrastructure

* **`Config` is read-only** and `Mapping`-compatible, so `cfg.train.lr`,
  `cfg["train"]`, and `dict(**cfg)` all work but accidental mutation cannot
  desync the object from the hash written to the leaderboard.
* **Lists replace rather than concatenate** on merge, so a child config can
  *shorten* `target_modules`.
* **`extends` is stripped before hashing**: two configs resolving to identical
  settings by different routes are the same experiment.
* **Torch is imported lazily** throughout, so `import amlc24`, the tests, and
  the EDA notebook all work on a CPU-only box with no ML stack.
* Paths are overridable via `AMLC24_*` env vars, which is how the test suite
  guarantees it never writes into the real `results/`.

### Open questions

* ~~Is the empty-value rate real?~~ **Answered:** it is 0.00%. See Run 000.
* Duplicate `image_link`s mean one photo can back several entities. The split
  stratifies on entity/emptiness but **does not group by image**, so the same
  photo can appear in both train and eval under different entities. The EDA
  says this affects a small share (255,906 distinct images for 263,859 rows,
  so ~3% sharing), but it is still unmeasured leakage.
* Is 256 visual tokens enough to read small print on packaging? Worth an
  ablation at 384 or 512 if VRAM allows at 4-bit.
* Does class weighting help macro F1 enough to justify any micro F1 it costs?
  One config flip to test.
* `reject_invalid_units` blanks ~0.2% of predictions whose label is literally
  that invalid unit (`item_volume` / `ounce`). Worth an ablation, though keeping
  it is safer for the competition's own sanity checker.

---

## Run 000 — EDA on the real `train.csv` (2026-09-14)

263,859 rows, 8 entities, 750 groups. Ran before any training, and it changed
four decisions. Full tables in `results/eda/`.

### 1. The number format assumption was wrong — and it was the expensive kind

The scaffolding guessed that labels were written without a trailing `.0`. They
are not:

| number shape | share | example |
|---|---|---|
| `d.0` (trailing `.0`) | **64.12%** | `500.0 gram` |
| real decimal | **27.89%** | `3.53 ounce` |
| bare integer | 7.99% | `50 gram` |

**92.01% of labels are exactly `str(float(x))`.** Six of the eight entities
(depth, height, item_volume, voltage, wattage, width) are *100%* float-style
with not one bare integer; only `item_weight` (18.9% bare) and
`maximum_weight_recommendation` (49.9% bare) mix the two.

The original `format_number` stripped `.0`, so it would have converted a
correct `500.0 gram` into `500 gram` — a false positive on ~92% of otherwise
correct predictions. Measured directly: feeding the ground-truth labels through
normalisation and checking how many survive unchanged,

* old (`int`) behaviour: **35.17%**
* new (`float`) default: **90.66%**
* with the bracket range rule too: **91.80%**

This is the single largest score lever in the repo and it came from the EDA, not
from the model. `number_format` is now a config flag defaulting to `float`.

The residual 7.99% is irreducible: nothing in the number itself says whether a
given item's label is `50` or `50.0`. Picking the 92% convention is the best
available single choice.

### 2. There are **zero** empty labels

Empty-value rate is 0.00% across all 263,859 rows. Two consequences:

* The emptiness axis of the stratification key is degenerate. Kept anyway — it
  costs nothing and guards a future file that does have them.
* **Blanking is never optimal.** With no empty ground truth, an empty
  prediction is always a false negative, and `F1 = 2TP/(2TP+FP+FN)` charges an
  FN and an FP identically. So a guess weakly dominates a blank: it costs the
  same when wrong and scores when right. The original `range_rule: blank`
  default was therefore provably losing.

### 3. Ranges use bracket notation, which the parser missed entirely

The labels write ranges as `[100.0, 240.0] volt` (3,276 rows, 1.24%), plus a
few hundred as `10 kilogram to 15 kilogram`. The original regex matched neither.

Since the truth *is* that bracket string, reproducing it can score a true
positive where blanking never can — so `range_rule` now defaults to `bracket`
and round-trips the dataset's exact notation. `blank`/`max`/`min` remain.

### 4. `entity_name` is heavily skewed — 31.5x

| entity | rows | share |
|---|---|---|
| item_weight | 102,786 | 38.95% |
| depth | 45,127 | 17.10% |
| width | 44,183 | 16.74% |
| height | 43,597 | 16.52% |
| voltage | 9,466 | 3.59% |
| wattage | 7,755 | 2.94% |
| item_volume | 7,682 | 2.91% |
| maximum_weight_recommendation | 3,263 | **1.24%** |

Unweighted, ~72% of the gradient comes from weight-and-dimension rows, and the
two entities with the most unit ambiguity (`item_volume` has 13 allowed units,
the most of any) get the least signal. Hence the class-weighted loss below.

### Other findings

* **`train.csv` has no `index` column** (only `test.csv` does). It is
  synthesised from row position, which is deterministic for a fixed file —
  but it ties the frozen split to this exact `train.csv`. If the organisers
  reissue it with rows added or reordered, ids shift and `verify_split` fails
  loudly, which is correct: the eval set would no longer be the same products.
* **Ground truth contains units outside `constants.py`** — 4,331 rows (1.64%),
  mostly `item_volume` labelled `ounce` (300 rows) where only `fluid ounce` is
  allowed. So `reject_invalid_units` can blank a prediction that exactly matches
  its label. Kept on (an invalid unit fails the competition's own sanity
  checker) but it is a measured ~0.2% local cost, and it is toggleable.
* **Image dedup saves only 3.0%** — 255,906 distinct images for 263,859 rows,
  far less sharing than assumed. The download is ~15k images either way.
* 5,914 labels (2.24%) use multi-word units (`fluid ounce`, `cubic foot`), so
  the parser must not split the unit on whitespace.

### What I changed as a result

1. `format_number` → float style, config flag `number_format` (default `float`)
2. `range_rule` → `bracket` default, with bracket-notation parsing added
3. Class-weighted loss (`train.class_weights`, `sqrt_inverse`)
4. Macro F1 + per-`group_id` breakdown added to every run's output

### Follow-ups from the official problem statement (2026-09-14)

Reading the competition description alongside the real archive surfaced three
more things, two of them bugs.

* **Exponent notation is explicitly invalid output.** The statement lists
  `"2.2e2 kilogram"` among the invalid examples. But `str(float(x))` — the
  convention the EDA told us to adopt — emits exponent form for extreme
  magnitudes: `1e20` rendered as `'1e+20'`. Every float path now routes through
  `_plain_decimal`, so output is always positional. Exponent *input* is now
  parsed and converted (`2.2e2 kilogram` → `220.0 kilogram`) rather than
  blanked, since blanking is a guaranteed false negative.
* **Exponent minus was being read as a range separator.** `1.5E-2 litre`
  parsed as the range `[1.5, 2.0]` — the regex backtracked, matched `E` as a
  unit and `-` as the separator, and invented a confident wrong answer. Fixed
  with a `(?![eE][-+]?\d)` guard after the low number.
* **Row count is checked by the grader but not by `sanity.py`.** The statement
  says a file with more or fewer rows than `test.csv` "won't be evaluated", and
  explicitly notes the shipped checker does not test this. `results/submission.py`
  does, and back-fills missing indices with empty predictions.

Also: the archive nests as `<slug>/student_resource 3/dataset/` (with a space),
two levels deeper than the original detector searched — hence the bounded
recursive search in `paths.py`. And the dataset's own `constants.py` now loads
on Kaggle; it matches the built-in fallback exactly, with no drift warnings.

---

## Class-weighted loss

`train/weighting.py`. Per-sample loss multiplier keyed on `entity_name`,
normalised to mean 1.0 over the training distribution so the scheme changes the
*balance* between classes but not the overall loss scale — and therefore does
not implicitly change the effective learning rate.

Schemes: `none`, `inverse`, `sqrt_inverse` (default), `effective`
(Cui et al. 2019). On the actual 10k training subset:

| entity | n | `sqrt_inverse` | `inverse` |
|---|---|---|---|
| item_weight | 3,896 | 0.642 | 0.381 |
| depth / width / height | ~1,680 | ~0.98 | ~0.88 |
| voltage | 359 | 2.114 | 3.807 |
| wattage | 293 | 2.340 | 3.807 |
| item_volume | 292 | 2.344 | 3.807 |
| maximum_weight_recommendation | 124 | **3.598** | 3.807 |
| **spread** | | **5.61x** | 10.0x (capped) |

**Why `sqrt_inverse` is the default.** Full `inverse` gives the rarest class a
~31x multiplier before capping. At `per_device_train_batch_size=1`, each
micro-batch is a single sample, so that multiplier lands on the whole gradient
for that step — and under fp16 the resulting magnitude swings are a loss-spike
and overflow risk. The square root keeps the largest multiplier near 5.6x,
which rebalances meaningfully without destabilising the run.

**Implementation note.** At batch size 1 the weighted loss is just
`outputs.loss * w` — no second forward, no extra logits copy. The general
per-sample path (`reduction="none"`) exists for batch > 1 but materialises an
fp32 view of a 152k-vocab logits tensor, which is why batch size 1 is a
correctness-adjacent choice here and not merely a memory convenience. Both
paths are tested against hand computation in `tests/test_weighted_loss.py`,
including that gradients scale linearly with the weight.

**Open question:** whether weighting actually helps. It should raise *macro* F1
(all entities equal) at some cost to *micro* F1 (dominated by `item_weight`).
Both are now reported every run, and `class_weights` is on the leaderboard, so
the ablation is a config flip: set `train.class_weights.enabled: false`.

---

## Run 001 — `run001_qwen2vl_8bit_10k`

**Status:** not yet run.

### Hypothesis

A 7B vision-language model already *reads* measurements off product packaging
competently; what it does not do is emit them in the exact string format the
metric demands. So a modest LoRA fine-tune on 10k examples should move F1
sharply — not by teaching the model to see, but by teaching it to answer in
`"<number> <unit>"` with a unit from the allowed list and nothing else.

Concretely, I expect:

1. **Raw F1 to improve a lot over zero-shot**, driven mostly by format
   compliance rather than better reading.
2. **The raw → post-processed gap to shrink as training progresses.** A large
   remaining gap would mean the fine-tune failed at exactly the thing it was
   supposed to learn.
3. **Per-entity F1 to vary widely.** `item_weight` and `item_volume` are usually
   printed prominently on packaging; `width`/`height`/`depth` are often not
   printed at all and must be inferred, so they should score worst.
4. **Unit-formatting errors to dominate the error analysis** — the
   `same_number_diff_unit` flag should be true for a large share of mismatch
   pairs. Each one is a missing alias and free score.

### Config rationale

* **8-bit over 4-bit.** 8-bit keeps more fidelity for reading small text, and
  the budget appears to fit (~13–14 GB of 16). 4-bit is the escape hatch and is
  a one-line config change.
* **`max_pixels = 256 × 28 × 28`.** The single decision that makes a 7B model
  trainable here. At Qwen2-VL's defaults a single image can produce thousands of
  visual tokens and OOM immediately. 256 tokens ≈ a 448×448 image, which is why
  images are cached at 448 px.
* **Vision tower frozen; LoRA on LM attention only.** The deficit is output
  format, which lives in the language model. Adapting the ViT would spend VRAM
  on the part that already works.
* **`r=16`.** Enough capacity for a formatting/vocabulary shift without the
  optimiser-state cost of a larger rank.
* **3 epochs at `lr=2e-4` with cosine decay and 3% warmup.** Standard for LoRA
  at this scale; 10k samples × 3 epochs ÷ effective batch 8 ≈ 3,750 steps, which
  fits the session limit with room for evaluation.
* **Completion-only loss.** Training on the prompt would waste capacity
  re-learning our own instructions and blunt exactly the formatting precision
  the metric rewards.

### Result

<!-- Fill in after the run:
     F1 raw / post, the delta, per-entity table, peak VRAM, wall-clock,
     and whether the 8-bit budget actually held. -->

### What I learned

<!-- Fill in after the run. Be specific about what was surprising, and what the
     next run should change as a result. If the hypothesis was wrong, say so
     plainly and say which part. -->

### Next

<!-- Candidate follow-ups, to be chosen based on the result above:
     - zero-shot baseline (`--zero-shot`) to size the fine-tune's actual gain
     - 4-bit + 25k samples: does more data beat more precision?
     - max_pixels 384/512: is 256 tokens enough to read small print?
     - prompt_v2_terse vs prompt_v1
     - ablate each post-processing rule to attribute the raw->post delta -->
