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
* **`range_rule` defaults to `blank`** as specified, with `max`/`min`
  available; how often it fires is counted and logged.
* **Ranges are detected before single values**, otherwise `"10 to 20 gram"`
  parses as a confident `"10 gram"`.
* **Number formatting**: no thousands separators, no trailing zeros, integers
  without `.0` — but this is a *hypothesis to check against the EDA format
  audit*, not an assumption. `run000_eda.ipynb` prints the audit prominently;
  if the real labels disagree, change `format_number` and re-run.
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

* Is the ~6% empty-value rate real, or an artefact of how the CSV was exported?
  The metric rewards abstention, so this directly affects the recall target.
* Duplicate `image_link`s mean one photo can back several entities. The current
  split stratifies on entity/emptiness but **does not group by image**, so the
  same photo can appear in both train and eval with different entities. Worth
  measuring whether that inflates the score.
* Is 256 visual tokens enough to read small print on packaging? Worth an
  ablation at 384 or 512 if VRAM allows at 4-bit.

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
