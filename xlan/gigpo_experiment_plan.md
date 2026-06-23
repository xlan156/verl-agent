# GiGPO DiscoveryWorld Curriculum Experiment Plan

## Goal

We want to test whether curriculum learning improves combinatorial generalization in DiscoveryWorld Combinatorial Chemistry.

In this codebase, `N` means the total chemical amount in the target solution, i.e. `sum(A, B, C, D) = N`, not the per-chemical maximum. With `CURRICULUM_ENABLED=False`, the environment samples direct fixed-amount tasks because `chemicalMinAmount = chemicalMaxAmount = MAX_CHEMICAL_N`.

Main question:

> Does training with easier stages `N=1,2` help the agent solve harder compositions `N>=3`, beyond what direct hard-task training can do?

## Shared Defaults

All new scripts call `xlan/gigpo_base.sh` so shared hyperparameters stay aligned.

| Setting | Default |
| --- | --- |
| Model | `Qwen2.5-0.5B-Instruct` SFT checkpoint |
| Scenario | `Combinatorial Chemistry` |
| Difficulty | `Challenge` |
| Train batch | `40` |
| Val batch | `40` |
| Group size | `2` |
| LR | `5e-8` |
| KL coef | `0.15` |
| Epochs | `40` |
| Max steps | `20` unless noted |
| Curriculum train fraction | `0.8` unless noted |
| Default curriculum mix | `[0.7,0.2,0.1]` |

Run seeds by overriding environment variables:

```bash
ENV_SEED=0 CURRICULUM_SEED=0 sbatch xlan/gigpo_curr_n3.sh
ENV_SEED=1 CURRICULUM_SEED=1 sbatch xlan/gigpo_curr_n3.sh
ENV_SEED=2 CURRICULUM_SEED=2 sbatch xlan/gigpo_curr_n3.sh
```

## First-Round Experiments

These are the highest-priority runs. They answer whether curriculum helps at all and whether the improvement transfers to harder tasks.

There are two different meanings of "curriculum" worth separating:

- **Mixture curriculum from SFT**: start from the SFT model and train with curriculum sampling up to stage `N`. This is what `xlan/gigpo_curr_n3.sh` does.
- **Staged curriculum from previous best checkpoint**: train `N=2`, choose the best checkpoint, then continue training `N=3` from that checkpoint. This is closer to a strict easy-to-hard curriculum.

For the main curriculum claim, prefer the staged version when possible. Keep the mixture version as a useful ablation.

| ID | Script | Train distribution | Purpose |
| --- | --- | --- | --- |
| C2 | `xlan/gigpo_curr_n2.sh` | curriculum up to N=2 | Reproduce/reference existing N=2 result under the unified base script |
| C3-mix | `xlan/gigpo_curr_n3.sh` | SFT -> curriculum up to N=3 | Mixture-curriculum ablation |
| C3-staged | `xlan/gigpo_curr_n3_from_c2.sh` | best C2 checkpoint -> curriculum up to N=3 | Main staged curriculum extension |
| D3 | `xlan/gigpo_direct_n3.sh` | direct fixed N=3, no curriculum | Hard-task baseline for C3 |
| C4-mix | `xlan/gigpo_curr_n4.sh` | SFT -> curriculum up to N=4 | Mixture-curriculum ablation |
| C4-staged | `xlan/gigpo_curr_n4_from_c3.sh` | best C3 checkpoint -> curriculum up to N=4 | Main staged N=4 extension |
| D4 | `xlan/gigpo_direct_n4.sh` | direct fixed N=4, no curriculum | Hard-task baseline for C4 |

Recommended first batch:

```bash
sbatch xlan/gigpo_curr_n3.sh
sbatch xlan/gigpo_direct_n3.sh
sbatch xlan/gigpo_curr_n4.sh
sbatch xlan/gigpo_direct_n4.sh
```

If GPU budget is tight, run `C2`, pick the best C2 checkpoint, then run staged C3 against direct D3:

```bash
sbatch xlan/gigpo_curr_n2.sh
C2_CKPT=checkpoints/GiGPO-discoveryworld/<c2-exp>/global_step_<best> sbatch xlan/gigpo_curr_n3_from_c2.sh
sbatch xlan/gigpo_direct_n3.sh
```

Then for N=4:

```bash
C3_CKPT=checkpoints/GiGPO-discoveryworld/<c3-exp>/global_step_<best> sbatch xlan/gigpo_curr_n4_from_c3.sh
sbatch xlan/gigpo_direct_n4.sh
```

## Replay / Mix-Ratio Ablation

These runs test whether lower-stage replay is important or whether training only the current stage is enough.

| ID | Script | Mix ratio | Hypothesis |
| --- | --- | --- | --- |
| C3-no-replay | `xlan/gigpo_curr_n3_no_replay.sh` | `[1.0,0.0,0.0]` | If this drops, replay prevents forgetting or stabilizes learning |
| C3-default | `xlan/gigpo_curr_n3.sh` | `[0.7,0.2,0.1]` | Current default |
| C3-more-replay | `xlan/gigpo_curr_n3_more_replay.sh` | `[0.5,0.3,0.2]` | More replay may improve transfer but slow hard-stage learning |

Run after the first-round N=3 comparison:

```bash
sbatch xlan/gigpo_curr_n3_no_replay.sh
sbatch xlan/gigpo_curr_n3_more_replay.sh
```

## Horizon Ablation

Harder N may fail because the agent cannot finish within 20 steps, not because it cannot plan. These runs separate planning failure from step-budget failure.

| ID | Script | Max steps | Compare against |
| --- | --- | --- | --- |
| C3-step40 | `xlan/gigpo_curr_n3_step40.sh` | `40` | `xlan/gigpo_curr_n3.sh` |
| D3-step40 | `xlan/gigpo_direct_n3_step40.sh` | `40` | `xlan/gigpo_direct_n3.sh` |
| C4-step40 | `xlan/gigpo_curr_n4_step40.sh` | `40` | `xlan/gigpo_curr_n4.sh` |

Interpretation:

If success improves strongly with step budget, report both success rate and average episode length. If success barely changes, the bottleneck is likely exploration/planning rather than horizon.

## Train/Val Composition Generalization

`CURRICULUM_TRAIN_FRACTION` controls how many composition states are used for training within each stage. Lower fractions create a harder held-out-composition validation split.

| ID | Script | Train fraction | Purpose |
| --- | --- | --- | --- |
| C3-trainfrac50 | `xlan/gigpo_curr_n3_trainfrac50.sh` | `0.5` | Stronger held-out composition test |
| C3-default | `xlan/gigpo_curr_n3.sh` | `0.8` | Default |
| C3-trainfrac90 | `xlan/gigpo_curr_n3_trainfrac90.sh` | `0.9` | More train coverage, easier validation |

Useful contrast:

```text
C3-trainfrac50 low success + C3-trainfrac90 high success
=> model may be memorizing seen composition patterns.

C3-trainfrac50 still good
=> stronger evidence for combinatorial generalization.
```

## Seed Robustness

For any claim you want to put in a report, run at least 3 seeds:

```bash
ENV_SEED=0 CURRICULUM_SEED=0 sbatch <script>
ENV_SEED=1 CURRICULUM_SEED=1 sbatch <script>
ENV_SEED=2 CURRICULUM_SEED=2 sbatch <script>
```

Priority for multi-seed:

1. `xlan/gigpo_curr_n3.sh`
2. `xlan/gigpo_direct_n3.sh`
3. `xlan/gigpo_curr_n3_more_replay.sh` or `xlan/gigpo_curr_n3_no_replay.sh`
4. `xlan/gigpo_curr_n4.sh`
5. `xlan/gigpo_direct_n4.sh`

## Reading Results

Do not judge a method only by its highest validation success rate. Use the highest point for checkpoint selection, but use curve stability and multi-seed averages for method comparison.

Recommended metrics:

| Metric | Use |
| --- | --- |
| `best_val_success` | Select the checkpoint for staged runs, e.g. C2 -> C3 |
| `final_val_success` | Check whether the run keeps its performance at the end |
| `avg_last3_val_success` | Main single-run reporting metric; average the last 3 validation points |
| `step_of_best` | Diagnose whether improvement happens early, late, or as a lucky spike |
| `best_final_gap` | Detect instability, forgetting, or overfitting |
| WandB curve shape | Explain learning speed, plateau, collapse, and replay effects |

For staged curriculum:

```text
C2 stage:
  Use best_val_success to choose the C2 checkpoint.

C3/C4 stage:
  Report avg_last3_val_success, final_val_success, and best_val_success.
  Use WandB curves to explain whether gains are stable or spiky.
```

For comparing methods, prefer:

```text
mean +/- std over seeds of avg_last3_val_success
```

over:

```text
maximum success over all checkpoints and all seeds
```

The maximum is still useful, but mainly as an upper-bound signal or checkpoint-selection rule. A method with one high spike and poor final/last-3 performance should be considered unstable.

Suggested interpretation patterns:

| Pattern | Interpretation |
| --- | --- |
| High best, low final | The model found a good policy briefly but did not stabilize |
| Curriculum rises earlier than direct | Curriculum improves learning efficiency |
| Curriculum final > direct final across seeds | Stronger evidence that curriculum improves performance |
| More replay has lower best but higher last-3 | Replay may stabilize learning |
| Step40 improves both curriculum and direct | Horizon was a bottleneck |
| Step40 improves only curriculum | Curriculum learned the task, but needed longer execution |
| Direct has high variance across `ENV_SEED` | Direct baseline depends heavily on sampled target composition |

## Suggested Reporting Table

Use one row per run and seed.

| Run | Seed | Curriculum | N | Mix | Train frac | Max steps | Best success | Final success | Last-3 avg success | Step of best | Avg reward | Invalid action rate | Avg episode length |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C3 | 0 | yes | 3 | 0.7/0.2/0.1 | 0.8 | 20 |  |  |  |  |  |  |  |
| D3 | 0 | no | 3 | none | n/a | 20 |  |  |  |  |  |  |  |

## Interpretation Checklist

Before making a conclusion, check:

- Does curriculum N=3 beat direct N=3 under the same seed budget?
- Does curriculum N=3 transfer better to validation compositions?
- Does more replay help or hurt hard-stage performance?
- Does increasing `MAX_STEP` explain the gap?
- Are improvements stable across at least 3 seeds?
- Is the gain visible in `avg_last3_val_success`, or only as a single best-success spike?
- Are invalid action rates lower for curriculum runs?

## Current Script Inventory

| Script | Role |
| --- | --- |
| `xlan/gigpo_base.sh` | Shared training command |
| `xlan/gigpo_curr_n2.sh` | Curriculum up to N=2 |
| `xlan/gigpo_curr_n3.sh` | Curriculum up to N=3 |
| `xlan/gigpo_curr_n3_from_c2.sh` | Continue N=3 curriculum from a C2 checkpoint |
| `xlan/gigpo_curr_n4.sh` | Curriculum up to N=4 |
| `xlan/gigpo_curr_n4_from_c3.sh` | Continue N=4 curriculum from a C3 checkpoint |
| `xlan/gigpo_direct_n3.sh` | Direct N=3 baseline |
| `xlan/gigpo_direct_n4.sh` | Direct N=4 baseline |
| `xlan/gigpo_curr_n3_no_replay.sh` | N=3 curriculum without replay |
| `xlan/gigpo_curr_n3_more_replay.sh` | N=3 curriculum with more replay |
| `xlan/gigpo_curr_n3_step40.sh` | N=3 curriculum with longer horizon |
| `xlan/gigpo_direct_n3_step40.sh` | Direct N=3 with longer horizon |
| `xlan/gigpo_curr_n4_step40.sh` | N=4 curriculum with longer horizon |
| `xlan/gigpo_curr_n3_trainfrac50.sh` | N=3 curriculum with harder held-out split |
| `xlan/gigpo_curr_n3_trainfrac90.sh` | N=3 curriculum with easier held-out split |
