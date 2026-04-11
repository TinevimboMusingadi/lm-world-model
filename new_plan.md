# Research Log & Forward Planning
## LM World Model — Phase 1 Training Run Reflection

**Author:** Tinevimbo Musingadi  
**Date:** April 11, 2026  
**Status:** Live training run (Step ~500/645)  

---

## 1. What We Are Doing Right Now & Why

### The Experiment
We are fine-tuning **Qwen2.5-1.5B-Instruct** on 5,974 Python execution traces (Condition B: full trace supervision). The model receives a short Python program and must predict every intermediate variable state at each line, followed by the final output.

### The Motivation
This is a direct empirical test of the central research question from the proposal:

> *Do language models, through next-token prediction, construct internal representations that constitute a model of how the world works?*

The specific bet is that by forcing the model to predict `[line=3] var_0=42, var_1=7` — i.e., the **exact machine state** at every step — we are training it to learn the *rules of computation*, not just pattern-match input-output pairs. If that works, it would show up as:

1. High OOD accuracy (it generalises to unseen code)
2. Low Levenshtein distance (its predicted traces are close to ground truth)
3. Linear probe R² > 0 (the residual stream actually encodes variable values)
4. Activation patching hotspots (specific layers causally carry state information)

---

## 2. Observed Training Dynamics

### Full Loss Curve (up to Step 500)

| Step | Loss | Phase |
|---|---|---|
| 25 | 0.6835 | Random → format template learnt |
| 50 | 0.1574 | 🔥 Format mastered (77% drop) |
| 75 | 0.1131 | Semantic learning begins |
| 100 | 0.1007 | Format perfectly stable |
| 125–300 | 0.089–0.094 | **Plateau Phase I** — slow grind |
| 300–500 | 0.086–0.089 | **Plateau Phase II** — essentially flat |

### Reading the Curve

The massive drop at step 25→50 shows the model immediately learnt *what a trace looks like* — the format, the XML tags, the line-by-line structure. This is not surprising; it is just template learning. The interesting question is what happens at 0.086 and beyond.

**No grokking observed in SFT.** The hoped-for sudden transition from memorisation to generalisation did not materialise in the supervised learning phase. This is important to record, but not fatal to the research. Grokking in algorithmic tasks (Powers et al.; Nanda et al.) typically requires:

1. Sufficient overparameterisation relative to dataset complexity
2. Long training beyond the point of near-zero training loss (generalisation happens *after* near-zero training loss)
3. Often, weight decay as the mechanism that pushes toward "cleaner" circuits

Our training loss has plateaued at ~0.086, which points to the model **not having fully converged** on the training data yet — the loss should be < 0.02 for grokking dynamics to be possible. This is because at 1.5B parameters with LoRA-16, we are using a relatively constrained adapter and only 3 epochs.

**Key insight:** Grokking may be the wrong thing to look for in SFT. The more interesting window is **RL training**, where the reward signal may create the sharp generalisation boundary that SFT didn't.

---

## 3. Immediate Next Steps (This Week)

### 3.1 Complete and Evaluate Current Run
Once training finishes (~Step 645):

1. **Run Stages 3A/3B** (evaluation cells) — get OA, Levenshtein, FES scores
2. **Run Stage 5A/5B** (Logit Lens + Linear Probe) — check if R² > 0 at any layer
3. **Run Stage 5C** (Activation Patching) — look for causal hotspots

These numbers are the actual research contribution from this run, regardless of the training loss plateau.

### 3.2 Hyperparameter Tuning for Run 2

The current run used defaults that were conservative (fast to run, T4-compatible). Before scaling to a bigger model, we should squeeze more out of 1.5B with better hyperparameters.

**Changes to try for Run 2:**

| Setting | Current | Proposed | Rationale |
|---|---|---|---|
| `num_train_epochs` | 3 | **10–15** | Grokking needs longer training beyond near-zero loss |
| `learning_rate` | 2e-4 | **5e-5** | Lower LR → smoother convergence, less early plateau |
| `lora_r` | 16 | **32** | More expressive adapter, better representational capacity |
| `weight_decay` | 0.01 | **0.1** | Weight decay is mechanistically tied to grokking emergence |
| `warmup_steps` | 50 | **100** | Gentler warmup for stability with longer run |
| `max_seq_length` | 1024 | **1024** | Keep same — traces fit fine |

**Grokking hypothesis for Run 2:** If we train for 15 epochs with higher weight decay, we may see the loss go near-zero on training data, then watch for validation accuracy to spike. This is the classic grokking signature. Run 2 should track train/val loss *separately* at every checkpoint.

---

## 4. Scaling Up Model Size (RQ4)

The proposal explicitly asks: *at what scale does execution simulation emerge?*

### Scale Ladder

| Model | Parameters | LoRA-R | Est. VRAM | Est. Time/Epoch (T4) |
|---|---|---|---|---|
| Qwen2.5-0.5B | 500M | 8 | ~3 GB | ~15 min |
| **Qwen2.5-1.5B** ← *here now* | 1.5B | 16 | ~6 GB | ~1.5 hrs |
| Qwen2.5-3B | 3B | 16 | ~9 GB | ~3 hrs |
| Qwen2.5-7B | 7B | 32 | ~18 GB | Needs A100 |

For RQ4 we need at minimum Qwen2.5-0.5B (baseline) and one comparison. **The cleanest paper result would be:**

> "Execution simulation did **not** emerge at 0.5B (OA < X%), emerged partially at 1.5B (OA = Y%), and was confirmed at 7B (OA = Z%)."

This gives a clean scaling curve and matches the structure of Anthropic's mechanistic interpretability papers.

**Practical path:** Run 0.5B on T4 (it's fast) as the baseline. For 7B, apply for Google Colab Pro or use the HIT computer lab GPU if available.

---

## 5. The RL Training Hypothesis

This is the most exciting forward-looking idea, and the one most likely to be novel.

### The Hypothesis

**SFT teaches the format. RL teaches the logic.**

In SFT, the model is optimised to produce tokens that match the ground truth trace character-by-character. This is a strong signal for *format* but a weak signal for *semantics* — the model doesn't need to "understand" why `var_0=42` comes after `var_0 = 21 * 2`; it just needs to predict the right tokens.

In GRPO-style RL, the reward is:

```
R = 0.35 × (output correct) + 0.40 × (trace similarity) + 0.25 × (steps correct %)
```

Crucially, the model only sees the reward *after generating the full trace*. This forces it to develop an internal strategy that produces correct states — it cannot rely on copying the next token; it must *simulate*.

**The grokking-during-RL hypothesis:** If a model trained with SFT to ~0.086 loss (good format, decent semantics) is then exposed to RL with the trace reward, we might see a delayed grokking event — a sudden jump in reward score — as the model transitions from "format-producing" to "state-simulating". This would be a major finding, aligned with DeepSeek-R1's observation that RL training produced emergent chain-of-thought reasoning not present in the SFT checkpoint.

### What to do

1. After Stage 3 evaluation, run the existing RL simulation cell (Stage 4) to see the reward distribution
2. If mean reward > 0.5, the model is already close enough to correct for RL to be viable
3. If mean reward < 0.3, do Run 2 (better SFT) before RL

For full GRPO training: use `trl.GRPOTrainer` with our `compute_reward` function as the reward model. This is ~50 lines of code on top of what already exists.

---

## 6. Validation Strategy

One thing the current run is missing that would significantly strengthen the research: **a clean train/validation loss split plotted separately over training.**

Right now we only log training loss. For the grokking analysis and for the paper, we need:

- `eval_dataset` passed to SFTTrainer with `eval_steps=50`
- Separate validation loss tracked in the W&B dashboard
- The **generalisation gap** plotted = `train_loss - val_loss`

When grokking occurs, train loss stays flat (near-zero) while val loss suddenly drops. Without tracking val loss, we can only see half the picture.

**For Run 2:** Add a `val_recs` split to the SFT training loop with `evaluation_strategy='steps'`.

---

## 7. Research Trajectory — Where This Could Go

### Short Term (1–3 months)
- Complete Phase 1 evaluation + MechInterp results
- Write up the Phase 1 findings as a standalone section
- Begin Phase 2 (MIS — Minimal Instruction Set): generate MIS traces, train same architecture, compare generalisation

### Medium Term (3–6 months)
- RL training on top of the best SFT checkpoint
- Scaling experiment: 0.5B vs 1.5B vs (ideally) 7B
- Submit findings as a short paper or technical report

### Long Term (6–12 months)
- Full Phase 2 MIS results: can the model generalise to instruction combinations it has never seen?
- Adversarial robustness test: obfuscated MIS programs with same semantics, different surface form
- If RL-grokking is confirmed: that becomes the primary contribution — the paper title shifts to something like *"Emergent World Models via Reinforcement Learning on Execution Traces"*

### The Paper Argument (if results are strong)

```
Claim: Sub-2B LMs develop genuine internal world models under trace supervision
Evidence:
  1. OA (OOD) >> chance baseline → generalisation, not memorisation
  2. Probe R² > 0 at intermediate layers → state is linearly encoded
  3. Activation patching hotspot → specific circuits are causally responsible
  4. [If RL works] Grokking in RL phase → emergent simulation, not just pattern matching
  5. Phase II MIS: generalises to unseen instruction set → not Python-specific
```

This structure matches top-tier venues (NeurIPS Mech Interp Workshop, ICLR, possibly ICML).

---

## 8. Open Questions to Resolve After This Run

1. **Does the current model (0.086 loss) have OA > 50% on test_indist?** → Determines if Run 2 is needed before MechInterp
2. **Is Probe R² > 0 at any layer?** → Core world model hypothesis test
3. **What is the mean RL reward on the current checkpoint?** → Determines if RL is viable now or needs better SFT first
4. **Does OA drop significantly at T=0.5 vs T=0?** → Fragility test; if large drop, model is brittle (pattern-matching)
5. **What is the OOD gap (InDist OA − OOD OA)?** → Small gap = genuine generalisation; large gap = memorisation

---

*This document is a living research log. Update after each major milestone.*
