Yep — and one key clarification up front:

* **Prompt length** is determined by your dataset + `ctx_bucket`.
* **Reasoning level (low/med/high)** mostly changes the **teacher output length** (CoT+answer), which is what will swing your token budget.

## 1) What are the **average CoT+Answer lengths** for low/med/high (gpt‑oss‑20b)?

OpenAI’s model card plots **accuracy vs average “CoT + Answer length”** across reasoning modes. From Figure 3 (AIME 2025 + GPQA Diamond), the rough averages for **gpt‑oss‑20b** are:

* **Low:** ~**1k** tokens
* **Medium:** ~**6–8k** tokens
* **High:** ~**20k+** on AIME and ~**32k** on GPQA

The model card also explicitly notes that **gpt-oss-20b uses over 20k CoT tokens per AIME problem on average** (i.e., high reasoning can get *very* long).

**Takeaway for budgeting:** at **high reasoning**, the 20B teacher can emit *very* long outputs, so cap or curriculum “high” carefully.

---

## 2) Yes — you should use a **reasoning-level curriculum**

Given those lengths, if you don’t gate “high,” it will dominate your token budget.

A simple curriculum that works well for budgeting:

* **Early:** mostly **low** (stability + cheap)
* **Main:** mostly **medium** (best value)
* **Late / selective:** **high** only on (a) hardest prompts, (b) long ctx buckets, (c) scorer-approved high-utility samples

(That matches what Nemotron-Math folks do conceptually: use buckets/curricula to avoid wasting long outputs on easy samples.)

---

## 3) Reverse-engineering: how many prompts do we need?

Let total token budget be **T = 750M**, counting:
[
t_{\text{example}} = t_{\text{prompt}} + t_{\text{teacher_output}}
]
(where teacher_output = analysis + final).

For each cell in your grid **(bucket b, ctx c)**:

1. **Token budget for the cell** (if you use token-share quotas):
   [
   T_{b,c} = T \cdot s_b \cdot s_c
   ]
   (where (s_b) is the skill-bucket token share, (s_c) is the ctx-bucket token share).

2. **Expected tokens per example**:
   [
   \mathbb{E}[t_{b,c}] = \mathbb{E}[t_{\text{prompt}}(c)] + \sum_{m \in {L,M,H}} p_{b,c}(m),\mathbb{E}[t_{\text{out}}(\text{teacher}, m)]
   ]

* (p_{b,c}(m)) is your curriculum mix of reasoning modes in that cell.
* (\mathbb{E}[t_{\text{out}}]) differs across low/med/high reasoning (numbers above).

3. **Number of prompts needed**:
   [
   N_{b,c} = \frac{T_{b,c}}{\mathbb{E}[t_{b,c}]}
   ]

### Important: your ctx mixes are “**token mixes**”

In your doc you wrote “most tokens short/medium,” so those percentages should be interpreted as **token share**, not sample share. That’s exactly why you need the formula above.

---

## 4) What I’d do next (practical, fast)

To make the reverse-engineering accurate, don’t guess — **measure** on your own prompt pools:

* Sample ~**1k prompts** per bucket (B1–B5), stratified by ctx bucket.
* For **gpt‑oss‑20b** and each reasoning mode (L/M/H), run with your intended `max_new_tokens` per ctx bucket.
* Record:

  * `prompt_tokens`
  * `teacher_output_tokens` (analysis + final)
* Compute mean / p50 / p90 per (teacher, mode, bucket, ctx).

Then plug those means into the equations above and you’ll have **prompt counts per bucket per ctx** immediately.

If you want, I can propose a **first-pass set of max_new_tokens caps per ctx bucket** (so “high” doesn’t explode your 750M budget) and we can run the math on paper to get initial prompt-count targets.
