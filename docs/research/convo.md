Distilling a Tiny-Active DeepSeekV3-MoE Student from GPT-OSS-120B (Conversation Notes)
0) Goal (the north star)
Build a tiny-yet-powerful model that is:
* STEM / math / code strong, but still elegant at writing
* native-CoT (reasoning traces are part of pretrain + KD, not just post-training)
* Competitive with Llama 3.2 1B/3B-class models and ideally pushes toward “Trinity Nano”-like quality
* Trained solo on 8×H100, with moat = data quality and smart distillation

1) Current student architecture (your stated plan)
Base backbone: DeepSeekV3-style MoE with major modifications:
* GRAPE-M positional encoding (replacing RoPE)
* Full GQA (max expressivity mindset)
* ReLU² activation (validated by your ablations)
* QK-norm
* Dense first layer
* Shared expert (stability scaffold)
* Optimizer: NorMuon + Polar Express (vs Newton–Schulz)
Model size/compute framing:
* Total params ~ 4.5B
* Active per token ~ 150M+
* Intent: “make the model as open as possible for learning” (expressivity-first)

2) Granular vs fine-grained MoE (working definitions)
* Granular MoE: large experts, fewer MoE layers, routing selects whole FFN-ish blocks
* Fine-grained MoE: many small experts, frequent routing, active params very small per token
Your design is fine-grained for your scale, prioritizing low active params and high capacity.

3) GPT-OSS takeaways we discussed (what’s worth stealing vs not)
Worth considering to copy (high ROI)
* Interleaved local/global attention (local window + full attention alternating)
    * Not purely “efficiency”; also a stability/regularization prior for long context and MoE routing.
* Attention sink bias
    * “Pay attention to nothing” safety valve that can stabilize representations upstream of the router.
Not necessary to copy
* The entire GPT-OSS architecture end-to-end (you have strong reasons for your own arch).
* GPT-OSS MoE routing style (your DeepSeekV3 router direction is fine; sink bias complements it).

4) Attention sink bias — what it is (and why it matters)
Concept:
* Lets each attention head opt out when nothing useful is present, reducing forced attention to junk.
* Helps especially with:
    * long contexts
    * local/global alternation
    * GQA sharing
    * MoE stability (cleaner residual stream → less routing volatility)
Key point:
* Does not cancel with router bias (different parts of the graph). They complement.
Kernel reality:
* Not always plug-and-play with every FlashAttention path.
* Efficient implementation typically needs access to per-query log-sum-exp (LSE) or custom support.
Action:
* Worth an ablation, especially if you see late-training drift or routing volatility.

5) Local/global attention: “better or just cheaper?”
We converged on:
* At 4k–16k, global-everywhere is already expressive.
* Local/global interleave becomes most valuable once you push context and attention cost dominates.
* It can improve quality indirectly by:
    * reducing “global attention noise”
    * stabilizing residual stream feeding the router
    * enabling longer ctx / larger batches / more tokens for same wall-clock
Decision direction:
* Global everywhere for the main base run.
* Save local/global ablation for later long-context phase (or gradually introduce it).

6) Sequence length plan (native-CoT aware)
We landed on:
* Main training ctx = 8192 (LLama-like; good default for capability-per-dollar)
* Native-CoT doesn’t force longer ctx by default; it mostly increases completion length, so manage with:
    * trace-length caps per reasoning mode (low/med/high)
    * packing multiple examples per 8192
    * bucket rare overflows into a later long-context stage
Long-context:
* Use YaRN later to go to 64k (GRAPE-M compatibility assumed in your plan)
* Avoid letting the long-tail of huge traces dictate the whole base budget.

7) Distillation strategy (base-first, not “pretraining business”)
Your priority: “How good can we make the base from GPT-OSS-120B?”
Key distillation framing:
* Start with off-policy (teacher targets/traces) for support-building.
* Add on-policy/online later once the student is competent (mode-seeking; fixes compounding error).
Token split proposal for a base run (ballpark):
* Mostly off-policy distill + some HQ raw, then a small on-policy tail.
* Keep on-policy low early because teacher throughput can bottleneck.
Native-CoT curriculum idea:
* Use GPT-OSS reasoning modes low → medium → high as curriculum progression.
* Also treat “trace length” as part of the curriculum (shorter early, longer later).

8) Token budget (you proposed)
We discussed “compute-optimal-ish” heuristics and agreed MoE scaling laws are messy; your stance:
* Start with 10B tokens as a serious first target, expecting you might need more later.
* Milestones: evaluate at intermediate checkpoints (e.g., 2B, 5B, 10B) and decide extension based on curves.

9) Tokenizer decision (math/code + native-CoT focus)
You’re worried about:
* number tokenization mismatch
* CoT formatting mismatch
We discussed trade:
* GPT-OSS tokenizer is ~200k vocab. It’s “clean” for matching teacher tokenization.
* Big vocab has real costs: embeddings + lm_head are large and dense each token.
Rule-of-thumb param cost:
* Extra params from vocab ~= vocab_size × hidden_size per matrix.
* If you untie embed and lm_head, you pay roughly 2× that (tied halves it).
Your current direction:
* “Bite the bullet” and use OpenAI’s tokenizer for cleanliness and reduced mismatch risk.

10) Weight tying vs untied (you’re leaning untied)
Why people untie:
* Input embedding and output head do different jobs; separate matrices can improve modeling, especially for rare/code/math tokens.
* Also enables different quantization/sharding/precision strategies.
Cost:
* Untying adds one extra vocab_size × hidden_size matrix (can be hundreds of millions params depending on hidden size).
Your rationale:
* You’re likely going “smaller hidden size + more layers,” and you observed this can make small models feel smarter.
* You’re willing to pay the cost.

11) “Cheating” initialization: upcycling from a dense checkpoint
We aligned on:
* You can start from an existing dense checkpoint and upcycle to MoE (replicate FFN weights to experts + noise; warm up router).
* You do NOT want to be constrained by exact seed model shapes/activations/tokenizer.
Key clarification:
* If the seed tokenizer differs from o200k, embeddings/lm_head won’t align. You can:
    * reinit embeddings/head in o200k and keep the rest as “feature prior,” and distill hard
    * OR use tokenizer transplantation tooling to bridge tokenization
We explicitly deprioritized:
* Naive “down-project GPT-OSS-20B weights by averaging” (not a reliable transformer compression trick).

12) MergeKit / tokenizer tooling (high-level)
We touched on:
* MergeKit is great for merging models with matching shapes, and has tokenizer tooling / tokensurgeon-style transplantation.
* It’s not a magic “shrink GPT-OSS-20B into your custom 5B” solution.
* Tokenizer transplantation (Arcee + tokensurgeon) can help avoid adopting a massive vocab while still doing token-level KD, but your current direction is to just use o200k for cleanliness.

13) Current “tentative” plan snapshot (what we converged to)
1. Tokenizer: likely adopt o200k for math/code/CoT fidelity.
2. ctx: train base at 8192.
3. Distill curriculum: low → medium → high reasoning modes; control trace length.
4. Data mix: heavy off-policy distill + some HQ raw; add on-policy later.
5. Long context: later YaRN to 64k, possibly with local/global ablations.
6. Ablations to run (high ROI):
    * attention sink bias
    * local/global attention introduced in long-context phase
    * shared expert schedule (constant vs warm-up/anneal)
    * tied vs untied head (since you’re leaning untied, validate)

14) Open questions / next decisions
* Final “base token budget” schedule: how much of 10B is distill vs raw?
* Online KD design: how to allocate teacher GPUs (1×7 vs 4×4 vs sync) without throughput collapse?
* Shared expert: permanent vs warm-up then anneal (stability vs specialization tradeoff)
* Decide whether local/global becomes part of the final model or stays as a long-context-only experiment.

15) Immediate next steps checklist
*  

End of notes.
