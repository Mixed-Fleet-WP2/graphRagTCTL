# Retrieval-Related Semantic Errors in Flat Vector RAG

These examples were generated using `rag_baseline/rag_index` embeddings (`text-embedding-3-small`) with top-5 cosine retrieval.

Definition used here:
- **Retrieval-related semantic error** = the top retrieved example has a pattern different from the intended pattern, with high cosine similarity, which can bias generation toward an incorrect UPPAAL structure.

Reference mapping used for interpretation:
- `safety_immediate_response` → `["A[]", "imply"]`
- `sequential_workflow` → `["-->"]`
- `eventual_completion` → `["A<>"]`
- `existential_reachability` → `["E<>"]`
- `universal_reachability` → `["A<>"]`
- `forbidden_state` → `["A[]", "not"]`
- `conditional_response` → `["-->"]`
- `time_bounded_constraint` → `["A[]", "imply"]`

---

## Example 1 — Exact query with wrong-pattern top retrieval

- **Constraint**: `Robot must be at base when battery is below 10%`
- **Expected pattern**: `safety_immediate_response`
- **Expected operators**: `A[]`, `imply`
- **Evidence source**: `rag_baseline/retrieval_score_proof_robot_exact.json`
- **Index hash (metadata SHA-256)**: `3583d82d1e46b52e1d502ad5853d93297afed966db164a8f122513dac2c6cf15`

### Top retrievals (flat vector, cosine)
1. `c029` — `conditional_response` — **0.6549**
2. `c022` — `time_bounded_constraint` — 0.6080
3. `c030` — `time_bounded_constraint` — 0.5975
4. `c012` — `conditional_response` — 0.5762
5. `c001` — `safety_immediate_response` — 0.5247

### Interpretation
- This is a **retrieval-related semantic failure case**: top-1 retrieval is not the expected pattern.
- Multiple wrong-pattern examples with high lexical overlap outrank the correct safety-immediate example.

---

## Example 2 — Failure variant A (entity shift: Drone → Robot)

- **Constraint**: `Robot must be at  base  when battery is below 10%`
- **Expected pattern**: `safety_immediate_response`
- **Evidence source**: `rag_baseline/retrieval_score_proof.json`
- **Index hash (metadata SHA-256)**: `acc30705c537337491fa8ec6a073cb8026b5a1272c442f736f3d6190792e95d2`

### Top retrievals (flat vector, cosine)
1. `c022` — `time_bounded_constraint` — **0.6120**
2. `c012` — `conditional_response` — 0.5778
3. `c001` — `safety_immediate_response` — 0.5202
4. `c017` — `eventual_completion` — 0.5060
5. `c010` — `forbidden_state` — 0.4767

### Why this failed
- Replacing `Drone` with `Robot` increases lexical affinity to `c022` and `c012`.
- The correct safety-immediate example (`c001`) is ranked third.

### Semantic failure mode
- Generation may introduce unintended time-bound structure (`clock <= T`) or leads-to semantics (`-->`) instead of strict immediate implication under `A[]`.

---

## Example 3 — Failure variant B (same intent, stronger trigger phrasing)

- **Constraint**: `When robot battery is low, robot must be at base immediately`
- **Expected pattern**: `safety_immediate_response`

### Top retrievals (flat vector, cosine)
1. `c022` — `time_bounded_constraint` — **0.6277**
2. `c012` — `conditional_response` — 0.5898
3. `c001` — `safety_immediate_response` — 0.5650
4. `c017` — `eventual_completion` — 0.5217
5. `c002` — `safety_immediate_response` — 0.4852

### Why this failed
- Tokens `when`, `battery low`, and `robot` align strongly with conditional/time-bounded examples.
- Correct safety-immediate evidence appears, but below two wrong-pattern examples.

### Semantic failure mode
- The model may produce either a delayed/conditional response (`-->`) or an unnecessary time-bounded response, despite the explicit adverb `immediately`.

---

## Example 4 — Failure variant C (conditional wording dominates)

- **Constraint**: `When battery is low, robot goes to base immediately`
- **Expected pattern**: `safety_immediate_response`

### Top retrievals (flat vector, cosine)
1. `c012` — `conditional_response` — **0.6386**
2. `c022` — `time_bounded_constraint` — 0.6039
3. `c001` — `safety_immediate_response` — 0.5557
4. `c017` — `eventual_completion` — 0.5252
5. `c020` — `sequential_workflow` — 0.4619

### Why this failed
- Verb phrase `goes to` and clause opener `when` strongly match `conditional_response` examples.
- Correct pattern remains in top-3 but is outscored by two wrong-pattern examples.

### Semantic failure mode
- Generation is likely to drift to `-->` semantics, weakening immediate-safety interpretation.

---

## Takeaway

Even with the same intended pattern (`safety_immediate_response`), small lexical shifts can cause high-scoring retrieval of different patterns in flat vector RAG. Therefore, cosine similarity alone is insufficient for pattern-faithful retrieval in this verification setting.
