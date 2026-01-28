# GraphRAG-TCTL Pipeline Architecture

## 🎯 **Goal**: Natural Language → UPPAAL Temporal Logic Query

---

### **📊 Pipeline Flow**

```
┌─────────────────┐
│  Natural Language  │  "Drone returns to base when battery < 10%"
│   Constraint       │
└────────┬───────────┘
         │
         ▼
┌────────────────────────────────────────────────────────────┐
│  STEP 1: LLM Classification (OpenAI GPT)                   │
│  ─────────────────────────────────────────────────────     │
│  • Identify pattern (safety, sequential, conditional, etc) │
│  • Extract keywords & temporal cues                        │
│  • Detect trigger/response conditions                      │
│  Output: Classification { pattern, keywords, cues }        │
└────────┬───────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────────┐
│  STEP 2: GraphRAG Retrieval (Neo4j Knowledge Graph)        │
│  ─────────────────────────────────────────────────────     │
│  • Fetch pattern template (A[], E<>, -->, imply)           │
│  • Retrieve operator semantics & usage rules              │
│  • Get similar examples (keyword-based ranking)            │
│  Output: GraphContext { pattern, operators, examples }    │
└────────┬───────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────────┐
│  STEP 3: LLM Generation (OpenAI GPT)                       │
│  ─────────────────────────────────────────────────────     │
│  • Apply pattern template to constraint                    │
│  • Use operator guidelines for correct syntax              │
│  • Learn from similar examples (few-shot)                  │
│  Output: { query, explanation, confidence }               │
└────────┬───────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────────┐
│  STEP 4: ANTLR Validation                                  │
│  ─────────────────────────────────────────────────────     │
│  • Parse query against UPPAAL grammar                      │
│  • Verify syntactic correctness                            │
│  Output: { antlr_ok: true/false, errors }                 │
└────────┬───────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────┐
│  UPPAAL Query   │  A[] (battery < 10 imply drone.location == base)
└─────────────────┘
```

---

### **🔑 Key Components**

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Classifier** | OpenAI GPT-5.1 | Parse NL → identify temporal pattern |
| **Knowledge Graph** | Neo4j | Store patterns, operators, examples |
| **Generator** | OpenAI GPT-5.1 | Template + context → UPPAAL query |
| **Validator** | ANTLR Parser | Ensure syntactic correctness |

---

### **🧠 Knowledge Graph Schema**

```
(Pattern)─[:USES_OPERATOR]→(Operator)
    ↑
    │
[:INSTANCE_OF]
    │
(Example: NL + UPPAAL pair)
```

**8 Pattern Types**: safety_immediate_response, sequential_workflow, eventual_completion, 
existential_reachability, universal_reachability, forbidden_state, 
conditional_response, time_bounded_constraint

---

### **✨ Why GraphRAG?**

1. **Deterministic Retrieval** - Pattern templates ensure consistency
2. **Operator Guidance** - When/when not to use specific operators
3. **Few-Shot Learning** - Similar examples improve generation quality
4. **Scalable** - Add new patterns/examples without retraining
5. **Explainable** - Traceable from NL → pattern → template → query

---

### **📈 Output Example**

**Input**: `"Drone returns to base when battery < 10%"`

**Output**:
```json
{
  "query": "A[] (battery < 10 imply drone.location == base)",
  "pattern_used": "conditional_response",
  "confidence": 0.95,
  "explanation": "Uses A[] with imply for immediate response constraint",
  "antlr_ok": true
}
```
