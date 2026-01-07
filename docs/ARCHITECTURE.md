# GraphRAG-TCTL Architecture

This document explains the end-to-end flow that converts a natural language (NL) atomic constraint into a structured UPPAAL query using a two-pass LLM strategy enhanced with knowledge-graph retrieval.

## 1. High-Level Flow

1. **User Input Layer**
   - Accepts an NL constraint describing safety, liveness, or timing guardrails for a cyber-physical system.
2. **LLM Analysis (First Pass)**
   - Claude (or another LLM) is prompted to classify the constraint into a known temporal pattern, extract key entities, identify temporal operators, and infer the user's intent.
   - Output conforms to `LLMAnalysisResult`.
3. **Knowledge Graph Retrieval**
   - Uses Neo4j + vector search to fetch:
     - Formal pattern templates (e.g., `A[] (condition imply action)`).
     - Operator documentation and best-practice snippets.
     - Similar historical examples for few-shot conditioning.
   - Combines symbolic queries (Cypher) and semantic similarity search.
4. **LLM Generation (Second Pass)**
   - Claude receives the original NL constraint plus the retrieved artifacts.
   - Produces a UPPAAL/TCTL query and natural-language rationale.
   - Output conforms to `GenerationResult`.
5. **Final Output Adapter**
   - Packages query, detected pattern, supporting context, and explanation for downstream systems (CLI, API, or UI).

## 2. Component Responsibilities

| Component | Responsibility | Key Inputs | Key Outputs |
| --- | --- | --- | --- |
| `UserConstraint` | Normalizes NL text and metadata | Raw user text | Clean text, domain |
| `ClaudeClient` | Two-pass LLM interface | Prompt fragments, context packs | `LLMAnalysisResult`, `GenerationResult` |
| `KnowledgeGraphClient` | Deterministic retrieval from Neo4j | Pattern type, entity hints | Templates, operator docs, labeled examples |
| `EnhancedVectorStore` | Semantic similarity search across textual artifacts stored in Neo4j (or external store) | Query embedding, filters | Ranked passages/examples |
| `RetrievalOrchestrator` | Bundles deterministic + semantic retrieval | `LLMAnalysisResult` | `RetrievalContext` |
| `UPPAALQueryBuilder` | Validates and post-processes LLM output | Generated text | Query string + diagnostics |
| `GraphRAGPipeline` | Orchestrates the entire workflow | `UserConstraint` | `PipelineOutput` |

## 3. Data Contracts

- `LLMAnalysisResult`
  - `pattern`: enum (e.g., `safety_immediate_response`).
  - `entities`: canonical variable identifiers.
  - `temporal_keywords`: lexemes used for prompting retrieval.
  - `intent`: normalized description (linkable to KG).
- `RetrievalContext`
  - `pattern_template`: canonical TCTL form from KG.
  - `operator_docs`: list of operator descriptions with citations.
  - `examples`: list of similar NL constraint + UPPAAL pairs.
  - `prompt_hints`: derived metadata for the second pass.
- `GenerationResult`
  - `query`: candidate UPPAAL query.
  - `pattern`: echoed pattern label.
  - `explanation`: textual justification.
  - `confidence`: heuristic quality metric.

Each dataclass is JSON-serializable to simplify API exposure.

## 4. Knowledge-Graph Layout

- `Pattern` nodes (`pattern_id`, `name`, `template`, `description`)
- `Operator` nodes (`symbol`, `semantics`, `usage_rules`)
- `Example` nodes (`nl_constraint`, `query`, `pattern_id`)
- Relationships:
  - `(Pattern)-[:USES]->(Operator)`
  - `(Example)-[:INSTANCE_OF]->(Pattern)`
  - `(Operator)-[:DOCUMENTED_IN]->(DocFragment)`

Vector indexes are attached to `Example` and `DocFragment` text to support similarity search without leaving Neo4j.

## 5. Deployment Considerations

- The LLM can be swapped by configuring a different provider in `ClaudeClient` because the interface uses a simple `generate(prompt, system)` contract.
- Neo4j connectivity is abstracted behind `KnowledgeGraphClient`; switching to a different KG or even a static JSON source only requires a new implementation of the same interface.
- The pipeline is stateless and can run inside a serverless function, FastAPI service, or CLI. All long-lived resources (Neo4j driver, embedding model) are injected via dependency containers defined in `config.py`.

## 6. Future Enhancements

- Add guarded decoding using a UPPAAL grammar to further trust the generated queries.
- Track provenance by logging every KG artifact that influenced the final answer.
- Incorporate constraint validation by simulating the automaton before emitting the final query.

