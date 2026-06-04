#!/usr/bin/env python3
"""
Flat RAG baseline pipeline  —  fair comparison with Graph RAG.

Architecture (mirrors pipeline.py step-by-step):
  Step 1: Classify constraint  →  same LLM call as Graph RAG
  Step 2: Flat embedding retrieval  →  cosine similarity over rag_index/
           (replaces Neo4j graph traversal)
  Step 3: Generate UPPAAL query  →  same prompt structure as Graph RAG
  Step 4: Validate with ANTLR   →  same validation service

Run:
    python pipeline_rag.py "Drone returns to base when battery < 10%"

Env vars required:
    OPENAI_API_KEY
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ─── Paths ────────────────────────────────────────────────────────────────────
_DIR = Path(__file__).parent
INDEX_DIR = _DIR / "rag_index"
EMBEDDING_MODEL = "text-embedding-3-small"

# ─── UPPAAL pattern enum (identical to Graph RAG pipeline) ────────────────────
PATTERN_ENUM = [
    "safety_immediate_response",
    "sequential_workflow",
    "eventual_completion",
    "existential_reachability",
    "universal_reachability",
    "forbidden_state",
    "conditional_response",
    "time_bounded_constraint",
]


# ─── Data classes (mirrors pipeline.py) ──────────────────────────────────────
@dataclass
class Classification:
    pattern: str
    keywords: List[str]
    cues: List[str]
    trigger_text: Optional[str] = None
    response_text: Optional[str] = None


@dataclass
class RetrievedContext:
    """Flat RAG equivalent of GraphContext."""
    chunks: List[Dict[str, Any]]   # top-k retrieved chunks (mixed types)


# ─── Step 1: Classify (identical to Graph RAG) ────────────────────────────────
def classify_constraint(client: OpenAI, constraint: str) -> Classification:
    print(f"\n{'='*70}")
    print("STEP 1: CLASSIFYING CONSTRAINT")
    print(f"{'='*70}")
    print(f"  Input: {constraint}")

    schema = {
        "name": "ConstraintClass",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "pattern": {"type": "string", "enum": PATTERN_ENUM},
                "keywords": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                "cues": {"type": "array", "items": {"type": "string"}},
                "trigger_text": {"type": ["string", "null"]},
                "response_text": {"type": ["string", "null"]},
            },
            "required": ["pattern", "keywords", "cues", "trigger_text", "response_text"],
        },
        "strict": True,
    }

    prompt = f"""
Classify this natural-language constraint into ONE pattern:

Patterns:
{PATTERN_ENUM}

Return:
- pattern (enum)
- keywords (for example retrieval)
- cues: tokens like ["when","after","then","eventually","never","must","can","<",">","%"]
- trigger_text/response_text if obvious (optional)

Constraint:
{constraint}
""".strip()

    resp = client.responses.create(
        model="gpt-5.1",
        input=prompt,
        text={
            "format": {
                "type": "json_schema",
                "name": schema["name"],
                "schema": schema["schema"],
                "strict": schema["strict"],
            }
        },
    )
    data = json.loads(resp.output_text)
    print(f"  ✓ Pattern   : {data['pattern']}")
    print(f"  ✓ Keywords  : {data['keywords']}")
    print(f"  ✓ Cues      : {data.get('cues', [])}")
    return Classification(
        pattern=data["pattern"],
        keywords=data["keywords"],
        cues=data.get("cues", []),
        trigger_text=data.get("trigger_text"),
        response_text=data.get("response_text"),
    )


# ─── Step 2: Flat RAG retrieval ───────────────────────────────────────────────
class FlatRAG:
    """
    Embeds the query and retrieves top-k chunks by cosine similarity.
    No graph structure, no typed relations — purely dense vector lookup.
    This is the key difference from Graph RAG.
    """

    def __init__(self, index_dir: Path) -> None:
        emb_path = index_dir / "embeddings.npy"
        meta_path = index_dir / "metadata.json"
        if not emb_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"Index not found at {index_dir}. "
                "Run build_index.py first."
            )
        self.embeddings: np.ndarray = np.load(str(emb_path))  # (N, dim)
        with open(meta_path) as f:
            self.metadata: List[Dict[str, Any]] = json.load(f)

        # L2-normalise for fast cosine via dot product
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings_norm = self.embeddings / (norms + 1e-10)

    def retrieve(
        self,
        client: OpenAI,
        query_text: str,
        k: int = 10,
    ) -> RetrievedContext:
        print(f"\n{'='*70}")
        print("STEP 2: FLAT RAG RETRIEVAL (cosine similarity)")
        print(f"{'='*70}")
        print(f"  Query: {query_text[:80]}...")

        # Embed query
        resp = client.embeddings.create(model=EMBEDDING_MODEL, input=[query_text])
        q_vec = np.array(resp.data[0].embedding, dtype=np.float32)
        q_norm = q_vec / (np.linalg.norm(q_vec) + 1e-10)

        # Cosine scores
        scores = self.embeddings_norm @ q_norm         # (N,)
        top_indices = np.argsort(scores)[::-1][:k]

        chunks = []
        for rank, idx in enumerate(top_indices):
            chunk = dict(self.metadata[idx])
            chunk["_score"] = float(scores[idx])
            chunks.append(chunk)
            ctype = chunk.get("type", "?")
            cid = chunk.get("id", "?")
            print(f"  [{rank+1:2d}] score={scores[idx]:.4f}  type={ctype:<10}  id={cid}")

        return RetrievedContext(chunks=chunks)


# ─── Step 3: Generate UPPAAL (same prompt structure as Graph RAG) ─────────────
def generate_uppaal(
    client: OpenAI,
    constraint: str,
    cls: Classification,
    ctx: RetrievedContext,
) -> Dict[str, Any]:
    print(f"\n{'='*70}")
    print("STEP 3: GENERATING UPPAAL QUERY")
    print(f"{'='*70}")

    out_schema = {
        "name": "UppaalOut",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "query": {"type": "string"},
                "explanation": {"type": "string"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "pattern_used": {"type": "string", "enum": PATTERN_ENUM},
                "operator_used": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["query", "explanation", "confidence", "pattern_used", "operator_used"],
        },
        "strict": True,
    }

    # ── Format retrieved chunks into context sections ──────────────────────
    pattern_chunks = [c for c in ctx.chunks if c["type"] == "pattern"]
    operator_chunks = [c for c in ctx.chunks if c["type"] == "operator"]
    example_chunks = [c for c in ctx.chunks if c["type"] == "example"]

    # Use the top matching pattern chunk if available, else fall back to classification
    pat_txt = ""
    if pattern_chunks:
        p = pattern_chunks[0]
        pat_txt = (
            f"Pattern: {p.get('name')}\n"
            f"Template: {p.get('template', 'N/A')}\n"
            f"When to use: {p.get('when_to_use', 'N/A')}"
        )
    else:
        pat_txt = f"Pattern (from classification): {cls.pattern}"

    ops_txt = "\n".join(
        f"- {o.get('name')}: {o.get('when_to_use','')} | avoid: {o.get('when_not_to_use','')}"
        for o in operator_chunks
    ) or "- (no operators retrieved)"

    ex_txt = "\n".join(
        f"- NL: {e.get('nl')} | UPPAAL: {e.get('uppaal')}"
        for e in example_chunks
    ) or "- (no examples retrieved)"

    instructions = f"""
Translate ONE natural-language constraint into ONE UPPAAL query.

Classifier output:
- pattern={cls.pattern}
- cues={cls.cues}
- keywords={cls.keywords}
- trigger_text={cls.trigger_text}
- response_text={cls.response_text}

Retrieved pattern context (RAG):
{pat_txt}

Retrieved operators (RAG):
{ops_txt}

Retrieved similar examples (RAG):
{ex_txt}

Constraint:
{constraint}

Rules:
- Follow the pattern template shape.
- Use "imply" for SAME-STATE immediate response (typically with A[]).
- Use "-->" for eventual triggered behavior (sequential_workflow / conditional_response).
- Use A<> for "must eventually happen" global liveness.
- Use E<> for "possible/can happen".
- Use A[] not(...) for "never/must not".

Return JSON only.
""".strip()

    resp = client.responses.create(
        model="gpt-5.1",
        instructions=instructions,
        input="Generate the UPPAAL query now.",
        text={
            "format": {
                "type": "json_schema",
                "name": out_schema["name"],
                "schema": out_schema["schema"],
                "strict": out_schema["strict"],
            }
        },
    )

    result = json.loads(resp.output_text)
    print(f"  ✓ Query      : {result['query']}")
    print(f"  ✓ Confidence : {result['confidence']:.2%}")
    print(f"  ✓ Pattern    : {result['pattern_used']}")
    return result


# ─── Step 4: ANTLR validation (identical to Graph RAG) ───────────────────────
def validate_antlr(query: str) -> Tuple[bool, Optional[str]]:
    print(f"\n{'='*70}")
    print("STEP 4: ANTLR VALIDATION")
    print(f"{'='*70}")
    if not query.strip():
        return False, "Empty query"
    try:
        response = requests.post(
            "http://127.0.0.1:5002/check-property",
            json={"property": query},
            timeout=5,
        )
        response.raise_for_status()
        result = response.json()
        valid = result.get("valid", False)
        errors = result.get("errors", [])
        if valid:
            print("  ✓ PASS")
            return True, None
        else:
            error_msg = "; ".join(errors) if errors else "Validation failed"
            print(f"  ✗ FAIL: {error_msg}")
            return False, error_msg
    except requests.exceptions.ConnectionError:
        print("  ⚠ ANTLR service not reachable")
        return False, "ANTLR parser service not reachable"
    except requests.exceptions.Timeout:
        return False, "Validation timeout"
    except Exception as e:
        return False, f"Validation error: {str(e)}"


# ─── CLI ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Flat RAG baseline: NL constraint → UPPAAL query"
    )
    parser.add_argument("constraint", type=str, help="Natural-language constraint")
    parser.add_argument("--k", type=int, default=10, help="Top-k chunks to retrieve")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY")

    client = OpenAI(api_key=api_key)
    rag = FlatRAG(INDEX_DIR)

    print("\n" + "=" * 70)
    print("FLAT RAG PIPELINE  (baseline comparison)")
    print("=" * 70)
    print(f"Constraint : {args.constraint}")
    print(f"Top-k      : {args.k}")

    cls = classify_constraint(client, args.constraint)

    # Build query text: combine constraint + keywords for richer embedding
    query_text = f"{args.constraint} {' '.join(cls.keywords)} {cls.pattern}"
    ctx = rag.retrieve(client, query_text, k=args.k)

    out = generate_uppaal(client, args.constraint, cls, ctx)

    ok, err = validate_antlr(out["query"])
    out["antlr_ok"] = ok
    out["antlr_error"] = err

    result_data = {
        "approach": "flat_rag",
        "input": args.constraint,
        "classification": {
            "pattern": cls.pattern,
            "keywords": cls.keywords,
            "cues": cls.cues,
            "trigger_text": cls.trigger_text,
            "response_text": cls.response_text,
        },
        "retrieval": {
            "k": args.k,
            "top_chunks": [
                {"id": c["id"], "type": c["type"], "score": round(c["_score"], 4)}
                for c in ctx.chunks
            ],
        },
        "result": out,
    }

    print("\n" + "=" * 70)
    print("RESULT")
    print("=" * 70)
    print(json.dumps(result_data, indent=2))


if __name__ == "__main__":
    main()
