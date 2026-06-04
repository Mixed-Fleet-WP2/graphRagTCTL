#!/usr/bin/env python3
"""
Benchmark the flat RAG baseline pipeline across the dataset and multiple LLMs.

Mirrors benchmark_pipeline.py exactly — same models, same dataset, same output
schema — but replaces Neo4j graph retrieval with flat cosine-similarity RAG.

Run:
    python benchmark_rag.py --folder ../dataset --start 0 --end 10

Env vars required:
    OPENAI_API_KEY
    ANTHROPIC_API_KEY      (optional — for Anthropic models)
    GOOGLE_API_KEY         (optional — for Gemini models)
    XAI_API_KEY            (optional — for Grok models)
    DEEPSEEK_API_KEY       (optional — for DeepSeek models)

Prerequisites:
    python build_index.py   ← must be run once before benchmarking
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv()

# ─── Dependencies (same as benchmark_pipeline.py) ────────────────────────────
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None

# ─── Paths ────────────────────────────────────────────────────────────────────
_DIR = Path(__file__).parent
INDEX_DIR = _DIR / "rag_index"
EMBEDDING_MODEL = "text-embedding-3-small"

# ─── Constants (identical to benchmark_pipeline.py) ──────────────────────────
MAX_RETRIES = 3
RETRY_DELAY = 2

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

MODELS_NO_TEMP = [
    "gpt-5", "gpt-5.1", "gpt-5-mini", "gpt-5-nano",
    "o1", "o1-mini",
    "deepseek-reasoner",
    "claude-sonnet-4-5", "claude-haiku-4-5",
]

OPENAI_RESPONSES_MODELS = [
    "gpt-5", "gpt-5.1", "gpt-5-mini", "gpt-5-nano",
]


# ─── Data classes ─────────────────────────────────────────────────────────────
from dataclasses import dataclass

@dataclass
class Classification:
    pattern: str
    keywords: List[str]
    cues: List[str]
    trigger_text: Optional[str] = None
    response_text: Optional[str] = None

@dataclass
class RetrievedContext:
    chunks: List[Dict[str, Any]]


# ─── Flat RAG retriever ───────────────────────────────────────────────────────
class FlatRAG:
    def __init__(self, index_dir: Path) -> None:
        emb_path = index_dir / "embeddings.npy"
        meta_path = index_dir / "metadata.json"
        if not emb_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"RAG index not found in {index_dir}. Run build_index.py first."
            )
        self.embeddings = np.load(str(emb_path)).astype(np.float32)
        with open(meta_path) as f:
            self.metadata: List[Dict[str, Any]] = json.load(f)
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings_norm = self.embeddings / (norms + 1e-10)

    def retrieve(self, client: Any, query_text: str, k: int = 10) -> RetrievedContext:
        resp = client.embeddings.create(model=EMBEDDING_MODEL, input=[query_text])
        q_vec = np.array(resp.data[0].embedding, dtype=np.float32)
        q_norm = q_vec / (np.linalg.norm(q_vec) + 1e-10)
        scores = self.embeddings_norm @ q_norm
        top_idx = np.argsort(scores)[::-1][:k]
        chunks = []
        for idx in top_idx:
            c = dict(self.metadata[idx])
            c["_score"] = float(scores[idx])
            chunks.append(c)
        return RetrievedContext(chunks=chunks)


# ─── Multi-provider LLM client (mirrors benchmark_pipeline.py) ───────────────
class LLMClient:
    def __init__(self, provider: str, model: str):
        self.provider = provider.lower()
        self.model = model.lower()

        if self.provider == "openai":
            self.client = OpenAI()
        elif self.provider == "anthropic":
            self.client = Anthropic()
        elif self.provider == "google":
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("Missing GOOGLE_API_KEY")
            genai.configure(api_key=api_key)
            self.client = None
        elif self.provider == "xai":
            self.client = None
            self.api_key = os.getenv("XAI_API_KEY")
        elif self.provider == "deepseek":
            self.client = None
            self.api_key = os.getenv("DEEPSEEK_API_KEY")
        elif self.provider == "ollama":
            self.client = None
        else:
            raise ValueError(f"Unknown provider: {provider}")

    # Returns the OpenAI client for embedding calls only
    @property
    def openai_client(self) -> Any:
        if self.provider == "openai":
            return self.client
        # For non-OpenAI models, embeddings still use OpenAI
        return OpenAI()

    def call(self, prompt: str, json_schema: Optional[Dict] = None) -> Tuple[str, float, int]:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                return self._single_call(prompt, json_schema)
            except Exception as e:
                print(f"[ERROR] {self.provider}/{self.model} attempt {attempt}: {e}")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)
                else:
                    return "", 0.0, 0

    def _single_call(self, prompt: str, json_schema: Optional[Dict] = None) -> Tuple[str, float, int]:
        start = time.time()
        allow_temp = not any(m in self.model for m in MODELS_NO_TEMP)
        use_responses = any(m in self.model for m in OPENAI_RESPONSES_MODELS)

        if self.provider == "openai":
            if use_responses and json_schema:
                resp = self.client.responses.create(
                    model=self.model,
                    input=prompt,
                    text={"format": {
                        "type": "json_schema",
                        "name": json_schema["name"],
                        "schema": json_schema["schema"],
                        "strict": json_schema.get("strict", True),
                    }},
                )
                return (getattr(resp, "output_text", "") or "").strip(), time.time() - start, 0
            elif use_responses:
                resp = self.client.responses.create(model=self.model, input=prompt)
                return (getattr(resp, "output_text", "") or "").strip(), time.time() - start, 0
            kwargs = {"model": self.model, "messages": [{"role": "user", "content": prompt}]}
            if allow_temp:
                kwargs["temperature"] = 0
            if json_schema:
                kwargs["response_format"] = {"type": "json_object"}
            resp = self.client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content.strip(), time.time() - start, resp.usage.total_tokens

        elif self.provider == "anthropic":
            kwargs = {"model": self.model, "max_tokens": 1024,
                      "messages": [{"role": "user", "content": prompt}]}
            if allow_temp:
                kwargs["temperature"] = 0
            resp = self.client.messages.create(**kwargs)
            text = resp.content[0].text if resp.content else ""
            return text.strip(), time.time() - start, 0

        elif self.provider == "google":
            model_obj = genai.GenerativeModel(self.model)
            resp = model_obj.generate_content(prompt)
            return resp.text.strip(), time.time() - start, 0

        elif self.provider in ("xai", "deepseek"):
            base_url = "https://api.x.ai/v1" if self.provider == "xai" else "https://api.deepseek.com/v1"
            resp = requests.post(
                f"{base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json={"model": self.model, "messages": [{"role": "user", "content": prompt}],
                      "temperature": 0},
                timeout=60,
            )
            resp.raise_for_status()
            r = resp.json()
            return r["choices"][0]["message"]["content"].strip(), time.time() - start, 0

        elif self.provider == "ollama":
            resp = requests.post(
                "http://localhost:11434/api/chat",
                json={"model": self.model, "messages": [{"role": "user", "content": prompt}],
                      "stream": False},
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"].strip(), time.time() - start, 0

        raise ValueError(f"Provider not implemented: {self.provider}")


# ─── JSON extraction helper ───────────────────────────────────────────────────
def extract_json(text: str) -> Optional[Dict]:
    if not text:
        return None
    # Try direct parse
    try:
        return json.loads(text)
    except Exception:
        pass
    # Try code block
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except Exception:
            pass
    # Try first { ... }
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    return None


# ─── ANTLR validation ─────────────────────────────────────────────────────────
def validate_antlr(query: str) -> Tuple[bool, Optional[str]]:
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
            return True, None
        return False, "; ".join(errors) if errors else "Validation failed"
    except requests.exceptions.ConnectionError:
        return False, "ANTLR parser service not reachable"
    except Exception as e:
        return False, f"Validation error: {str(e)}"


# ─── Step 1: Classify ─────────────────────────────────────────────────────────
def classify_constraint(
    llm: LLMClient,
    constraint: str,
) -> Tuple[Optional[Classification], Dict]:
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

    prompt = f"""Classify this natural-language constraint into ONE pattern.

Patterns: {PATTERN_ENUM}

Return JSON with: pattern, keywords, cues, trigger_text, response_text

Constraint: {constraint}"""

    text, elapsed, tokens = llm.call(prompt, json_schema=schema)
    data = extract_json(text)
    if not data:
        return None, {"error": "Failed to parse classification", "timing": elapsed}

    try:
        cls = Classification(
            pattern=data["pattern"],
            keywords=data.get("keywords", []),
            cues=data.get("cues", []),
            trigger_text=data.get("trigger_text"),
            response_text=data.get("response_text"),
        )
        return cls, {"pattern": cls.pattern, "timing": elapsed, "tokens": tokens}
    except Exception as e:
        return None, {"error": str(e), "timing": elapsed}


# ─── Step 2: Flat RAG retrieve ────────────────────────────────────────────────
def retrieve_rag(
    rag: FlatRAG,
    embed_client: Any,
    constraint: str,
    cls: Classification,
    k: int = 10,
) -> Tuple[RetrievedContext, Dict]:
    query_text = f"{constraint} {' '.join(cls.keywords)} {cls.pattern}"
    ctx = rag.retrieve(embed_client, query_text, k=k)
    return ctx, {
        "k": k,
        "top_scores": [round(c["_score"], 4) for c in ctx.chunks],
        "types": [c["type"] for c in ctx.chunks],
    }


# ─── Step 3: Generate UPPAAL ──────────────────────────────────────────────────
def generate_uppaal(
    llm: LLMClient,
    constraint: str,
    cls: Classification,
    ctx: RetrievedContext,
    robot_names: Optional[List[str]] = None,
) -> Tuple[Optional[Dict], Dict]:
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

    pattern_chunks = [c for c in ctx.chunks if c["type"] == "pattern"]
    operator_chunks = [c for c in ctx.chunks if c["type"] == "operator"]
    example_chunks = [c for c in ctx.chunks if c["type"] == "example"]

    pat_txt = ""
    if pattern_chunks:
        p = pattern_chunks[0]
        pat_txt = (f"Pattern: {p.get('name')}\n"
                   f"Template: {p.get('template', 'N/A')}\n"
                   f"When to use: {p.get('when_to_use', 'N/A')}")
    else:
        pat_txt = f"Pattern (from classification): {cls.pattern}"

    ops_txt = "\n".join(
        f"- {o.get('name')}: {o.get('when_to_use', '')} | avoid: {o.get('when_not_to_use', '')}"
        for o in operator_chunks
    ) or "- (no operators retrieved)"

    ex_txt = "\n".join(
        f"- NL: {e.get('nl')} | UPPAAL: {e.get('uppaal')}"
        for e in example_chunks
    ) or "- (no examples retrieved)"

    robot_hint = (
        f"\nRobot names (use EXACTLY as provided): {robot_names}" if robot_names else ""
    )

    prompt = f"""Translate ONE natural-language constraint into ONE UPPAAL query.

Classifier output:
- pattern={cls.pattern}, cues={cls.cues}, keywords={cls.keywords}
- trigger_text={cls.trigger_text}, response_text={cls.response_text}
{robot_hint}

Retrieved pattern context (flat RAG):
{pat_txt}

Retrieved operators (flat RAG):
{ops_txt}

Retrieved similar examples (flat RAG):
{ex_txt}

Constraint: {constraint}

Rules:
- Follow the pattern template shape.
- Use "imply" for SAME-STATE immediate response (typically with A[]).
- Use "-->" for eventual triggered behavior.
- Use A<> for "must eventually happen" global liveness.
- Use E<> for "possible/can happen".
- Use A[] not(...) for "never/must not".
- Use robot names EXACTLY as given.

Return JSON with: query, explanation, confidence, pattern_used, operator_used"""

    text, elapsed, tokens = llm.call(prompt, json_schema=out_schema)
    raw = extract_json(text)
    if not raw:
        return None, {"error": "Failed to parse generation", "timing": elapsed, "raw_text": text}

    # Normalize field names (handle model variations)
    result = {
        "query": raw.get("query") or raw.get("uppaal_query") or raw.get("translated_query") or "",
        "explanation": raw.get("explanation") or raw.get("reasoning") or "",
        "confidence": raw.get("confidence", 0.0),
        "pattern_used": raw.get("pattern_used") or raw.get("pattern") or cls.pattern,
        "operator_used": raw.get("operator_used", []),
    }
    return result, {"timing": elapsed, "tokens": tokens, "raw_text": text}


# ─── Full pipeline for one constraint ─────────────────────────────────────────
def run_pipeline(
    llm: LLMClient,
    rag: FlatRAG,
    embed_client: Any,
    constraint: str,
    robot_names: Optional[List[str]] = None,
    k: int = 10,
) -> Dict[str, Any]:
    start = time.time()
    result: Dict[str, Any] = {
        "approach": "flat_rag",
        "provider": llm.provider,
        "model": llm.model,
        "constraint": constraint,
        "classification": None,
        "retrieval": None,
        "generation": None,
        "validation": None,
        "total_time": 0,
        "success": False,
        "error": None,
    }
    try:
        # Step 1: Classify
        cls, cls_meta = classify_constraint(llm, constraint)
        result["classification"] = cls_meta
        if not cls:
            result["error"] = "Classification failed"
            result["total_time"] = time.time() - start
            return result

        # Step 2: Retrieve (flat RAG — key difference vs Graph RAG)
        ctx, ret_meta = retrieve_rag(rag, embed_client, constraint, cls, k=k)
        result["retrieval"] = ret_meta

        # Step 3: Generate
        uppaal_result, gen_meta = generate_uppaal(llm, constraint, cls, ctx, robot_names)
        result["generation"] = gen_meta
        if not uppaal_result:
            result["error"] = "UPPAAL generation failed"
            result["total_time"] = time.time() - start
            return result

        # Step 4: Validate
        query = uppaal_result.get("query", "")
        valid, err = validate_antlr(query)
        result["validation"] = {"valid": valid, "error": err, "query": query}
        result["generation"]["uppaal_result"] = uppaal_result
        result["success"] = True

    except Exception as e:
        result["error"] = str(e)

    result["total_time"] = time.time() - start
    return result


# ─── Benchmark a folder ───────────────────────────────────────────────────────
def benchmark_folder(
    folder: str,
    models: List[Dict[str, str]],
    rag: FlatRAG,
    embed_client: Any,
    start: int = 0,
    end: Optional[int] = None,
    output_dir: str = "rag_benchmarks",
    k: int = 10,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    all_files = sorted([f for f in os.listdir(folder) if f.endswith(".json")])
    files = all_files[start:end]

    print(f"\n{'='*80}")
    print("FLAT RAG BENCHMARK")
    print(f"{'='*80}")
    print(f"Scenarios : {start} → {start + len(files)}")
    print(f"Models    : {len(models)}")
    print(f"k         : {k}")
    print(f"Output    : {output_dir}")
    print(f"{'='*80}\n")

    for file in files:
        scenario_path = os.path.join(folder, file)
        with open(scenario_path) as f:
            scenario = json.load(f)

        scenario_id = scenario.get("id", os.path.splitext(file)[0])
        out_path = os.path.join(output_dir, f"{scenario_id}_rag.json")

        print(f"\n{'='*80}")
        print(f"Scenario: {file} ({scenario_id})")
        print(f"{'='*80}")

        if os.path.exists(out_path):
            with open(out_path) as f:
                scenario_result = json.load(f)
        else:
            scenario_result = {
                "id": scenario_id,
                "domain": scenario.get("domain", ""),
                "prompt": scenario.get("prompt", ""),
                "robots": scenario.get("robots", []),
                "constraints": scenario.get("constraints", []),
                "results": [],
            }

        constraints = scenario.get("constraints", [])
        robot_names = scenario.get("robots", [])

        for cfg in models:
            model_key = f"{cfg['provider']}/{cfg['model']}"
            print(f"\n  Model: {model_key}")

            model_entry = next(
                (e for e in scenario_result["results"]
                 if e.get("provider") == cfg["provider"].lower()
                 and e.get("model") == cfg["model"].lower()),
                None,
            )
            if model_entry is None:
                model_entry = {
                    "provider": cfg["provider"].lower(),
                    "model": cfg["model"].lower(),
                    "results": [],
                }
                scenario_result["results"].append(model_entry)

            for idx, constraint in enumerate(constraints):
                already_done = any(
                    r.get("constraint_index") == idx for r in model_entry["results"]
                )
                if already_done:
                    print(f"    [{idx}] SKIP")
                    continue

                print(f"    [{idx}] RUN: {constraint[:65]}...")
                try:
                    llm = LLMClient(cfg["provider"], cfg["model"])
                    res = run_pipeline(llm, rag, embed_client, constraint,
                                       robot_names=robot_names, k=k)
                    res["constraint_index"] = idx
                    res.pop("provider", None)
                    res.pop("model", None)
                    model_entry["results"].append(res)
                    with open(out_path, "w") as f:
                        json.dump(scenario_result, f, indent=2)
                    status = "✓" if res["success"] else "✗"
                    print(f"    {status} ({res['total_time']:.2f}s)")
                except Exception as e:
                    print(f"    ✗ ERROR: {e}")
                    model_entry["results"].append({
                        "constraint_index": idx,
                        "constraint": constraint,
                        "error": str(e),
                        "success": False,
                        "approach": "flat_rag",
                    })
                    with open(out_path, "w") as f:
                        json.dump(scenario_result, f, indent=2)

    print(f"\n{'='*80}")
    print("BENCHMARK COMPLETE ✓")
    print(f"{'='*80}")


# ─── CLI ──────────────────────────────────────────────────────────────────────
MODELS = [
    {"provider": "openai",    "model": "gpt-5.1"},
    {"provider": "openai",    "model": "gpt-5-mini"},
    {"provider": "openai",    "model": "gpt-5-nano"},
    {"provider": "openai",    "model": "gpt-4o"},
    {"provider": "anthropic", "model": "claude-sonnet-4-5-20250929"},
    {"provider": "anthropic", "model": "claude-haiku-4-5-20251001"},
    {"provider": "xai",       "model": "grok-code-fast-1"},
    {"provider": "xai",       "model": "grok-3-mini"},
    {"provider": "google",    "model": "gemini-2.0-flash"},
    {"provider": "google",    "model": "gemini-2.5-pro"},
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark flat RAG pipeline across LLMs (comparison vs Graph RAG)"
    )
    parser.add_argument("--folder", type=str, required=True, help="Dataset folder")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--output", type=str, default="rag_benchmarks")
    parser.add_argument("--k", type=int, default=10, help="Top-k chunks to retrieve")
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: Missing OPENAI_API_KEY")
        sys.exit(1)

    # Embeddings always use OpenAI regardless of generation model
    embed_client = OpenAI(api_key=api_key)

    rag = FlatRAG(INDEX_DIR)

    benchmark_folder(
        folder=args.folder,
        models=MODELS,
        rag=rag,
        embed_client=embed_client,
        start=args.start,
        end=args.end,
        output_dir=args.output,
        k=args.k,
    )
