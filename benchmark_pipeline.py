#!/usr/bin/env python3
"""
Benchmark the complete pipeline (classification → GraphRAG → UPPAAL generation) across all LLMs.

Run from terminal:
  python benchmark_pipeline.py --folder dataset --start 0 --end 10

Env vars required:
  OPENAI_API_KEY
  ANTHROPIC_API_KEY
  GOOGLE_API_KEY
  XAI_API_KEY
  DEEPSEEK_API_KEY
  NEO4J_URI
  NEO4J_USER
  NEO4J_PASSWORD
"""

import os
import re
import json
import time
import argparse
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

# Import Neo4j and requests
from neo4j import GraphDatabase
import requests
from dotenv import load_dotenv
load_dotenv()

# ==========================================
#  OPTIONAL RETRY SETTINGS
# ==========================================
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

# -------------------------------
# 1. MULTI-PROVIDER CLIENTS
# -------------------------------
try:
    from openai import OpenAI
except:
    OpenAI = None

try:
    from anthropic import Anthropic
except:
    Anthropic = None

try:
    import google.generativeai as genai
except:
    genai = None

# ================================
#  MODELS THAT DO NOT SUPPORT TEMPERATURE
# ================================
MODELS_NO_TEMP = [
    "gpt-5", "gpt-5.1", "gpt-5-mini", "gpt-5-nano",
    "o1", "o1-mini",
    "deepseek-reasoner",
    "claude-sonnet-4-5", "claude-haiku-4-5"
]

# ================================
#  OPENAI RESPONSES API MODELS
# ================================
OPENAI_RESPONSES_MODELS = [
    "gpt-5", "gpt-5.1", "gpt-5-mini", "gpt-5-nano"
]

# ==========================================
#  PATTERN ENUM (MUST MATCH PIPELINE)
# ==========================================
PATTERN_ENUM = [
    "safety_immediate_response",
    "sequential_workflow",
    "eventual_completion",
    "existential_reachability",
    "universal_reachability",
    "forbidden_state",
    "conditional_response",
    "time_bounded_constraint"
]

# ==========================================
#  DATACLASSES (FROM PIPELINE)
# ==========================================
@dataclass
class Classification:
    pattern: str
    keywords: List[str]
    cues: List[str]
    trigger_text: Optional[str] = None
    response_text: Optional[str] = None


@dataclass
class PatternNode:
    name: str
    description: str
    template: str
    when_to_use: str
    decision_rules: List[str]


@dataclass
class GraphContext:
    pattern: PatternNode
    operators: List[Dict[str, Any]]
    examples: List[Dict[str, Any]]


# ==========================================
#  NEO4J GRAPHRAG CLIENT (FROM PIPELINE)
# ==========================================
class GraphRAG:
    def __init__(self, uri: str, user: str, password: str) -> None:
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        self.driver.close()

    def get_pattern(self, pattern_name: str) -> PatternNode:
        cypher = """
        MATCH (p:Pattern {name:$pattern})
        RETURN p.name AS name,
               p.description AS description,
               p.template AS template,
               p.when_to_use AS when_to_use,
               coalesce(p.decision_rules, []) AS decision_rules
        """
        with self.driver.session() as session:
            rec = session.run(cypher, pattern=pattern_name).single()
            if not rec:
                raise ValueError(f"Pattern not found in Neo4j: {pattern_name}")
            return PatternNode(
                name=rec["name"],
                description=rec.get("description") or "",
                template=rec.get("template") or "",
                when_to_use=rec.get("when_to_use") or "",
                decision_rules=list(rec.get("decision_rules") or []),
            )

    def get_operators_for_pattern(self, pattern_name: str) -> List[Dict[str, Any]]:
        cypher = """
        MATCH (p:Pattern {name:$pattern})-[:USES_OPERATOR]->(o:Operator)
        RETURN collect(DISTINCT {
          name: o.name,
          description: o.description,
          semantics: o.semantics,
          when_to_use: o.when_to_use,
          when_not_to_use: o.when_not_to_use,
          examples: coalesce(o.examples, [])
        }) AS ops
        """
        with self.driver.session() as session:
            rec = session.run(cypher, pattern=pattern_name).single()
            ops = (rec and rec.get("ops")) or []
        filtered_ops = [o for o in ops if o.get("name")]
        return filtered_ops

    def get_examples(self, pattern_name: str, keywords: List[str], k: int = 5) -> List[Dict[str, Any]]:
        cypher = """
        MATCH (e:Example)-[:INSTANCE_OF]->(p:Pattern {name:$pattern})
        WITH e,
             size([kw IN coalesce(e.keywords, []) WHERE kw IN $keywords]) AS overlap
        ORDER BY overlap DESC
        RETURN collect({
          id: e.id,
          nl: e.nl,
          uppaal: e.uppaal,
          overlap: overlap
        })[..$k] AS examples
        """
        with self.driver.session() as session:
            rec = session.run(cypher, pattern=pattern_name, keywords=keywords, k=k).single()
            examples = (rec and rec.get("examples")) or []
        return examples

    def retrieve(self, cls: Classification, k_examples: int = 5) -> GraphContext:
        p = self.get_pattern(cls.pattern)
        ops = self.get_operators_for_pattern(cls.pattern)
        exs = self.get_examples(cls.pattern, cls.keywords, k=k_examples)
        return GraphContext(pattern=p, operators=ops, examples=exs)


# ==========================================
#  ANTLR VALIDATION
# ==========================================
def validate_antlr(query: str) -> Tuple[bool, Optional[str]]:
    """Validate UPPAAL query using external ANTLR parser service."""
    if not query.strip():
        return False, "Empty query"
    
    try:
        response = requests.post(
            "http://127.0.0.1:5002/check-property",
            json={"property": query},
            timeout=5
        )
        response.raise_for_status()
        result = response.json()
        
        valid = result.get("valid", False)
        errors = result.get("errors", [])
        
        if valid:
            return True, None
        else:
            error_msg = "; ".join(errors) if errors else "Validation failed"
            return False, error_msg
            
    except requests.exceptions.ConnectionError:
        return False, "ANTLR parser service not reachable"
    except requests.exceptions.Timeout:
        return False, "Validation timeout"
    except Exception as e:
        return False, f"Validation error: {str(e)}"


# ==========================================
#  MULTI-PROVIDER LLM CLIENT
# ==========================================
class LLMClient:
    def __init__(self, provider: str, model: str):
        self.provider = provider.lower()
        self.model = model.lower()

        if self.provider == "openai":
            if OpenAI is None:
                raise ImportError("openai package not installed")
            self.client = OpenAI()

        elif self.provider == "anthropic":
            if Anthropic is None:
                raise ImportError("anthropic package not installed")
            self.client = Anthropic()

        elif self.provider == "google":
            if genai is None:
                raise ImportError("google-generativeai package not installed")
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("Missing GOOGLE_API_KEY environment variable")
            genai.configure(api_key=api_key)

        elif self.provider == "xai":
            self.client = None
            self.api_key = os.getenv("XAI_API_KEY")
            if not self.api_key:
                raise ValueError("Missing XAI_API_KEY environment variable")

        elif self.provider == "deepseek":
            self.client = None
            self.api_key = os.getenv("DEEPSEEK_API_KEY")
            if not self.api_key:
                raise ValueError("Missing DEEPSEEK_API_KEY environment variable")

        elif self.provider == "ollama":
            self.client = None

        else:
            raise ValueError(f"Unknown provider: {provider}")

    def call(self, prompt: str, json_schema: Optional[Dict] = None) -> Tuple[str, float, int]:
        """Call LLM with optional JSON schema for structured outputs."""
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                return self._single_call(prompt, json_schema)
            except Exception as e:
                print(f"[ERROR] {self.provider}/{self.model} failed (attempt {attempt}/{MAX_RETRIES}): {e}")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)
                else:
                    return "", 0.0, 0

    def _single_call(self, prompt: str, json_schema: Optional[Dict] = None) -> Tuple[str, float, int]:
        start = time.time()

        allow_temp = not any(m in self.model for m in MODELS_NO_TEMP)
        use_responses = any(m in self.model for m in OPENAI_RESPONSES_MODELS)

        # ------------------------------
        # OPENAI
        # ------------------------------
        if self.provider == "openai":
            if use_responses and json_schema:
                # Use Responses API with structured outputs
                resp = self.client.responses.create(
                    model=self.model,
                    input=prompt,
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": json_schema["name"],
                            "schema": json_schema["schema"],
                            "strict": json_schema.get("strict", True)
                        }
                    }
                )
                text = getattr(resp, "output_text", "") or ""
                return text.strip(), time.time() - start, 0
            
            elif use_responses:
                resp = self.client.responses.create(
                    model=self.model,
                    input=prompt
                )
                text = getattr(resp, "output_text", "") or ""
                return text.strip(), time.time() - start, 0

            # Chat completions API
            kwargs = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
            }
            if allow_temp:
                kwargs["temperature"] = 0
            
            if json_schema and not use_responses:
                kwargs["response_format"] = {"type": "json_object"}

            resp = self.client.chat.completions.create(**kwargs)
            text = resp.choices[0].message.content or ""
            tokens = getattr(resp.usage, "total_tokens", 0) or 0
            return text.strip(), time.time() - start, tokens

        # ------------------------------
        # ANTHROPIC
        # ------------------------------
        if self.provider == "anthropic":
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )
            if not resp.content:
                return "", time.time() - start, 0
            block = resp.content[0]
            text = block.text if hasattr(block, "text") else str(block)
            return (text or "").strip(), time.time() - start, 0

        # ------------------------------
        # GOOGLE GEMINI
        # ------------------------------
        if self.provider == "google":
            model = genai.GenerativeModel(self.model)
            resp = model.generate_content(prompt)
            text = getattr(resp, "text", "") or ""
            return text.strip(), time.time() - start, 0

        # ------------------------------
        # XAI GROK
        # ------------------------------
        if self.provider == "xai":
            url = "https://api.x.ai/v1/chat/completions"
            headers = {"Authorization": f"Bearer {self.api_key}"}
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}]
            }
            if json_schema:
                payload["response_format"] = {"type": "json_object"}
                
            raw = requests.post(url, json=payload, headers=headers).json()

            try:
                text = raw["choices"][0]["message"]["content"]
                return (text or "").strip(), time.time() - start, 0
            except:
                if "message" in raw and "content" in raw["message"]:
                    return str(raw["message"]["content"]).strip(), time.time() - start, 0
                print("[WARNING] Grok returned unexpected format:", raw)
                return "", time.time() - start, 0

        # ------------------------------
        # DEEPSEEK
        # ------------------------------
        if self.provider == "deepseek":
            url = "https://api.deepseek.com/v1/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}]
            }
            if json_schema:
                payload["response_format"] = {"type": "json_object"}
                
            raw = requests.post(url, headers=headers, json=payload).json()

            try:
                text = raw["choices"][0]["message"]["content"]
                return (text or "").strip(), time.time() - start, 0
            except:
                print("[WARNING] DeepSeek returned unexpected response:", raw)
                return "", time.time() - start, 0

        # ------------------------------
        # OLLAMA
        # ------------------------------
        if self.provider == "ollama":
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }
            if json_schema:
                payload["format"] = "json"
                
            raw = requests.post(
                "http://localhost:11434/api/generate",
                json=payload
            ).json()
            text = raw.get("response", "") or ""
            return text.strip(), time.time() - start, 0

        raise ValueError("Unsupported provider")


# ==========================================
#  EXTRACT JSON FROM LLM RESPONSE
# ==========================================
def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Extract JSON object from LLM response that might have extra text."""
    if not text:
        return None

    text = text.strip()

    # Try direct parse
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except:
        pass

    # Try to find JSON object
    m = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, flags=re.DOTALL)
    if m:
        try:
            parsed = json.loads(m.group(0))
            if isinstance(parsed, dict):
                return parsed
        except:
            pass

    return None


# ==========================================
#  PIPELINE STEPS
# ==========================================
def classify_constraint(llm: LLMClient, constraint: str) -> Tuple[Optional[Classification], Dict[str, Any]]:
    """Step 1: Classify constraint into pattern."""
    
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

Output ONLY valid JSON matching this schema.

Constraint:
{constraint}
""".strip()

    text, elapsed, tokens = llm.call(prompt, json_schema=schema)
    
    metadata = {
        "step": "classification",
        "timing": elapsed,
        "tokens": tokens,
        "raw_text": text
    }
    
    data = extract_json(text)
    if not data:
        metadata["error"] = "Failed to parse JSON"
        return None, metadata
    
    try:
        cls = Classification(
            pattern=data["pattern"],
            keywords=data["keywords"],
            cues=data.get("cues", []),
            trigger_text=data.get("trigger_text"),
            response_text=data.get("response_text"),
        )
        return cls, metadata
    except Exception as e:
        metadata["error"] = f"Invalid classification structure: {e}"
        return None, metadata


def generate_uppaal(llm: LLMClient, constraint: str, cls: Classification, ctx: GraphContext, robot_names: Optional[List[str]] = None) -> Tuple[Optional[Dict], Dict[str, Any]]:
    """Step 3: Generate UPPAAL query."""
    
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

    pattern_txt = f"""
Pattern: {ctx.pattern.name}
Description: {ctx.pattern.description}
Template: {ctx.pattern.template}
When to use: {ctx.pattern.when_to_use}
Decision rules: {ctx.pattern.decision_rules}
""".strip()

    ops_txt = "\n".join(
        f"- {o['name']}: when_to_use={o.get('when_to_use')} | when_not_to_use={o.get('when_not_to_use')}"
        for o in ctx.operators
    ) or "- (no operators linked; rely on pattern template)"

    ex_txt = "\n".join(
        f"- ex[{e.get('id')}]: NL={e.get('nl')} | UPPAAL={e.get('uppaal')}"
        for e in ctx.examples
    ) or "- (no examples)"

    robot_context = ""
    if robot_names:
        robot_context = f"""

Available robot/agent names in this scenario:
{', '.join(robot_names)}

IMPORTANT: Use these EXACT robot names in your query. Do NOT invent names like Robot1, Robot2, AgentA, AgentB.
"""

    instructions = f"""
Translate ONE natural-language constraint into ONE UPPAAL query.

Classifier output:
- pattern={cls.pattern}
- cues={cls.cues}
- keywords={cls.keywords}
- trigger_text={cls.trigger_text}
- response_text={cls.response_text}{robot_context}

GraphRAG pattern context:
{pattern_txt}

Operators for this pattern:
{ops_txt}

Similar examples:
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
- When robot names are provided, use them EXACTLY as given. Do NOT use generic names.
- For multiple unnamed entities, use suffix A/B (e.g., RobotA, RobotB) only if no names provided.

CRITICAL: Return ONLY valid JSON with these EXACT fields:
{{
  "query": "the UPPAAL query string",
  "explanation": "why this query matches the constraint",
  "confidence": 0.85,
  "pattern_used": "{cls.pattern}",
  "operator_used": ["-->"]
}}

Do NOT use field names like "uppaal_query", "reasoning", or any other variations.
Use EXACTLY: query, explanation, confidence, pattern_used, operator_used
""".strip()

    text, elapsed, tokens = llm.call(instructions, json_schema=out_schema)
    
    metadata = {
        "step": "generation",
        "timing": elapsed,
        "tokens": tokens,
        "raw_text": text
    }
    
    result = extract_json(text)
    if not result:
        metadata["error"] = "Failed to parse JSON"
        return None, metadata
    
    # Normalize field names (handle models that use different field names)
    normalized = {}
    
    # Map query field (handle variations)
    if "query" in result:
        normalized["query"] = result["query"]
    elif "uppaal_query" in result:
        normalized["query"] = result["uppaal_query"]
    elif "UPPAAL_query" in result:
        normalized["query"] = result["UPPAAL_query"]
    elif "translated_query" in result:
        normalized["query"] = result["translated_query"]
    elif "translation" in result:
        normalized["query"] = result["translation"]
    else:
        normalized["query"] = ""
    
    # Map explanation field
    if "explanation" in result:
        normalized["explanation"] = result["explanation"]
    elif "reasoning" in result:
        normalized["explanation"] = result["reasoning"]
    else:
        normalized["explanation"] = result.get("explanation", "")
    
    # Map other required fields with defaults
    normalized["confidence"] = result.get("confidence", 0.0)
    normalized["pattern_used"] = result.get("pattern_used", result.get("pattern", cls.pattern))
    normalized["operator_used"] = result.get("operator_used", [])
    
    return normalized, metadata


# ==========================================
#  RUN COMPLETE PIPELINE FOR ONE CONSTRAINT
# ==========================================
def run_pipeline(
    llm: LLMClient,
    kg: GraphRAG,
    constraint: str,
    robot_names: Optional[List[str]] = None,
    k_examples: int = 5
) -> Dict[str, Any]:
    """Run complete pipeline: classify → retrieve → generate → validate."""
    
    start_time = time.time()
    result = {
        "provider": llm.provider,
        "model": llm.model,
        "constraint": constraint,
        "classification": None,
        "graph_context": None,
        "generation": None,
        "validation": None,
        "total_time": 0,
        "success": False,
        "error": None
    }
    
    try:
        # Step 1: Classification
        cls, cls_meta = classify_constraint(llm, constraint)
        result["classification"] = cls_meta
        
        if not cls:
            result["error"] = "Classification failed"
            result["total_time"] = time.time() - start_time
            return result
        
        # Step 2: GraphRAG retrieval
        try:
            ctx = kg.retrieve(cls, k_examples=k_examples)
            result["graph_context"] = {
                "pattern": ctx.pattern.name,
                "operators_count": len(ctx.operators),
                "examples_count": len(ctx.examples)
            }
        except Exception as e:
            result["error"] = f"GraphRAG retrieval failed: {e}"
            result["total_time"] = time.time() - start_time
            return result
        
        # Step 3: UPPAAL generation
        uppaal_result, gen_meta = generate_uppaal(llm, constraint, cls, ctx, robot_names)
        result["generation"] = gen_meta
        
        if not uppaal_result:
            result["error"] = "UPPAAL generation failed"
            result["total_time"] = time.time() - start_time
            return result
        
        # Step 4: Validation
        query = uppaal_result.get("query", "")
        valid, error = validate_antlr(query)
        result["validation"] = {
            "valid": valid,
            "error": error,
            "query": query
        }
        
        result["generation"]["uppaal_result"] = uppaal_result
        result["success"] = True
        
    except Exception as e:
        result["error"] = str(e)
    
    result["total_time"] = time.time() - start_time
    return result


# ==========================================
#  BENCHMARK DATASET FOLDER
# ==========================================
def benchmark_folder(
    folder: str,
    models: List[Dict[str, str]],
    kg: GraphRAG,
    start: int = 0,
    end: Optional[int] = None,
    output_dir: str = "pipeline_benchmarks",
    k_examples: int = 5
):
    """Run pipeline benchmark across all scenarios and models."""
    
    os.makedirs(output_dir, exist_ok=True)

    all_files = sorted([f for f in os.listdir(folder) if f.endswith(".json")])
    files = all_files[start:end]

    print(f"\n{'='*80}")
    print(f"BENCHMARKING PIPELINE")
    print(f"{'='*80}")
    print(f"Scenarios: {start} → {start + len(files)}")
    print(f"Models: {len(models)}")
    print(f"Dataset folder: {folder}")
    print(f"Output folder: {output_dir}")
    print(f"{'='*80}\n")

    for file in files:
        scenario_path = os.path.join(folder, file)
        with open(scenario_path, "r") as f:
            scenario = json.load(f)

        scenario_id = scenario.get("id", os.path.splitext(file)[0])
        out_path = os.path.join(output_dir, f"{scenario_id}_pipeline.json")

        print(f"\n{'='*80}")
        print(f"Scenario: {file} ({scenario_id})")
        print(f"{'='*80}")

        # Load existing results if present
        if os.path.exists(out_path):
            with open(out_path, "r") as f:
                scenario_result = json.load(f)
        else:
            scenario_result = {
                "id": scenario_id,
                "domain": scenario.get("domain", ""),
                "prompt": scenario.get("prompt", ""),
                "robots": scenario.get("robots", []),
                "constraints": scenario.get("constraints", []),
                "results": []
            }

        constraints = scenario.get("constraints", [])
        
        # Process by model first (all constraints for one model, then next model)
        for cfg in models:
            model_key = f"{cfg['provider']}/{cfg['model']}"
            print(f"\n{'─'*80}")
            print(f"MODEL: {model_key}")
            print(f"{'─'*80}")
            
            # Find or create model entry in results
            model_entry = None
            for entry in scenario_result["results"]:
                if (entry.get("provider") == cfg["provider"].lower() and 
                    entry.get("model") == cfg["model"].lower()):
                    model_entry = entry
                    break
            
            if model_entry is None:
                model_entry = {
                    "provider": cfg["provider"].lower(),
                    "model": cfg["model"].lower(),
                    "results": []
                }
                scenario_result["results"].append(model_entry)
            
            for idx, constraint in enumerate(constraints):
                print(f"\n  Constraint {idx}: {constraint[:70]}...")
                
                # Check if this constraint already processed for this model
                existing = [
                    r for r in model_entry["results"]
                    if r.get("constraint_index") == idx
                ]
                
                if existing:
                    print(f"    → SKIP (already exists)")
                    continue

                print(f"    → RUN")
                
                try:
                    llm = LLMClient(cfg["provider"], cfg["model"])
                    robot_names = scenario.get("robots", [])
                    result = run_pipeline(llm, kg, constraint, robot_names=robot_names, k_examples=k_examples)
                    result["constraint_index"] = idx
                    
                    # Remove provider/model from individual result (already in parent)
                    result.pop("provider", None)
                    result.pop("model", None)
                    
                    model_entry["results"].append(result)
                    
                    # Save incrementally
                    with open(out_path, "w") as f:
                        json.dump(scenario_result, f, indent=2)
                    
                    status = "✓ SUCCESS" if result["success"] else "✗ FAILED"
                    print(f"    {status} (time: {result['total_time']:.2f}s)")
                    
                except Exception as e:
                    print(f"    ✗ ERROR: {e}")
                    error_result = {
                        "constraint_index": idx,
                        "constraint": constraint,
                        "error": str(e),
                        "success": False
                    }
                    model_entry["results"].append(error_result)
                    
                    with open(out_path, "w") as f:
                        json.dump(scenario_result, f, indent=2)

    print(f"\n{'='*80}")
    print("BENCHMARK COMPLETE ✓")
    print(f"{'='*80}\n")


# ==========================================
#  CLI
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark complete pipeline across all LLMs")
    parser.add_argument("--folder", type=str, required=True, help="Dataset folder of scenario JSON files")
    parser.add_argument("--start", type=int, default=0, help="Start index")
    parser.add_argument("--end", type=int, default=None, help="End index (exclusive)")
    parser.add_argument("--output", type=str, default="pipeline_benchmarks", help="Output directory")
    parser.add_argument("--k", type=int, default=3, help="Number of examples to retrieve")
    args = parser.parse_args()

    # All LLMs from multi_llm_constraint_classifier.py
    MODELS = [
        {"provider": "openai", "model": "gpt-5.1"},
        {"provider": "openai", "model": "gpt-5-mini"},
        {"provider": "openai", "model": "gpt-5-nano"},
        {"provider": "openai", "model": "gpt-4o"},
        {"provider": "anthropic", "model": "claude-sonnet-4-5-20250929"},
        {"provider": "anthropic", "model": "claude-haiku-4-5-20251001"},
        {"provider": "xai", "model": "grok-code-fast-1"},
        {"provider": "xai", "model": "grok-3-mini"},
        # {"provider": "deepseek", "model": "deepseek-reasoner"},
        {"provider": "google", "model": "gemini-2.0-flash"},
        {"provider": "google", "model": "gemini-2.5-pro"},
        # Local Ollama models:
        {"provider": "ollama", "model": "deepseek-r1:14b"},
    ]

    # Neo4j connection
    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_user = os.getenv("NEO4J_USER")
    neo4j_password = os.getenv("NEO4J_PASSWORD")

    if not all([neo4j_uri, neo4j_user, neo4j_password]):
        print("ERROR: Missing Neo4j environment variables (NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)")
        exit(1)

    kg = GraphRAG(neo4j_uri, neo4j_user, neo4j_password)

    try:
        benchmark_folder(
            folder=args.folder,
            models=MODELS,
            kg=kg,
            start=args.start,
            end=args.end,
            output_dir=args.output,
            k_examples=args.k
        )
    finally:
        kg.close()
