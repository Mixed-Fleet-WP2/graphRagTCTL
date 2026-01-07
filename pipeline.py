#!/usr/bin/env python3
"""
Run from terminal:

  python pipeline.py "Drone returns to base when battery < 10%"

Env vars required:
  OPENAI_API_KEY
  NEO4J_URI
  NEO4J_USER
  NEO4J_PASSWORD

Install:
  pip install openai neo4j
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests
from neo4j import GraphDatabase
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

# ANSI color codes
class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    GRAY = '\033[90m'

# Custom colored formatter
class ColoredFormatter(logging.Formatter):
    FORMATS = {
        logging.DEBUG: Colors.GRAY + '%(asctime)s - DEBUG - %(message)s' + Colors.RESET,
        logging.INFO: Colors.CYAN + '%(asctime)s - INFO - %(message)s' + Colors.RESET,
        logging.WARNING: Colors.YELLOW + '%(asctime)s - WARNING - %(message)s' + Colors.RESET,
        logging.ERROR: Colors.RED + '%(asctime)s - ERROR - %(message)s' + Colors.RESET,
        logging.CRITICAL: Colors.RED + Colors.BOLD + '%(asctime)s - CRITICAL - %(message)s' + Colors.RESET,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt='%Y-%m-%d %H:%M:%S')
        return formatter.format(record)

# Setup logging with color
handler = logging.StreamHandler(sys.stderr)
handler.setFormatter(ColoredFormatter())
logging.basicConfig(
    level=logging.INFO,
    handlers=[handler]
)
logger = logging.getLogger(__name__)

def log_step(step_num: int, title: str):
    """Log a step header with color"""
    logger.info(Colors.BOLD + Colors.MAGENTA + "=" * 80 + Colors.RESET)
    logger.info(Colors.BOLD + Colors.MAGENTA + f"STEP {step_num}: {title}" + Colors.RESET)
    logger.info(Colors.BOLD + Colors.MAGENTA + "=" * 80 + Colors.RESET)

def log_success(message: str):
    """Log a success message"""
    logger.info(Colors.GREEN + "✓ " + message + Colors.RESET)

def log_highlight(label: str, value: Any):
    """Log a key-value pair with highlighting"""
    logger.info(Colors.YELLOW + f"  {label}: " + Colors.WHITE + f"{value}" + Colors.RESET)

def colorize_json(obj: Any, indent: int = 0) -> str:
    """Recursively colorize JSON output"""
    spaces = "  " * indent
    
    if isinstance(obj, dict):
        lines = [Colors.WHITE + "{" + Colors.RESET]
        items = list(obj.items())
        for i, (key, value) in enumerate(items):
            comma = "," if i < len(items) - 1 else ""
            colored_key = f'{spaces}  {Colors.CYAN}"{key}"{Colors.RESET}: '
            if isinstance(value, (dict, list)):
                lines.append(colored_key)
                lines.append(colorize_json(value, indent + 1) + comma)
            elif isinstance(value, str):
                lines.append(colored_key + Colors.GREEN + f'"{value}"' + Colors.RESET + comma)
            elif isinstance(value, bool):
                lines.append(colored_key + Colors.MAGENTA + str(value).lower() + Colors.RESET + comma)
            elif isinstance(value, (int, float)):
                lines.append(colored_key + Colors.YELLOW + str(value) + Colors.RESET + comma)
            elif value is None:
                lines.append(colored_key + Colors.GRAY + "null" + Colors.RESET + comma)
            else:
                lines.append(colored_key + str(value) + comma)
        lines.append(spaces + Colors.WHITE + "}" + Colors.RESET)
        return "\n".join(lines)
    
    elif isinstance(obj, list):
        if not obj:
            return Colors.WHITE + "[]" + Colors.RESET
        lines = [Colors.WHITE + "[" + Colors.RESET]
        for i, item in enumerate(obj):
            comma = "," if i < len(obj) - 1 else ""
            if isinstance(item, (dict, list)):
                lines.append(spaces + "  " + colorize_json(item, indent + 1) + comma)
            elif isinstance(item, str):
                lines.append(spaces + "  " + Colors.GREEN + f'"{item}"' + Colors.RESET + comma)
            elif isinstance(item, bool):
                lines.append(spaces + "  " + Colors.MAGENTA + str(item).lower() + Colors.RESET + comma)
            elif isinstance(item, (int, float)):
                lines.append(spaces + "  " + Colors.YELLOW + str(item) + Colors.RESET + comma)
            elif item is None:
                lines.append(spaces + "  " + Colors.GRAY + "null" + Colors.RESET + comma)
            else:
                lines.append(spaces + "  " + str(item) + comma)
        lines.append(spaces + Colors.WHITE + "]" + Colors.RESET)
        return "\n".join(lines)
    
    elif isinstance(obj, str):
        return Colors.GREEN + f'"{obj}"' + Colors.RESET
    elif isinstance(obj, bool):
        return Colors.MAGENTA + str(obj).lower() + Colors.RESET
    elif isinstance(obj, (int, float)):
        return Colors.YELLOW + str(obj) + Colors.RESET
    elif obj is None:
        return Colors.GRAY + "null" + Colors.RESET
    else:
        return str(obj)

PATTERN_ENUM = [
    "safety_immediate_response",
    "sequential_workflow",
    "eventual_completion",
    "reachability",
    "forbidden_state",
    "conditional_response",
    "time_bounded_constraint"
]


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


# -----------------------------
# Neo4j GraphRAG
# -----------------------------

class GraphRAG:
    def __init__(self, uri: str, user: str, password: str) -> None:
        logger.info(f"Connecting to Neo4j at {Colors.BLUE}{uri}{Colors.RESET}...")
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        log_success("Neo4j connection established")

    def close(self) -> None:
        logger.info("Closing Neo4j connection")
        self.driver.close()

    def get_pattern(self, pattern_name: str) -> PatternNode:
        logger.info(f"Retrieving pattern: {Colors.YELLOW}{pattern_name}{Colors.RESET}")
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
                logger.error(f"Pattern not found in Neo4j: {pattern_name}")
                raise ValueError(f"Pattern not found in Neo4j: {pattern_name}")
            log_success(f"Pattern retrieved: {rec['name']} - {rec.get('description', '')[:50]}...")
            return PatternNode(
                name=rec["name"],
                description=rec.get("description") or "",
                template=rec.get("template") or "",
                when_to_use=rec.get("when_to_use") or "",
                decision_rules=list(rec.get("decision_rules") or []),
            )

    def get_operators_for_pattern(self, pattern_name: str) -> List[Dict[str, Any]]:
        """
        Requires (Pattern)-[:USES_OPERATOR]->(Operator)
        If you didn't create this relationship, return [] and the LLM will rely on pattern template.
        """
        logger.info(f"Retrieving operators for pattern: {Colors.YELLOW}{pattern_name}{Colors.RESET}")
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
        log_success(f"Found {Colors.BOLD}{len(filtered_ops)}{Colors.RESET}{Colors.GREEN} operators: {Colors.WHITE}{[o['name'] for o in filtered_ops]}{Colors.RESET}")
        return filtered_ops

    def get_examples(self, pattern_name: str, keywords: List[str], k: int = 5) -> List[Dict[str, Any]]:
        logger.info(f"Retrieving top-{Colors.BOLD}{k}{Colors.RESET}{Colors.CYAN} examples for pattern: {Colors.YELLOW}{pattern_name}{Colors.RESET}")
        log_highlight("Search keywords", keywords)
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
        log_success(f"Retrieved {Colors.BOLD}{len(examples)}{Colors.RESET}{Colors.GREEN} examples")
        for i, ex in enumerate(examples[:3], 1):  # Log first 3
            logger.info(f"  {Colors.BLUE}Example {i}{Colors.RESET} [{Colors.GRAY}{ex.get('id')}{Colors.RESET}]: overlap={Colors.YELLOW}{ex.get('overlap')}{Colors.RESET}, NL={ex.get('nl', '')[:60]}...")
        return examples

    def retrieve(self, cls: Classification, k_examples: int = 5) -> GraphContext:
        log_step(2, "RETRIEVING GRAPH CONTEXT")
        p = self.get_pattern(cls.pattern)
        ops = self.get_operators_for_pattern(cls.pattern)
        exs = self.get_examples(cls.pattern, cls.keywords, k=k_examples)
        log_success(f"Graph context retrieved: {Colors.BOLD}{len(ops)}{Colors.RESET}{Colors.GREEN} operators, {Colors.BOLD}{len(exs)}{Colors.RESET}{Colors.GREEN} examples")
        return GraphContext(pattern=p, operators=ops, examples=exs)


# -----------------------------
# OpenAI: classify + generate
# -----------------------------

def classify_constraint(client: OpenAI, constraint: str) -> Classification:
    log_step(1, "CLASSIFYING CONSTRAINT")
    log_highlight("Input", constraint)
    
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

    logger.info(f"{Colors.BLUE}Calling OpenAI API for classification...{Colors.RESET}")
    resp = client.responses.create(
        model="gpt-5.1",
        input=prompt,
        text={
            "format": {
                "type": "json_schema",
                "name": schema["name"],
                "schema": schema["schema"],
                "strict": schema["strict"]
            }
        },
    )
    data = json.loads(resp.output_text)
    
    log_success("Classification complete")
    log_highlight("Pattern", data['pattern'])
    log_highlight("Keywords", data['keywords'])
    log_highlight("Cues", data.get('cues', []))
    log_highlight("Trigger", data.get('trigger_text'))
    log_highlight("Response", data.get('response_text'))
    
    return Classification(
        pattern=data["pattern"],
        keywords=data["keywords"],
        cues=data.get("cues", []),
        trigger_text=data.get("trigger_text"),
        response_text=data.get("response_text"),
    )


def generate_uppaal(client: OpenAI, constraint: str, cls: Classification, ctx: GraphContext) -> Dict[str, Any]:
    log_step(3, "GENERATING UPPAAL QUERY")
    
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

    instructions = f"""
Translate ONE natural-language constraint into ONE UPPAAL query.

Classifier output:
- pattern={cls.pattern}
- cues={cls.cues}
- keywords={cls.keywords}
- trigger_text={cls.trigger_text}
- response_text={cls.response_text}

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

Return JSON only.
""".strip()

    logger.info(f"{Colors.BLUE}Calling OpenAI API for UPPAAL generation...{Colors.RESET}")
    log_highlight("Pattern template", ctx.pattern.template)
    
    resp = client.responses.create(
        model="gpt-5.1",
        instructions=instructions,
        input="Generate the UPPAAL query now.",
        text={
            "format": {
                "type": "json_schema",
                "name": out_schema["name"],
                "schema": out_schema["schema"],
                "strict": out_schema["strict"]
            }
        },
    )
    
    result = json.loads(resp.output_text)
    log_success("UPPAAL query generated")
    logger.info(f"{Colors.GREEN}  Query: {Colors.BOLD}{Colors.WHITE}{result['query']}{Colors.RESET}")
    log_highlight("Confidence", f"{result['confidence']:.2%}")
    log_highlight("Explanation", result['explanation'])
    
    return result


# -----------------------------
# ANTLR Validation
# -----------------------------

def validate_antlr_stub(query: str) -> Tuple[bool, Optional[str]]:
    """Validate UPPAAL query using external ANTLR parser service."""
    if not query.strip():
        return False, "Empty query"
    
    try:
        logger.info(f"{Colors.BLUE}Validating query with ANTLR parser...{Colors.RESET}")
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
        logger.warning(f"{Colors.YELLOW}ANTLR parser service not available at http://127.0.0.1:5002{Colors.RESET}")
        return False, "ANTLR parser service not reachable"
    except requests.exceptions.Timeout:
        logger.warning(f"{Colors.YELLOW}ANTLR parser service timeout{Colors.RESET}")
        return False, "Validation timeout"
    except Exception as e:
        logger.warning(f"{Colors.YELLOW}Validation error: {e}{Colors.RESET}")
        return False, f"Validation error: {str(e)}"


# -----------------------------
# CLI
# -----------------------------

def require_env(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise RuntimeError(f"Missing env var: {name}")
    return v


def main() -> None:
    parser = argparse.ArgumentParser(description="NL constraint -> Pattern classify -> GraphRAG -> UPPAAL query")
    parser.add_argument("constraint", type=str, help="Natural-language constraint string")
    parser.add_argument("--k", type=int, default=5, help="Top-k examples to retrieve from Neo4j")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging (DEBUG level)")
    args = parser.parse_args()

    # Adjust logging level if verbose
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    logger.info(Colors.BOLD + Colors.CYAN + "=" * 80 + Colors.RESET)
    logger.info(Colors.BOLD + Colors.CYAN + "🚀 PIPELINE STARTING" + Colors.RESET)
    logger.info(Colors.BOLD + Colors.CYAN + "=" * 80 + Colors.RESET)
    log_highlight("Constraint", args.constraint)
    log_highlight("Top-k examples", args.k)
    logger.info(Colors.BOLD + Colors.CYAN + "=" * 80 + Colors.RESET)

    # env
    logger.info(f"{Colors.BLUE}Loading environment variables...{Colors.RESET}")
    openai_key = require_env("OPENAI_API_KEY")
    neo4j_uri = require_env("NEO4J_URI")
    neo4j_user = require_env("NEO4J_USER")
    neo4j_password = require_env("NEO4J_PASSWORD")

    logger.info(f"{Colors.BLUE}Initializing OpenAI client...{Colors.RESET}")
    client = OpenAI(api_key=openai_key)
    kg = GraphRAG(neo4j_uri, neo4j_user, neo4j_password)

    try:
        cls = classify_constraint(client, args.constraint)
        ctx = kg.retrieve(cls, k_examples=args.k)
        out = generate_uppaal(client, args.constraint, cls, ctx)

        log_step(4, "VALIDATING QUERY")
        ok, err = validate_antlr_stub(out["query"])
        out["antlr_ok"] = ok
        out["antlr_error"] = err
        if ok:
            log_success(f"Validation: {Colors.BOLD}PASS{Colors.RESET}")
        else:
            logger.warning(f"Validation: {Colors.BOLD}FAIL{Colors.RESET}")
        if err:
            logger.warning(f"Validation error: {err}")

        logger.info(Colors.BOLD + Colors.GREEN + "=" * 80 + Colors.RESET)
        logger.info(Colors.BOLD + Colors.GREEN + "✨ PIPELINE COMPLETED SUCCESSFULLY ✨" + Colors.RESET)
        logger.info(Colors.BOLD + Colors.GREEN + "=" * 80 + Colors.RESET)

        # Print a clean result JSON to stdout with colors
        result_data = {
            "input": args.constraint,
            "classification": {
                "pattern": cls.pattern,
                "keywords": cls.keywords,
                "cues": cls.cues,
                "trigger_text": cls.trigger_text,
                "response_text": cls.response_text,
            },
            "graph_context": {
                "pattern": ctx.pattern.__dict__,
                "operators_count": len(ctx.operators),
                "examples_count": len(ctx.examples),
            },
            "result": out,
        }
        
        print("\n" + Colors.BOLD + Colors.WHITE + "📊 RESULT:" + Colors.RESET)
        print(colorize_json(result_data))

    except Exception as e:
        logger.error(Colors.BOLD + Colors.RED + "=" * 80 + Colors.RESET)
        logger.error(Colors.BOLD + Colors.RED + f"❌ PIPELINE FAILED: {e}" + Colors.RESET)
        logger.error(Colors.BOLD + Colors.RED + "=" * 80 + Colors.RESET)
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    finally:
        kg.close()


if __name__ == "__main__":
    main()
