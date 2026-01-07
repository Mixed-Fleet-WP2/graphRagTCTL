import os
import re
import json
import time
import argparse
from typing import Dict, List, Any, Optional, Tuple

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

import requests

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
#  YOUR PATTERNS (CAN EDIT FREELY)
# ==========================================
PATTERNS = [
    {
        "name": "safety_immediate_response",
        "description": "Immediate response to dangerous conditions",
        "template": "A[] (trigger_condition imply safety_action)",
        "when_to_use": "When response must be instantaneous. No delay allowed between trigger and response.",
        "decision_rules": ["Keywords: immediately, when, must, stop, abort, pause"]
    },
    {
        "name": "sequential_workflow",
        "description": "One action eventually follows another",
        "template": "action1 --> action2",
        "when_to_use": "For multi-step processes where one step leads to another with possible delays.",
        "decision_rules": ["Keywords: after, once, then, eventually, follows"]
    },
    {
        "name": "eventual_completion",
        "description": "Task must eventually complete",
        "template": "A<> final_state",
        "when_to_use": "For liveness properties where something must eventually happen.",
        "decision_rules": ["Keywords: eventually, will, must complete, finish"]
    },
    {
        "name": "reachability",
        "description": "Check if state is possible",
        "template": "E<> target_state",
        "when_to_use": "To verify that something CAN happen or is POSSIBLE.",
        "decision_rules": ["Keywords: possible, can, able to, feasible"]
    },
    {
        "name": "forbidden_state",
        "description": "State must never occur",
        "template": "A[] not (bad_state)",
        "when_to_use": "To prevent dangerous or invalid states from ever happening.",
        "decision_rules": ["Keywords: never, cannot, must not, prohibited"]
    },
    {
        "name": "conditional_response",
        "description": "If condition, then eventual response",
        "template": "condition --> response",
        "when_to_use": "For triggered behaviors where a condition leads to eventual action.",
        "decision_rules": ["Keywords: if, when, causes, triggers, leads to"]
    }
]

# ==========================================
#  PROMPT (STRICT JSON OUTPUT)
# ==========================================
CLASSIFY_PROMPT = """
You are a robotics verification assistant.

TASK:
Classify each constraint into EXACTLY ONE pattern from the provided list.

IMPORTANT:
- You must choose the best matching pattern.
- If multiple seem possible, choose the most specific one:
  forbidden_state > safety_immediate_response > conditional_response > sequential_workflow > eventual_completion > reachability

OUTPUT RULES (STRICT):
- Output ONLY valid JSON (no markdown, no code blocks).
- Output must be a JSON array.
- Each element must be an object with keys:
  "index" (int), "pattern_name" (string), "template" (string)

NO extra keys. NO explanation text.

Robots: {robots}

Scenario:
{scenario}

Patterns (choose one name):
{patterns_json}

Constraints to classify:
{constraints_block}

Output JSON:
"""


# ==========================================
#  ROBUST JSON EXTRACTOR
# ==========================================
def extract_json_array(text: str) -> Optional[List[Dict[str, Any]]]:
    """
    Tries to parse a JSON array from raw model text.
    Handles cases where model adds extra text before/after.
    """
    if not text:
        return None

    text = text.strip()

    # Direct parse
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except:
        pass

    # Try to find the first JSON array substring
    # This finds the first '[' ... matching last ']' (greedy)
    m = re.search(r"\[\s*{.*}\s*\]", text, flags=re.DOTALL)
    if m:
        try:
            parsed = json.loads(m.group(0))
            if isinstance(parsed, list):
                return parsed
        except:
            return None

    return None


def normalize_pattern_name(name: str) -> str:
    return (name or "").strip().lower()


def build_patterns_json(patterns: List[Dict[str, Any]]) -> str:
    # Keep it compact but informative for the LLM
    slim = []
    for p in patterns:
        slim.append({
            "name": p["name"],
            "template": p["template"],
            "when_to_use": p["when_to_use"],
            "decision_rules": p["decision_rules"]
        })
    return json.dumps(slim, indent=2)


def constraints_block(constraints: List[str]) -> str:
    # Make indexing explicit
    lines = []
    for i, c in enumerate(constraints):
        lines.append(f"{i}: {c}")
    return "\n".join(lines)


# ==========================================
#  LLM CLIENT (ROBUST)
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

    def call(self, prompt: str) -> Tuple[str, float, int]:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                return self._single_call(prompt)
            except Exception as e:
                print(f"[ERROR] {self.provider}/{self.model} failed (attempt {attempt}/{MAX_RETRIES}): {e}")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)
                else:
                    return "", 0.0, 0

    def _single_call(self, prompt: str) -> Tuple[str, float, int]:
        start = time.time()

        allow_temp = not any(m in self.model for m in MODELS_NO_TEMP)
        use_responses = any(m in self.model for m in OPENAI_RESPONSES_MODELS)

        # ------------------------------
        # OPENAI
        # ------------------------------
        if self.provider == "openai":
            if use_responses:
                resp = self.client.responses.create(
                    model=self.model,
                    input=prompt
                )
                text = getattr(resp, "output_text", "") or ""
                return text.strip(), time.time() - start, 0

            kwargs = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
            }
            if allow_temp:
                kwargs["temperature"] = 0

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
                max_tokens=1200,
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
            raw = requests.post(url, json=payload, headers=headers).json()

            # Standard format
            try:
                text = raw["choices"][0]["message"]["content"]
                return (text or "").strip(), time.time() - start, 0
            except:
                pass

            # Fallbacks
            if "message" in raw and "content" in raw["message"]:
                return str(raw["message"]["content"]).strip(), time.time() - start, 0

            if "output" in raw and "text" in raw["output"]:
                return str(raw["output"]["text"]).strip(), time.time() - start, 0

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
            raw = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": self.model, "prompt": prompt}
            ).json()
            text = raw.get("response", "") or ""
            return text.strip(), time.time() - start, 0

        raise ValueError("Unsupported provider")


# ==========================================
#  PROCESS ONE SCENARIO (ONE MODEL)
# ==========================================
def classify_constraints(llm: LLMClient, scenario: Dict[str, Any]) -> Dict[str, Any]:
    robots = ", ".join(scenario.get("robots", []))
    scenario_text = scenario.get("prompt", "")
    constraints = scenario.get("constraints", [])

    prompt = CLASSIFY_PROMPT.format(
        robots=robots,
        scenario=scenario_text,
        patterns_json=build_patterns_json(PATTERNS),
        constraints_block=constraints_block(constraints),
    )

    text, elapsed, tokens = llm.call(prompt)

    parsed = extract_json_array(text)
    if parsed is None:
        # Hard fallback: mark all as "unparsed"
        return {
            "provider": llm.provider,
            "model": llm.model,
            "timing": elapsed,
            "tokens": tokens,
            "raw_text": text,
            "classified": [],
            "parse_error": True
        }

    # Validate shape + normalize
    allowed = {p["name"] for p in PATTERNS}
    out = []
    for obj in parsed:
        idx = obj.get("index", None)
        pname = normalize_pattern_name(obj.get("pattern_name", ""))
        template = obj.get("template", "")

        # Keep only allowed keys and safe values
        if not isinstance(idx, int):
            continue
        if pname not in allowed:
            # If LLM returns unknown, keep it but flag it
            out.append({
                "index": idx,
                "constraint": constraints[idx] if 0 <= idx < len(constraints) else "",
                "pattern_name": pname,
                "template": template,
                "invalid_pattern": True
            })
        else:
            out.append({
                "index": idx,
                "constraint": constraints[idx] if 0 <= idx < len(constraints) else "",
                "pattern_name": pname,
                "template": template
            })

    # Sort by index for consistency
    out.sort(key=lambda x: x["index"])

    return {
        "provider": llm.provider,
        "model": llm.model,
        "timing": elapsed,
        "tokens": tokens,
        "classified": out,
        "parse_error": False
    }


# ==========================================
#  MAIN LOOP OVER DATASET FOLDER
# ==========================================
def run_folder(
    folder: str,
    models: List[Dict[str, str]],
    start: int = 0,
    end: Optional[int] = None,
    scenario_output: str = "scenario_classifications"
):
    os.makedirs(scenario_output, exist_ok=True)

    all_files = sorted([f for f in os.listdir(folder) if f.endswith(".json")])
    files = all_files[start:end]

    print(f"\nRunning scenarios {start} → {start + len(files)}")
    print(f"Dataset folder: {folder}")
    print(f"Output folder: {scenario_output}")

    for file in files:
        scenario_path = os.path.join(folder, file)
        with open(scenario_path, "r") as f:
            scenario = json.load(f)

        scenario_id = scenario.get("id", os.path.splitext(file)[0])
        out_path = os.path.join(scenario_output, f"{scenario_id}_classification.json")

        print(f"\n=== Scenario {file} ({scenario_id}) ===")

        # Load existing scenario output if present
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

        existing = {(r.get("provider"), r.get("model")) for r in scenario_result.get("results", [])}

        for cfg in models:
            key = (cfg["provider"].lower(), cfg["model"].lower())

            if key in existing:
                print(f"  → SKIP (already exists): {cfg['provider']} / {cfg['model']}")
                continue

            print(f"  → RUN: {cfg['provider']} / {cfg['model']}")
            llm = LLMClient(cfg["provider"], cfg["model"])
            result = classify_constraints(llm, scenario)
            scenario_result["results"].append(result)

            # Save incrementally after each model
            with open(out_path, "w") as f:
                json.dump(scenario_result, f, indent=2)

            print(f"    Saved incremental update → {out_path}")

    print("\nDone ✅")


# ==========================================
#  CLI
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True, help="Dataset folder of scenario JSON files")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--scenario_output", type=str, default="scenario_classifications")
    args = parser.parse_args()

    MODELS = [
        {"provider": "openai", "model": "gpt-5.1"},
        {"provider": "openai", "model": "gpt-5-mini"},
        {"provider": "openai", "model": "gpt-5-nano"},
        {"provider": "openai", "model": "gpt-4o"},
        {"provider": "anthropic", "model": "claude-sonnet-4-5-20250929"},
        {"provider": "anthropic", "model": "claude-haiku-4-5-20251001"},
        {"provider": "xai", "model": "grok-code-fast-1"},
        {"provider": "xai", "model": "grok-3-mini"},
        {"provider": "deepseek", "model": "deepseek-reasoner"},
        {"provider": "google", "model": "gemini-2.0-flash"},
        {"provider": "google", "model": "gemini-2.5-pro"},
        # Optional local baseline:
        # {"provider": "ollama", "model": "llama3.1"},
    ]

    run_folder(
        folder=args.folder,
        models=MODELS,
        start=args.start,
        end=args.end,
        scenario_output=args.scenario_output
    )
