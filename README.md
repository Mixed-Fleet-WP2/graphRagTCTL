# GraphRAG TCTL Pipeline

> 🚀 Natural Language to UPPAAL Query Translation using Graph-Augmented Retrieval and LLM

A sophisticated pipeline that translates natural language constraints into formal UPPAAL temporal logic queries using a combination of pattern classification, knowledge graph retrieval, and LLM-based generation.

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Pipeline Steps](#pipeline-steps)
- [Temporal Patterns](#temporal-patterns)
- [Examples](#examples)
- [Project Structure](#project-structure)
- [Advanced Usage](#advanced-usage)

---

## 🎯 Overview

This pipeline transforms natural language safety constraints, workflow specifications, and temporal requirements into formal UPPAAL queries that can be used for model checking and verification.

**Example:**
```bash
Input:  "if Drone battery < 10% it should be in charging"
Output: (Drone.battery < 10) --> Drone.charging
```

---

## 🏗️ Architecture

The pipeline uses a **4-step GraphRAG approach**:

```
┌─────────────────────────────────────────────────────────────┐
│  Step 1: CLASSIFY CONSTRAINT                                │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ LLM analyzes natural language and determines:        │   │
│  │ • Pattern type (safety, workflow, liveness, etc.)    │   │
│  │ • Keywords for retrieval                             │   │
│  │ • Temporal cues (when, eventually, never, etc.)      │   │
│  │ • Trigger and response components                    │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 2: RETRIEVE GRAPH CONTEXT                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Neo4j Knowledge Graph provides:                      │   │
│  │ • Pattern templates and rules                        │   │
│  │ • Operator semantics and usage guidelines            │   │
│  │ • Top-k similar examples (keyword-based)             │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 3: GENERATE UPPAAL QUERY                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ LLM synthesizes query using:                         │   │
│  │ • Classification results                             │   │
│  │ • Pattern template                                   │   │
│  │ • Operator guidelines                                │   │
│  │ • Similar examples as few-shot demonstrations        │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 4: VALIDATE QUERY                                     │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ • Syntax validation (ANTLR parser)                   │   │
│  │ • Semantic checks                                    │   │
│  │ • Confidence scoring                                 │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## ✨ Key Features

- **🎨 Colorful Logging**: Beautiful, color-coded terminal output for easy tracking
- **🧠 Pattern-Based Classification**: Automatically identifies the temporal pattern type
- **📊 Knowledge Graph Integration**: Retrieves relevant context from Neo4j
- **🔍 Semantic Retrieval**: Finds similar examples based on keyword matching
- **✅ Structured Output**: JSON schema validation for reliable LLM responses
- **🎯 High Accuracy**: Combines rule-based templates with LLM flexibility

---

## 📦 Installation

### Prerequisites

- Python 3.8+
- Neo4j Database (local or cloud)
- OpenAI API key

### Setup

1. **Clone the repository**
   ```bash
   cd graphRagTCTL
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   
   Create a `.env` file in the project root:
   ```bash
   OPENAI_API_KEY=your_openai_api_key
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USER=neo4j
   NEO4J_PASSWORD=your_password
   ```

4. **Seed the knowledge graph**
   ```bash
   python seed_kg.py --seed-all
   ```

   This creates:
   - 6 pattern templates
   - 7 operator definitions
   - 20+ example constraints
   - Relationships between patterns and operators

---

## 🚀 Quick Start

### Basic Usage

```bash
python pipeline.py "Drone returns to base when battery < 10%"
```

### With Custom Parameters

```bash
# Retrieve top 10 examples instead of default 5
python pipeline.py "Robot stops when obstacle detected" --k 10

# Enable verbose logging
python pipeline.py "System eventually reaches stable state" --verbose
```

### Output

The pipeline outputs a JSON result to stdout and colorful logs to stderr:

```json
{
  "input": "Drone returns to base when battery < 10%",
  "classification": {
    "pattern": "safety_immediate_response",
    "keywords": ["drone", "return", "base", "battery", "when"],
    "cues": ["when", "<", "%"],
    "trigger_text": "battery < 10%",
    "response_text": "return to base"
  },
  "graph_context": {
    "pattern": {
      "name": "safety_immediate_response",
      "template": "A[] (trigger_condition imply safety_action)",
      ...
    },
    "operators_count": 2,
    "examples_count": 5
  },
  "result": {
    "query": "A[] (Drone.battery < 10 imply Drone.returning_to_base)",
    "explanation": "Uses A[] with imply for immediate safety response...",
    "confidence": 0.95,
    "pattern_used": "safety_immediate_response",
    "operator_used": ["A[]", "imply"],
    "antlr_ok": true
  }
}
```

---

## 🔄 Pipeline Steps

### Step 1: Classification

The LLM analyzes the natural language constraint and extracts:

- **Pattern**: Which temporal pattern best fits (safety, workflow, liveness, etc.)
- **Keywords**: Important terms for example retrieval
- **Cues**: Temporal indicators (when, eventually, never, must, can)
- **Trigger/Response**: Condition and action components

### Step 2: Graph Retrieval

The Neo4j knowledge graph provides:

- **Pattern Template**: The UPPAAL query structure for the identified pattern
- **Operators**: Relevant temporal operators with usage guidelines
- **Examples**: Top-k similar constraints with their UPPAAL translations

### Step 3: Query Generation

The LLM synthesizes the final UPPAAL query using:

- Classification results
- Pattern template as a structural guide
- Operator semantics and decision rules
- Similar examples as few-shot demonstrations

### Step 4: Validation

The generated query undergoes:

- Syntax validation (ANTLR parser stub)
- Confidence assessment
- Operator usage verification

---

## 🎯 Temporal Patterns

## 🎯 Temporal Patterns

The pipeline supports 6 built-in temporal patterns:

### 1. Safety Immediate Response
**Template:** `A[] (trigger_condition imply safety_action)`  
**When to use:** Immediate reaction required for safety-critical conditions  
**Keywords:** immediately, when, must, stop, abort, pause

**Example:**
- Input: "Robot must stop immediately when obstacle is detected"
- Output: `A[] (Robot.obstacle_detected imply Robot.stopped)`

### 2. Sequential Workflow
**Template:** `action1 --> action2`  
**When to use:** Multi-step processes where one action eventually follows another  
**Keywords:** after, once, then, eventually, follows

**Example:**
- Input: "Once scanning is done, UGV will collect the container"
- Output: `Drone.scanning_done --> UGV.collecting`

### 3. Eventual Completion
**Template:** `A<> final_state`  
**When to use:** Liveness properties - something must eventually happen  
**Keywords:** eventually, will, must complete, finish

**Example:**
- Input: "Both drones will eventually complete their missions"
- Output: `A<> (DroneA.mission_complete && DroneB.mission_complete)`

### 4. Reachability
**Template:** `E<> target_state`  
**When to use:** Verify that something CAN happen or is POSSIBLE  
**Keywords:** possible, can, able to, feasible

**Example:**
- Input: "Can the drone reach the target location"
- Output: `E<> Drone.at_target`

### 5. Forbidden State
**Template:** `A[] not (bad_state)`  
**When to use:** Prevent dangerous or invalid states from occurring  
**Keywords:** never, cannot, must not, prohibited

**Example:**
- Input: "Robot cannot move while charging"
- Output: `A[] not (Robot.charging && Robot.moving)`

### 6. Conditional Response
**Template:** `condition --> response`  
**When to use:** Triggered behaviors with eventual response  
**Keywords:** if, when, causes, triggers, leads to

**Example:**
- Input: "When battery is low, robot goes to charging station"
- Output: `(Robot.battery < 20) --> Robot.at_charging_station`

---

## 🎨 Colorful Logging

The pipeline features beautiful, color-coded logging for easy tracking:

- 🔵 **Blue**: API calls and operations in progress
- 🟢 **Green**: Successful completions and validations
- 🟡 **Yellow**: Highlights and important values
- 🔴 **Red**: Errors and failures
- 🟣 **Magenta**: Step headers and major sections
- ⚪ **White**: Data values and query outputs

Example output:
```
================================================================================
STEP 1: CLASSIFYING CONSTRAINT
================================================================================
2024-12-22 10:30:15 - INFO - Input: Drone returns to base when battery < 10%
2024-12-22 10:30:15 - INFO - Calling OpenAI API for classification...
2024-12-22 10:30:16 - INFO - ✓ Classification complete
2024-12-22 10:30:16 - INFO - Pattern: safety_immediate_response
2024-12-22 10:30:16 - INFO - Keywords: ['drone', 'return', 'base', 'battery', 'when']
```

---

## 📚 Examples

### Example 1: Safety Constraint
```bash
python pipeline.py "Welder pauses when temperature exceeds 100 degrees"
```
**Output:** `A[] (Welder.temperature > 100 imply Welder.paused)`

### Example 2: Workflow
```bash
python pipeline.py "After inspection finishes, welder starts welding"
```
**Output:** `InspectorDrone.inspection_complete --> Welder.welding`

### Example 3: Liveness
```bash
python pipeline.py "All inspection tasks will eventually be completed"
```
**Output:** `A<> InspectionBot.all_tasks_done`

### Example 4: Forbidden State
```bash
python pipeline.py "System never allows operation with high vibration"
```
**Output:** `A[] not (vibration > 5.0 && System.operating)`

---

## 📁 Project Structure

```
graphRagTCTL/
├── pipeline.py              # Main GraphRAG pipeline orchestration
├── seed_kg.py              # Neo4j knowledge graph seeding script
├── multi_llm_constraint_classifier.py  # Alternative classifier
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (create this)
├── README.md              # This file
│
├── dataset/               # Training/test scenarios
│   ├── scenario_001.json
│   ├── scenario_002.json
│   └── ...
│
├── scenario_classifications/  # Generated classifications
│   ├── scenario_001_classification.json
│   └── ...
│
└── docs/
    └── ARCHITECTURE.md    # Detailed architecture documentation
```

---

## ⚙️ Advanced Usage

### Custom Knowledge Graph Queries

You can modify the retrieval queries in `pipeline.py`:

```python
# Adjust example retrieval
exs = self.get_examples(cls.pattern, cls.keywords, k=10)  # Get top-10

# Custom pattern filtering
cypher = """
MATCH (e:Example)-[:INSTANCE_OF]->(p:Pattern {name:$pattern})
WHERE e.qualityScore > 0.8
...
"""
```

### Extending Patterns

Add new patterns to `seed_kg.py`:

```python
PATTERNS.append({
    "name": "my_custom_pattern",
    "description": "Description of when to use this pattern",
    "template": "UPPAAL template structure",
    "when_to_use": "Guidance for classification",
    "decision_rules": ["keyword1", "keyword2"]
})
```

### Batch Processing

Process multiple constraints:

```bash
#!/bin/bash
while IFS= read -r constraint; do
  python pipeline.py "$constraint" >> results.jsonl
done < constraints.txt
```

### Integration with Model Checker

```python
import subprocess

# Generate query
result = subprocess.run(
    ["python", "pipeline.py", "Your constraint here"],
    capture_output=True,
    text=True
)

query_data = json.loads(result.stdout)
uppaal_query = query_data["result"]["query"]

# Feed to UPPAAL verifier
# ... your verification code ...
```

---

## 🔧 Troubleshooting

### Issue: "Pattern not found in Neo4j"
**Solution:** Run `python seed_kg.py --seed-all` to populate the knowledge graph

### Issue: OpenAI API errors
**Solution:** 
- Check your API key in `.env`
- Verify you have credits available
- Check if you have access to the model (gpt-5.1 or your configured model)

### Issue: Neo4j connection refused
**Solution:**
- Ensure Neo4j is running: `sudo systemctl start neo4j` (Linux) or start via Neo4j Desktop
- Check the URI in `.env` matches your Neo4j instance
- Verify credentials are correct

### Issue: No examples retrieved
**Solution:**
- Check that examples were seeded: `python seed_kg.py --seed-examples`
- Verify keyword overlap - try broader constraint descriptions

---

## 📊 Knowledge Graph Schema

```
(Pattern)
  ├─[:USES_OPERATOR]─>(Operator)
  │
  └─[:INSTANCE_OF]<─(Example)
                     └─[:USES_OPERATOR]─>(Operator)
```

**Nodes:**
- `Pattern`: Template definitions and usage rules
- `Operator`: UPPAAL operators with semantics
- `Example`: Labeled NL→UPPAAL constraint pairs

**Properties:**
- Pattern: name, description, template, when_to_use, decision_rules[]
- Operator: name, description, semantics, when_to_use, when_not_to_use, examples[]
- Example: id, nl, uppaal, keywords[], explanation, qualityScore

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

1. **More Patterns**: Add specialized temporal patterns
2. **Better Validation**: Integrate full ANTLR parser for UPPAAL syntax
3. **Vector Search**: Add embedding-based retrieval alongside keyword matching
4. **Fine-tuning**: Collect high-quality examples for model fine-tuning
5. **Multi-modal**: Support diagrams/state machines as input

---

## 📄 License

[Specify your license here]

---

## 📞 Contact

[Your contact information]

---

## 🙏 Acknowledgments

- UPPAAL model checker team for the temporal logic specification
- Neo4j for graph database platform
- OpenAI for language model capabilities

---

## 🔗 Related Projects

- [UPPAAL](https://uppaal.org/) - Model checking tool
- [Neo4j](https://neo4j.com/) - Graph database
- [LangChain](https://langchain.com/) - LLM application framework

---

**Happy Formal Verification! 🚀✨**

```bash
python main.py
```

The default configuration uses deterministic mock services, so no external credentials are required. To connect to real infrastructure:

1. Update `PipelineSettings` with your Claude API key and Neo4j credentials.
2. Instantiate `GraphRAGPipeline` with `use_mock_services=False`.

```python
settings = PipelineSettings(
    use_mock_services=False,
    claude_api_key="sk-...",
    neo4j_uri="neo4j://<host>:7687",
    neo4j_user="neo4j",
    neo4j_password="password",
)
pipeline = GraphRAGPipeline(settings=settings)
```

## Using Real Claude + Neo4j

1. Install the runtime dependencies (Anthropic SDK for Claude + Neo4j Python driver):

```bash
pip install -r requirements.txt
```

2. Set the required environment variables (replace placeholders with your credentials):

```bash
export ANTHROPIC_API_KEY="sk-..."          # Claude API key
export NEO4J_URI="neo4j://<host>:7687"     # e.g., bolt or neo4j protocol URL
export NEO4J_USER="neo4j"                  # DB username
export NEO4J_PASSWORD="your-password"      # DB password
```

3. Instantiate the pipeline with `use_mock_services=False` so the real Claude + Neo4j clients are used:

```python
from graphrag_tctl import GraphRAGPipeline, PipelineSettings

settings = PipelineSettings(use_mock_services=False)
pipeline = GraphRAGPipeline(settings=settings)
result = pipeline.run("Drone returns to base when battery drops below 10%")
```

If the API key is missing, `ClaudeClient` raises `ValueError`, which helps catch misconfiguration early.

## Extending

- Add new temporal patterns or operator documentation inside Neo4j (or `LocalKnowledgeGraph` for quick prototyping).
- Implement a production-grade vector store by swapping `EnhancedVectorStore` with any embedding-backed solution.
- Integrate with a REST or FastAPI front-end by importing `GraphRAGPipeline` and exposing its `.run()` method.
