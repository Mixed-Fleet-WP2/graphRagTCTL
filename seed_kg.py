

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()


# -----------------------------
# Your canonical data
# -----------------------------

PATTERNS: List[Dict[str, Any]] = [
    {
        "name": "safety_immediate_response",
        "description": "Immediate response to dangerous conditions",
        "template": "A[] (trigger_condition imply safety_action)",
        "when_to_use": "When response must be instantaneous. No delay allowed between trigger and response.",
        "decision_rules": ["Keywords: immediately, when, must, stop, abort, pause"],
    },
    {
        "name": "sequential_workflow",
        "description": "One action eventually follows another",
        "template": "action1 --> action2",
        "when_to_use": "For multi-step processes where one step leads to another with possible delays.",
        "decision_rules": ["Keywords: after, once, then, eventually, follows"],
    },
    {
        "name": "eventual_completion",
        "description": "Task must eventually complete",
        "template": "A<> final_state",
        "when_to_use": "For liveness properties where something must eventually happen.",
        "decision_rules": ["Keywords: eventually, will, must complete, finish"],
    },
    {
        "name": "reachability",
        "description": "Check if state is reachable",
        "template": "E<> target_state OR A<> target_state",
        "when_to_use": "To verify reachability. Use E<> for existential (there exists a path), A<> for universal (all paths can reach).",
        "decision_rules": [
            "Use E<> when: possible, can, might, feasible, exists a way",
            "Use A<> when: all paths, always possible, guaranteed reachable, must be able to reach",
            "E<> = weaker (at least one path)",
            "A<> = stronger (every path)"
        ],
    },
    {
        "name": "forbidden_state",
        "description": "State must never occur",
        "template": "A[] not (bad_state)",
        "when_to_use": "To prevent dangerous or invalid states from ever happening.",
        "decision_rules": ["Keywords: never, cannot, must not, prohibited"],
    },
    {
        "name": "conditional_response",
        "description": "If condition, then eventual response",
        "template": "condition --> response",
        "when_to_use": "For triggered behaviors where a condition leads to eventual action.",
        "decision_rules": ["Keywords: if, when, causes, triggers, leads to"],
    },
    {
        "name": "time_bounded_constraint",
        "description": "Action must occur within specific time duration",
        "template": "A[] (state imply clock <= T) or A[] (state && not target_state imply clock <= T)",
        "when_to_use": "When there's a maximum time allowed in a state or for a response. Enforces that time cannot exceed limit.",
        "decision_rules": [
            "Keywords: within X time, after X time (meaning 'no more than'), before X time, timeout",
            "Intent: enforce deadline, maximum duration",
            "Pattern: Use 'imply clock <= T' to enforce upper time bound",
            "Clock resets: At state entry or triggering condition"
        ]
    }
]

# Your operator catalog (extended with && and not because your examples use them)
OPERATORS: List[Dict[str, Any]] = [
    {
        "name": "A[]",
        "description": "Always - property holds in ALL reachable states at ALL times",
        "semantics": "universal_invariant",
        "when_to_use": "For safety properties. Things that must ALWAYS be true or NEVER happen. Use with 'imply' for immediate responses.",
        "when_not_to_use": "Don't use for eventual actions or when delay is acceptable.",
        "examples": ["A[] not deadlock", "A[] (battery < 10 imply charging)"],
    },
    {
        "name": "E<>",
        "description": "Possibly - there exists at least one path where property eventually holds",
        "semantics": "existential_reachability",
        "when_to_use": "To check if something CAN happen or is POSSIBLE.",
        "when_not_to_use": "Don't use when something MUST happen (use A<> instead).",
        "examples": ["E<> Robot.at_goal", "E<> (DroneA.done && DroneB.done)"],
    },
    {
        "name": "A<>",
        "description": "Eventually - property eventually holds on ALL execution paths",
        "semantics": "universal_reachability",
        "when_to_use": "For liveness properties. Things that MUST eventually happen.",
        "when_not_to_use": "Don't use for conditional responses (use --> instead).",
        "examples": ["A<> mission_complete", "A<> Robot.at_base"],
    },
    {
        "name": "-->",
        "description": "Leads-to - if left becomes true, right will eventually become true",
        "semantics": "progress",
        "when_to_use": "For sequential workflows and conditional eventual responses.",
        "when_not_to_use": "Don't use for immediate responses (use A[] imply instead).",
        "examples": ["scanning_done --> collecting", "(battery < 10) --> charging"],
    },
    {
        "name": "imply",
        "description": "Logical implication in the SAME state (synchronous)",
        "semantics": "immediate_implication",
        "when_to_use": "For immediate responses. If condition is true, consequence must be true RIGHT NOW. Always used inside A[].",
        "when_not_to_use": "Don't use for eventual responses (use --> instead).",
        "examples": ["A[] (emergency imply stopped)", "A[] (overload imply not operating)"],
    },
   
]

PATTERN_TO_OPERATORS: Dict[str, List[str]] = {
    "safety_immediate_response": ["A[]", "imply"],
    "sequential_workflow": ["-->"],
    "eventual_completion": ["A<>"],
    "reachability": ["E<>", "A<>"],
    "forbidden_state": ["A[]", "not"],
    "conditional_response": ["-->"],
    "time_bounded_constraint": ["A[]", "imply"]
}

# Your constraints/examples (will be inserted as :Example)
CONSTRAINT_EXAMPLES: List[Dict[str, Any]] = [
    {
        "id": "c001",
        "nl": "Drone must return to base when battery is below 10%",
        "query": "A[] (Drone.battery < 10 imply Drone.returning_to_base)",
        "pattern": "safety_immediate_response",
        "operators": ["A[]", "imply"],
        "keywords": ["drone", "return", "base", "battery", "below", "when", "must"],
        "explanation": "Use A[] with imply for immediate safety response. No delay allowed."
    },
    {
        "id": "c002",
        "nl": "Robot must stop immediately when obstacle is detected",
        "query": "A[] (Robot.obstacle_detected imply Robot.stopped)",
        "pattern": "safety_immediate_response",
        "operators": ["A[]", "imply"],
        "keywords": ["robot", "stop", "immediately", "obstacle", "detected", "when", "must"],
        "explanation": "Immediate safety response requires A[] with imply."
    },
    {
        "id": "c003",
        "nl": "Once scanning is done, UGV will eventually collect the container",
        "query": "Drone.scanning_done --> UGV.collecting",
        "pattern": "sequential_workflow",
        "operators": ["-->"],
        "keywords": ["scanning", "done", "ugv", "collect", "container", "eventually", "once"],
        "explanation": "Use --> for sequential workflow where one action eventually follows another."
    },
    {
        "id": "c004",
        "nl": "Both drones will eventually complete their missions",
        "query": "A<> (DroneA.mission_complete && DroneB.mission_complete)",
        "pattern": "eventual_completion",
        "operators": ["A<>", "&&"],
        "keywords": ["both", "drones", "eventually", "complete", "missions", "will"],
        "explanation": "Use A<> for liveness - missions MUST eventually complete."
    },
    {
        "id": "c005",
        "nl": "It's possible for both robots to reach the charging station",
        "query": "E<> (RobotA.at_charging_station && RobotB.at_charging_station)",
        "pattern": "reachability",
        "operators": ["E<>", "&&"],
        "keywords": ["possible", "both", "robots", "reach", "charging", "station"],
        "explanation": "Use E<> to check if something is POSSIBLE."
    },
    {
        "id": "c006",
        "nl": "System never allows operation with high vibration",
        "query": "A[] not (vibration > 5.0 && System.operating)",
        "pattern": "forbidden_state",
        "operators": ["A[]", "not", "&&"],
        "keywords": ["never", "allows", "operation", "high", "vibration", "system"],
        "explanation": "Use A[] not to forbid dangerous state combinations."
    },
    {
        "id": "c007",
        "nl": "Visibility below 30% causes both drones to surface",
        "query": "(visibility < 30) --> (DroneA.at_surface && DroneB.at_surface)",
        "pattern": "conditional_response",
        "operators": ["-->", "&&"],
        "keywords": ["visibility", "below", "causes", "drones", "surface", "both"],
        "explanation": "Use --> for conditional eventual response."
    },
    {
        "id": "c008",
        "nl": "PickerBot aborts picking when load exceeds limit",
        "query": "A[] (PickerBot.load > limit imply not PickerBot.picking)",
        "pattern": "safety_immediate_response",
        "operators": ["A[]", "imply", "not"],
        "keywords": ["pickerbot", "abort", "load", "exceeds", "limit", "when"],
        "explanation": "Immediate abort requires A[] imply. Use 'not picking' to express abortion."
    },
    {
        "id": "c009",
        "nl": "After inspection finishes, welder starts welding",
        "query": "InspectorDrone.inspection_complete --> Welder.welding",
        "pattern": "sequential_workflow",
        "operators": ["-->"],
        "keywords": ["after", "inspection", "finishes", "welder", "starts", "welding"],
        "explanation": "Sequential workflow - one task eventually follows another."
    },
    {
        "id": "c010",
        "nl": "Robot cannot move while charging",
        "query": "A[] not (Robot.charging && Robot.moving)",
        "pattern": "forbidden_state",
        "operators": ["A[]", "not", "&&"],
        "keywords": ["robot", "cannot", "move", "while", "charging"],
        "explanation": "Forbidden state - charging and moving cannot happen simultaneously."
    },
    {
        "id": "c011",
        "nl": "System eventually reaches stable state",
        "query": "A<> System.stable",
        "pattern": "eventual_completion",
        "operators": ["A<>"],
        "keywords": ["system", "eventually", "reaches", "stable", "state"],
        "explanation": "Liveness property - system must eventually stabilize."
    },
    {
        "id": "c012",
        "nl": "When battery is low, robot goes to charging station",
        "query": "(Robot.battery < 20) --> Robot.at_charging_station",
        "pattern": "conditional_response",
        "operators": ["-->"],
        "keywords": ["when", "battery", "low", "robot", "charging", "station", "goes"],
        "explanation": "Conditional eventual response - triggered by low battery."
    },
    {
        "id": "c013",
        "nl": "Welder pauses when temperature exceeds 100 degrees",
        "query": "A[] (Welder.temperature > 100 imply Welder.paused)",
        "pattern": "safety_immediate_response",
        "operators": ["A[]", "imply"],
        "keywords": ["welder", "pause", "temperature", "exceeds", "when"],
        "explanation": "Immediate safety response to prevent overheating."
    },
    {
        "id": "c014",
        "nl": "Can the drone reach the target location",
        "query": "E<> Drone.at_target",
        "pattern": "reachability",
        "operators": ["E<>"],
        "keywords": ["can", "drone", "reach", "target", "location"],
        "explanation": "Reachability check - is it possible to reach target?"
    },
    {
        "id": "c015",
        "nl": "System never deadlocks",
        "query": "A[] not deadlock",
        "pattern": "forbidden_state",
        "operators": ["A[]", "not"],
        "keywords": ["system", "never", "deadlock"],
        "explanation": "Standard deadlock freedom property."
    },
    {
        "id": "c016",
        "nl": "All inspection tasks will eventually be completed",
        "query": "A<> InspectionBot.all_tasks_done",
        "pattern": "eventual_completion",
        "operators": ["A<>"],
        "keywords": ["all", "inspection", "tasks", "eventually", "completed", "will"],
        "explanation": "Liveness - all tasks must eventually finish."
    },
    {
        "id": "c017",
        "nl": "Robot will eventually return to base",
        "query": "A<> Robot.at_base",
        "pattern": "eventual_completion",
        "operators": ["A<>"],
        "keywords": ["robot", "eventually", "return", "base", "will"],
        "explanation": "Liveness property - robot must eventually return."
    },
    {
        "id": "c018",
        "nl": "Two robots cannot occupy same position simultaneously",
        "query": "A[] not (RobotA.position == RobotB.position)",
        "pattern": "forbidden_state",
        "operators": ["A[]", "not"],
        "keywords": ["two", "robots", "cannot", "occupy", "same", "position", "simultaneously"],
        "explanation": "Collision avoidance - forbid same position."
    },
    {
        "id": "c019",
        "nl": "If error detected, system enters recovery mode",
        "query": "System.error_detected --> System.recovery_mode",
        "pattern": "conditional_response",
        "operators": ["-->"],
        "keywords": ["if", "error", "detected", "system", "recovery", "mode", "enters"],
        "explanation": "Conditional response - error triggers recovery."
    },
    {
        "id": "c020",
        "nl": "Charging completes then robot resumes operation",
        "query": "Robot.charging_complete --> Robot.operating",
        "pattern": "sequential_workflow",
        "operators": ["-->"],
        "keywords": ["charging", "completes", "robot", "resumes", "operation", "then"],
        "explanation": "Sequential workflow - operation follows charging."
    },
    {
        "id": "c021",
        "nl": "Drone must stop after 10 minutes of flying",
        "query": "A[] (Drone.flying imply clock <= 10)",
        "pattern": "time_bounded_constraint",
        "operators": ["A[]", "imply"],
        "keywords": ["drone", "stop", "after", "minutes", "flying", "10"],
        "explanation": "Drone cannot remain in flying state for more than 10 minutes. Clock resets when flying state is entered."
    },
    {
        "id": "c022",
        "nl": "Robot must reach base within 20 seconds when battery is low",
        "query": "A[] (Robot.battery < 10 && not Robot.at_base imply clock <= 20)",
        "pattern": "time_bounded_constraint",
        "operators": ["A[]", "imply", "&&", "not"],
        "keywords": ["robot", "reach", "base", "within", "seconds", "battery", "low", "20"],
        "explanation": "When battery is low and robot is not at base, clock cannot exceed 20 seconds. Clock resets when battery drops below 10%."
    },
    {
        "id": "c023",
        "nl": "System must respond to alarm within 5 seconds",
        "query": "A[] (System.alarm_active && not System.responded imply clock <= 5)",
        "pattern": "time_bounded_constraint",
        "operators": ["A[]", "imply", "&&", "not"],
        "keywords": ["system", "respond", "alarm", "within", "seconds", "5"],
        "explanation": "While alarm is active and not yet responded, clock cannot exceed 5 seconds. Clock resets when alarm triggers."
    },
    {
        "id": "c024",
        "nl": "Charging must complete within 30 minutes",
        "query": "A[] (Robot.charging imply clock <= 30)",
        "pattern": "time_bounded_constraint",
        "operators": ["A[]", "imply"],
        "keywords": ["charging", "complete", "within", "minutes", "30"],
        "explanation": "Robot cannot remain in charging state for more than 30 minutes. Clock resets when charging starts."
    },
    {
        "id": "c025",
        "nl": "Emergency stop must occur within 2 seconds of collision detection",
        "query": "A[] (collision_detected && not emergency_stopped imply clock <= 2)",
        "pattern": "time_bounded_constraint",
        "operators": ["A[]", "imply", "&&", "not"],
        "keywords": ["emergency", "stop", "within", "seconds", "collision", "detection", "2"],
        "explanation": "After collision detected, if not yet stopped, clock cannot exceed 2 seconds. Clock resets at collision detection."
    },
    {
        "id": "c026",
        "nl": "All paths guarantee the robot can reach the safe zone",
        "query": "A<> Robot.at_safe_zone",
        "pattern": "reachability",
        "operators": ["A<>"],
        "keywords": ["all", "paths", "guarantee", "robot", "reach", "safe", "zone"],
        "explanation": "Universal reachability - on every execution path, the robot can eventually reach the safe zone."
    },
    {
        "id": "c027",
        "nl": "Every execution path allows the drone to return home",
        "query": "A<> Drone.at_home",
        "pattern": "reachability",
        "operators": ["A<>"],
        "keywords": ["every", "execution", "path", "allows", "drone", "return", "home"],
        "explanation": "Universal reachability - all paths must be able to reach home state."
    },
    {
        "id": "c028",
        "nl": "System is guaranteed to be able to reach idle state",
        "query": "A<> System.idle",
        "pattern": "reachability",
        "operators": ["A<>"],
        "keywords": ["system", "guaranteed", "able", "reach", "idle", "state"],
        "explanation": "Universal reachability - from any state, it's always possible to eventually reach idle."
    }
]


# -----------------------------
# Neo4j seeder
# -----------------------------

def require_env(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise RuntimeError(f"Missing env var: {name}")
    return v


class KGSeeder:
    def __init__(self, uri: str, user: str, password: str) -> None:
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        self.driver.close()

    def create_constraints(self) -> None:
        cyphers = [
            "CREATE CONSTRAINT pattern_name IF NOT EXISTS FOR (p:Pattern) REQUIRE p.name IS UNIQUE",
            "CREATE CONSTRAINT operator_name IF NOT EXISTS FOR (o:Operator) REQUIRE o.name IS UNIQUE",
            "CREATE CONSTRAINT example_id IF NOT EXISTS FOR (e:Example) REQUIRE e.id IS UNIQUE",
        ]
        with self.driver.session() as session:
            for c in cyphers:
                session.run(c)

    def upsert_patterns(self, patterns: List[Dict[str, Any]]) -> None:
        cypher = """
        UNWIND $patterns AS p
        MERGE (n:Pattern {name: p.name})
        SET n.description = p.description,
            n.template = p.template,
            n.when_to_use = p.when_to_use,
            n.decision_rules = coalesce(p.decision_rules, []),
            n.updatedAt = datetime()
        """
        with self.driver.session() as session:
            session.run(cypher, patterns=patterns)

    def upsert_operators(self, operators: List[Dict[str, Any]]) -> None:
        cypher = """
        UNWIND $operators AS o
        MERGE (n:Operator {name: o.name})
        SET n.description = o.description,
            n.semantics = o.semantics,
            n.when_to_use = o.when_to_use,
            n.when_not_to_use = o.when_not_to_use,
            n.examples = coalesce(o.examples, []),
            n.updatedAt = datetime()
        """
        with self.driver.session() as session:
            session.run(cypher, operators=operators)

    def link_pattern_operators(self, mapping: Dict[str, List[str]]) -> None:
        cypher = """
        UNWIND $rows AS row
        MATCH (p:Pattern {name: row.pattern})
        MATCH (o:Operator {name: row.operator})
        MERGE (p)-[:USES_OPERATOR]->(o)
        """
        rows = [{"pattern": p, "operator": op} for p, ops in mapping.items() for op in ops]
        with self.driver.session() as session:
            session.run(cypher, rows=rows)

    def upsert_examples_and_links(self, examples: List[Dict[str, Any]]) -> None:
        """
        Writes Example nodes and links:
          (Example)-[:INSTANCE_OF]->(Pattern)
          (Example)-[:USES_OPERATOR]->(Operator)  [optional but useful]
        """
        cypher = """
        UNWIND $examples AS e

        MERGE (x:Example {id: e.id})
        SET x.nl = e.nl,
            x.uppaal = e.query,
            x.pattern = e.pattern,
            x.keywords = coalesce(e.keywords, []),
            x.explanation = coalesce(e.explanation, ""),
            x.updatedAt = datetime()

        WITH x, e
        MATCH (p:Pattern {name: e.pattern})
        MERGE (x)-[:INSTANCE_OF]->(p)

        WITH x, e
        UNWIND coalesce(e.operators, []) AS opName
        MATCH (o:Operator {name: opName})
        MERGE (x)-[:USES_OPERATOR]->(o)
        """
        with self.driver.session() as session:
            session.run(cypher, examples=examples)

    def stats(self) -> Dict[str, int]:
        cypher = """
        MATCH (p:Pattern)
        WITH count(p) AS patterns
        MATCH (o:Operator)
        WITH patterns, count(o) AS operators
        MATCH (e:Example)
        WITH patterns, operators, count(e) AS examples
        MATCH ()-[r:USES_OPERATOR]->()
        WITH patterns, operators, examples, count(r) AS uses_operator_rels
        MATCH ()-[r2:INSTANCE_OF]->()
        RETURN patterns, operators, examples, uses_operator_rels, count(r2) AS instance_of_rels
        """
        with self.driver.session() as session:
            rec = session.run(cypher).single()
            return dict(rec)


def main() -> None:
    uri = require_env("NEO4J_URI")
    user = require_env("NEO4J_USER")
    password = require_env("NEO4J_PASSWORD")

    parser = argparse.ArgumentParser(description="Seed Neo4j KG with patterns/operators/examples.")
    parser.add_argument("--seed-all", action="store_true", help="Constraints + patterns + operators + links + examples")
    parser.add_argument("--create-constraints", action="store_true", help="Create uniqueness constraints")
    parser.add_argument("--seed-patterns", action="store_true", help="Upsert Pattern nodes")
    parser.add_argument("--seed-operators", action="store_true", help="Upsert Operator nodes")
    parser.add_argument("--link-pattern-operators", action="store_true", help="Create Pattern->Operator links")
    parser.add_argument("--seed-examples", action="store_true", help="Upsert embedded Example nodes and links")
    args = parser.parse_args()

    seeder = KGSeeder(uri, user, password)
    try:
        if args.seed_all:
            seeder.create_constraints()
            seeder.upsert_patterns(PATTERNS)
            seeder.upsert_operators(OPERATORS)
            seeder.link_pattern_operators(PATTERN_TO_OPERATORS)
            seeder.upsert_examples_and_links(CONSTRAINT_EXAMPLES)
        else:
            if args.create_constraints:
                seeder.create_constraints()
            if args.seed_patterns:
                seeder.upsert_patterns(PATTERNS)
            if args.seed_operators:
                seeder.upsert_operators(OPERATORS)
            if args.link_pattern_operators:
                seeder.link_pattern_operators(PATTERN_TO_OPERATORS)
            if args.seed_examples:
                seeder.upsert_examples_and_links(CONSTRAINT_EXAMPLES)

        print(json.dumps({"status": "ok", "stats": seeder.stats()}, indent=2))

    finally:
        seeder.close()


if __name__ == "__main__":
    main()
