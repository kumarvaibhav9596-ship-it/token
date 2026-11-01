"""Evaluate decisions, evidence, required facts, and tool names on supplied practice data."""
from __future__ import annotations

import csv
import re
import tempfile
from pathlib import Path
from unittest.mock import patch

from agent import SupportAgent
from tools import tools as mock_tools


ROOT = Path(__file__).resolve().parent
if not (ROOT / "practice").is_dir() and (ROOT.parent / "practice").is_dir():
    ROOT = ROOT.parent


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def fact_groups(value: str) -> list[tuple[str, ...]]:
    return [tuple(part.strip().lower() for part in group.split("/") if part.strip())
            for group in value.split(";") if group.strip()]


def check_facts(answer: str, required: str, forbidden: str) -> tuple[list[tuple[str, ...]], list[tuple[str, ...]]]:
    text = answer.lower()
    missing = [options for options in fact_groups(required)
               if not any(option in text for option in options)]
    present = [options for options in fact_groups(forbidden)
               if any(option in text for option in options)]
    return missing, present


def expected_tool_calls(value: str) -> list[tuple]:
    calls = []
    for part in filter(None, (item.strip() for item in value.split(";"))):
        lookup = re.fullmatch(
            r"lookup_order\(order_id=(\d+)(?:,\s*verification_value=(\d+))?\)", part)
        if lookup:
            calls.append(("lookup_order", lookup.group(1), lookup.group(2)))
        elif part == "create_ticket()":
            calls.append(("create_ticket",))
        else:
            raise ValueError(f"Unsupported expected tool call: {part}")
    return calls


def check_actual_tool_calls(expected: list[tuple], actual: list[tuple], row: dict[str, str]) -> bool:
    if len(expected) != len(actual):
        return False
    for expected_call, actual_call in zip(expected, actual):
        if expected_call[0] != actual_call[0]:
            return False
        if expected_call[0] == "lookup_order":
            if expected_call[1:] != actual_call[1:]:
                return False
        elif expected_call[0] == "create_ticket":
            _name, summary, order_id, reason, intent = actual_call
            if not summary or not reason or intent not in ("human_handover", "damaged_item"):
                return False
            if row["order_id"] and order_id != row["order_id"]:
                return False
            if "Detected intent:" not in summary or "Checks made:" not in summary or "Escalation reason:" not in summary:
                return False
    return True


def main() -> int:
    questions = read_rows(ROOT / "practice" / "questions.csv")
    conversations = read_rows(ROOT / "practice" / "conversations.csv")
    failures: list[str] = []
    question_decisions = question_evidence = question_required = question_forbidden = 0
    question_required_total = question_forbidden_total = 0
    conversation_decisions = conversation_tools = conversation_required = conversation_forbidden = 0
    conversation_required_total = conversation_forbidden_total = 0

    with tempfile.TemporaryDirectory(prefix="zentromart-eval-") as temp_dir:
        ticket_file = str(Path(temp_dir) / "tickets_created.csv")
        actual_calls: list[tuple] = []
        original_lookup = mock_tools.lookup_order
        original_create = mock_tools.create_ticket

        def tracked_lookup(order_id, verification_value=None):
            actual_calls.append(("lookup_order", str(order_id),
                                 None if verification_value is None else str(verification_value)))
            return original_lookup(order_id, verification_value)

        def tracked_create(summary, order_id=None, reason=None, intent=None):
            actual_calls.append(("create_ticket", summary, order_id, reason, intent))
            return original_create(summary, order_id=order_id, reason=reason, intent=intent)

        with patch.object(mock_tools, "_TICKETS", ticket_file), \
             patch.object(mock_tools, "lookup_order", side_effect=tracked_lookup), \
             patch.object(mock_tools, "create_ticket", side_effect=tracked_create):
            agent = SupportAgent(llm_enabled=False)
            for row in questions:
                result = agent.respond(f"practice-{row['id']}", row["question"])
                accepted = {value.strip() for value in row["acceptable_decisions"].split(";") if value.strip()}
                if not accepted:
                    accepted = {row["expected_decision"]}
                decision_ok = result["decision"] in accepted or (
                    row["answerable"] == "False" and result["decision"] == "refuse")
                question_decisions += decision_ok

                gold = {value.strip() for value in row["gold_evidence"].split(";") if value.strip()}
                actual = {source["id"] for source in result["sources"]}
                evidence_ok = bool(gold & actual) if gold else not actual
                question_evidence += evidence_ok

                missing, present = check_facts(result["answer"], row["must_include"], row["must_not_include"])
                required_count = len(fact_groups(row["must_include"]))
                forbidden_count = len(fact_groups(row["must_not_include"]))
                question_required += required_count - len(missing)
                question_required_total += required_count
                question_forbidden += forbidden_count - len(present)
                question_forbidden_total += forbidden_count
                if not (decision_ok and evidence_ok and not missing and not present):
                    failures.append(f"question {row['id']}")

            for row in conversations:
                calls_before = len(actual_calls)
                result = agent.respond(row["conversation_id"], row["customer_message"])
                decision_ok = result["decision"] == row["expected_decision"]
                conversation_decisions += decision_ok
                expected = expected_tool_calls(row["expected_tool_calls"])
                actual = actual_calls[calls_before:]
                tool_ok = check_actual_tool_calls(expected, actual, row)
                conversation_tools += tool_ok

                missing, present = check_facts(result["answer"], row["must_include"], row["must_not_include"])
                required_count = len(fact_groups(row["must_include"]))
                forbidden_count = len(fact_groups(row["must_not_include"]))
                conversation_required += required_count - len(missing)
                conversation_required_total += required_count
                conversation_forbidden += forbidden_count - len(present)
                conversation_forbidden_total += forbidden_count
                if not (decision_ok and tool_ok and not missing and not present):
                    failures.append(f"conversation {row['conversation_id']} turn {row['turn']}")

    print(f"Questions ({len(questions)}): decisions {question_decisions}/{len(questions)}, "
          f"gold evidence overlap {question_evidence}/{len(questions)}, "
          f"required facts {question_required}/{question_required_total}, "
          f"forbidden facts {question_forbidden}/{question_forbidden_total}.")
    print(f"Conversations ({len(conversations)} turns): decisions {conversation_decisions}/{len(conversations)}, "
          f"actual tool calls and arguments {conversation_tools}/{len(conversations)}, "
          f"required facts {conversation_required}/{conversation_required_total}, "
          f"forbidden facts {conversation_forbidden}/{conversation_forbidden_total}.")
    print("Order IDs and submitted verification values are compared with the actual mock-tool arguments.")
    if failures:
        print("Needs review: " + ", ".join(failures))
        return 1
    print("PASS: all configured practice checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
