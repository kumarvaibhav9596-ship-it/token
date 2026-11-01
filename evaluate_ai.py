"""Compare the API-backed prompting-only baseline with the same RAG practice set."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import llm_adapter
from agent import SupportAgent


ROOT = Path(__file__).resolve().parent
if not (ROOT / "practice").is_dir() and (ROOT.parent / "practice").is_dir():
    ROOT = ROOT.parent


def read_questions() -> list[dict[str, str]]:
    with (ROOT / "practice" / "questions.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def fact_groups(value: str) -> list[tuple[str, ...]]:
    return [tuple(option.strip().lower() for option in group.split("/") if option.strip())
            for group in value.split(";") if group.strip()]


def facts_ok(answer: str, row: dict[str, str]) -> bool:
    text = answer.lower()
    required = fact_groups(row.get("must_include", ""))
    forbidden = fact_groups(row.get("must_not_include", ""))
    has_required = all(any(option in text for option in group) for group in required)
    has_forbidden = any(any(option in text for option in group) for group in forbidden)
    return has_required and not has_forbidden


def expected_decision_ok(decision: str, row: dict[str, str]) -> bool:
    answerable = row.get("answerable", "True").casefold() == "true"
    if not answerable:
        return decision == "refuse" or decision in {
            value.strip() for value in row.get("acceptable_decisions", "").split(";") if value.strip()
        }
    accepted = {value.strip() for value in row.get("acceptable_decisions", "").split(";") if value.strip()}
    if not accepted:
        accepted = {row.get("expected_decision", "answer")}
    return decision in accepted


def score_answer(answer: str, decision: str, sources: set[str], row: dict[str, str], rag: bool) -> dict[str, bool | float]:
    gold = {value.strip() for value in row.get("gold_evidence", "").split(";") if value.strip()}
    decision_ok = expected_decision_ok(decision, row)
    facts_okay = facts_ok(answer, row)
    checks: dict[str, bool | float] = {
        "expected_decision": decision_ok,
        "required_and_forbidden_facts": facts_okay,
        "question_correct": decision_ok and facts_okay,
    }
    if rag:
        overlap = gold & sources
        checks.update({
            "gold_evidence_overlap": bool(overlap) if gold else not sources,
            "gold_evidence_all_present": gold.issubset(sources) if gold else not sources,
            "gold_evidence_recall": len(overlap) / len(gold) if gold else (1.0 if not sources else 0.0),
        })
    return checks


def _case_complete(item: dict[str, Any], mode: str) -> bool:
    required = {
        "baseline": ("step1_prompting_only",),
        "rag": ("step2_rag",),
        "both": ("step1_prompting_only", "step2_rag"),
    }[mode]
    return all(field in item for field in required)


def _save_results(output_path: Path, results: dict[str, Any]) -> None:
    """Atomically checkpoint results so a failed live run can resume safely."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(output_path.name + ".tmp")
    temp_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp_path, output_path)


def _new_results(mode: str, questions: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "mode": mode,
        "status": "running",
        "model": llm_adapter.model_name(),
        "embedding_model": llm_adapter.embeddings_model_name(),
        "total_questions": len(questions),
        "completed_cases": 0,
        "cases": [],
    }


def _resume_results(output_path: Path, mode: str, questions: list[dict[str, str]]) -> dict[str, Any]:
    if not output_path.is_file():
        raise RuntimeError("No saved evaluation exists at that output path to resume.")
    try:
        results = json.loads(output_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("The saved evaluation file is not valid JSON and cannot be resumed.") from error
    if not isinstance(results, dict):
        raise RuntimeError("The saved evaluation has an invalid structure and cannot be resumed.")
    expected_questions = {row["id"]: row["question"] for row in questions}
    saved_cases = results.get("cases")
    if (results.get("schema_version") != 1 or results.get("mode") != mode
            or results.get("model") != llm_adapter.model_name()
            or results.get("embedding_model") != llm_adapter.embeddings_model_name()
            or not isinstance(saved_cases, list)):
        raise RuntimeError("The saved evaluation uses a different mode, model, or output schema; start a fresh run instead.")
    seen: set[str] = set()
    for item in saved_cases:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str) or item.get("id") not in expected_questions:
            raise RuntimeError("The saved evaluation contains an unknown case and cannot be resumed.")
        if item.get("question") != expected_questions[item["id"]] or item["id"] in seen:
            raise RuntimeError("The saved questions do not match the current practice set.")
        seen.add(item["id"])
    order = {row["id"]: index for index, row in enumerate(questions)}
    results["cases"].sort(key=lambda item: order[item["id"]])
    results["status"] = "running"
    results.pop("error_type", None)
    return results


def retrieve_rag_evidence(agent: SupportAgent, question: str) -> list[dict[str, str]]:
    """Retrieve current policy chunks and a catalog row when the text identifies a product."""
    sections = agent._retrieve_semantic(question)
    evidence = [{"id": section.section_id, "title": section.title,
                 "type": section.doc_type, "status": section.status,
                 "effective_from": section.effective_from, "text": section.searchable_text}
                for section in sections]

    lowered = question.casefold()
    warranty_question = any(term in lowered for term in ("warranty", "assured", "वारंटी", "अश्योर्ड"))
    product = agent._find_product(question, for_warranty=warranty_question)
    if product:
        products_by_sku = {row["sku"]: row for row in agent.products}
        matched_skus = product.get("_matched_skus", [product.get("sku", "")])
        fields = ("sku", "title", "brand", "return_category", "price_inr",
                  "fulfilled_by", "assured", "warranty_months")
        for sku in matched_skus:
            row = products_by_sku.get(sku)
            if not row:
                continue
            catalog = {field: row.get(field, "") for field in fields}
            evidence.append({
                "id": f"products.csv:{sku}",
                "title": row.get("title", "Product catalog record"),
                "type": "catalog", "status": "current", "effective_from": "",
                "text": json.dumps(catalog, ensure_ascii=False, sort_keys=True),
            })
    return evidence


def run(mode: str, output_path: Path, resume: bool = False) -> int:
    if not llm_adapter.configured():
        key_name = "GEMINI_API_KEY" if llm_adapter.provider_name() == "gemini" else "OPENAI_API_KEY"
        raise RuntimeError(f"{key_name} is not available in this terminal. Set it without sharing it in chat, then rerun.")
    questions = read_questions()
    if mode not in ("both", "baseline", "rag"):
        raise ValueError("mode must be 'both', 'baseline', or 'rag'.")
    results = _resume_results(output_path, mode, questions) if resume else _new_results(mode, questions)
    existing = {item["id"]: item for item in results["cases"]}
    agent = SupportAgent(llm_enabled=True) if mode in ("both", "rag") else None
    _save_results(output_path, results)

    def checkpoint() -> None:
        results["completed_cases"] = sum(_case_complete(item, mode) for item in results["cases"])
        _save_results(output_path, results)

    try:
        for row in questions:
            item = existing.get(row["id"])
            if item is None:
                item = {"id": row["id"], "question": row["question"]}
                results["cases"].append(item)
                existing[row["id"]] = item
                checkpoint()
            if mode in ("both", "baseline") and "step1_prompting_only" not in item:
                baseline = llm_adapter.generate_grounded_answer(row["question"], [], prompting_only=True)
                item["step1_prompting_only"] = {
                    "answer": baseline["answer"],
                    "decision": baseline["decision"], "supported": baseline["supported"],
                    "engine": "openai_api",
                    "checks": score_answer(baseline["answer"], baseline["decision"], set(), row, rag=False),
                }
                checkpoint()
            if mode in ("both", "rag") and "step2_rag" not in item:
                assert agent is not None
                evidence = retrieve_rag_evidence(agent, row["question"])
                if evidence:
                    rag = llm_adapter.generate_grounded_answer(row["question"], evidence)
                    engine = "openai_api"
                else:
                    rag = {
                        "answer": "I couldn't confirm that from the supplied policies. I can connect you to a human support colleague if you'd like.",
                        "decision": "refuse", "supported": False, "source_ids": [],
                    }
                    engine = "deterministic_abstention_no_evidence"
                item["step2_rag"] = {
                    "answer": rag["answer"], "decision": rag["decision"], "supported": rag["supported"],
                    "sources": rag["source_ids"], "engine": engine,
                    "checks": score_answer(rag["answer"], rag["decision"], set(rag["source_ids"]), row, rag=True),
                }
                checkpoint()
    except Exception as error:
        results["status"] = "interrupted"
        # Keep provider/request details out of the local report; the completed case data is preserved.
        results["error_type"] = type(error).__name__
        checkpoint()
        raise

    results["status"] = "complete"
    checkpoint()
    for step, field in (("Step 1 prompting-only", "step1_prompting_only"), ("Step 2 RAG", "step2_rag")):
        if mode == "baseline" and field != "step1_prompting_only":
            continue
        if mode == "rag" and field != "step2_rag":
            continue
        cases = [item[field] for item in results["cases"]]
        overall = sum(case["checks"]["question_correct"] for case in cases)
        decisions = sum(case["checks"]["expected_decision"] for case in cases)
        facts = sum(case["checks"]["required_and_forbidden_facts"] for case in cases)
        if field == "step2_rag":
            evidence = sum(case["checks"]["gold_evidence_overlap"] for case in cases)
            exact = sum(case["checks"]["gold_evidence_all_present"] for case in cases)
            recall = sum(case["checks"]["gold_evidence_recall"] for case in cases) / len(cases)
            print(f"{step}: answer correctness {overall}/{len(cases)}; decisions {decisions}/{len(cases)}; fact checks {facts}/{len(cases)}; gold-source hit {evidence}/{len(cases)}; exact gold-source coverage {exact}/{len(cases)}; mean gold-source recall {recall:.1%}")
        else:
            print(f"{step}: answer correctness {overall}/{len(cases)}; decisions {decisions}/{len(cases)}; fact checks {facts}/{len(cases)}; source evidence not applicable")
    print(f"Saved evaluation answers, evidence, and checks to {output_path.resolve()}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate GPT-4.1 prompting-only and RAG answers on the 52 supplied questions.")
    parser.add_argument("--mode", choices=("both", "baseline", "rag"), default="both")
    parser.add_argument("--output", type=Path, default=ROOT / "ai_eval_results.json")
    parser.add_argument("--resume", action="store_true", help="Resume a prior interrupted run from its saved output file.")
    args = parser.parse_args()
    try:
        return run(args.mode, args.output, resume=args.resume)
    except (RuntimeError, ValueError, llm_adapter.LLMError) as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
