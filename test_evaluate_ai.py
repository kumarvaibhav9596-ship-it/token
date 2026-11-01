import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import evaluate_ai
import evaluate_practice


def sample_question():
    return {
        "id": "TEST-1",
        "question": "Can I return this item?",
        "answerable": "True",
        "acceptable_decisions": "answer",
        "expected_decision": "answer",
        "must_include": "eligible/allowed",
        "must_not_include": "five business days/5 business days",
        "gold_evidence": "RET-1; RET-2",
    }


class EvaluationScoringTests(unittest.TestCase):
    def test_prompting_baseline_score_does_not_fail_for_missing_evidence(self):
        checks = evaluate_ai.score_answer(
            "The return is allowed.", "answer", set(), sample_question(), rag=False)

        self.assertTrue(checks["question_correct"])
        self.assertNotIn("gold_evidence_overlap", checks)
        self.assertNotIn("gold_evidence_all_present", checks)

    def test_forbidden_alternatives_are_all_checked_in_both_evaluators(self):
        row = sample_question()
        self.assertFalse(evaluate_ai.facts_ok(
            "The return is allowed within five business days.", row))

        missing, present = evaluate_practice.check_facts(
            "The return is allowed within five business days.",
            row["must_include"], row["must_not_include"])
        self.assertEqual([], missing)
        self.assertEqual([("five business days", "5 business days")], present)

    def test_rag_reports_gold_hit_all_sources_and_recall_separately(self):
        row = sample_question()
        checks = evaluate_ai.score_answer(
            "The return is allowed.", "answer", {"RET-1"}, row, rag=True)

        self.assertTrue(checks["question_correct"])
        self.assertTrue(checks["gold_evidence_overlap"])
        self.assertFalse(checks["gold_evidence_all_present"])
        self.assertEqual(0.5, checks["gold_evidence_recall"])

    def test_rag_evidence_adds_matched_catalog_row_without_using_gold_labels(self):
        agent = evaluate_ai.SupportAgent(llm_enabled=False)
        policy = next(section for section in agent.sections if section.section_id == "WAR-2")
        question = "SKU ELE-0012 pe ZentroMart Assured hai kya? Claim kaise karein?"

        with patch.object(agent, "_retrieve_semantic", return_value=[policy]):
            evidence = evaluate_ai.retrieve_rag_evidence(agent, question)

        by_id = {item["id"]: item for item in evidence}
        self.assertIn("WAR-2", by_id)
        self.assertIn("products.csv:ELE-0012", by_id)
        self.assertIn('"assured": "True"', by_id["products.csv:ELE-0012"]["text"])


class EvaluationResumeTests(unittest.TestCase):
    def test_interrupted_comparison_checkpoints_and_resumes_without_repeating_step1(self):
        row = sample_question()
        section = SimpleNamespace(
            section_id="RET-1", title="Returns", doc_type="policy",
            status="current", effective_from="2026-01-01",
            searchable_text="Current returns policy.",
        )
        agent = SimpleNamespace(
            _retrieve_semantic=lambda _question: [section],
            _find_product=lambda _question, for_warranty=False: None,
            products=[],
        )
        baseline_result = {
            "answer": "The return is allowed.", "decision": "answer", "supported": True,
        }
        rag_result = {
            "answer": "The return is allowed. [RET-1]", "decision": "answer",
            "supported": True, "source_ids": ["RET-1"],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "evaluation.json"
            first_calls = []

            def fail_during_rag(_question, _evidence, prompting_only=False):
                if prompting_only:
                    first_calls.append("baseline")
                    return baseline_result
                first_calls.append("rag")
                raise RuntimeError("temporary provider failure with private details")

            with patch.object(evaluate_ai, "read_questions", return_value=[row]), \
                    patch.object(evaluate_ai, "SupportAgent", return_value=agent), \
                    patch.object(evaluate_ai.llm_adapter, "configured", return_value=True), \
                    patch.object(evaluate_ai.llm_adapter, "model_name", return_value="test-model"), \
                    patch.object(evaluate_ai.llm_adapter, "generate_grounded_answer",
                                 side_effect=fail_during_rag):
                with self.assertRaisesRegex(RuntimeError, "temporary provider failure"):
                    evaluate_ai.run("both", output)

            partial = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual("interrupted", partial["status"])
            self.assertEqual(0, partial["completed_cases"])
            self.assertEqual("RuntimeError", partial["error_type"])
            self.assertNotIn("private details", output.read_text(encoding="utf-8"))
            self.assertEqual(["baseline", "rag"], first_calls)

            resume_calls = []

            def complete_rag(_question, _evidence, prompting_only=False):
                if prompting_only:
                    resume_calls.append("baseline")
                    return baseline_result
                resume_calls.append("rag")
                return rag_result

            with patch.object(evaluate_ai, "read_questions", return_value=[row]), \
                    patch.object(evaluate_ai, "SupportAgent", return_value=agent), \
                    patch.object(evaluate_ai.llm_adapter, "configured", return_value=True), \
                    patch.object(evaluate_ai.llm_adapter, "model_name", return_value="test-model"), \
                    patch.object(evaluate_ai.llm_adapter, "generate_grounded_answer",
                                 side_effect=complete_rag):
                self.assertEqual(0, evaluate_ai.run("both", output, resume=True))

            final = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual("complete", final["status"])
            self.assertEqual(1, final["completed_cases"])
            self.assertEqual(["rag"], resume_calls)
            self.assertTrue(final["cases"][0]["step1_prompting_only"]["checks"]["question_correct"])
            self.assertTrue(final["cases"][0]["step2_rag"]["checks"]["gold_evidence_overlap"])


if __name__ == "__main__":
    unittest.main()
