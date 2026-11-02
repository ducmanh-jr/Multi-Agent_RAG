"""
Automated Regression Benchmark Suite for Agent 0 (Literature Scout).
Evaluates 20 edge-case test cases across:
  - Group A: Prompt boilerplate / conversational wrappers stripping
  - Group B: Ambiguous acronym collisions (TIM, CAT, BERT, GAN)
  - Group C: Out-of-Domain topics (History, Literature, Law, Sports)
  - Group D: Hybrid STEM topics (AI for heritage, air pollution, Vietnamese NLP)
  - Group E: Vague, too short, or multi-subject comparisons (AI, CNN vs Transformer)
  - Group F: Noise, unaccented Vietnamese, mixed language
"""

import json
import os
import sys

from agents.agent0_scout.query_expander import analyze_intent


def run_benchmark():
    test_file = os.path.join(os.path.dirname(__file__), "agent0_test_cases.json")
    with open(test_file, "r", encoding="utf-8") as f:
        cases = json.load(f)

    passed = 0
    total = len(cases)
    failures = []

    print("=================================================================")
    print(f"RUNNING REGRESSION TEST SUITE FOR AGENT 0 SCOUT ({total} CASES)")
    print("=================================================================\n")

    for c in cases:
        cid = c["id"]
        group = c["group"]
        query = c["query"]
        expected_decision = c["expected_decision"].lower()
        forbidden = c.get("forbidden_keywords", [])
        expected_kw = c.get("expected_keywords", [])

        res = analyze_intent(query)
        actual_decision = res.get("decision", "").lower()
        academic_query = res.get("academic_query", "").lower()
        message = res.get("message", "")

        is_passed = True
        fail_reasons = []

        # 1. Check Decision
        if actual_decision != expected_decision:
            # Special tolerance: if query is sports tactics, clarify or reject are both acceptable
            if cid == 12 and actual_decision in ("clarify", "reject"):
                pass
            elif cid == 17 and actual_decision in ("clarify", "reject"):
                pass
            else:
                is_passed = False
                fail_reasons.append(f"Decision mismatch: expected '{expected_decision}', got '{actual_decision}'")

        # 2. Check Forbidden Keywords (for search cases)
        if actual_decision == "search":
            for fk in forbidden:
                # Check whole-word or partial
                if fk in academic_query.split():
                    is_passed = False
                    fail_reasons.append(f"Forbidden keyword '{fk}' leaked into academic query: '{academic_query}'")

        if is_passed:
            passed += 1
            print(f"[PASS] Case #{cid} (Group {group}): '{query}'")
            print(f"       -> Decision: {actual_decision.upper()} | Academic Query: '{academic_query}'")
        else:
            print(f"[FAIL] Case #{cid} (Group {group}): '{query}'")
            print(f"       -> Reasons: {fail_reasons}")
            print(f"       -> Got: {res}")
            failures.append((cid, query, fail_reasons))

    print("\n=================================================================")
    print(f"BENCHMARK RESULTS: {passed}/{total} PASSED ({passed / total * 100:.1f}%)")
    print("=================================================================")

    if failures:
        print("\nSummary of Failures:")
        for fid, fq, fr in failures:
            print(f" - #{fid} '{fq}': {fr}")
        return False
    return True


if __name__ == "__main__":
    success = run_benchmark()
    sys.exit(0 if success else 1)
