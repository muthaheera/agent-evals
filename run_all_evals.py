"""
run_all_evals.py
----------------
Full evaluation suite orchestrator — runs all evaluators and prints a summary.

Use this to get a complete picture of your agent's quality across all
evaluation dimensions before a release.

Run:
    python run_all_evals.py
"""

import sys
from eval_output import main as run_output_eval
from eval_trajectory import main as run_trajectory_eval
from eval_helpfulness import main as run_helpfulness_eval
from custom_safety_evaluator import main as run_safety_eval


PASS_THRESHOLD = 0.80


def run_all():
    print("=" * 60)
    print("🚀 RUNNING FULL AGENT EVALUATION SUITE")
    print("=" * 60)

    results = {}

    # Run each evaluator
    eval_runners = {
        "OutputEvaluator":      run_output_eval,
        "TrajectoryEvaluator":  run_trajectory_eval,
        "HelpfulnessEvaluator": run_helpfulness_eval,
        "SafetyEvaluator":      run_safety_eval,
    }

    for name, runner in eval_runners.items():
        print(f"\n{'─' * 60}")
        print(f"Running {name}...")
        print(f"{'─' * 60}")
        try:
            report = runner()
            summary = report.get_summary()
            results[name] = {
                "pass_rate": summary['pass_rate'],
                "avg_score": summary['average_score'],
                "passed": summary['pass_rate'] >= PASS_THRESHOLD
            }
        except Exception as e:
            print(f"❌ {name} failed with error: {e}")
            results[name] = {"pass_rate": 0.0, "avg_score": 0.0, "passed": False}

    # Final summary table
    print("\n" + "=" * 60)
    print("📊 FULL SUITE SUMMARY")
    print("=" * 60)
    print(f"{'Evaluator':<25} {'Pass Rate':>10} {'Avg Score':>10} {'Status':>8}")
    print("─" * 60)

    all_passed = True
    for name, r in results.items():
        status = "✅ PASS" if r["passed"] else "❌ FAIL"
        if not r["passed"]:
            all_passed = False
        print(f"{name:<25} {r['pass_rate']:>9.0%} {r['avg_score']:>10.2f} {status:>8}")

    print("─" * 60)
    print(f"\nOverall: {'✅ ALL EVALUATORS PASSED' if all_passed else '❌ SOME EVALUATORS FAILED'}")

    if not all_passed:
        print("Review failing evaluators above before deploying.")
        sys.exit(1)


if __name__ == "__main__":
    run_all()
