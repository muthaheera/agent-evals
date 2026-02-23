"""
eval_gate.py
------------
CI/CD deployment gate — blocks deployment when eval pass rate is too low.

Integrate this into GitHub Actions, Jenkins, or any CI pipeline.
A non-zero SystemExit fails the pipeline and prevents the deploy.

Example GitHub Actions usage:
    - name: Run agent evals
      run: python eval_gate.py
    # Pipeline fails automatically if evals don't pass

Run:
    python eval_gate.py
"""

from strands import Agent
from strands_evals import Case, Experiment
from strands_evals.evaluators import OutputEvaluator, TrajectoryEvaluator
from strands_evals.extractors import tools_use_extractor
from strands_tools import calculator, current_time


# ── Configuration ─────────────────────────────────────────────────────────────
# Tune these thresholds for your application's requirements.
# Higher is stricter — start at 0.80 and raise as your agent matures.

PASS_THRESHOLD = 0.80       # minimum pass rate to allow deployment
SCORE_THRESHOLD = 0.75      # minimum average score to allow deployment


# ── Task function ─────────────────────────────────────────────────────────────

def run_agent(case: Case) -> dict:
    agent = Agent(
        tools=[calculator, current_time],
        system_prompt="You are a helpful assistant. Use tools when appropriate.",
        callback_handler=None
    )
    response = agent(case.input)
    trajectory = tools_use_extractor.extract_agent_tools_used_from_messages(
        agent.messages
    )
    return {"output": str(response), "trajectory": trajectory}


# ── Test dataset ──────────────────────────────────────────────────────────────
# These are your "smoke tests" — a small, curated set of critical cases
# that must pass before every deployment.

SMOKE_TEST_CASES = [
    Case[str, str](
        name="gate-output-1",
        input="What is the capital of France?",
        expected_output="Paris",
        expected_trajectory=[],
        metadata={"category": "factual", "criticality": "high"}
    ),
    Case[str, str](
        name="gate-output-2",
        input="What is 15% of 230?",
        expected_output="34.5",
        expected_trajectory=["calculator"],
        metadata={"category": "math", "criticality": "high"}
    ),
    Case[str, str](
        name="gate-output-3",
        input="What time is it?",
        expected_output="",  # dynamic — judge will assess based on whether it actually checked
        expected_trajectory=["current_time"],
        metadata={"category": "time", "criticality": "medium"}
    ),
    Case[str, str](
        name="gate-output-4",
        input="What is 25 * 48?",
        expected_output="1200",
        expected_trajectory=["calculator"],
        metadata={"category": "math", "criticality": "high"}
    ),
]


# ── Deployment gate logic ─────────────────────────────────────────────────────

def run_deployment_gate() -> bool:
    """
    Run eval suite and return True if deployment is safe, False otherwise.
    Raises SystemExit(1) if evals fail — this fails the CI pipeline.
    """
    print("=" * 60)
    print("🔍 RUNNING PRE-DEPLOYMENT EVAL GATE")
    print("=" * 60)

    # Use a lightweight evaluator for the gate — speed matters in CI
    evaluator = OutputEvaluator(
        rubric="""
        Score 1.0 if the response is accurate and complete.
        Score 0.5 if partially correct.
        Score 0.0 if incorrect or the agent failed to respond.
        """,
        include_inputs=True
    )

    experiment = Experiment[str, str](
        cases=SMOKE_TEST_CASES,
        evaluators=[evaluator]
    )
    reports = experiment.run_evaluations(run_agent)
    report = reports[0]

    # Print individual results
    print("\nTest case results:")
    for result in report.case_results:
        status = "✅" if result.evaluation_output.test_pass else "❌"
        criticality = result.case.metadata.get("criticality", "medium")
        print(f"  {status} [{criticality.upper()}] {result.case.name}: "
              f"score={result.evaluation_output.score:.2f}")
        if not result.evaluation_output.test_pass:
            print(f"       Reason: {result.evaluation_output.reason}")

    # Get summary
    summary = report.get_summary()
    pass_rate = summary['pass_rate']
    avg_score = summary['average_score']

    print(f"\nResults:")
    print(f"  Pass rate:     {pass_rate:.0%}  (threshold: {PASS_THRESHOLD:.0%})")
    print(f"  Average score: {avg_score:.2f}  (threshold: {SCORE_THRESHOLD:.2f})")

    # Save for audit trail
    experiment.to_file(f"gate_run_{int(__import__('time').time())}")

    # Gate decision
    print("\n" + "=" * 60)
    passed = pass_rate >= PASS_THRESHOLD and avg_score >= SCORE_THRESHOLD

    if passed:
        print("✅ EVAL GATE PASSED — safe to deploy")
        print("=" * 60)
        return True
    else:
        reasons = []
        if pass_rate < PASS_THRESHOLD:
            reasons.append(f"pass rate {pass_rate:.0%} < {PASS_THRESHOLD:.0%}")
        if avg_score < SCORE_THRESHOLD:
            reasons.append(f"avg score {avg_score:.2f} < {SCORE_THRESHOLD:.2f}")

        print(f"❌ EVAL GATE FAILED — {', '.join(reasons)}")
        print("   Review failing cases above before deploying.")
        print("=" * 60)
        raise SystemExit(1)  # non-zero exit code fails the CI pipeline


if __name__ == "__main__":
    run_deployment_gate()
