"""
eval_output.py
--------------
Evaluates final answer quality using OutputEvaluator (LLM-as-a-Judge).

The LLM judge (Claude 4 via Amazon Bedrock) scores each response against
a rubric you define. Use this as your starting point — it's the simplest
evaluator and a good baseline before adding trajectory or trace-based evals.

Run:
    python eval_output.py
"""

from strands import Agent
from strands_evals import Case, Experiment
from strands_evals.evaluators import OutputEvaluator


# ── 1. Task function ──────────────────────────────────────────────────────────
# This is the function the SDK calls for each test case.
# It runs your agent and returns the string response.
# callback_handler=None suppresses console output for cleaner eval runs.

def run_agent(case: Case) -> str:
    agent = Agent(
        system_prompt="You are a helpful assistant. Be accurate and concise.",
        callback_handler=None
    )
    response = agent(case.input)
    return str(response)


# ── 2. Test dataset ───────────────────────────────────────────────────────────
# Each Case has: name, input, expected_output, and optional metadata.
# Use metadata categories to slice results by type later.

test_cases = [
    Case[str, str](
        name="factual-1",
        input="What is the capital of France?",
        expected_output="The capital of France is Paris.",
        metadata={"category": "factual"}
    ),
    Case[str, str](
        name="reasoning-1",
        input="If 5 machines make 5 widgets in 5 minutes, how long for 100 machines to make 100 widgets?",
        expected_output="5 minutes. Each machine makes 1 widget per 5 minutes regardless of total count.",
        metadata={"category": "reasoning"}
    ),
    Case[str, str](
        name="math-1",
        input="What is 15% of 230?",
        expected_output="34.5",
        metadata={"category": "math"}
    ),
    Case[str, str](
        name="explanation-1",
        input="Explain what hallucination means in the context of LLMs.",
        expected_output="Hallucination in LLMs refers to when the model generates confident-sounding information that is factually incorrect or not grounded in its input or training data.",
        metadata={"category": "explanation"}
    ),
]


# ── 3. Evaluator ──────────────────────────────────────────────────────────────
# The rubric tells the LLM judge what to look for.
# Be specific — vague rubrics produce inconsistent, unreliable scores.

evaluator = OutputEvaluator(
    rubric="""
    Evaluate the agent's response on three criteria:
    1. Accuracy     — Is the answer factually correct?
    2. Completeness — Does it fully address the question?
    3. Clarity      — Is it easy to understand?

    Score 1.0 if all three criteria are met excellently.
    Score 0.5 if some criteria are partially met.
    Score 0.0 if the response is incorrect, incomplete, or confusing.
    """,
    include_inputs=True  # pass the original question to the judge for context
)


# ── 4. Run experiment ─────────────────────────────────────────────────────────

def main():
    print("Running OutputEvaluator...\n")

    experiment = Experiment[str, str](cases=test_cases, evaluators=[evaluator])
    reports = experiment.run_evaluations(run_agent)

    # Pretty-print results to console
    reports[0].run_display()

    # Save to ./experiment_files/output_evaluation.json for later analysis
    experiment.to_file("output_evaluation")
    print("\nExperiment saved to ./experiment_files/output_evaluation.json")

    # Print programmatic summary
    summary = reports[0].get_summary()
    print(f"\nPass rate:     {summary['pass_rate']:.0%}")
    print(f"Average score: {summary['average_score']:.2f}")

    # Per-case breakdown
    print("\nPer-case results:")
    for result in reports[0].case_results:
        status = "✅" if result.evaluation_output.test_pass else "❌"
        print(f"  {status} {result.case.name}: {result.evaluation_output.score:.2f}")
        print(f"     {result.evaluation_output.reason}")

    return reports[0]


if __name__ == "__main__":
    main()
