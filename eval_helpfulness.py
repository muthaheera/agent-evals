"""
eval_helpfulness.py
-------------------
Evaluates agent helpfulness using full OpenTelemetry execution traces.

The HelpfulnessEvaluator captures every internal span of the agent's
execution — reasoning steps, tool calls, intermediate outputs — and
scores on a 7-level helpfulness scale. This is the most comprehensive
evaluator and closest to what a human reviewer would assess.

CRITICAL: Always pass trace_attributes with session IDs. Without them,
spans from different test cases contaminate each other in the memory
exporter and your scores will be wrong.

Run:
    python eval_helpfulness.py
"""

from strands import Agent
from strands_evals import Case, Experiment
from strands_evals.evaluators import HelpfulnessEvaluator
from strands_evals.telemetry import StrandsEvalsTelemetry
from strands_evals.mappers import StrandsInMemorySessionMapper
from strands_tools import calculator


# ── 1. Set up in-memory OpenTelemetry exporter ────────────────────────────────
# This captures all agent spans during evaluation runs.
# Must be initialized before running any agents.

telemetry = StrandsEvalsTelemetry().setup_in_memory_exporter()


# ── 2. Task function ──────────────────────────────────────────────────────────
# Three key steps:
#   a) Clear the exporter before each test case (prevents contamination)
#   b) Pass trace_attributes with the case's session_id
#   c) Map finished spans to a session object for the evaluator

def run_agent_with_trace(case: Case) -> dict:
    # ALWAYS clear before each test case
    telemetry.in_memory_exporter.clear()

    agent = Agent(
        tools=[calculator],
        # REQUIRED: isolates spans per test case in the memory exporter
        trace_attributes={
            "gen_ai.conversation.id": case.session_id,
            "session.id": case.session_id
        },
        callback_handler=None
    )
    response = agent(case.input)

    # Collect all spans captured during this agent run
    finished_spans = telemetry.in_memory_exporter.get_finished_spans()

    # Map spans to a structured session object
    mapper = StrandsInMemorySessionMapper()
    session = mapper.map_to_session(finished_spans, session_id=case.session_id)

    return {"output": str(response), "trajectory": session}


# ── 3. Test dataset ───────────────────────────────────────────────────────────
# No expected_output needed — HelpfulnessEvaluator assesses quality directly.
# Include edge cases to surface system prompt weaknesses.

test_cases = [
    Case[str, str](
        name="practical-calculation",
        input="Calculate the 18% tip on a $45.67 restaurant bill.",
        metadata={"category": "practical"}
    ),
    Case[str, str](
        name="educational-stepbystep",
        input="What does 2 to the power of 8 equal? Show the calculation step by step.",
        metadata={"category": "educational"}
    ),
    Case[str, str](
        name="ambiguous-request",
        input="Help me with numbers.",
        # Tests whether the agent asks for clarification on vague input
        # vs making assumptions and proceeding
        metadata={"category": "edge_case"}
    ),
    Case[str, str](
        name="multi-step-problem",
        input="I have $1,200 to invest. If I split it equally between 3 funds and each grows by 7%, how much do I have in total after the growth?",
        metadata={"category": "multi_step"}
    ),
]


# ── 4. Run experiment ─────────────────────────────────────────────────────────

def main():
    print("Running HelpfulnessEvaluator (trace-based)...\n")
    print("Note: This evaluator captures OpenTelemetry spans — it may be")
    print("slower than OutputEvaluator due to trace collection overhead.\n")

    evaluator = HelpfulnessEvaluator()

    experiment = Experiment[str, str](cases=test_cases, evaluators=[evaluator])
    reports = experiment.run_evaluations(run_agent_with_trace)

    reports[0].run_display()
    experiment.to_file("helpfulness_evaluation")
    print("\nExperiment saved to ./experiment_files/helpfulness_evaluation.json")

    summary = reports[0].get_summary()
    print(f"\nPass rate:     {summary['pass_rate']:.0%}")
    print(f"Average score: {summary['average_score']:.2f}")

    print("\nPer-case results:")
    for result in reports[0].case_results:
        status = "✅" if result.evaluation_output.test_pass else "❌"
        print(f"  {status} {result.case.name}: {result.evaluation_output.score:.2f}")
        print(f"     {result.evaluation_output.reason}")

    return reports[0]


if __name__ == "__main__":
    main()
