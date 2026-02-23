"""
eval_trajectory.py
------------------
Evaluates tool selection and call sequence using TrajectoryEvaluator.

This goes beyond checking the final answer — it verifies HOW the agent
got there: which tools it called, in what order, and whether it called
unnecessary tools or missed required ones.

Key: always use tools_use_extractor (not manual message parsing) to extract
trajectories. It prevents context overflow on long agent conversations.

Run:
    python eval_trajectory.py
"""

from strands import Agent
from strands_evals import Case, Experiment
from strands_evals.evaluators import TrajectoryEvaluator
from strands_evals.extractors import tools_use_extractor
from strands_tools import calculator, current_time


# ── 1. Task function ──────────────────────────────────────────────────────────
# Must return a dict with both "output" (str) and "trajectory" (tool list).
# The trajectory is extracted from agent.messages after the run completes.

def run_agent_with_tools(case: Case) -> dict:
    agent = Agent(
        tools=[calculator, current_time],
        system_prompt="You are a helpful assistant. Always use tools when appropriate.",
        callback_handler=None
    )
    response = agent(case.input)

    # Extract the actual sequence of tool names the agent called
    trajectory = tools_use_extractor.extract_agent_tools_used_from_messages(
        agent.messages
    )

    return {"output": str(response), "trajectory": trajectory}


# ── 2. Test dataset ───────────────────────────────────────────────────────────
# expected_trajectory is the list of tool names the agent SHOULD call.
# The TrajectoryEvaluator compares actual vs expected using the rubric.

test_cases = [
    Case[str, str](
        name="single-tool-math",
        input="What is 15% of 230?",
        expected_trajectory=["calculator"],       # only calculator needed
        metadata={"category": "math"}
    ),
    Case[str, str](
        name="single-tool-time",
        input="What time is it right now?",
        expected_trajectory=["current_time"],     # should NOT also call calculator
        metadata={"category": "time"}
    ),
    Case[str, str](
        name="multi-tool",
        input="What time is it, and what is 25 * 48?",
        expected_trajectory=["current_time", "calculator"],
        metadata={"category": "multi_tool"}
    ),
    Case[str, str](
        name="no-tool-needed",
        input="What is the capital of France?",
        expected_trajectory=[],                   # should answer from knowledge — no tools
        metadata={"category": "no_tool"}
    ),
    Case[str, str](
        name="tool-bias-test",
        input="What is the color of the sky?",
        expected_trajectory=[],                   # tests whether agent over-uses tools
        metadata={"category": "no_tool"}
    ),
]


# ── 3. Evaluator ──────────────────────────────────────────────────────────────
# The rubric defines scoring criteria for tool usage quality.
# The judge has access to three built-in scoring helpers:
#   - exact_match_scorer: full sequence must match exactly
#   - in_order_match_scorer: correct tools in order (subset ok)
#   - any_order_match_scorer: correct tools used, order flexible

evaluator = TrajectoryEvaluator(
    rubric="""
    Evaluate tool usage on three criteria:
    1. Correct selection — Were the right tools chosen for this task?
    2. Proper sequence   — Were tools called in a logical order?
    3. Efficiency        — Were unnecessary tools avoided?

    Use the built-in scoring helpers:
    - exact_match_scorer for cases requiring a precise sequence
    - in_order_match_scorer for ordered subsets
    - any_order_match_scorer for flexible ordering

    Score 1.0 for optimal tool usage.
    Score 0.5 for correct tools used but suboptimal sequence or extras called.
    Score 0.0 for wrong tools used, critical tools missed, or excessive calls.
    """,
    include_inputs=True
)

# Provide tool descriptions so the judge understands what each tool does.
# is_short=True keeps descriptions concise to avoid context overflow.
sample_agent = Agent(tools=[calculator, current_time])
tool_descriptions = tools_use_extractor.extract_tools_description(
    sample_agent, is_short=True
)
evaluator.update_trajectory_description(tool_descriptions)


# ── 4. Run experiment ─────────────────────────────────────────────────────────

def main():
    print("Running TrajectoryEvaluator...\n")

    experiment = Experiment[str, str](cases=test_cases, evaluators=[evaluator])
    reports = experiment.run_evaluations(run_agent_with_tools)

    reports[0].run_display()
    experiment.to_file("trajectory_evaluation")
    print("\nExperiment saved to ./experiment_files/trajectory_evaluation.json")

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
