"""
generate_tests.py
-----------------
Auto-generates a test suite using ExperimentGenerator.

Instead of writing test cases by hand, describe your agent's tools
and purpose — the SDK uses an LLM to generate a diverse set of test
cases automatically. Useful for:
  - Bootstrapping a new eval suite quickly
  - Discovering edge cases you hadn't thought of
  - Generating regression tests after adding new tools

Tip: save generated experiments to file and version-control them.
Don't regenerate on every run — it costs tokens and produces different
cases each time, making results non-comparable across runs.

Run:
    python generate_tests.py
"""

import asyncio
from strands_evals.generators import ExperimentGenerator
from strands_evals.evaluators import TrajectoryEvaluator


# ── Describe your agent's tools and purpose ───────────────────────────────────
# The more specific you are, the better the generated test cases.
# Include: tool names, what each does, and the agent's overall purpose.

TOOL_CONTEXT = """
Available tools for this agent:
- calculator(expression: str) -> float: Evaluate mathematical expressions
- search_knowledge_base(query: str) -> str: Search internal engineering documents
- current_time() -> str: Get the current date and time in UTC

The agent is a research assistant for software engineers at a hardware
company. It helps engineers find technical information from internal
documents, perform calculations, and answer time-sensitive questions.

Typical use cases:
- Looking up specifications in technical documents
- Performing unit conversions and calculations
- Checking timestamps and scheduling
- Multi-step problems combining search and calculation
"""


# ── Generate test suite ───────────────────────────────────────────────────────

async def generate_test_suite():
    print("Generating test suite from context description...")
    print("(This calls the LLM — may take 30-60 seconds)\n")

    generator = ExperimentGenerator[str, str](str, str)

    experiment = await generator.from_context_async(
        context=TOOL_CONTEXT,
        num_cases=10,                     # generate 10 test cases
        evaluator=TrajectoryEvaluator,    # evaluator type to configure cases for
        task_description="Research assistant for hardware engineers with calculation, search, and time tools",
        num_topics=3                      # distribute across 3 topic categories
    )

    print(f"Generated {len(experiment.cases)} test cases:\n")
    for case in experiment.cases:
        trajectory_hint = f" → expected tools: {case.expected_trajectory}" if hasattr(case, 'expected_trajectory') and case.expected_trajectory else ""
        print(f"  [{case.name}] {case.input[:80]}...{trajectory_hint}")

    # Save to file — reuse this instead of regenerating
    experiment.to_file("generated_test_suite")
    print("\n✅ Saved to ./experiment_files/generated_test_suite.json")
    print("   Load with: Experiment.from_file('generated_test_suite')")

    return experiment


# ── Load and run a previously saved experiment ────────────────────────────────

def load_and_run_saved():
    """
    Example of loading a previously generated experiment and running it.
    Use this in your CI pipeline instead of regenerating every time.
    """
    from strands import Agent
    from strands_evals import Experiment
    from strands_evals.extractors import tools_use_extractor
    from strands_tools import calculator, current_time

    def run_agent(case):
        agent = Agent(
            tools=[calculator, current_time],
            system_prompt="You are a helpful research assistant.",
            callback_handler=None
        )
        response = agent(case.input)
        trajectory = tools_use_extractor.extract_agent_tools_used_from_messages(
            agent.messages
        )
        return {"output": str(response), "trajectory": trajectory}

    try:
        experiment = Experiment.from_file("generated_test_suite")
        print(f"Loaded {len(experiment.cases)} saved test cases")
        reports = experiment.run_evaluations(run_agent)
        reports[0].run_display()
    except FileNotFoundError:
        print("No saved experiment found. Run generate_test_suite() first.")


if __name__ == "__main__":
    # Generate and save
    asyncio.run(generate_test_suite())

    # Optionally run the generated suite immediately
    # load_and_run_saved()
