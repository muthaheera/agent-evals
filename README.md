# 🤖 agent-evals

**Practical agent evaluation using the Strands Agents Evals SDK**

A companion repo to the Medium article series:
> *"Evaluating Strands Agents with the Official Evals SDK"* — [Read on Medium](#)


---

## 📖 What This Repo Covers

This repo demonstrates a production-grade agent evaluation pipeline using the official **[Strands Agents Evals SDK](https://strandsagents.com/latest/documentation/docs/user-guide/evals-sdk/quickstart/)**. It covers:

| File | What it demonstrates |
|---|---|
| `eval_output.py` | OutputEvaluator — LLM-as-a-Judge for answer quality |
| `eval_trajectory.py` | TrajectoryEvaluator — tool selection & call sequence |
| `eval_helpfulness.py` | HelpfulnessEvaluator — OpenTelemetry trace-based scoring |
| `custom_safety_evaluator.py` | Custom rule-based safety evaluator |
| `generate_tests.py` | Auto-generate test cases with ExperimentGenerator |
| `eval_gate.py` | CI/CD deployment gate with pass/fail threshold |
| `run_all_evals.py` | Orchestrator — runs the full evaluation suite end-to-end |

---

## 🧠 Why Agent Evaluation Is Different

Testing regular software is simple: given input X, does function return Y?

Agents require evaluating **multiple independent quality dimensions simultaneously**:

- **Output quality** — Is the final answer accurate and complete?
- **Tool selection accuracy** — Did the agent pick the right tools?
- **Tool call sequence** — Were tools called in a logical order?
- **Faithfulness** — Is the response grounded in tool outputs, or hallucinated?
- **Helpfulness** — Did the agent genuinely accomplish what the user needed?
- **Safety** — Did it avoid harmful or inappropriate outputs?

An agent can produce a correct answer ✅ while using the wrong tool ❌. It can complete a task ✅ while hallucinating facts not in its sources ❌. Each dimension fails independently — which is why this repo uses multiple evaluators.

---

## ⚡ Quickstart

### 1. Prerequisites

- Python 3.10+
- AWS account with Amazon Bedrock access
- Claude models enabled in your Bedrock region ([how to enable](https://docs.aws.amazon.com/bedrock/latest/userguide/model-access-modify.html))

### 2. Install

```bash
git clone https://github.com/muthaheera/agent-evals.git
cd agent-evals

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 3. Configure AWS credentials

```bash
aws configure
# or
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_DEFAULT_REGION=us-east-1
```

### 4. Run your first evaluation

```bash
# Run a single evaluator
python eval_output.py

# Run the full suite
python run_all_evals.py
```

---

## 🔍 Evaluator Reference

### OutputEvaluator
Uses LLM-as-a-Judge (Claude 4 via Bedrock) to score final answer quality against a rubric you define.

```python
from strands_evals import Case, Experiment
from strands_evals.evaluators import OutputEvaluator

evaluator = OutputEvaluator(
    rubric="Score 1.0 if accurate and complete, 0.5 if partial, 0.0 if wrong.",
    include_inputs=True
)
experiment = Experiment[str, str](cases=test_cases, evaluators=[evaluator])
reports = experiment.run_evaluations(your_task_function)
```

### TrajectoryEvaluator
Verifies which tools the agent called and in what order. Always use `tools_use_extractor` to extract trajectories from agent message history.

```python
from strands_evals.evaluators import TrajectoryEvaluator
from strands_evals.extractors import tools_use_extractor

# Your task function must return a dict with "output" and "trajectory"
def run_agent(case):
    agent = Agent(tools=[...], callback_handler=None)
    response = agent(case.input)
    trajectory = tools_use_extractor.extract_agent_tools_used_from_messages(agent.messages)
    return {"output": str(response), "trajectory": trajectory}
```

### HelpfulnessEvaluator
Deep trace-based evaluation using OpenTelemetry spans. Scores on a 7-level scale.

> ⚠️ **Always pass `trace_attributes` with session IDs** — without them, spans from different test cases contaminate each other in the memory exporter.

```python
agent = Agent(
    tools=[...],
    trace_attributes={
        "gen_ai.conversation.id": case.session_id,
        "session.id": case.session_id
    },
    callback_handler=None
)
```

### Custom Evaluator
Rule-based logic that doesn't need an LLM judge — ideal for safety checks, PII detection, or business guardrails.

```python
from strands_evals.evaluators import Evaluator
from strands_evals.types import EvaluationData, EvaluationOutput

class SafetyEvaluator(Evaluator[str, str]):
    def evaluate(self, evaluation_case: EvaluationData[str, str]) -> EvaluationOutput:
        # your deterministic logic here
        return EvaluationOutput(score=1.0, test_pass=True, reason="...", label="safe")
```

---

## 📊 Reading Results

```python
# Individual case results
for result in reports[0].case_results:
    print(f"{result.case.name}: {result.evaluation_output.score:.2f}")
    print(f"  Passed: {result.evaluation_output.test_pass}")
    print(f"  Reason: {result.evaluation_output.reason}")

# Summary statistics
summary = reports[0].get_summary()
print(f"Pass rate:     {summary['pass_rate']:.0%}")
print(f"Average score: {summary['average_score']:.2f}")
```

### CI/CD deployment gate

```python
if summary['pass_rate'] < 0.80:
    raise SystemExit(1)  # blocks the pipeline, prevents deployment
```

---

## 📁 Repo Structure

```
agent-evals/
├── README.md
├── requirements.txt
│
├── eval_output.py              # OutputEvaluator — answer quality
├── eval_trajectory.py          # TrajectoryEvaluator — tool usage
├── eval_helpfulness.py         # HelpfulnessEvaluator — trace-based
├── custom_safety_evaluator.py  # Custom rule-based evaluator
├── generate_tests.py           # Auto-generate test cases
├── eval_gate.py                # CI/CD gate with threshold
└── run_all_evals.py            # Full suite orchestrator
```

Saved experiments are written to `./experiment_files/` (git-ignored).

---

## 🔗 Related Resources

- **Strands Evals SDK Docs** — [strandsagents.com](https://strandsagents.com/latest/documentation/docs/user-guide/evals-sdk/quickstart/)
- **Available Evaluators** — [Output](https://strandsagents.com/latest/documentation/docs/user-guide/evals-sdk/evaluators/output_evaluator/) · [Trajectory](https://strandsagents.com/latest/documentation/docs/user-guide/evals-sdk/evaluators/trajectory_evaluator/) · [Helpfulness](https://strandsagents.com/latest/documentation/docs/user-guide/evals-sdk/evaluators/helpfulness_evaluator/) · [Faithfulness](https://strandsagents.com/latest/documentation/docs/user-guide/evals-sdk/evaluators/faithfulness_evaluator/)
- **Medium Article** — [Read the full guide](#)
- **Author LinkedIn** — [Muthaheera Yasmeena](https://www.linkedin.com/in/muthaheera-yasmeena-belgur-shamiullah-9954ab29)

---

## 📌 Coming Next

**Part 2 — Scaling to 10,000+ Evals with AWS ECS Fargate + Docker**
- Containerizing the eval runner
- Parallel execution across large datasets
- CloudWatch dashboard for continuous production monitoring

⭐ Star this repo to get notified when Part 2 drops.

---

## 🙏 Contributing

Found a bug or want to add an example? PRs are welcome. Open an issue first to discuss what you'd like to change.

---

