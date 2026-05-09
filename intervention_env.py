import json

from base.data import Data
from base.env import ToolEnv
from episode import generate_episode
from eval import evaluate, extract_layers
from prompts import build_system_prompt
from tools import dispatch_tool
from toon import encode as toon_encode
from vectors import VectorRegister

_LAYER_TOOLS = {
    "patch_sweep",
    "patch_component",
    "get_activations",
    "attention_pattern",
    "compare_weights",
}
_ALL_SCAN_TOOLS = {"scan_all_layers", "scan_residual_stream"}


class InterventionEnv(ToolEnv):
    def __init__(self, budget=40):
        super().__init__("intervention")
        self.budget = budget
        self._episode = None
        self._register = None
        self._tool_count = 0
        self._done = False

    def reset(self, data: Data) -> str:
        self._hypothesis_called = False
        self._queried_layers = set()
        self._all_layers_scanned = False
        self._episode = {
            "intervention": data.intervention,
            "samples": data.samples,
            "difficulty": data.difficulty,
            "seed": data.seed,
            "_sample_cursor": 5,
        }
        self._register = VectorRegister()
        self._tool_count = 0
        self._done = False
        return "Please investigate what intervention was applied to the modified model. Use the tools available to you to identify the type, location, and mechanism of the change."

    def step(self, action: str):
        if self._done:
            return ("episode already done", 0.0, True, {"done": True})

        try:
            parsed = json.loads(action)
            name = parsed["name"]
            args = parsed.get("args", {})
        except (json.JSONDecodeError, KeyError):
            return ("invalid action", 0.0, False, {"invalid": True})

        self._tool_count += 1
        try:
            result = dispatch_tool(name, args, self._episode, self._register)
        except Exception as e:
            return (
                f"tool error: {e}",
                0.0,
                False,
                {"tool": name, "tool_count": self._tool_count, "error": True},
            )

        if name == "state_hypothesis":
            self._hypothesis_called = True
        elif name in _LAYER_TOOLS and "layer" in args:
            self._queried_layers.add(int(args["layer"]))
        elif name in _ALL_SCAN_TOOLS:
            self._all_layers_scanned = True

        if name == "submit_report":
            report = args.get("report", "")
            score = evaluate(report, self._episode["intervention"], self._tool_count)

            violations = 0
            if not self._hypothesis_called:
                violations += 1
            observed = (
                set(range(12)) if self._all_layers_scanned else self._queried_layers
            )
            hallucinated = extract_layers(report) - observed
            violations += len(hallucinated)
            score["total"] = max(0.0, score["total"] - 0.1 * violations)
            score["policy_violations"] = violations

            self._done = True
            return (
                report,
                score["total"],
                True,
                {
                    "score": score,
                    "tool_count": self._tool_count,
                    "policy_violations": violations,
                },
            )

        obs = toon_encode(result)

        if self._tool_count >= self.budget:
            score = evaluate("", self._episode["intervention"], self._tool_count)
            self._done = True
            return (
                "budget exhausted",
                score["total"],
                True,
                {"score": score, "tool_count": self._tool_count},
            )

        return (obs, 0.0, False, {"tool": name, "tool_count": self._tool_count})

    def generate(self, num_of_questions=100, max_attempts=100, difficulty=1, **kwargs):
        seed = kwargs.get("seed", 0)
        results = []
        attempts = 0
        while len(results) < num_of_questions and attempts < max_attempts:
            ep = generate_episode(difficulty, seed + attempts)
            attempts += 1
            if ep["samples"]:
                results.append(
                    Data(
                        samples=ep["samples"],
                        intervention=ep["intervention"],
                        difficulty=difficulty,
                        seed=seed + attempts - 1,
                    )
                )
        return results
