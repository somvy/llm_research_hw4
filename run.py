import os
import sys
from episode import generate_episode, run_episode, make_openai_agent
from supervisor import make_openai_supervisor

GLM_URL = "https://api.us-west-2.modal.direct/v1"
GLM_MODEL = "zai-org/GLM-5-FP8"
LOG_DIR = "logs"


def check_env():
    missing = []
    if not os.environ.get("GLM_API_KEY"):
        missing.append("GLM_API_KEY")
    if not os.environ.get("CUSTOM_PROXY"):
        missing.append("CUSTOM_PROXY")
    if missing:
        print(f"\nWARNING: env vars not set: {', '.join(missing)}")
        print("API calls will likely fail.\n")


def make_agent():
    api_key = os.environ.get("GLM_API_KEY", "")
    return make_openai_agent(base_url=GLM_URL, api_key=api_key, model_name=GLM_MODEL)


def make_supervisor():
    api_key = os.environ.get("GLM_API_KEY", "")
    return make_openai_supervisor(base_url=GLM_URL, api_key=api_key, model_name=GLM_MODEL)


def run_single(difficulty=1, seed=0, verbose=False):
    check_env()
    ep = generate_episode(difficulty, seed)

    if not ep["samples"]:
        print("\nNo divergent samples found, skipping episode")
        return None

    log_path = None
    if verbose:
        os.makedirs(LOG_DIR, exist_ok=True)
        log_path = os.path.join(LOG_DIR, f"episode_d{difficulty}_s{seed}.txt")
        print(f"\nLogging to {log_path}")

    agent = make_agent()
    supervisor = make_supervisor()
    score = run_episode(ep, agent, log_path=log_path, supervisor_fn=supervisor)

    print(f"\n--- Episode Results ---")
    print(f"Difficulty: {difficulty}")
    print(f"Intervention: {ep['intervention']['type']}")
    print(f"Ground truth components: {ep['intervention']['ground_truth']['components']}")
    for k, v in score.items():
        print(f"  {k}: {v}")
    return score


def run_batch(difficulties=None, seeds_per_difficulty=3, verbose=False):
    check_env()
    if difficulties is None:
        difficulties = [1, 2, 3, 4, 5]

    all_scores = []
    for d in difficulties:
        for s in range(seeds_per_difficulty):
            print(f"\n{'='*60}")
            print(f"Running: difficulty={d}, seed={s}")
            print(f"{'='*60}")
            score = run_single(d, s, verbose=verbose)
            if score:
                score["difficulty"] = d
                score["seed"] = s
                all_scores.append(score)

    print(f"\n{'='*60}")
    print("BATCH RESULTS")
    print(f"{'='*60}")
    if all_scores:
        avg_total = sum(s["total"] for s in all_scores) / len(all_scores)
        print(f"Average score: {avg_total}")
        for d in difficulties:
            d_scores = [s for s in all_scores if s["difficulty"] == d]
            if d_scores:
                avg = sum(s["total"] for s in d_scores) / len(d_scores)
                print(f"  Difficulty {d}: {avg} (n={len(d_scores)})")

    return all_scores


def run_smoke_test():
    from vectors import VectorRegister
    from tools import dispatch_tool
    from eval import evaluate

    print("\n--- Smoke Test ---")
    ep = generate_episode(difficulty=1, seed=42)

    if not ep["samples"]:
        print("\nNo samples generated, trying different seed...")
        ep = generate_episode(difficulty=1, seed=0)

    if not ep["samples"]:
        print("\nStill no samples, smoke test failed")
        return

    print(f"\nIntervention: {ep['intervention']['type']}")
    print(f"Ground truth: {ep['intervention']['ground_truth']}")
    print(f"\nSample count: {len(ep['samples'])}")
    s = ep["samples"][0]
    print(f"Top sample input: {s['input'][:80]}...")
    print(f"  Base output: {s['base_output'][:60]}...")
    print(f"  Modified output: {s['modified_output'][:60]}...")

    register = VectorRegister()
    gt = ep["intervention"]["ground_truth"]
    gt_layers = list(gt["layers"])
    gt_type = gt["type"]
    text = ep["samples"][0]["input"]
    L = gt_layers[0]

    print(f"\nRunning patch_sweep on layer {L}...")
    result = dispatch_tool("patch_sweep", {
        "text": text, "layer": L, "metric": "kl", "direction": "base_to_modified"
    }, ep, register)
    print(f"Top 3 components by |delta|:")
    for r in result["results"][:3]:
        print(f"  {r['component']}: delta={r['delta']}")

    print(f"\nRunning get_activations on layer {L}...")
    result = dispatch_tool("get_activations", {
        "text": text, "layer": L, "component": "resid_post",
        "token_pos": -1, "model": "both"
    }, ep, register)
    act = result["activations"][0]
    print(f"  cosine_sim: {act['cosine_sim']}")
    print(f"  l2_diff: {act['l2_diff']}")

    proj = dispatch_tool("project_to_vocab", {
        "vector_id": act["diff_vector_id"], "top_k": 5
    }, ep, register)
    print(f"Top tokens: {[(p['token'], round(p['logit'], 2)) for p in proj['projections']]}")

    fake_report = f"""
    FINDINGS:
    - Intervention type: {gt_type}
    - Affected layers: {', '.join(f'layer {l}' for l in gt_layers)}
    - Affected components: {', '.join(f'{c}' for _, c in gt["components"])}
    - Mechanism: The intervention modifies the model at the specified location
    - Confidence: high
    """
    score = evaluate(fake_report, ep["intervention"], 5)
    print(f"\nEval on perfect report:")
    for k, v in score.items():
        print(f"  {k}: {v}")

    print("\nSmoke test complete")


if __name__ == "__main__":
    args = sys.argv[1:]
    verbose = "--verbose" in args or "-v" in args
    args = [a for a in args if a not in ("--verbose", "-v")]

    if args and args[0] == "smoke":
        run_smoke_test()
    elif args and args[0] == "single":
        d = int(args[1]) if len(args) > 1 else 1
        s = int(args[2]) if len(args) > 2 else 0
        run_single(d, s, verbose=verbose)
    elif args and args[0] == "batch":
        run_batch(verbose=verbose)
    else:
        print("\nUsage:")
        print("  .venv/bin/python run.py smoke                       # quick self-test (no API)")
        print("  .venv/bin/python run.py single [d] [s] [-v]        # single episode")
        print("  .venv/bin/python run.py batch [-v]                  # full batch")
        print("\nFlags:")
        print("  -v / --verbose    write full agent trace to logs/episode_d{d}_s{s}.txt")
        print("\nSet GLM_API_KEY env var before running single/batch.")
