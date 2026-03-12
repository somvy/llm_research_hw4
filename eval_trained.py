"""
Evaluate trained LoRA vs base model on the intervention env.
Logs full traces with COT to logs_base/ and logs_grpo/.

Usage:
    uv run python eval_trained.py
    uv run python eval_trained.py --difficulties 1 2 3 --seeds 5
    uv run python eval_trained.py --lora-only   # skip base model eval
    uv run python eval_trained.py --base-only
"""
import copy
import json
import re
import sys
import os

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from intervention_env import InterventionEnv
from prompts import TOOL_SCHEMAS, build_system_prompt
from train import parse_tool_call, to_vllm_tools
from eval import evaluate

MODEL = "Qwen/Qwen3-8B"
LORA_PATH = "grpo_lora"
BUDGET = 15
SAMPLING = SamplingParams(temperature=0.0, max_tokens=2048)  # greedy for eval


def load_llm():
    return LLM(
        model=MODEL,
        enable_lora=True,
        max_lora_rank=64,
        gpu_memory_utilization=0.88,
        max_model_len=16384,
        trust_remote_code=True,
        dtype="bfloat16",
        enforce_eager=True,
    )


def extract_think(text):
    """Split <think>...</think> from visible text. Returns (cot, text)."""
    m = re.match(r'\s*<think>(.*?)</think>\s*(.*)', text, re.DOTALL)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return None, text


def rollout_single(llm, env, data, budget, lora_request=None, log_path=None):
    tools = to_vllm_tools(TOOL_SCHEMAS)
    obs = env.reset(data)
    # no /no_think so COT is enabled
    sys_prompt = build_system_prompt(data.samples[:5], budget=budget)
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": obs},
    ]
    call_idx = 0
    final_score = None

    log = open(log_path, "w") if log_path else None

    def write(text):
        if log:
            log.write(text + "\n")
            log.flush()

    if log:
        gt = data.intervention["ground_truth"]
        write(f"=== EPISODE: difficulty={data.difficulty} seed={data.seed} ===")
        write(f"Intervention: {data.intervention['type']}")
        write(f"Ground truth: {gt['components']}")
        write(f"Ground truth layers: {gt['layers']}")
        write("\n--- SYSTEM PROMPT ---")
        write(sys_prompt)
        write("--- END SYSTEM PROMPT ---\n")

    for _ in range(budget):
        outputs = llm.chat(
            [messages],
            sampling_params=SAMPLING,
            use_tqdm=False,
            lora_request=lora_request,
            tools=tools,
        )
        raw_text = outputs[0].outputs[0].text
        cot, visible = extract_think(raw_text)
        name, args, prefix = parse_tool_call(visible)

        if log and cot:
            write(f"\n[TURN {call_idx}] AGENT REASONING:")
            write(cot)

        if name:
            tc_id = f"tc_{call_idx}"
            call_idx += 1
            if log:
                if prefix:
                    write(f"\n[TURN {call_idx-1}] AGENT THOUGHT:")
                    write(prefix)
                write(f"\n[CALL {call_idx}] {name}")
                write(f"  args: {json.dumps(args, default=str)}")
            messages.append({
                "role": "assistant",
                "content": prefix,
                "tool_calls": [{
                    "id": tc_id,
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args)},
                }],
            })
            obs, reward, done, info = env.step(json.dumps({"name": name, "args": args}))
            if log:
                write(f"  result: {obs[:2000]}{'...' if len(obs) > 2000 else ''}")
            if done:
                final_score = info.get("score")
                if log:
                    report = args.get("report", "")
                    write(f"\n[REPORT]\n{report}")
                    write(f"\n[SCORE] {json.dumps(final_score, default=str)}")
                break
            messages.append({"role": "tool", "tool_call_id": tc_id, "content": obs})
        else:
            if log and visible:
                write(f"\n[TURN {call_idx}] AGENT THOUGHT:")
                write(visible)
            messages.append({"role": "assistant", "content": visible})
            messages.append({
                "role": "user",
                "content": "Please use a tool to continue your investigation, or call submit_report when ready.",
            })

    if final_score is None:
        final_score = evaluate("", data.intervention, call_idx)
        if log:
            write(f"\n[BUDGET EXHAUSTED]")
            write(f"[SCORE] {json.dumps(final_score, default=str)}")

    if log:
        log.close()

    return final_score


def run_eval(llm, episodes, budget, lora_request=None, log_dir=None):
    env = InterventionEnv(budget=budget)
    results = []
    tag = "grpo" if lora_request else "base"
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    for data in episodes:
        log_path = None
        if log_dir:
            log_path = os.path.join(log_dir, f"episode_d{data.difficulty}_s{data.seed}.txt")
        score = rollout_single(llm, copy.deepcopy(env), data, budget, lora_request, log_path)
        score["difficulty"] = data.difficulty
        score["seed"] = data.seed
        results.append(score)
        print(f"  d={data.difficulty} s={data.seed} | total={score['total']:.3f} "
              f"type={score['type_accuracy']:.0f} comp_f1={score['component_f1']:.2f} "
              f"layer_iou={score['layer_iou']:.2f} facts={score['key_fact_score']:.2f} "
              f"calls={score['tool_calls_used']} [{tag}]")
    return results


def print_summary(results, label):
    keys = ["total", "type_accuracy", "component_f1", "layer_iou", "key_fact_score", "efficiency"]
    n = len(results)
    if not n:
        return
    avgs = {k: sum(r[k] for r in results) / n for k in keys}
    print(f"\n{label} (n={n}):")
    print(f"  total={avgs['total']:.3f}  type={avgs['type_accuracy']:.3f}  "
          f"comp_f1={avgs['component_f1']:.3f}  layer_iou={avgs['layer_iou']:.3f}  "
          f"facts={avgs['key_fact_score']:.3f}  eff={avgs['efficiency']:.3f}")
    for d in sorted(set(r["difficulty"] for r in results)):
        d_res = [r for r in results if r["difficulty"] == d]
        avg_total = sum(r["total"] for r in d_res) / len(d_res)
        avg_type = sum(r["type_accuracy"] for r in d_res) / len(d_res)
        avg_f1 = sum(r["component_f1"] for r in d_res) / len(d_res)
        print(f"  d={d}: total={avg_total:.3f}  type={avg_type:.3f}  comp_f1={avg_f1:.3f} (n={len(d_res)})")


def main():
    args = sys.argv[1:]
    lora_only = "--lora-only" in args
    base_only = "--base-only" in args

    difficulties = [1, 2, 3, 4]
    seeds = 3

    if "--difficulties" in args:
        idx = args.index("--difficulties")
        difficulties = []
        i = idx + 1
        while i < len(args) and not args[i].startswith("--"):
            difficulties.append(int(args[i]))
            i += 1

    if "--seeds" in args:
        idx = args.index("--seeds")
        seeds = int(args[idx + 1])

    print(f"\nEval config: difficulties={difficulties} seeds_per_d={seeds} budget={BUDGET}")

    env = InterventionEnv(budget=BUDGET)
    episodes = []
    for d in difficulties:
        for s in range(seeds):
            items = env.generate(num_of_questions=1, max_attempts=10, difficulty=d, seed=s)
            if items:
                episodes.append(items[0])
            else:
                print(f"  WARNING: could not generate episode d={d} s={s}")

    print(f"\nGenerated {len(episodes)} episodes")
    print("\nLoading vLLM...")
    llm = load_llm()

    base_results = []
    lora_results = []

    if not lora_only:
        print("\n--- Base model ---")
        base_results = run_eval(llm, episodes, BUDGET, lora_request=None, log_dir="logs_base")

    if not base_only:
        lora_req = LoRARequest("trained", 1, LORA_PATH)
        print("\n--- Trained LoRA ---")
        lora_results = run_eval(llm, episodes, BUDGET, lora_request=lora_req, log_dir="logs_grpo")

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    if base_results:
        print_summary(base_results, "Base model")
    if lora_results:
        print_summary(lora_results, "Trained LoRA")

    if base_results and lora_results:
        keys = ["total", "type_accuracy", "component_f1", "layer_iou", "key_fact_score"]
        n = len(base_results)
        print(f"\nDelta (trained - base):")
        for k in keys:
            delta = sum(lora_results[i][k] - base_results[i][k] for i in range(n)) / n
            sign = "+" if delta >= 0 else ""
            print(f"  {k}: {sign}{delta:.3f}")


if __name__ == "__main__":
    main()
