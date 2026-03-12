import json
import random
import torch
import torch.nn.functional as F
from toon import encode as toon_encode

from model import get_model, tokenize, generate_greedy, top_k_probs, compute_kl
from interventions import (
    difficulty_to_tier, TIER_POOLS, sample_intervention_params, make_intervention,
)
from vectors import VectorRegister
from tools import dispatch_tool
from prompts import INPUT_POOL, build_system_prompt, TOOL_SCHEMAS
from eval import evaluate


def generate_behavior_samples(intervention, rng, n_candidates=200, n_keep=20):
    m = get_model()
    from model import run_modified
    candidates = rng.sample(INPUT_POOL, min(n_candidates, len(INPUT_POOL)))

    # first pass: compute KL for all candidates
    kl_scores = []
    with torch.no_grad():
        for text in candidates:
            tokens = tokenize(text)
            base_logits = m(tokens)
            mod_logits = run_modified(tokens, intervention)
            pos = -1
            base_probs = F.softmax(base_logits[0, pos], dim=-1)
            mod_probs = F.softmax(mod_logits[0, pos], dim=-1)
            kl = F.kl_div(
                torch.log(mod_probs + 1e-10), base_probs, reduction="sum"
            ).item()
            kl_scores.append((text, kl))

    # take top candidates by KL (adaptive threshold)
    kl_scores.sort(key=lambda x: x[1], reverse=True)
    top_candidates = [(t, kl) for t, kl in kl_scores[:n_keep * 2] if kl > 1e-4]

    samples = []
    with torch.no_grad():
        for text, kl in top_candidates[:n_keep]:
            tokens = tokenize(text)
            base_logits = m(tokens)
            mod_logits = run_modified(tokens, intervention)
            pos = -1
            base_out = generate_greedy(text, max_tokens=50)
            mod_out = generate_greedy(text, max_tokens=50, intervention=intervention)

            samples.append({
                "input": text,
                "base_output": base_out,
                "modified_output": mod_out,
                "base_top5": top_k_probs(base_logits[0, pos]),
                "modified_top5": top_k_probs(mod_logits[0, pos]),
                "divergence": kl,
            })

    samples.sort(key=lambda s: s["divergence"], reverse=True)
    return samples[:n_keep]


def generate_episode(difficulty, seed):
    rng = random.Random(seed)
    tier = difficulty_to_tier(difficulty)
    itype = rng.choice(TIER_POOLS[tier])

    params = sample_intervention_params(itype, rng)
    intervention = make_intervention(itype, params)

    print(f"\nGenerating episode: difficulty={difficulty} type={itype} seed={seed}")
    samples = generate_behavior_samples(intervention, rng)
    print(f"\nGenerated {len(samples)} behavior samples")

    return {
        "intervention": intervention,
        "samples": samples,
        "difficulty": difficulty,
        "seed": seed,
        "_sample_cursor": 5,
    }

def run_episode(episode, agent_fn, budget=40, log_path=None, supervisor_fn=None):
    if supervisor_fn:
        episode["_supervisor_fn"] = supervisor_fn
    register = VectorRegister()
    tool_count = 0
    log = open(log_path, "w") if log_path else None

    def write(text):
        if log:
            log.write(text + "\n")
            log.flush()

    initial_samples = episode["samples"][:5]
    sys_prompt = build_system_prompt(initial_samples, budget=budget)

    if log:
        write(f"=== EPISODE: difficulty={episode['difficulty']} seed={episode['seed']} ===")
        write(f"Intervention: {episode['intervention']['type']}")
        write(f"Ground truth: {episode['intervention']['ground_truth']['components']}")
        write(f"Ground truth layers: {episode['intervention']['ground_truth']['layers']}")
        write("\n--- SYSTEM PROMPT ---")
        write(sys_prompt)
        write("--- END SYSTEM PROMPT ---\n")

    messages = [{"role": "system", "content": sys_prompt}]
    messages.append({
        "role": "user",
        "content": "Please investigate what intervention was applied to the modified model. Use the tools available to you to identify the type, location, and mechanism of the change."
    })

    while tool_count < budget:
        response = agent_fn(messages, TOOL_SCHEMAS)

        if response.get("reasoning"):
            if log:
                write(f"\n[TURN {tool_count}] AGENT REASONING:")
                write(response["reasoning"])

        if response.get("text"):
            messages.append({"role": "assistant", "content": response["text"]})
            if log:
                write(f"\n[TURN {tool_count}] AGENT THOUGHT:")
                write(response["text"])

        tool_calls = response.get("tool_calls", [])
        if not tool_calls:
            messages.append({
                "role": "user",
                "content": "Please use a tool to continue your investigation, or call submit_report when ready."
            })
            continue

        for tc in tool_calls:
            tool_count += 1
            name = tc["name"]
            args = tc["args"]

            if log:
                write(f"\n[CALL {tool_count}] {name}")
                write(f"  args: {json.dumps(args, default=str)}")

            result = dispatch_tool(name, args, episode, register)

            if name == "submit_report":
                report = args.get("report", "")
                score = evaluate(report, episode["intervention"], tool_count)
                if log:
                    write(f"\n[REPORT]\n{report}")
                    write(f"\n[SCORE] {json.dumps(score, default=str)}")
                    log.close()
                return score

            result_str = toon_encode(result)
            episode.setdefault("_tool_log", []).append({
                "tool": name,
                "result": result_str[:300],
            })
            if log:
                write(f"  result: {result_str[:2000]}{'...' if len(result_str) > 2000 else ''}")

            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": tc.get("id", f"call_{tool_count}"), "name": name, "args": args}],
            })
            messages.append({
                "role": "tool",
                "tool_call_id": tc.get("id", f"call_{tool_count}"),
                "content": result_str,
            })

            if tool_count >= budget:
                break

    score = evaluate("", episode["intervention"], budget)
    if log:
        write(f"\n[BUDGET EXHAUSTED]")
        write(f"[SCORE] {json.dumps(score, default=str)}")
        log.close()
    return score


# ---- Agent function adapters ----

def make_anthropic_agent(api_key=None, model_name="claude-sonnet-4-20250514"):
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    def convert_tools(schemas):
        return [
            {"name": s["name"], "description": s["description"], "input_schema": s["input_schema"]}
            for s in schemas
        ]

    def agent_fn(messages, tool_schemas):
        # convert messages to anthropic format
        sys_msg = None
        api_msgs = []
        for m in messages:
            if m["role"] == "system":
                sys_msg = m["content"]
            elif m["role"] == "user":
                api_msgs.append({"role": "user", "content": m["content"]})
            elif m["role"] == "assistant":
                if m.get("tool_calls"):
                    tc = m["tool_calls"][0]
                    api_msgs.append({
                        "role": "assistant",
                        "content": [
                            {"type": "tool_use", "id": tc["id"], "name": tc["name"], "input": tc["args"]}
                        ],
                    })
                elif m.get("content"):
                    api_msgs.append({"role": "assistant", "content": m["content"]})
            elif m["role"] == "tool":
                api_msgs.append({
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}
                    ],
                })

        response = client.messages.create(
            model=model_name,
            max_tokens=4096,
            system=sys_msg or "",
            messages=api_msgs,
            tools=convert_tools(tool_schemas),
        )

        result = {"text": None, "tool_calls": []}
        for block in response.content:
            if block.type == "text":
                result["text"] = block.text
            elif block.type == "tool_use":
                result["tool_calls"].append({
                    "id": block.id,
                    "name": block.name,
                    "args": block.input,
                })
        return result

    return agent_fn


def make_openai_agent(base_url, api_key, model_name="zai-org/GLM-5-FP8"):
    import json as _json
    from openai import OpenAI
    from httpx import Client
    import os 

    http_client = Client(proxy=os.environ.get("CUSTOM_PROXY"))

    client = OpenAI(base_url=base_url, api_key=api_key, http_client=http_client)

    def convert_tools(schemas):
        return [
            {
                "type": "function",
                "function": {
                    "name": s["name"],
                    "description": s["description"],
                    "parameters": s["input_schema"],
                },
            }
            for s in schemas
        ]

    def agent_fn(messages, tool_schemas):
        api_msgs = []
        for m in messages:
            if m["role"] == "system":
                api_msgs.append({"role": "system", "content": m["content"]})
            elif m["role"] == "user":
                api_msgs.append({"role": "user", "content": m["content"]})
            elif m["role"] == "assistant":
                if m.get("tool_calls"):
                    tc = m["tool_calls"][0]
                    api_msgs.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": _json.dumps(tc["args"]),
                            },
                        }],
                    })
                else:
                    api_msgs.append({"role": "assistant", "content": m.get("content", "")})
            elif m["role"] == "tool":
                api_msgs.append({
                    "role": "tool",
                    "tool_call_id": m["tool_call_id"],
                    "content": m["content"],
                })

        for _attempt in range(5):
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=api_msgs,
                    tools=convert_tools(tool_schemas),
                    tool_choice="auto",
                    max_tokens=4096,
                )
                break
            except Exception as e:
                if _attempt == 4:
                    raise
                print(f"\nRequest failed ({e}), retrying...")
                import time; time.sleep(3)

        msg = response.choices[0].message
        reasoning = getattr(msg, "reasoning_content", None)
        result = {"text": msg.content, "reasoning": reasoning, "tool_calls": []}
        if msg.tool_calls:
            for tc in msg.tool_calls:
                result["tool_calls"].append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "args": _json.loads(tc.function.arguments),
                })
        return result

    return agent_fn

