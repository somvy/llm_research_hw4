import copy
import json
import os
import random
import re
import sys
import threading

import torch
import wandb
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from base.data import Data
from intervention_env import InterventionEnv
from prompts import TOOL_SCHEMAS, build_system_prompt

# --- Config ---
MODEL        = "Qwen/Qwen3-8B"
LORA_RANK    = 64
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
BUDGET       = 15
GROUP_SIZE   = 4
BATCH_EPS    = 2
LR           = 5e-6
KL_COEF      = 0.01
MAX_STEPS    = 300
SYNC_EVERY   = 20
LOG_EVERY    = 5
DIFFICULTIES = [1, 2]
LORA_TMP_DIR = "/tmp/grpo_lora"
# vLLM is initialized first and claims cuda:0; training model goes on cuda:1
TRAIN_DEVICE = "cuda:1"

SAMPLING = SamplingParams(temperature=0.7, max_tokens=1024)
_lora_request = None
_lora_uid = 0


# --- vLLM in-process ---

def load_vllm():
    return LLM(
        model=MODEL,
        enable_lora=True,
        max_lora_rank=LORA_RANK,
        gpu_memory_utilization=0.88,
        max_model_len=16384,
        trust_remote_code=True,
        dtype="bfloat16",
        enforce_eager=True,  # disable CUDA graphs — workaround for LoRA capture bug on some architectures
    )


def sync_lora(model, tokenizer, step):
    global _lora_request, _lora_uid
    os.makedirs(LORA_TMP_DIR, exist_ok=True)
    model.save_pretrained(LORA_TMP_DIR)
    tokenizer.save_pretrained(LORA_TMP_DIR)
    _lora_uid += 1
    _lora_request = LoRARequest("grpo_adapter", _lora_uid, LORA_TMP_DIR)
    print(f"\nLoRA synced at step {step}")


# --- Model loading ---

def load_model_and_lora():
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.bfloat16,
        device_map={"": TRAIN_DEVICE},
        trust_remote_code=True,
    )
    lora_cfg = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_RANK * 2,
        target_modules=LORA_TARGETS,
        lora_dropout=0.0,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model.print_trainable_parameters()
    return model, tokenizer


# --- Tools ---

def to_vllm_tools(schemas):
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


# --- Rollout ---

def parse_tool_call(text):
    m = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', text, re.DOTALL)
    if not m:
        return None, None, text.strip()
    try:
        call = json.loads(m.group(1))
        name = call.get("name")
        args = call.get("arguments", {})
        if isinstance(args, str):
            args = json.loads(args)
        prefix = text[:m.start()].strip() or None
        return name, args, prefix
    except (json.JSONDecodeError, KeyError):
        return None, None, text.strip()


def rollout_batch(llm, envs, data_list, budget):
    """Roll out multiple episodes in parallel, batching all active episodes per turn."""
    tools = to_vllm_tools(TOOL_SCHEMAS)
    eps = []
    for env, data in zip(envs, data_list):
        obs = env.reset(data)
        sys_prompt = build_system_prompt(data.samples[:5], budget=budget) + "\n/no_think"
        eps.append({
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": obs},
            ],
            "env": env,
            "reward": 0.0,
            "tool_count": 0,
            "call_idx": 0,
            "done": False,
        })

    for _ in range(budget):
        pending = [ep for ep in eps if not ep["done"]]
        if not pending:
            break

        outputs = llm.chat(
            [ep["messages"] for ep in pending],
            sampling_params=SAMPLING,
            use_tqdm=False,
            lora_request=_lora_request,
            tools=tools,
        )

        for ep, out in zip(pending, outputs):
            text = out.outputs[0].text
            name, args, prefix = parse_tool_call(text)

            if name:
                tc_id = f"tc_{ep['call_idx']}"
                ep["call_idx"] += 1
                ep["messages"].append({
                    "role": "assistant",
                    "content": prefix,
                    "tool_calls": [{
                        "id": tc_id,
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(args)},
                    }],
                })
                obs, reward, done, info = ep["env"].step(json.dumps({"name": name, "args": args}))
                ep["reward"] = reward
                ep["tool_count"] = info.get("tool_count", ep["tool_count"])
                ep["messages"].append({"role": "tool", "tool_call_id": tc_id, "content": obs})
                if done:
                    ep["done"] = True
            else:
                ep["messages"].append({"role": "assistant", "content": text})
                ep["messages"].append({
                    "role": "user",
                    "content": "Please use a tool to continue your investigation, or call submit_report when ready.",
                })

    return [(ep["messages"], ep["reward"], ep["tool_count"]) for ep in eps]


def collect_step(llm, env, episodes_data, budget):
    """GROUP_SIZE rollouts per episode, all batched together."""
    envs = [copy.deepcopy(env) for _ in episodes_data for _ in range(GROUP_SIZE)]
    data = [d for d in episodes_data for _ in range(GROUP_SIZE)]
    return rollout_batch(llm, envs, data, budget)


# --- GRPO loss ---

def traj_to_ids(tokenizer, messages):
    ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="pt",
    )
    return ids  # (1, seq_len)


def build_assistant_mask(tokenizer, input_ids):
    ids = input_ids[0].tolist()
    mask = [0] * len(ids)

    im_start = tokenizer.encode("<|im_start|>", add_special_tokens=False)
    im_end   = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    asst_tok = tokenizer.encode("assistant", add_special_tokens=False)

    n = len(ids)
    i = 0
    while i < n:
        if ids[i:i+len(im_start)] == im_start:
            role_start = i + len(im_start)
            if ids[role_start:role_start+len(asst_tok)] == asst_tok:
                content_start = role_start + len(asst_tok) + 1
                j = content_start
                while j < n:
                    if ids[j:j+len(im_end)] == im_end:
                        for k in range(content_start, j):
                            mask[k] = 1
                        i = j
                        break
                    j += 1
        i += 1

    return torch.tensor(mask, dtype=torch.float32)


MAX_TRAIN_TOKENS = 8192


def compute_traj_logprob(model, tokenizer, messages, device):
    input_ids = traj_to_ids(tokenizer, messages).to(device)
    if input_ids.shape[1] > MAX_TRAIN_TOKENS:
        input_ids = input_ids[:, -MAX_TRAIN_TOKENS:]
    mask = build_assistant_mask(tokenizer, input_ids).to(device)

    logits = model(input_ids=input_ids).logits  # (1, seq, vocab)

    log_probs = torch.log_softmax(logits[0, :-1], dim=-1)
    token_ids = input_ids[0, 1:]
    tok_log_probs = log_probs.gather(1, token_ids.unsqueeze(1)).squeeze(1)
    assistant_mask = mask[1:]

    return (tok_log_probs * assistant_mask).sum()


def compute_loss(model, tokenizer, trajectories, rewards):
    device = TRAIN_DEVICE
    r = torch.tensor(rewards, dtype=torch.float32)
    advantages = (r - r.mean()) / (r.std() + 1e-8)

    total_loss_val = 0.0
    total_kl = 0.0
    n = len(trajectories)

    model.train()
    for msgs, adv in zip(trajectories, advantages):
        adv = adv.to(device)

        lp = compute_traj_logprob(model, tokenizer, msgs, device)

        model.disable_adapter_layers()
        with torch.no_grad():
            ref_lp = compute_traj_logprob(model, tokenizer, msgs, device)
        model.enable_adapter_layers()

        kl = lp.detach() - ref_lp
        total_kl += kl.item()

        loss_i = (-adv * lp + KL_COEF * kl) / n
        loss_i.backward()
        total_loss_val += loss_i.item()

    return total_loss_val, total_kl / n


# --- WandB logging ---

def summarize_traj(messages):
    parts = []
    for m in messages:
        if m["role"] == "assistant" and m.get("tool_calls"):
            tc = m["tool_calls"][0]["function"]
            try:
                args = json.loads(tc["arguments"])
            except Exception:
                args = {}
            brief = {k: v for k, v in args.items() if k in ("layer", "component", "text", "n", "metric")}
            parts.append(f"[{tc['name']} {brief}]")
        elif m["role"] == "assistant" and m.get("content"):
            content = m["content"] or ""
            if "FINDINGS" in content:
                idx = content.index("FINDINGS")
                parts.append(content[idx:idx+500])
    return " ".join(parts)[:2000]


def log_to_wandb(step, episodes, all_trajectories, all_rewards):
    r = torch.tensor(all_rewards)
    wandb.log({
        "reward/mean": r.mean().item(),
        "reward/max":  r.max().item(),
        "reward/min":  r.min().item(),
        "step": step,
    }, step=step)

    table = wandb.Table(columns=["step", "difficulty", "seed", "reward", "trajectory"])
    for data, msgs, rwd in zip(
        [ep for ep in episodes for _ in range(GROUP_SIZE)],
        all_trajectories,
        all_rewards,
    ):
        table.add_data(step, data.difficulty, data.seed, rwd, summarize_traj(msgs))

    wandb.log({"generations": table}, step=step)


# --- Main ---

def sample_episodes(env, n, step):
    episodes = []
    for i in range(n):
        d = random.choice(DIFFICULTIES)
        items = env.generate(num_of_questions=1, max_attempts=10, difficulty=d, seed=step * 1000 + i)
        if items:
            episodes.append(items[0])
    return episodes


def run_smoke(llm):
    print("\n--- Smoke test: single rollout ---")
    env = InterventionEnv(budget=BUDGET)
    items = env.generate(num_of_questions=1, max_attempts=10, difficulty=1, seed=0)
    if not items:
        print("\nFailed to generate episode")
        return
    data = items[0]
    results = rollout_batch(llm, [env], [data], BUDGET)
    msgs, reward, tool_count = results[0]
    print(f"\nReward: {reward}")
    print(f"Tool calls: {tool_count}")
    print(f"\nTrajectory summary:\n{summarize_traj(msgs)}")
    print("\nSmoke test done")


def train(steps=MAX_STEPS, smoke=False):
    print("\nLoading vLLM on cuda:0...")
    llm = load_vllm()
    print("\nvLLM ready")

    if smoke:
        run_smoke(llm)
        return

    print(f"\nLoading training model on {TRAIN_DEVICE}...")
    model, tokenizer = load_model_and_lora()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR,
    )

    wandb.init(
        project="intervention-grpo",
        config={
            "model": MODEL,
            "lora_rank": LORA_RANK,
            "budget": BUDGET,
            "group_size": GROUP_SIZE,
            "batch_eps": BATCH_EPS,
            "lr": LR,
            "kl_coef": KL_COEF,
            "max_steps": steps,
            "difficulties": DIFFICULTIES,
        },
    )

    sync_lora(model, tokenizer, step=0)

    env = InterventionEnv(budget=BUDGET)

    # Async pipeline: rollout step N+1 on GPU 0 while training step N on GPU 1
    next_result = [None]
    ready = threading.Event()

    def do_rollout(step):
        eps = sample_episodes(env, BATCH_EPS, step)
        results = collect_step(llm, env, eps, BUDGET) if eps else []
        next_result[0] = (eps, results)
        ready.set()

    # Prime step 0
    do_rollout(0)
    ready.clear()

    for step in range(steps):
        episodes, results = next_result[0]

        # Start next rollout immediately — runs on GPU 0 while we train on GPU 1
        if step + 1 < steps:
            ready.clear()
            threading.Thread(target=do_rollout, args=(step + 1,), daemon=True).start()

        if not results:
            if step + 1 < steps:
                ready.wait()
            continue

        all_trajectories = [r[0] for r in results]
        all_rewards = [r[1] for r in results]

        loss, kl = compute_loss(model, tokenizer, all_trajectories, all_rewards)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        r = torch.tensor(all_rewards)
        print(f"\nstep {step} | loss={loss:.4f} | kl={kl:.4f} | reward mean={r.mean():.3f} max={r.max():.3f}")

        wandb.log({"loss": loss, "kl": kl, "step": step}, step=step)

        if step % SYNC_EVERY == 0:
            sync_lora(model, tokenizer, step)

        if step % LOG_EVERY == 0:
            log_to_wandb(step, episodes, all_trajectories, all_rewards)

        # Wait for next rollout if training finished first
        if step + 1 < steps:
            ready.wait()

    wandb.finish()


if __name__ == "__main__":
    args = sys.argv[1:]
    smoke = "--smoke" in args
    steps = MAX_STEPS
    for a in args:
        if a.startswith("--steps="):
            steps = int(a.split("=")[1])
        elif a == "--steps" and args.index(a) + 1 < len(args):
            steps = int(args[args.index(a) + 1])
    train(steps=steps, smoke=smoke)
