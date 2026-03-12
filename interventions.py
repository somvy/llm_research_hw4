import random
import torch
import torch.nn.functional as F
from model import get_model, N_LAYERS, N_HEADS, D_MODEL, compute_mean_cache
from prompts import INPUT_POOL

TIER_POOLS = {
    1: ["head_ablation", "mlp_ablation", "mean_ablation"],
    2: ["steering_vector", "rank1_edit", "multi_ablation"],
    3: ["conditional_steering", "distributed_finetune"],
}

_mean_cache = None


def get_mean_cache():
    global _mean_cache
    if _mean_cache is None:
        rng = random.Random(42)
        prompts = rng.sample(INPUT_POOL, min(200, len(INPUT_POOL)))
        _mean_cache = compute_mean_cache(prompts)
    return _mean_cache


def difficulty_to_tier(d):
    if d <= 3:
        return 1
    if d <= 6:
        return 2
    return 3


# --- Hook factories ---

def make_head_ablation_hooks(layer, head):
    def hooks_fn(tokens):
        def hook(value, hook):
            value[:, :, head, :] = 0.0
            return value
        return [(f"blocks.{layer}.attn.hook_result", hook)]
    return hooks_fn


def make_mlp_ablation_hooks(layer):
    def hooks_fn(tokens):
        def hook(value, hook):
            return torch.zeros_like(value)
        return [(f"blocks.{layer}.hook_mlp_out", hook)]
    return hooks_fn


def make_mean_ablation_hooks(layer, component):
    mc = get_mean_cache()
    if component.startswith("head."):
        head = int(component.split(".")[1])
        mean_val = mc[(layer, component)]
        def hooks_fn(tokens):
            def hook(value, hook):
                value[:, :, head, :] = mean_val.to(value.device)
                return value
            return [(f"blocks.{layer}.attn.hook_result", hook)]
    else:
        mean_val = mc[(layer, "mlp")]
        def hooks_fn(tokens):
            def hook(value, hook):
                return mean_val.to(value.device).expand_as(value)
            return [(f"blocks.{layer}.hook_mlp_out", hook)]
    return hooks_fn


def make_steering_hooks(layer, direction, scale):
    def hooks_fn(tokens):
        def hook(value, hook):
            value = value + scale * direction.to(value.device)
            return value
        return [(f"blocks.{layer}.hook_resid_pre", hook)]
    return hooks_fn


def make_conditional_steering_hooks(layer, direction, scale, trigger_ids):
    trigger_set = set(int(t) for t in trigger_ids)
    def hooks_fn(tokens):
        tok_set = set(tokens[0].tolist())
        if not tok_set.intersection(trigger_set):
            return []
        def hook(value, hook):
            value = value + scale * direction.to(value.device)
            return value
        return [(f"blocks.{layer}.hook_resid_pre", hook)]
    return hooks_fn


def make_multi_ablation_hooks(ablations):
    sub_fns = []
    for abl in ablations:
        if abl["type"] == "head_ablation":
            sub_fns.append(make_head_ablation_hooks(abl["layer"], abl["head"]))
        elif abl["type"] == "mlp_ablation":
            sub_fns.append(make_mlp_ablation_hooks(abl["layer"]))
        elif abl["type"] == "mean_ablation":
            sub_fns.append(make_mean_ablation_hooks(abl["layer"], abl["component"]))

    def hooks_fn(tokens):
        hooks = []
        for fn in sub_fns:
            hooks.extend(fn(tokens))
        return hooks
    return hooks_fn


# --- Steering direction generation ---

def sample_steering_direction(rng, method="semantic"):
    m = get_model()
    if method == "semantic":
        pairs = [
            (" good", " bad"), (" happy", " sad"), (" love", " hate"),
            (" beautiful", " ugly"), (" smart", " stupid"), (" kind", " cruel"),
            (" honest", " dishonest"), (" brave", " cowardly"),
            (" positive", " negative"), (" safe", " dangerous"),
        ]
        tok_a, tok_b = rng.choice(pairs)
        id_a = m.tokenizer.encode(tok_a)[0]
        id_b = m.tokenizer.encode(tok_b)[0]
        direction = (m.W_E[id_b] - m.W_E[id_a]).cpu()
    else:
        direction = torch.randn(D_MODEL)
    direction = direction / direction.norm()
    return direction.detach().float()


# --- Rank-1 edit ---

def make_rank1_edit(layer, key_dir, val_dir, scale):
    # key_dir: [d_mlp], val_dir: [d_model]
    # W_out shape: [d_mlp, d_model], delta = scale * outer(key_dir, val_dir)
    delta = scale * torch.outer(key_dir.cpu(), val_dir.cpu())
    param_name = f"blocks.{layer}.mlp.W_out"
    return {param_name: delta.detach().float()}


def sample_rank1_edit_params(rng, layer):
    m = get_model()
    d_mlp = m.cfg.d_mlp  # 3072 for gpt2

    # key direction: random unit vector in MLP hidden space
    key_dir = torch.randn(d_mlp)
    key_dir = key_dir / key_dir.norm()

    # value direction: difference between two token embeddings in vocab space
    pairs = [
        (" Paris", " London"), (" cat", " dog"), (" water", " fire"),
        (" yes", " no"), (" left", " right"), (" old", " new"),
    ]
    tok_a, tok_b = rng.choice(pairs)
    id_a = m.tokenizer.encode(tok_a)[0]
    id_b = m.tokenizer.encode(tok_b)[0]
    val_dir = (m.W_U[:, id_b] - m.W_U[:, id_a]).cpu()
    val_dir = val_dir / val_dir.norm()

    scale = rng.uniform(50.0, 200.0)
    return key_dir.detach().float(), val_dir.detach().float(), scale


# --- Distributed finetune (weight deltas via gradient steps) ---

def make_finetune_deltas(layers, component, lr, n_steps, texts, seed):
    m = get_model()
    torch.manual_seed(seed)

    # collect parameters to finetune
    params = []
    orig_values = {}
    for L in layers:
        if component == "mlp":
            names = [f"blocks.{L}.mlp.W_in", f"blocks.{L}.mlp.W_out"]
        else:
            names = [f"blocks.{L}.attn.W_Q", f"blocks.{L}.attn.W_K", f"blocks.{L}.attn.W_V", f"blocks.{L}.attn.W_O"]
        for n in names:
            p = _get_param_by_name(m, n)
            orig_values[n] = p.data.clone()
            p.requires_grad_(True)
            params.append((n, p))

    optimizer = torch.optim.SGD([p for _, p in params], lr=lr)

    for step in range(n_steps):
        text = texts[step % len(texts)]
        tokens = m.to_tokens(text)
        logits = m(tokens)
        # simple next-token loss on the text
        loss = F.cross_entropy(logits[0, :-1], tokens[0, 1:])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # compute deltas and restore
    deltas = {}
    for n, p in params:
        deltas[n] = (p.data - orig_values[n]).detach().float()
        p.data.copy_(orig_values[n])
        p.requires_grad_(False)

    return deltas


def _get_param_by_name(mdl, dotpath):
    obj = mdl
    for attr in dotpath.split("."):
        obj = getattr(obj, attr)
    return obj


# --- Intervention generation ---

def make_intervention(itype, params):
    if itype == "head_ablation":
        L, H = params["layer"], params["head"]
        return {
            "type": itype,
            "params": params,
            "hooks_fn": make_head_ablation_hooks(L, H),
            "weight_deltas": None,
            "ground_truth": {
                "type": "head_ablation",
                "components": {(L, f"head.{H}")},
                "layers": {L},
                "key_facts": {
                    "intervention_type": ["ablat", "zero"],
                    "layer": [f"layer {L}", f"L{L}"],
                    "component": [f"head {H}", f"head.{H}", f"{L}.{H}"],
                },
                "optimal_calls": 3,
            },
        }

    if itype == "mlp_ablation":
        L = params["layer"]
        return {
            "type": itype,
            "params": params,
            "hooks_fn": make_mlp_ablation_hooks(L),
            "weight_deltas": None,
            "ground_truth": {
                "type": "mlp_ablation",
                "components": {(L, "mlp")},
                "layers": {L},
                "key_facts": {
                    "intervention_type": ["ablat", "zero"],
                    "layer": [f"layer {L}", f"L{L}"],
                    "component": ["mlp", "MLP"],
                },
                "optimal_calls": 3,
            },
        }

    if itype == "mean_ablation":
        L, comp = params["layer"], params["component"]
        return {
            "type": itype,
            "params": params,
            "hooks_fn": make_mean_ablation_hooks(L, comp),
            "weight_deltas": None,
            "ground_truth": {
                "type": "mean_ablation",
                "components": {(L, comp)},
                "layers": {L},
                "key_facts": {
                    "intervention_type": ["ablat", "mean"],
                    "layer": [f"layer {L}", f"L{L}"],
                    "component": [comp],
                },
                "optimal_calls": 4,
            },
        }

    if itype == "steering_vector":
        L = params["layer"]
        return {
            "type": itype,
            "params": params,
            "hooks_fn": make_steering_hooks(L, params["direction"], params["scale"]),
            "weight_deltas": None,
            "ground_truth": {
                "type": "steering_vector",
                "components": {(L, "resid")},
                "layers": {L},
                "key_facts": {
                    "intervention_type": ["steer", "vector", "direction"],
                    "layer": [f"layer {L}", f"L{L}"],
                },
                "optimal_calls": 6,
            },
        }

    if itype == "rank1_edit":
        L = params["layer"]
        deltas = make_rank1_edit(L, params["key_dir"], params["val_dir"], params["scale"])
        return {
            "type": itype,
            "params": params,
            "hooks_fn": lambda tokens: [],
            "weight_deltas": deltas,
            "ground_truth": {
                "type": "rank1_edit",
                "components": {(L, "mlp")},
                "layers": {L},
                "key_facts": {
                    "intervention_type": ["edit", "rank", "ROME", "weight"],
                    "layer": [f"layer {L}", f"L{L}"],
                    "component": ["mlp", "MLP"],
                },
                "optimal_calls": 7,
            },
        }

    if itype == "multi_ablation":
        ablations = params["ablations"]
        comps = set()
        layers = set()
        for a in ablations:
            L = a["layer"]
            layers.add(L)
            if a["type"] == "head_ablation":
                comps.add((L, f"head.{a['head']}"))
            elif a["type"] in ("mlp_ablation", "mean_ablation"):
                comps.add((L, a.get("component", "mlp")))
        return {
            "type": itype,
            "params": params,
            "hooks_fn": make_multi_ablation_hooks(ablations),
            "weight_deltas": None,
            "ground_truth": {
                "type": "multi_ablation",
                "components": comps,
                "layers": layers,
                "key_facts": {
                    "intervention_type": ["ablat", "multi"],
                    "layers": [f"layer {L}" for L in sorted(layers)],
                },
                "optimal_calls": 8,
            },
        }

    if itype == "conditional_steering":
        L = params["layer"]
        return {
            "type": itype,
            "params": params,
            "hooks_fn": make_conditional_steering_hooks(
                L, params["direction"], params["scale"], params["trigger_ids"]
            ),
            "weight_deltas": None,
            "ground_truth": {
                "type": "conditional_steering",
                "components": {(L, "resid")},
                "layers": {L},
                "key_facts": {
                    "intervention_type": ["conditional", "steer", "trigger"],
                    "layer": [f"layer {L}", f"L{L}"],
                },
                "optimal_calls": 10,
            },
        }

    if itype == "distributed_finetune":
        layers = params["layers"]
        deltas = make_finetune_deltas(
            layers, params["component"], params["lr"],
            params["n_steps"], params["texts"], params["seed"]
        )
        return {
            "type": itype,
            "params": params,
            "hooks_fn": lambda tokens: [],
            "weight_deltas": deltas,
            "ground_truth": {
                "type": "distributed_finetune",
                "components": {(L, params["component"]) for L in layers},
                "layers": set(layers),
                "key_facts": {
                    "intervention_type": ["finetun", "distributed", "train"],
                    "layers": [f"layer {L}" for L in sorted(layers)],
                    "component": [params["component"]],
                },
                "optimal_calls": 15,
            },
        }

    raise ValueError(f"Unknown intervention type: {itype}")


def sample_intervention_params(itype, rng):
    if itype == "head_ablation":
        return {"layer": rng.randint(0, N_LAYERS - 1), "head": rng.randint(0, N_HEADS - 1)}

    if itype == "mlp_ablation":
        return {"layer": rng.randint(0, N_LAYERS - 1)}

    if itype == "mean_ablation":
        L = rng.randint(0, N_LAYERS - 1)
        if rng.random() < 0.5:
            comp = f"head.{rng.randint(0, N_HEADS - 1)}"
        else:
            comp = "mlp"
        return {"layer": L, "component": comp}

    if itype == "steering_vector":
        L = rng.randint(2, N_LAYERS - 2)  # avoid very early/late layers
        direction = sample_steering_direction(rng, method="semantic")
        scale = rng.uniform(2.0, 8.0)
        return {"layer": L, "direction": direction, "scale": scale}

    if itype == "rank1_edit":
        L = rng.randint(3, N_LAYERS - 2)
        key_dir, val_dir, scale = sample_rank1_edit_params(rng, L)
        return {"layer": L, "key_dir": key_dir, "val_dir": val_dir, "scale": scale}

    if itype == "multi_ablation":
        n = rng.randint(2, 3)
        ablations = []
        used_layers = set()
        for _ in range(n):
            L = rng.randint(0, N_LAYERS - 1)
            while L in used_layers:
                L = rng.randint(0, N_LAYERS - 1)
            used_layers.add(L)
            if rng.random() < 0.6:
                H = rng.randint(0, N_HEADS - 1)
                ablations.append({"type": "head_ablation", "layer": L, "head": H})
            else:
                ablations.append({"type": "mlp_ablation", "layer": L, "component": "mlp"})
        return {"ablations": ablations}

    if itype == "conditional_steering":
        m = get_model()
        L = rng.randint(2, N_LAYERS - 2)
        direction = sample_steering_direction(rng, method="semantic")
        scale = rng.uniform(3.0, 10.0)
        trigger_words = [" the", " a", " is", " was", " are", " has", " had", " will",
                         " can", " do", " not", " but", " and", " or", " if"]
        chosen = rng.sample(trigger_words, rng.randint(1, 3))
        trigger_ids = [m.tokenizer.encode(w)[0] for w in chosen]
        return {"layer": L, "direction": direction, "scale": scale,
                "trigger_ids": trigger_ids, "trigger_words": chosen}

    if itype == "distributed_finetune":
        n_layers = rng.randint(2, 4)
        all_layers = list(range(N_LAYERS))
        layers = sorted(rng.sample(all_layers, n_layers))
        comp = rng.choice(["mlp", "attn"])
        lr = rng.uniform(1e-4, 5e-4)
        texts = rng.sample(INPUT_POOL, min(20, len(INPUT_POOL)))
        return {"layers": layers, "component": comp, "lr": lr,
                "n_steps": 20, "texts": texts, "seed": rng.randint(0, 10000)}

    raise ValueError(f"Unknown intervention type: {itype}")
