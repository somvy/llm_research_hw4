import re
import torch
import torch.nn.functional as F
from model import (
    get_model, tokenize, top_k_probs, run_base, run_base_with_cache,
    run_modified, run_modified_with_cache, run_modified_with_extra_hooks,
    generate_greedy, compute_kl, N_HEADS,
)


# ---- Observation Tools ----

def tool_get_behavior_samples(ep, register, n=5):
    idx = ep.get("_sample_cursor", 5)  # first 5 are in system prompt
    samples = ep["samples"][idx:idx + n]
    ep["_sample_cursor"] = idx + len(samples)
    return {"samples": samples, "remaining": len(ep["samples"]) - ep["_sample_cursor"]}


def tool_test_input(ep, register, text=""):
    m = get_model()
    tokens = tokenize(text)
    # truncate to 128 tokens
    if tokens.shape[1] > 128:
        tokens = tokens[:, :128]

    intervention = ep["intervention"]
    base_logits = run_base(tokens)
    mod_logits = run_modified(tokens, intervention)

    base_out = generate_greedy(text, max_tokens=30)
    mod_out = generate_greedy(text, max_tokens=30, intervention=intervention)

    pos = -1
    return {
        "base_output": base_out,
        "modified_output": mod_out,
        "base_top5": top_k_probs(base_logits[0, pos]),
        "modified_top5": top_k_probs(mod_logits[0, pos]),
        "divergence_score": compute_kl(base_logits, mod_logits, pos),
    }


# ---- Localization Tools ----

def _compute_metric(logits, pos, metric, metric_args):
    m = get_model()
    if pos is None or pos == -1:
        pos = -1
    if metric == "kl":
        # KL needs two distributions — but for patch_sweep we compare before/after
        # so we just return the logits vector for later comparison
        # Actually return the full logit vector at pos for KL computation
        return logits[0, pos]
    elif metric == "logit_diff":
        tok_a = m.tokenizer.encode(metric_args["token_a"])[0]
        tok_b = m.tokenizer.encode(metric_args["token_b"])[0]
        return (logits[0, pos, tok_a] - logits[0, pos, tok_b]).item()


def tool_patch_sweep(ep, register, text="", layer=0, target_token_pos=None,
                     metric="kl", metric_args=None, direction="base_to_modified"):
    tokens = tokenize(text)
    intervention = ep["intervention"]
    pos = target_token_pos if target_token_pos is not None else -1

    with torch.no_grad():
        base_logits, base_cache = run_base_with_cache(tokens)
        mod_logits, mod_cache = run_modified_with_cache(tokens, intervention)

    if direction == "base_to_modified":
        source_cache = base_cache
        before_logits = mod_logits
        run_target = lambda hooks: run_modified_with_extra_hooks(tokens, intervention, hooks)
    else:
        source_cache = mod_cache
        before_logits = base_logits
        run_target = lambda hooks: run_base(tokens, hooks=hooks)

    if metric == "kl":
        before_probs = F.softmax(before_logits[0, pos], dim=-1)
    else:
        before_val = _compute_metric(before_logits, pos, metric, metric_args)

    results = []

    # sweep heads
    hook_name = f"blocks.{layer}.attn.hook_result"
    source_result = source_cache[hook_name].clone()

    for h in range(N_HEADS):
        cached_head = source_result[:, :, h, :].clone()

        def patch_fn(value, hook, _h=h, _cv=cached_head):
            value[:, :, _h, :] = _cv.to(value.device)
            return value

        with torch.no_grad():
            patched_logits = run_target([(hook_name, patch_fn)])

        if metric == "kl":
            after_probs = F.softmax(patched_logits[0, pos], dim=-1)
            before_kl = 0.0  # reference point
            after_kl = F.kl_div(
                torch.log(after_probs + 1e-10), before_probs, reduction="sum"
            ).item()
            results.append({
                "component": f"head.{h}",
                "metric_before_patch": before_kl,
                "metric_after_patch": after_kl,
                "delta": after_kl - before_kl,
            })
        else:
            after_val = _compute_metric(patched_logits, pos, metric, metric_args)
            results.append({
                "component": f"head.{h}",
                "metric_before_patch": before_val,
                "metric_after_patch": after_val,
                "delta": after_val - before_val,
            })

    # sweep MLP
    mlp_hook = f"blocks.{layer}.hook_mlp_out"
    cached_mlp = source_cache[mlp_hook].clone()

    def patch_mlp(value, hook, _cv=cached_mlp):
        return _cv.to(value.device)

    with torch.no_grad():
        patched_logits = run_target([(mlp_hook, patch_mlp)])

    if metric == "kl":
        after_probs = F.softmax(patched_logits[0, pos], dim=-1)
        after_kl = F.kl_div(
            torch.log(after_probs + 1e-10), before_probs, reduction="sum"
        ).item()
        results.append({
            "component": "mlp",
            "metric_before_patch": 0.0,
            "metric_after_patch": after_kl,
            "delta": after_kl,
        })
    else:
        after_val = _compute_metric(patched_logits, pos, metric, metric_args)
        results.append({
            "component": "mlp",
            "metric_before_patch": before_val,
            "metric_after_patch": after_val,
            "delta": after_val - before_val,
        })

    results.sort(key=lambda r: abs(r["delta"]), reverse=True)

    deltas = [abs(r["delta"]) for r in results]
    top = deltas[0]
    second = deltas[1] if len(deltas) > 1 else 0.0
    if top < 1e-4:
        return {"layer": layer, "note": "NO SIGNAL at this layer."}
    # only return components with signal
    sig_results = [r for r in results if abs(r["delta"]) > 1e-4]
    out = {"layer": layer, "results": sig_results}
    if second > 0 and top / second > 10:
        out["note"] = f"STRONG SIGNAL: {results[0]['component']} dominates by {top/second:.0f}x. Use get_activations(model='both') to confirm."
    return out


def tool_patch_component(ep, register, text="", layer=0, component="mlp",
                         direction="base_to_modified"):
    tokens = tokenize(text)
    intervention = ep["intervention"]
    m = get_model()

    with torch.no_grad():
        base_logits, base_cache = run_base_with_cache(tokens)
        mod_logits, mod_cache = run_modified_with_cache(tokens, intervention)

    if direction == "base_to_modified":
        source_cache = base_cache
        orig_logits = mod_logits
        run_target = lambda hooks: run_modified_with_extra_hooks(tokens, intervention, hooks)
    else:
        source_cache = mod_cache
        orig_logits = base_logits
        run_target = lambda hooks: run_base(tokens, hooks=hooks)

    # create patch hook
    if component.startswith("head."):
        head = int(component.split(".")[1])
        hook_name = f"blocks.{layer}.attn.hook_result"
        cached = source_cache[hook_name][:, :, head, :].clone()
        def patch_fn(value, hook, _h=head, _cv=cached):
            value[:, :, _h, :] = _cv.to(value.device)
            return value
    elif component == "mlp":
        hook_name = f"blocks.{layer}.hook_mlp_out"
        cached = source_cache[hook_name].clone()
        def patch_fn(value, hook, _cv=cached):
            return _cv.to(value.device)
    elif component == "resid_pre":
        hook_name = f"blocks.{layer}.hook_resid_pre"
        cached = source_cache[hook_name].clone()
        def patch_fn(value, hook, _cv=cached):
            return _cv.to(value.device)
    elif component == "resid_post":
        hook_name = f"blocks.{layer}.hook_resid_post"
        cached = source_cache[hook_name].clone()
        def patch_fn(value, hook, _cv=cached):
            return _cv.to(value.device)
    else:
        return {"error": f"Unknown component: {component}"}

    with torch.no_grad():
        patched_logits = run_target([(hook_name, patch_fn)])

    pos = -1
    kl = F.kl_div(
        F.log_softmax(patched_logits[0, pos], dim=-1),
        F.softmax(orig_logits[0, pos], dim=-1),
        reduction="sum"
    ).item()

    orig_out = m.tokenizer.decode(orig_logits[0].argmax(dim=-1).tolist())
    patched_out = m.tokenizer.decode(patched_logits[0].argmax(dim=-1).tolist())

    return {
        "original_output": orig_out,
        "patched_output": patched_out,
        "original_top5": top_k_probs(orig_logits[0, pos]),
        "patched_top5": top_k_probs(patched_logits[0, pos]),
        "kl_divergence": kl,
    }


def tool_scan_all_layers(ep, register, text="", metric="kl", metric_args=None,
                          direction="base_to_modified", top_n=5):
    tokens = tokenize(text)
    intervention = ep["intervention"]
    pos = -1

    with torch.no_grad():
        base_logits, base_cache = run_base_with_cache(tokens)
        mod_logits, mod_cache = run_modified_with_cache(tokens, intervention)

    if direction == "base_to_modified":
        source_cache = base_cache
        before_logits = mod_logits
        run_target = lambda hooks: run_modified_with_extra_hooks(tokens, intervention, hooks)
    else:
        source_cache = mod_cache
        before_logits = base_logits
        run_target = lambda hooks: run_base(tokens, hooks=hooks)

    if metric == "kl":
        before_probs = F.softmax(before_logits[0, pos], dim=-1)

    all_results = []

    for layer in range(12):
        hook_name = f"blocks.{layer}.attn.hook_result"
        source_result = source_cache[hook_name].clone()

        for h in range(N_HEADS):
            cached_head = source_result[:, :, h, :].clone()
            def patch_fn(value, hook, _h=h, _cv=cached_head):
                value[:, :, _h, :] = _cv.to(value.device)
                return value
            with torch.no_grad():
                patched_logits = run_target([(hook_name, patch_fn)])
            if metric == "kl":
                after_probs = F.softmax(patched_logits[0, pos], dim=-1)
                delta = F.kl_div(
                    torch.log(after_probs + 1e-10), before_probs, reduction="sum"
                ).item()
            else:
                before_val = _compute_metric(before_logits, pos, metric, metric_args)
                after_val = _compute_metric(patched_logits, pos, metric, metric_args)
                delta = after_val - before_val
            if abs(delta) > 1e-4:
                all_results.append({"layer": layer, "component": f"head.{h}", "delta": delta})

        mlp_hook = f"blocks.{layer}.hook_mlp_out"
        cached_mlp = source_cache[mlp_hook].clone()
        def patch_mlp(value, hook, _cv=cached_mlp):
            return _cv.to(value.device)
        with torch.no_grad():
            patched_logits = run_target([(mlp_hook, patch_mlp)])
        if metric == "kl":
            after_probs = F.softmax(patched_logits[0, pos], dim=-1)
            delta = F.kl_div(
                torch.log(after_probs + 1e-10), before_probs, reduction="sum"
            ).item()
        else:
            before_val = _compute_metric(before_logits, pos, metric, metric_args)
            after_val = _compute_metric(patched_logits, pos, metric, metric_args)
            delta = after_val - before_val
        if abs(delta) > 1e-4:
            all_results.append({"layer": layer, "component": "mlp", "delta": delta})

    # sweep resid_pre at each layer
    for layer in range(12):
        resid_hook = f"blocks.{layer}.hook_resid_pre"
        cached_resid = source_cache[resid_hook].clone()
        def patch_resid(value, hook, _cv=cached_resid):
            return _cv.to(value.device)
        with torch.no_grad():
            patched_logits = run_target([(resid_hook, patch_resid)])
        if metric == "kl":
            after_probs = F.softmax(patched_logits[0, pos], dim=-1)
            delta = F.kl_div(
                torch.log(after_probs + 1e-10), before_probs, reduction="sum"
            ).item()
        else:
            before_val = _compute_metric(before_logits, pos, metric, metric_args)
            after_val = _compute_metric(patched_logits, pos, metric, metric_args)
            delta = after_val - before_val
        if abs(delta) > 1e-4:
            all_results.append({"layer": layer, "component": "resid_pre", "delta": delta})

    all_results.sort(key=lambda r: abs(r["delta"]), reverse=True)
    top = all_results[:top_n]

    out = {"top_components": top, "total_with_signal": len(all_results)}
    if top:
        best = top[0]
        second = top[1]["delta"] if len(top) > 1 else 0
        if second > 0 and abs(best["delta"]) / abs(second) > 5:
            out["note"] = f"STRONG SIGNAL: L{best['layer']} {best['component']} dominates by {abs(best['delta'])/abs(second):.0f}x. Use get_activations(model='both') to confirm."
        if best["component"] == "resid_pre":
            out["note"] = (out.get("note", "") +
                f" resid_pre dominates — likely a steering_vector or conditional_steering injected at layer {best['layer']}."
                " Check if some inputs show zero divergence (conditional_steering) vs all inputs diverge (steering_vector).")
    else:
        out["note"] = "NO SIGNAL across any layer. The intervention may be subtle — try different input text or check residual stream."
    return out


def tool_scan_residual_stream(ep, register, text="", metric="kl", metric_args=None,
                               direction="base_to_modified"):
    tokens = tokenize(text)
    intervention = ep["intervention"]

    with torch.no_grad():
        base_logits, base_cache = run_base_with_cache(tokens)
        mod_logits, mod_cache = run_modified_with_cache(tokens, intervention)

    results = []
    for layer in range(12):
        hook = f"blocks.{layer}.hook_resid_pre"
        base_act = base_cache[hook][0, -1, :]
        mod_act = mod_cache[hook][0, -1, :]
        l2_diff = (mod_act - base_act).norm().item()
        cos = F.cosine_similarity(base_act.unsqueeze(0), mod_act.unsqueeze(0)).item()
        results.append({
            "layer": layer,
            "base_norm": base_act.norm().item(),
            "modified_norm": mod_act.norm().item(),
            "l2_diff": l2_diff,
            "cosine_sim": cos,
        })

    # find biggest jump in l2_diff between consecutive layers
    diffs = [r["l2_diff"] for r in results]
    jumps = []
    for i in range(1, len(diffs)):
        jump = diffs[i] - diffs[i - 1]
        if jump > 0.1:
            jumps.append({"layer": i, "jump": jump, "l2_diff_before": diffs[i - 1], "l2_diff_after": diffs[i]})
    jumps.sort(key=lambda j: j["jump"], reverse=True)

    out = {"residual_stream": results}
    if jumps:
        out["biggest_jumps"] = jumps[:3]
        top_jump = jumps[0]
        out["note"] = (
            f"Biggest l2_diff jump at layer {top_jump['layer']} "
            f"({top_jump['l2_diff_before']:.3f} -> {top_jump['l2_diff_after']:.3f}). "
            f"The intervention likely acts at layer {top_jump['layer']-1} or injects into resid_pre of layer {top_jump['layer']}. "
            f"Use get_activations(layer={top_jump['layer']}, component='resid_pre', model='both') to confirm."
        )
    return out


# ---- Inspection Tools ----

def _extract_activation(cache, layer, component, token_pos):
    if component.startswith("head."):
        head = int(component.split(".")[1])
        act = cache[f"blocks.{layer}.attn.hook_result"]
        if token_pos is not None:
            return act[0, token_pos, head, :]  # [d_model]
        return act[0, :, head, :]  # [seq, d_model]
    elif component == "mlp":
        act = cache[f"blocks.{layer}.hook_mlp_out"]
    elif component == "resid_pre":
        act = cache[f"blocks.{layer}.hook_resid_pre"]
    elif component == "resid_post":
        act = cache[f"blocks.{layer}.hook_resid_post"]
    else:
        raise ValueError(f"Unknown component: {component}")

    if token_pos is not None:
        return act[0, token_pos, :]
    return act[0, :, :]


def tool_get_activations(ep, register, text="", layer=0, component="resid_post",
                         token_pos=None, model="both"):
    tokens = tokenize(text)
    m = get_model()
    intervention = ep["intervention"]
    tok_strs = [m.tokenizer.decode(t.item()) for t in tokens[0]]

    results = []

    def process_one(cache, model_name, pos):
        act = _extract_activation(cache, layer, component, pos)
        if act.dim() == 1:
            vid = register.store(act, f"{model_name}_L{layer}_{component}_p{pos}")
            return [{
                "position": pos if pos is not None else -1,
                "token": tok_strs[pos] if pos is not None else tok_strs[-1],
                "l2_norm": act.norm().item(),
                "vector_id": vid,
            }]
        else:
            out = []
            for p in range(act.shape[0]):
                vid = register.store(act[p], f"{model_name}_L{layer}_{component}_p{p}")
                out.append({
                    "position": p,
                    "token": tok_strs[p] if p < len(tok_strs) else "?",
                    "l2_norm": act[p].norm().item(),
                    "vector_id": vid,
                })
            return out

    with torch.no_grad():
        if model in ("base", "both"):
            _, base_cache = run_base_with_cache(tokens)
        if model in ("modified", "both"):
            _, mod_cache = run_modified_with_cache(tokens, intervention)

    if model == "base":
        return {"model": "base", "activations": process_one(base_cache, "base", token_pos)}
    if model == "modified":
        return {"model": "modified", "activations": process_one(mod_cache, "modified", token_pos)}

    # both
    base_acts = process_one(base_cache, "base", token_pos)
    mod_acts = process_one(mod_cache, "modified", token_pos)

    combined = []
    ablated_tokens = []
    for ba, ma in zip(base_acts, mod_acts):
        bv = register.get(ba["vector_id"])
        mv = register.get(ma["vector_id"])
        diff = mv - bv
        diff_vid = register.store(diff, f"diff_L{layer}_{component}_p{ba['position']}")
        cos = F.cosine_similarity(bv.unsqueeze(0), mv.unsqueeze(0)).item()
        combined.append({
            "position": ba["position"],
            "token": ba["token"],
            "base_l2_norm": ba["l2_norm"],
            "modified_l2_norm": ma["l2_norm"],
            "base_vector_id": ba["vector_id"],
            "modified_vector_id": ma["vector_id"],
            "cosine_sim": cos,
            "l2_diff": diff.norm().item(),
            "diff_vector_id": diff_vid,
        })
        if ba["l2_norm"] > 0 and ma["l2_norm"] == 0:
            ablated_tokens.append(ba["token"])

    result = {"model": "both", "activations": combined}
    if ablated_tokens:
        result["note"] = (
            f"ABLATION DETECTED: modified model output is zero at {ablated_tokens}. "
            f"Use project_to_vocab on the base_vector_id at the last token to characterize "
            f"what signal was removed, then include l2_norm and direction in your report."
        )
    return result


def tool_attention_pattern(ep, register, text="", layer=0, head=0, model="both"):
    tokens = tokenize(text)
    m = get_model()
    intervention = ep["intervention"]
    tok_strs = [m.tokenizer.decode(t.item()) for t in tokens[0]]
    hook_name = f"blocks.{layer}.attn.hook_pattern"
    long_input = len(tok_strs) > 30

    with torch.no_grad():
        if model in ("base", "both"):
            _, base_cache = run_base_with_cache(tokens)
            base_pattern = base_cache[hook_name][0, head].cpu()  # [q, k]
        if model in ("modified", "both"):
            _, mod_cache = run_modified_with_cache(tokens, intervention)
            mod_pattern = mod_cache[hook_name][0, head].cpu()

    result = {"tokens": tok_strs}

    if model == "base":
        if long_input:
            result["note"] = "Input > 30 tokens, pattern omitted"
        else:
            result["pattern"] = base_pattern.tolist()
        return result

    if model == "modified":
        if long_input:
            result["note"] = "Input > 30 tokens, pattern omitted"
        else:
            result["pattern"] = mod_pattern.tolist()
        return result

    # both
    diff = (mod_pattern - base_pattern).abs()
    flat = diff.flatten()
    topk = min(10, flat.numel())
    vals, idxs = flat.topk(topk)
    max_diff_positions = []
    seq_len = diff.shape[0]
    for i in range(topk):
        q = idxs[i].item() // seq_len
        k = idxs[i].item() % seq_len
        max_diff_positions.append({
            "q_pos": q, "k_pos": k,
            "q_token": tok_strs[q] if q < len(tok_strs) else "?",
            "k_token": tok_strs[k] if k < len(tok_strs) else "?",
            "diff": vals[i].item(),
        })

    if not long_input:
        result["pattern_base"] = base_pattern.tolist()
        result["pattern_modified"] = mod_pattern.tolist()
    result["max_diff_positions"] = max_diff_positions
    return result


# ---- Vector Operation Tools ----

def tool_project_to_vocab(ep, register, vector_id="", top_k=10):
    m = get_model()
    vec = register.get(vector_id).to(m.W_U.device)
    logits = vec @ m.W_U + m.b_U  # [d_vocab]

    top_vals, top_idx = logits.topk(top_k)
    bot_vals, bot_idx = (-logits).topk(top_k)

    projections = [
        {"token": m.tokenizer.decode(top_idx[i].item()), "logit": top_vals[i].item(), "rank": i + 1}
        for i in range(top_k)
    ]
    bottom = [
        {"token": m.tokenizer.decode(bot_idx[i].item()), "logit": -bot_vals[i].item(), "rank": i + 1}
        for i in range(top_k)
    ]
    return {"projections": projections, "bottom_k": bottom}


def tool_vector_dot(ep, register, vector_id_a="", vector_id_b=""):
    va = register.get(vector_id_a)
    vb = register.get(vector_id_b)
    cos = F.cosine_similarity(va.unsqueeze(0), vb.unsqueeze(0)).item()
    dot = (va * vb).sum().item()
    return {
        "cosine_similarity": cos,
        "dot_product": dot,
        "norm_a": va.norm().item(),
        "norm_b": vb.norm().item(),
    }


def tool_vector_arithmetic(ep, register, operations=None):
    vid, norm = register.arithmetic(operations)
    return {"vector_id": vid, "l2_norm": norm}


def tool_steer_and_run(ep, register, text="", vector_id="", scale=1.0,
                       layer=0, token_pos="all", target_model="base"):
    m = get_model()
    tokens = tokenize(text)
    vec = register.get(vector_id).to("cuda")
    intervention = ep["intervention"]

    def steer_hook(value, hook):
        if token_pos == "all":
            value = value + scale * vec
        else:
            p = int(token_pos)
            value[:, p, :] = value[:, p, :] + scale * vec
        return value

    steer = [(f"blocks.{layer}.hook_resid_pre", steer_hook)]

    with torch.no_grad():
        if target_model == "base":
            orig_logits = run_base(tokens)
            steered_logits = run_base(tokens, hooks=steer)
        else:
            orig_logits = run_modified(tokens, intervention)
            steered_logits = run_modified(tokens, intervention, extra_hooks=steer)

    pos = -1
    kl = F.kl_div(
        F.log_softmax(steered_logits[0, pos], dim=-1),
        F.softmax(orig_logits[0, pos], dim=-1),
        reduction="sum"
    ).item()

    orig_out = m.tokenizer.decode(orig_logits[0].argmax(dim=-1).tolist())
    steered_out = m.tokenizer.decode(steered_logits[0].argmax(dim=-1).tolist())

    return {
        "original_output": orig_out,
        "steered_output": steered_out,
        "original_top5": top_k_probs(orig_logits[0, pos]),
        "steered_top5": top_k_probs(steered_logits[0, pos]),
        "kl_divergence": kl,
    }


# ---- Communication Tools ----

def tool_state_hypothesis(ep, register, hypothesis=""):
    supervisor_fn = ep.get("_supervisor_fn")
    if supervisor_fn:
        return supervisor_fn(
            ep["intervention"]["ground_truth"],
            hypothesis,
            ep.get("_tool_log", []),
        )

    gt = ep["intervention"]["ground_truth"]
    gt_type = gt["type"]
    gt_layers = gt["layers"]
    gt_comps = gt["components"]

    # parse hypothesis
    layer_matches = set(int(x) for x in re.findall(r"layer\s*(\d+)|L(\d+)", hypothesis, re.I)
                        for x in x if x)
    head_matches = set(int(x) for x in re.findall(r"head\s*(\d+)|head\.(\d+)", hypothesis, re.I)
                       for x in x if x)

    type_keywords = {
        "head_ablation": ["ablat", "zero", "head ablat"],
        "mlp_ablation": ["ablat", "zero", "mlp ablat"],
        "mean_ablation": ["mean", "ablat"],
        "steering_vector": ["steer", "vector", "direction"],
        "rank1_edit": ["edit", "rank", "rome", "weight"],
        "multi_ablation": ["multi", "several", "multiple"],
        "conditional_steering": ["conditional", "trigger"],
        "distributed_finetune": ["finetun", "train", "distributed"],
    }

    mentioned_type = None
    hyp_lower = hypothesis.lower()
    for tname, keywords in type_keywords.items():
        if any(k in hyp_lower for k in keywords):
            mentioned_type = tname
            break

    correct_type = mentioned_type == gt_type
    correct_layer = bool(layer_matches & gt_layers)
    adjacent_layer = any(abs(l - gl) <= 1 for l in layer_matches for gl in gt_layers) if layer_matches else False

    if not layer_matches and not head_matches and not mentioned_type:
        return {"follow_up": "Can you narrow down to specific layers or components?"}

    if correct_layer:
        # check component
        correct_comp = False
        for gl, gc in gt_comps:
            if gc.startswith("head."):
                h = int(gc.split(".")[1])
                if h in head_matches and gl in layer_matches:
                    correct_comp = True
            elif "mlp" in hyp_lower and gl in layer_matches:
                correct_comp = True

        if correct_comp:
            if correct_type:
                return {"follow_up": "Can you characterize the direction or magnitude of the change?"}
            return {"follow_up": "What do you think the intervention type is?"}
        if correct_type:
            return {"follow_up": "Can you localize it further to specific components?"}
        return {"follow_up": "You're looking at the right area. What specific component is affected?"}

    if adjacent_layer and not correct_layer:
        return {"follow_up": "Interesting. Have you checked nearby layers?"}

    if correct_type and not correct_layer:
        return {"follow_up": "The type seems right. Can you narrow down the layer?"}

    if layer_matches and not correct_layer:
        return {"follow_up": "What evidence supports that specific location?"}

    return {"follow_up": "Can you be more specific about which layer and component?"}


def tool_submit_report(ep, register, report=""):
    return {"status": "episode_terminated", "report": report}


def tool_compare_weights(ep, register, layer=0, component="mlp"):
    deltas = ep["intervention"].get("weight_deltas")
    if not deltas:
        return {"layer": layer, "component": component, "weight_modified": False,
                "note": "Intervention has no weight deltas (activation-only, e.g. steering or ablation)"}

    if component == "mlp":
        param_names = [f"blocks.{layer}.mlp.W_in", f"blocks.{layer}.mlp.W_out"]
    elif component == "attn":
        param_names = [f"blocks.{layer}.attn.W_Q", f"blocks.{layer}.attn.W_K",
                       f"blocks.{layer}.attn.W_V", f"blocks.{layer}.attn.W_O"]
    else:
        return {"error": f"Unknown component '{component}', use 'mlp' or 'attn'"}

    found = []
    for name in param_names:
        if name in deltas:
            delta = deltas[name].float()
            rank = torch.linalg.matrix_rank(delta).item() if delta.dim() == 2 else None
            found.append({
                "param": name.split(".")[-1],
                "shape": list(delta.shape),
                "l2_norm": delta.norm().item(),
                "rank": rank,
            })

    if not found:
        return {"layer": layer, "component": component, "weight_modified": False,
                "note": "No weight delta at this layer/component"}

    return {"layer": layer, "component": component, "weight_modified": True, "params": found}


def tool_find_trigger_inputs(ep, register, n=5):
    from prompts import INPUT_POOL
    import random

    intervention = ep["intervention"]
    rng = random.Random(ep.get("seed", 0) + 99)
    pool = rng.sample(INPUT_POOL, min(50, len(INPUT_POOL)))

    results = []
    with torch.no_grad():
        for text in pool:
            tokens = tokenize(text)
            if tokens.shape[1] > 64:
                tokens = tokens[:, :64]
            base_logits = run_base(tokens)
            mod_logits = run_modified(tokens, intervention)
            results.append({"text": text, "divergence": compute_kl(base_logits, mod_logits)})

    results.sort(key=lambda x: x["divergence"], reverse=True)
    zero_frac = sum(1 for r in results if r["divergence"] < 0.001) / len(results)
    return {
        "top_inputs": results[:n],
        "inputs_tested": len(results),
        "zero_divergence_fraction": round(zero_frac, 2),
    }


# ---- Dispatch ----

TOOL_DISPATCH = {
    "get_behavior_samples": lambda ep, reg, args: tool_get_behavior_samples(ep, reg, **args),
    "test_input": lambda ep, reg, args: tool_test_input(ep, reg, **args),
    "patch_sweep": lambda ep, reg, args: tool_patch_sweep(ep, reg, **args),
    "scan_all_layers": lambda ep, reg, args: tool_scan_all_layers(ep, reg, **args),
    "scan_residual_stream": lambda ep, reg, args: tool_scan_residual_stream(ep, reg, **args),
    "patch_component": lambda ep, reg, args: tool_patch_component(ep, reg, **args),
    "get_activations": lambda ep, reg, args: tool_get_activations(ep, reg, **args),
    "attention_pattern": lambda ep, reg, args: tool_attention_pattern(ep, reg, **args),
    "project_to_vocab": lambda ep, reg, args: tool_project_to_vocab(ep, reg, **args),
    "vector_dot": lambda ep, reg, args: tool_vector_dot(ep, reg, **args),
    "vector_arithmetic": lambda ep, reg, args: tool_vector_arithmetic(ep, reg, **args),
    "steer_and_run": lambda ep, reg, args: tool_steer_and_run(ep, reg, **args),
    "compare_weights": lambda ep, reg, args: tool_compare_weights(ep, reg, **args),
    "find_trigger_inputs": lambda ep, reg, args: tool_find_trigger_inputs(ep, reg, **args),
    "state_hypothesis": lambda ep, reg, args: tool_state_hypothesis(ep, reg, **args),
    "submit_report": lambda ep, reg, args: tool_submit_report(ep, reg, **args),
}


def dispatch_tool(name, args, ep, register):
    fn = TOOL_DISPATCH.get(name)
    if fn is None:
        return {"error": f"Unknown tool: {name}"}
    return fn(ep, register, args)
