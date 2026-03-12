import torch
import torch.nn.functional as F
from contextlib import contextmanager
from transformer_lens import HookedTransformer

N_LAYERS = 12
N_HEADS = 12
D_MODEL = 768

_model = None


def get_model():
    global _model
    if _model is None:
        print("\nLoading GPT-2 small...")
        _model = HookedTransformer.from_pretrained("gpt2", device="cuda")
        _model.cfg.use_attn_result = True
        print("\nModel loaded")
    return _model


def tokenize(text):
    m = get_model()
    return m.to_tokens(text)


def top_k_probs(logits, k=5):
    probs = F.softmax(logits, dim=-1)
    vals, idxs = probs.topk(k)
    m = get_model()
    return [(m.tokenizer.decode(idxs[i].item()), vals[i].item()) for i in range(k)]


def run_base(tokens, hooks=None):
    m = get_model()
    if hooks:
        return m.run_with_hooks(tokens, fwd_hooks=hooks)
    return m(tokens)


def run_base_with_cache(tokens):
    m = get_model()
    logits, cache = m.run_with_cache(tokens, return_cache_object=True)
    return logits, cache


def run_modified(tokens, intervention, extra_hooks=None):
    m = get_model()
    hooks = intervention["hooks_fn"](tokens)
    if extra_hooks:
        hooks = hooks + extra_hooks
    deltas = intervention.get("weight_deltas")
    if deltas:
        with weight_delta_ctx(m, deltas):
            return m.run_with_hooks(tokens, fwd_hooks=hooks) if hooks else m(tokens)
    if hooks:
        return m.run_with_hooks(tokens, fwd_hooks=hooks)
    return m(tokens)


def run_modified_with_cache(tokens, intervention):
    m = get_model()
    hooks = intervention["hooks_fn"](tokens)
    deltas = intervention.get("weight_deltas")
    if deltas:
        with weight_delta_ctx(m, deltas):
            if hooks:
                # run_with_hooks doesn't return cache, so we add hooks manually
                for name, fn in hooks:
                    m.add_hook(name, fn)
                try:
                    logits, cache = m.run_with_cache(tokens, return_cache_object=True)
                finally:
                    m.reset_hooks()
                return logits, cache
            else:
                return m.run_with_cache(tokens, return_cache_object=True)
    if hooks:
        for name, fn in hooks:
            m.add_hook(name, fn)
        try:
            logits, cache = m.run_with_cache(tokens, return_cache_object=True)
        finally:
            m.reset_hooks()
        return logits, cache
    return m.run_with_cache(tokens, return_cache_object=True)


def run_modified_with_extra_hooks(tokens, intervention, extra_hooks):
    m = get_model()
    hooks = intervention["hooks_fn"](tokens) + extra_hooks
    deltas = intervention.get("weight_deltas")
    if deltas:
        with weight_delta_ctx(m, deltas):
            return m.run_with_hooks(tokens, fwd_hooks=hooks)
    return m.run_with_hooks(tokens, fwd_hooks=hooks)


@contextmanager
def weight_delta_ctx(mdl, deltas):
    for name, delta in deltas.items():
        param = _get_param(mdl, name)
        param.data.add_(delta.to(param.device))
    try:
        yield
    finally:
        for name, delta in deltas.items():
            param = _get_param(mdl, name)
            param.data.sub_(delta.to(param.device))


def _get_param(mdl, dotpath):
    obj = mdl
    for attr in dotpath.split("."):
        obj = getattr(obj, attr)
    return obj


def generate_greedy(text, max_tokens=50, intervention=None):
    m = get_model()
    tokens = m.to_tokens(text)
    for _ in range(max_tokens):
        if intervention:
            logits = run_modified(tokens, intervention)
        else:
            logits = m(tokens)
        next_tok = logits[0, -1].argmax(dim=-1, keepdim=True).unsqueeze(0)
        tokens = torch.cat([tokens, next_tok], dim=1)
        if next_tok.item() == m.tokenizer.eos_token_id:
            break
    # return only the generated part
    input_len = m.to_tokens(text).shape[1]
    return m.tokenizer.decode(tokens[0, input_len:].tolist())


def compute_kl(logits_a, logits_b, pos=-1):
    p = F.softmax(logits_a[0, pos], dim=-1)
    q = F.log_softmax(logits_b[0, pos], dim=-1)
    return F.kl_div(q, p, reduction="sum").item()


def compute_mean_cache(prompts):
    m = get_model()
    print("\nComputing mean activation cache...")
    sums = {}
    count = 0
    with torch.no_grad():
        for p in prompts:
            tokens = m.to_tokens(p)
            _, cache = m.run_with_cache(tokens, return_cache_object=True)
            for L in range(N_LAYERS):
                # heads
                hook_result = cache[f"blocks.{L}.attn.hook_result"]
                for h in range(N_HEADS):
                    key = (L, f"head.{h}")
                    val = hook_result[:, :, h, :].mean(dim=(0, 1))
                    sums[key] = sums.get(key, torch.zeros(D_MODEL, device=val.device)) + val
                # mlp
                mlp_out = cache[f"blocks.{L}.hook_mlp_out"]
                key = (L, "mlp")
                val = mlp_out.mean(dim=(0, 1))
                sums[key] = sums.get(key, torch.zeros(D_MODEL, device=val.device)) + val
            count += 1
    mean_cache = {k: v / count for k, v in sums.items()}
    print(f"\nMean cache computed over {count} prompts")
    return mean_cache
