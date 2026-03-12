import re


def extract_findings(report):
    # look for FINDINGS section
    match = re.search(r"FINDINGS:(.*)", report, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1)
    return report


def extract_type(report):
    findings = extract_findings(report).lower()
    types = [
        "head_ablation", "mlp_ablation", "mean_ablation",
        "steering_vector", "rank1_edit", "multi_ablation",
        "conditional_steering", "distributed_finetune", "adversarial_patch",
    ]
    for t in types:
        if t in findings:
            return t
    # fuzzy matching
    if "head" in findings and "ablat" in findings:
        return "head_ablation"
    if "mlp" in findings and "ablat" in findings:
        return "mlp_ablation"
    if "mean" in findings and "ablat" in findings:
        return "mean_ablation"
    if "steer" in findings and "condition" in findings:
        return "conditional_steering"
    if "steer" in findings or "direction" in findings:
        return "steering_vector"
    if "rank" in findings or "edit" in findings or "rome" in findings:
        return "rank1_edit"
    if "multi" in findings:
        return "multi_ablation"
    if "finetun" in findings or "train" in findings:
        return "distributed_finetune"
    if "ablat" in findings:
        return "head_ablation"  # default ablation guess
    return None


def extract_components(report):
    findings = extract_findings(report)
    comps = set()

    # extract all mentioned layers and components separately, then combine
    layers = set()
    for m in re.finditer(r"layer\s*(\d+)", findings, re.I):
        layers.add(int(m.group(1)))
    for m in re.finditer(r"L(\d+)", findings):
        layers.add(int(m.group(1)))

    # direct patterns: "layer X head Y" or "X.Y" on same line
    for m in re.finditer(r"layer\s*(\d+).*?head[\s.](\d+)", findings, re.I):
        comps.add((int(m.group(1)), f"head.{m.group(2)}"))
    for m in re.finditer(r"L(\d+).*?head[\s.](\d+)", findings, re.I):
        comps.add((int(m.group(1)), f"head.{m.group(2)}"))
    for m in re.finditer(r"(\d+)\.(\d+)", findings):
        L, H = int(m.group(1)), int(m.group(2))
        if 0 <= L <= 11 and 0 <= H <= 11:
            comps.add((L, f"head.{H}"))

    # same-line patterns
    for m in re.finditer(r"layer\s*(\d+).*?mlp", findings, re.I):
        comps.add((int(m.group(1)), "mlp"))
    for m in re.finditer(r"L(\d+).*?mlp", findings, re.I):
        comps.add((int(m.group(1)), "mlp"))
    for m in re.finditer(r"layer\s*(\d+).*?resid", findings, re.I):
        comps.add((int(m.group(1)), "resid"))
    for m in re.finditer(r"L(\d+).*?resid", findings, re.I):
        comps.add((int(m.group(1)), "resid"))

    # fallback: if layers found and component keywords mentioned separately, combine them
    if not comps and layers:
        findings_lower = findings.lower()
        has_head = re.search(r"head[\s.](\d+)", findings, re.I)
        if has_head:
            h = int(has_head.group(1))
            for L in layers:
                comps.add((L, f"head.{h}"))
        if "mlp" in findings_lower:
            for L in layers:
                comps.add((L, "mlp"))
        if "resid" in findings_lower:
            for L in layers:
                comps.add((L, "resid"))

    return comps


def extract_layers(report):
    findings = extract_findings(report)
    layers = set()
    for m in re.finditer(r"layer\s*(\d+)", findings, re.I):
        layers.add(int(m.group(1)))
    for m in re.finditer(r"L(\d+)", findings):
        layers.add(int(m.group(1)))
    return layers


def compute_f1(predicted, ground_truth):
    if not predicted and not ground_truth:
        return 1.0
    if not predicted or not ground_truth:
        return 0.0
    inter = predicted & ground_truth
    precision = len(inter) / len(predicted)
    recall = len(inter) / len(ground_truth)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate(report, intervention, tool_calls_used):
    gt = intervention["ground_truth"]

    # 1. intervention type accuracy (0.25)
    pred_type = extract_type(report)
    type_score = 1.0 if pred_type == gt["type"] else 0.0

    # 2. component localization F1 (0.30)
    pred_comps = extract_components(report)
    f1 = compute_f1(pred_comps, gt["components"])

    # 3. layer accuracy IoU (0.15)
    pred_layers = extract_layers(report)
    gt_layers = gt["layers"]
    union = pred_layers | gt_layers
    layer_iou = len(pred_layers & gt_layers) / len(union) if union else 0.0

    # 4. key fact extraction (0.20)
    report_lower = report.lower()
    key_facts = gt.get("key_facts", {})
    if key_facts:
        found = 0
        for fact_name, patterns in key_facts.items():
            if any(p.lower() in report_lower for p in patterns):
                found += 1
        fact_score = found / len(key_facts)
    else:
        fact_score = 0.0

    # 5. efficiency (0.10)
    optimal = gt.get("optimal_calls", 3)
    budget = 40
    eff = max(0.0, 1.0 - (tool_calls_used - optimal) / budget)

    total = (
        0.25 * type_score
        + 0.30 * f1
        + 0.15 * layer_iou
        + 0.20 * fact_score
        + 0.10 * eff
    )

    return {
        "total": total,
        "type_accuracy": type_score,
        "component_f1": f1,
        "layer_iou": layer_iou,
        "key_fact_score": fact_score,
        "efficiency": eff,
        "predicted_type": pred_type,
        "predicted_components": list(pred_comps),
        "predicted_layers": list(pred_layers),
        "tool_calls_used": tool_calls_used,
    }
