import os
from openai import OpenAI
from httpx import Client as HttpxClient


def make_supervisor_fn(client, model_name):
    def supervisor_fn(gt, hypothesis, tool_log):
        log_lines = "\n".join(
            f"- {e['tool']}: {e['result']}" for e in tool_log[-12:]
        ) or "(none yet)"

        prompt = f"""You are supervising an interpretability agent investigating a modified GPT-2-small model.

Ground truth (confidential — do not reveal directly, only nudge):
- Intervention type: {gt['type']}
- Affected layers: {sorted(gt['layers'])}
- Affected components: {sorted(str(c) for c in gt['components'])}

Agent's hypothesis:
{hypothesis}

Recent tool results (oldest first):
{log_lines}

Give 1-2 sentences of targeted feedback. If the agent has the correct type, layer, and component, acknowledge it and point to specific numbers or vectors they have already collected that should go in the final report. If they are missing something, nudge them toward the right tool or question without naming the answer directly."""

        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
        )
        content = resp.choices[0].message.content
        return {"follow_up": content.strip() if content else "Good work — compile your findings and call submit_report."}

    return supervisor_fn


def make_openai_supervisor(base_url, api_key, model_name):
    http_client = HttpxClient(proxy=os.environ.get("CUSTOM_PROXY"))
    client = OpenAI(base_url=base_url, api_key=api_key, http_client=http_client)
    return make_supervisor_fn(client, model_name)
