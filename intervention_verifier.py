from base.verifier import TrajectoryVerifier
from base.data import Data


class InterventionVerifier(TrajectoryVerifier):
    def verify_trajectory(self, env, data: Data, actions, max_steps=None):
        env.reset(data)
        total_reward = 0.0
        info_trace = []
        invalid_actions = 0
        limit = max_steps if max_steps is not None else len(actions)

        for action in actions[:limit]:
            obs, reward, done, info = env.step(action)
            total_reward += reward
            info_trace.append(info)
            if info.get("invalid"):
                invalid_actions += 1
            if done:
                break

        steps = len(info_trace)
        final_info = info_trace[-1] if info_trace else {}
        score = final_info.get("score", {})

        return {
            "success": score.get("total", 0.0) >= 0.5,
            "total_reward": total_reward,
            "steps": steps,
            "tool_calls": steps,
            "policy_violations": final_info.get("policy_violations", 0),
            "terminated_early": done and steps < len(actions),
            "invalid_actions": invalid_actions,
            "info_trace": info_trace,
            "score": score,
        }
