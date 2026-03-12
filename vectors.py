import torch


class VectorRegister:
    def __init__(self):
        self.vecs = {}
        self._n = 0

    def store(self, t, label=""):
        vid = f"v_{self._n:04d}_{label}"
        self._n += 1
        self.vecs[vid] = t.detach().clone().float().cpu()
        return vid

    def get(self, vid):
        return self.vecs[vid]

    def arithmetic(self, ops):
        result = torch.zeros(768)
        for op in ops:
            v = self.get(op["vector_id"])
            s = op.get("scalar", 1.0)
            if op["op"] == "add":
                result = result + s * v
            elif op["op"] == "sub":
                result = result - s * v
            elif op["op"] == "scale":
                result = s * result
        vid = self.store(result, "arith")
        return vid, result.norm().item()
