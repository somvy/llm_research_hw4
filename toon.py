def _scalar(v):
    if isinstance(v, bool): return "true" if v else "false"
    if v is None: return "null"
    if isinstance(v, float): return f"{v:.5g}"
    return str(v)


def _col_val(v):
    s = _scalar(v)
    return f'"{s}"' if ("," in s or "\n" in s) else s


def _uniform_dict_keys(lst):
    if not lst or not all(isinstance(x, dict) for x in lst): return None
    keys = list(lst[0].keys())
    if not all(list(x.keys()) == keys for x in lst): return None
    for item in lst:
        for v in item.values():
            if isinstance(v, (dict, list, tuple)):
                return None
    return keys


def _uniform_tuple_rows(lst):
    if not lst or not all(isinstance(x, (tuple, list)) for x in lst): return False
    n = len(lst[0])
    return all(len(x) == n and all(not isinstance(v, (dict, list, tuple)) for v in x) for x in lst)


def encode(obj, _indent=0):
    pad = "  " * _indent

    if isinstance(obj, (list, tuple)) and not isinstance(obj, str):
        lst = list(obj)
        if not lst:
            return "[]"
        if all(not isinstance(x, (dict, list, tuple)) for x in lst):
            return f"[{len(lst)}]: " + ",".join(_col_val(x) for x in lst)
        keys = _uniform_dict_keys(lst)
        if keys:
            header = "{" + ",".join(keys) + "}"
            rows = [pad + "  " + ",".join(_col_val(item[k]) for k in keys) for item in lst]
            return f"[{len(lst)}]{header}:\n" + "\n".join(rows)
        if _uniform_tuple_rows(lst):
            rows = [pad + "  " + ",".join(_col_val(v) for v in row) for row in lst]
            return f"[{len(lst)}]:\n" + "\n".join(rows)
        # fallback: one item per line
        lines = []
        for item in lst:
            if isinstance(item, (dict, list, tuple)) and not isinstance(item, str):
                enc = encode(item, _indent + 1)
                enc_lines = enc.split("\n")
                inner_pad = "  " * (_indent + 1)
                first = enc_lines[0][len(inner_pad):] if enc_lines[0].startswith(inner_pad) else enc_lines[0]
                lines.append(pad + "- " + first)
                lines.extend(enc_lines[1:])
            else:
                lines.append(pad + "- " + _scalar(item))
        return "\n".join(lines)

    elif isinstance(obj, dict):
        lines = []
        for k, v in obj.items():
            if isinstance(v, (list, tuple)) and not isinstance(v, str):
                if not v:
                    lines.append(f"{pad}{k}: []")
                else:
                    lst_repr = encode(v, _indent + 1)
                    if lst_repr.startswith("["):
                        lines.append(f"{pad}{k}{lst_repr}")
                    else:
                        lines.append(f"{pad}{k}:")
                        lines.append(lst_repr)
            elif isinstance(v, dict):
                lines.append(f"{pad}{k}:")
                lines.append(encode(v, _indent + 1))
            else:
                lines.append(f"{pad}{k}: {_scalar(v)}")
        return "\n".join(lines)

    else:
        return _scalar(obj)
