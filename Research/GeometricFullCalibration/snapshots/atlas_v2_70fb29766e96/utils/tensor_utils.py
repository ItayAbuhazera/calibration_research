import torch


def coerce_to_tensor(out):
    """Coerce common structured outputs into a torch.Tensor or return None.

    Handles:
    - torch.Tensors directly
    - HuggingFace-style BaseModelOutput dataclasses (attrs: last_hidden_state, pooler_output, logits, hidden_states)
    - nested tuples/lists/dicts by recursively searching for the first tensor-like
    """
    if torch.is_tensor(out):
        return out

    # HuggingFace-style dataclasses
    for attr in ("last_hidden_state", "pooler_output", "logits", "hidden_states"):
        if hasattr(out, attr):
            v = getattr(out, attr)
            if attr == "hidden_states" and isinstance(v, (list, tuple)) and len(v) > 0:
                v = v[-1]
            if torch.is_tensor(v):
                return v
            # nested
            if isinstance(v, (list, tuple, dict)):
                t = coerce_to_tensor(v)
                if t is not None:
                    return t

    # tuple/list: take first tensor (or recurse)
    if isinstance(out, (list, tuple)):
        for x in out:
            if torch.is_tensor(x):
                return x
            t = coerce_to_tensor(x)
            if t is not None:
                return t

    # dict: try known keys then any value
    if isinstance(out, dict):
        for k in ("last_hidden_state", "pooler_output", "logits", "hidden_states"):
            if k in out:
                t = coerce_to_tensor(out[k])
                if t is not None:
                    return t
        for v in out.values():
            if torch.is_tensor(v):
                return v
            t = coerce_to_tensor(v)
            if t is not None:
                return t

    return None


