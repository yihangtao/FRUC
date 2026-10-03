"""Normalize the checkpoint containers used by the public FRUC entry points."""
from collections.abc import Mapping


def extract_state_dict(checkpoint):
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Expected a state dictionary or a checkpoint mapping.")
    state = checkpoint
    for key in ("model_state_dict", "state_dict", "model"):
        if key in checkpoint and isinstance(checkpoint[key], Mapping):
            state = checkpoint[key]
            break
    if not state or any(not isinstance(key, str) for key in state):
        raise ValueError("Checkpoint does not contain a non-empty parameter dictionary.")
    return {key.removeprefix("module."): value for key, value in state.items()}
