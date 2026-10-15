"""Validated timed inputs for real replay; separate data, not model policy."""
import math


def validate_schedule(document, duration):
    if not isinstance(document, dict) or set(document) != {"initial_prompt", "requests"}:
        raise ValueError("schedule requires initial_prompt and requests")
    def prompt(value):
        if not isinstance(value, str) or not value.strip() or len(value) > 5000:
            raise ValueError("invalid schedule prompt")
        return value
    initial = prompt(document["initial_prompt"])
    if not isinstance(document["requests"], list) or len(document["requests"]) < 2:
        raise ValueError("replay requires at least two condition updates")
    previous = -1.
    requests = []
    for item in document["requests"]:
        if not isinstance(item, dict) or set(item) != {"at_seconds", "prompt"}:
            raise ValueError("schedule request requires at_seconds and prompt")
        when = item["at_seconds"]
        if type(when) not in (float, int) or not math.isfinite(when) or not previous <= when < duration:
            raise ValueError("schedule times must be finite, ordered and within replay")
        if when < 0:
            raise ValueError("schedule times must be nonnegative")
        requests.append((float(when), prompt(item["prompt"])))
        previous = when
    return initial, requests
