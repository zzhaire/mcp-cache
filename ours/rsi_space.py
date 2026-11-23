"""Search space for RSI over M1–M4 scheduler policies.

The space is read from ``ours/scheduler.schema.json``: every leaf parameter
tagged ``x-tier`` L1 (structural switch) or L2 (numeric value) is tunable, and
its range is the schema's range. L0 (``locked``) parameters are never part of
the space, and every generated candidate must still pass the full policy
validation, so a search cannot produce an unsafe or malformed policy.

A candidate is always one change away from its parent (coordinate search):
numbers are scaled by the pre-registered factors and clipped to the schema
range, booleans are flipped, enums take each other value, rule sets toggle one
member, and capacity shares move a fixed slice between modules so they keep
summing to 1.
"""
import copy
import json

from exp_core.scheduler_policy import SCHEMA_PATH, validate

TUNABLE_TIERS = ("L1", "L2")
SHARE_PATH = "capacity.share"
MODULES = ("M1", "M2", "M3", "M4")


def load_schema(path=SCHEMA_PATH):
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_ref(node, schema):
    while "$ref" in node:
        target = schema
        for part in node["$ref"].lstrip("#/").split("/"):
            target = target[part]
        node = {**target, **{k: v for k, v in node.items() if k != "$ref"}}
    return node


def tunable_space(schema=None):
    """``{dotted.path: spec}`` for every L1/L2 leaf of the scheduler schema."""
    schema = schema or load_schema()
    space = {}

    def walk(node, path, tier):
        node = _resolve_ref(node, schema)
        tier = node.get("x-tier", tier)
        if "const" in node:
            return
        types = node.get("type")
        types = types if isinstance(types, list) else [types]
        if "object" in types and "properties" in node:
            for key, child in node["properties"].items():
                walk(child, path + [key], tier)
            return
        if tier not in TUNABLE_TIERS or not path:
            return
        name = ".".join(path)
        if "object" in types:
            return  # free-form maps (e.g. fresh_class_overrides) are not searched
        if "array" in types:
            items = _resolve_ref(node.get("items", {}), schema)
            if "enum" in items:
                space[name] = {"tier": tier, "kind": "set", "values": list(items["enum"])}
            return
        if "enum" in node:
            space[name] = {"tier": tier, "kind": "enum", "values": list(node["enum"])}
        elif "boolean" in types:
            space[name] = {"tier": tier, "kind": "boolean"}
        elif "integer" in types or "number" in types:
            spec = {"tier": tier, "kind": "integer" if "integer" in types else "number",
                    "nullable": "null" in types}
            if "minimum" in node:
                spec["min"] = node["minimum"]
            if "exclusiveMinimum" in node:
                spec["min_exclusive"] = node["exclusiveMinimum"]
            if "maximum" in node:
                spec["max"] = node["maximum"]
            space[name] = spec

    walk({"type": "object", "properties": schema["properties"]}, [], None)
    return space


def get_value(policy, path):
    node = policy
    for part in path.split("."):
        node = node[part]
    return node


def with_value(policy, path, value):
    out = copy.deepcopy(policy)
    node = out
    parts = path.split(".")
    for part in parts[:-1]:
        node = node[part]
    node[parts[-1]] = value
    return out


def _numeric_values(current, spec, factors):
    if current is None:
        return []  # "no limit" stays as is; the search does not invent a bound
    low = spec.get("min", spec.get("min_exclusive"))
    high = spec.get("max")
    out = []
    for factor in factors:
        value = current * factor
        if current == 0:
            # Multiplying zero goes nowhere: step to the smallest useful value.
            value = 1 if factor > 1 else None
        if value is None:
            continue
        if spec["kind"] == "integer":
            value = int(round(value))
            if value == current:
                value = current + (1 if factor > 1 else -1)
        if low is not None:
            value = max(low, value)
        if high is not None:
            value = min(high, value)
        if "min_exclusive" in spec and value <= spec["min_exclusive"]:
            continue
        if value != current and value not in out:
            out.append(value)
    return out


def _share_neighbours(policy, slice_size):
    share = get_value(policy, SHARE_PATH)
    out = []
    for target in MODULES:
        donor = max((m for m in MODULES if m != target), key=lambda m: share[m])
        if share[donor] - slice_size < 0:
            continue
        moved = dict(share)
        moved[target] = round(moved[target] + slice_size, 6)
        moved[donor] = round(moved[donor] - slice_size, 6)
        out.append((f"+{target}", with_value(policy, SHARE_PATH, moved)))
    return out


def neighbours(policy, path, space, factors=(0.5, 2), share_slice=0.1):
    """Valid one-change neighbours of ``policy`` along ``path``: [(label, policy)]."""
    check_axes([path], space)
    if path == SHARE_PATH:
        candidates = _share_neighbours(policy, share_slice)
    else:
        spec, current = space[path], get_value(policy, path)
        if spec["kind"] == "boolean":
            values = [not current]
        elif spec["kind"] == "enum":
            values = [value for value in spec["values"] if value != current]
        elif spec["kind"] == "set":
            values = [sorted(set(current) ^ {member}) for member in spec["values"]]
        else:
            values = _numeric_values(current, spec, factors)
        candidates = [(_label(value), with_value(policy, path, value)) for value in values]
    valid = []
    for label, candidate in candidates:
        try:
            validate(candidate)
        except ValueError:
            continue  # e.g. a switch that scheduler-v0 does not implement
        valid.append((label, candidate))
    return valid


def _label(value):
    if isinstance(value, list):
        return "[" + ",".join(value) + "]"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value).lower() if isinstance(value, bool) else str(value)


def check_axes(axes, space):
    """Reject axes outside the tunable space (L0, unknown or share leaves)."""
    for path in axes:
        if path == SHARE_PATH:
            continue
        if path.startswith(SHARE_PATH + "."):
            raise ValueError(f"tune {SHARE_PATH} as a whole, not {path}")
        if path not in space:
            raise ValueError(f"{path} is not a tunable (L1/L2) scheduler parameter")
