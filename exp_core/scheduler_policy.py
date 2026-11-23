"""Load, merge and validate M1–M4 scheduler policies (docs/m1-m4-scheduler.md §7).

A scheduler spec is one of:
- a preset name from configs/scheduler-presets.json ("default", "ablation_current", ...);
- a path to a JSON file holding a full policy, or ``{"base": <preset>, "overrides": {...}}``;
- a dict in either of those two shapes.

The resolved policy is always a complete, schema-valid dict so that its hash
covers every value the scheduler uses.
"""
import copy
import json
from pathlib import Path

from exp_core.settings import CONFIGS, ROOT

SCHEMA_PATH = ROOT / "ours" / "scheduler.schema.json"
PRESETS_PATH = CONFIGS / "scheduler-presets.json"
VERSION = "scheduler-v0"


def deep_merge(base, overrides):
    out = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_presets(path=PRESETS_PATH):
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    presets = {"default": copy.deepcopy(document["default"])}
    for name, overrides in document.get("presets", {}).items():
        merged = deep_merge(document["default"], overrides)
        merged["preset"] = name
        presets[name] = merged
    return presets


def validate(policy):
    """Raise ValueError when the policy breaks the schema or a loader rule."""
    import jsonschema
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    errors = sorted(jsonschema.Draft202012Validator(schema).iter_errors(policy),
                    key=lambda error: list(error.path))
    if errors:
        raise ValueError("scheduler policy invalid: " + "; ".join(
            f"{'/'.join(map(str, e.path)) or '<root>'}: {e.message}" for e in errors[:5]))
    share = sum(policy["capacity"]["share"].values())
    if abs(share - 1) > 1e-6:
        raise ValueError(f"scheduler policy invalid: capacity.share sums to {share}, not 1")
    if policy["recovery"]["allow_revalidate"]:
        raise ValueError("scheduler policy invalid: recovery.allow_revalidate is not "
                         "implemented in scheduler-v0")
    return policy


def resolve(spec, presets_path=PRESETS_PATH):
    """Return a complete, validated scheduler policy, or None for no spec."""
    if spec is None:
        return None
    presets = load_presets(presets_path)
    if isinstance(spec, str):
        if spec in presets:
            return validate(presets[spec])
        path = Path(spec)
        if not path.is_file():
            raise ValueError(f"unknown scheduler preset or file: {spec}")
        spec = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(spec, dict):
        raise ValueError("scheduler spec must be a preset name, a path or a JSON object")
    if spec.get("version") == VERSION:
        return validate(copy.deepcopy(spec))
    unknown = set(spec) - {"base", "overrides"}
    if unknown:
        raise ValueError(f"scheduler spec has unexpected keys: {', '.join(sorted(unknown))}")
    base = spec.get("base", "default")
    if base not in presets:
        raise ValueError(f"unknown scheduler base preset: {base}")
    merged = deep_merge(presets[base], spec.get("overrides", {}))
    return validate(merged)


def apply_to_arm(policy):
    """Resolve ``policy["scheduler"]`` in place and make it the single source of
    the publication settings. Returns the policy (unchanged without a scheduler)."""
    spec = policy.get("scheduler")
    if spec is None:
        return policy
    scheduler = resolve(spec)
    publish = scheduler["publish"]
    policy["scheduler"] = scheduler
    policy.update(publish_read_results=publish["read_results"],
                  publish_budget_bytes=publish["budget_bytes"],
                  publish_operations=publish["operations"],
                  context_dedup=publish["context_dedup"])
    return policy
