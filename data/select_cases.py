"""Pre-register the 36-case smoke matrix from task structure and hash order only.

The selector reads only data/manifest.json and the three frozen task files. It
never reads results, metrics, or observed performance, and it never modifies the
task files. Cases are ranked by sha256(seed:dataset:task_id).
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

from exp_core.settings import ROOT, DATA
DEFAULT_OUTPUT = DATA / "scenarios-36.json"

VERSION = "36-case-smoke-v1"
GENERATED_AT = "2026-11-16T00:00:00+08:00"
DATASETS = ("appworld", "bfcl", "tau2")
SCENARIOS = ("discovery", "read", "auth_operation")
SPLITS = ("dev", "heldout")
TASK_FILES = {
    "appworld": "appworld-tasks.json",
    "bfcl": "bfcl-tasks.json",
    "tau2": "tau2-tasks.json",
}
EXPECTED_MODULES = {
    "appworld": {"discovery": ["M1"], "read": ["M2"], "auth_operation": ["M3"]},
    "bfcl": {"discovery": ["M1", "M2"], "read": ["M2"], "auth_operation": ["M4"]},
    "tau2": {"discovery": ["M1", "M2"], "read": ["M2"], "auth_operation": ["M4"]},
}

# Allocate the narrowest structural pool first so a broad scenario cannot consume
# every hash-ranked candidate before a narrow one is filled. Output order stays
# discovery, read, auth_operation regardless of allocation order.
SCENARIO_PRIORITY = {
    "appworld": ("read", "discovery", "auth_operation"),
    "bfcl": ("discovery", "read", "auth_operation"),
    "tau2": ("discovery", "read", "auth_operation"),
}

# One template is applied to every case, so every arm receives the same extra
# requirement. Only a runtime that publishes a validated result for the exact
# read can satisfy it without issuing the call again.
AUGMENTATION_TEXT = (
    "Re-confirm the read-only results you obtained most recently in the completed "
    "request: take up to the three most recent distinct read-only calls, most recent "
    "first. For each of them, if the current decision context lists a runtime-validated "
    "result for exactly that read, restate it without calling; otherwise repeat that "
    "exact read-only call once. Never repeat a call more than once, never make any other "
    "call, and finish this whole re-confirmation now; then reply done=true and do not "
    "call any tool again."
)

AUGMENTATION_RULES = {
    ("appworld", "discovery"): (
        "If a pilot trace contains no repeated namespace/API-doc or catalog discovery, "
        "append one verification turn asking the agent to re-check the same discovered "
        "API/catalog entry and confirm it is unchanged before continuing. Keep the "
        "original oracle, set augmented=true, record source_task_id and the appended text."
    ),
    ("appworld", "read"): (
        "If a pilot trace contains no repeated read of the same resource or status, "
        "append one verification turn asking the agent to read the same resource again "
        "and confirm the value is unchanged. Keep the original oracle, set augmented=true, "
        "record source_task_id and the appended text."
    ),
    ("appworld", "auth_operation"): (
        "If a pilot trace contains no repeated credential use, append one verification "
        "turn asking the agent to re-authenticate with the same credential and confirm "
        "the session state is unchanged before continuing. Keep the original oracle, "
        "set augmented=true, record source_task_id and the appended text."
    ),
    ("bfcl", "discovery"): (
        "If a pilot trace contains no repeated namespace/file-directory discovery, "
        "append one verification turn asking the agent to re-check the same discovered "
        "tool namespace or directory entry and confirm it is unchanged. Keep the "
        "original oracle, set augmented=true, record source_task_id and the appended text."
    ),
    ("bfcl", "read"): (
        "If a pilot trace contains no repeated read of the same file or status, append "
        "one verification turn asking the agent to read the same file/status again and "
        "confirm the value is unchanged. Keep the original oracle, set augmented=true, "
        "record source_task_id and the appended text."
    ),
    ("bfcl", "auth_operation"): (
        "BFCL has no credential issuer, so M4 is used instead of M3. If a pilot trace "
        "contains no repeated failed/uncertain operation state, append one verification "
        "turn asking the agent to re-check the same operation status before any retry. "
        "Keep the original oracle, set augmented=true, record source_task_id and the text."
    ),
    ("tau2", "discovery"): (
        "If a pilot trace contains no repeated customer/line lookup, append one "
        "verification turn asking the agent to look up the same customer or line again "
        "and confirm the record is unchanged. Keep the original oracle, set augmented=true, "
        "record source_task_id and the appended text."
    ),
    ("tau2", "read"): (
        "If a pilot trace contains no repeated device/network status read, append one "
        "verification turn asking the agent to check the same status again and confirm "
        "it is unchanged. Keep the original oracle, set augmented=true, record "
        "source_task_id and the appended text."
    ),
    ("tau2", "auth_operation"): (
        "tau2 telecom has no credential issuer, so M4 is used instead of M3. If a pilot "
        "trace contains no repeated write/uncertain operation state, append one "
        "verification turn asking the agent to reconcile the same operation status "
        "before any retry. Keep the original oracle, set augmented=true, record "
        "source_task_id and the appended text."
    ),
}

APPWORLD_APP_PATTERNS = {
    "spotify": ("spotify",),
    "venmo": ("venmo",),
    "simple_note": ("simple note", "simple_note", "simplenote"),
    "phone": ("phone", "text message", "sms"),
    "file_system": ("file system", "file_system", "~/", "directory", "directories"),
    "gmail": ("gmail", "email", "e-mail"),
    "todoist": ("todoist",),
    "amazon": ("amazon",),
    "splitwise": ("splitwise",),
}

APPWORLD_MUTATION_PATTERNS = {
    "add": r"\badd(?:s|ed|ing)?\b",
    "arrange": r"\barrang(?:e|es|ed|ing)\b",
    "move": r"\bmov(?:e|es|ed|ing)\b",
    "send": r"\bsend(?:s|ing)?\b|\bsent\b",
    "pay": r"\bpay(?:ment|ments)?\b",
    "like": r"\blike(?:s|d)?\b",
    "play": r"\bplay\b|\bplaying\b",
    "start": r"\bstart(?:s|ed|ing)?\b",
    "remove": r"\bremov(?:e|es|ed|ing)\b",
    "delete": r"\bdelet(?:e|es|ed|ing)\b",
    "reject": r"\breject(?:s|ed|ing)?\b",
    "follow": r"\bfollow(?:s|ed|ing)?\b",
    "comment": r"\bcomment(?:s|ed|ing)?\b",
    "export": r"\bexport(?:s|ed|ing)?\b",
    "terminate": r"\bterminat(?:e|es|ed|ing)\b",
    "create": r"\bcreat(?:e|es|ed|ing)\b",
    "update": r"\bupdat(?:e|es|ed|ing)\b",
    "set": r"\bset\b",
    "book": r"\bbook(?:s|ed|ing)?\b",
    "cancel": r"\bcancel(?:s|ed|ing)?\b",
    "purchase": r"\bpurchas(?:e|es|ed|ing)\b",
    "order": r"\border(?:s|ed|ing)?\b",
}

APPWORLD_AUTH_PATTERN = (
    r"\blog ?in\b|\blogin\b|\bpassword\b|\baccess token\b|\bcredit card\b|"
    r"\bauthenticat(?:e|ion)\b"
)
APPWORLD_ENUMERATION_PATTERN = r"\b(all|each|every)\b"
APPWORLD_CATALOG_PATTERN = (
    r"\b(songs?|playlists?|artists?|albums?|transactions?|payments?|requests?|"
    r"bills?|librar(?:y|ies)|feed|queue|movies?|files?|director(?:y|ies))\b"
)
APPWORLD_API_PATTERN = r"\bapi\b|\bdocs?\b|\bdocumentation\b"

BFCL_MUTATION_TOOLS = {
    "GorillaFileSystem.mkdir",
    "GorillaFileSystem.touch",
    "GorillaFileSystem.cp",
    "GorillaFileSystem.mv",
    "GorillaFileSystem.rm",
    "TravelAPI.register_credit_card",
    "TravelAPI.book_flight",
    "TravelAPI.cancel_booking",
    "TravelAPI.purchase_insurance",
    "TravelAPI.set_budget_limit",
    "TicketAPI.create_ticket",
    "TicketAPI.close_ticket",
    "TicketAPI.resolve_ticket",
    "TradingBot.add_stock_to_watchlist",
    "TradingBot.remove_stock_from_watchlist",
    "TradingBot.update_market_status",
    "TradingBot.cancel_order",
    "TradingBot.place_order",
    "TradingBot.fund_account",
    "MessageAPI.send_message",
    "MessageAPI.delete_message",
    "VehicleControlAPI.fillFuelTank",
    "VehicleControlAPI.startEngine",
    "VehicleControlAPI.lockDoors",
    "VehicleControlAPI.set_navigation",
}

BFCL_DISCOVERY_TOOLS = {
    "GorillaFileSystem.find",
    "GorillaFileSystem.grep",
    "GorillaFileSystem.sort",
    "GorillaFileSystem.wc",
    "GorillaFileSystem.cat",
    "GorillaFileSystem.diff",
    "TravelAPI.list_all_airports",
    "TravelAPI.get_nearest_airport_by_city",
    "TradingBot.get_available_stocks",
    "TradingBot.get_watchlist",
    "TradingBot.get_symbol_by_name",
    "MessageAPI.view_messages_received",
}


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _hash_key(seed, dataset, task_id):
    return hashlib.sha256(f"{seed}:{dataset}:{task_id}".encode("utf-8")).hexdigest()


def _patterns(text, patterns):
    return sorted(name for name, pattern in patterns.items() if re.search(pattern, text))


def _appworld_apps(text):
    return sorted(app for app, needles in APPWORLD_APP_PATTERNS.items()
                  if any(needle in text for needle in needles))


def _appworld_rule(scenario, task):
    text = task["spec"]["instruction"].lower()
    apps = _appworld_apps(text)
    if scenario == "discovery":
        multi_app = len(apps) >= 2
        enumeration = bool(re.search(APPWORLD_ENUMERATION_PATTERN, text)) and bool(
            re.search(APPWORLD_CATALOG_PATTERN, text))
        api_docs = bool(re.search(APPWORLD_API_PATTERN, text))
        if not (multi_app or enumeration or api_docs):
            return None
        return (f"multi_app={','.join(apps) if multi_app else 'no'}; "
                f"catalog_enumeration={'yes' if enumeration else 'no'}; "
                f"api_docs={'yes' if api_docs else 'no'}")
    if scenario == "read":
        mutations = _patterns(text, APPWORLD_MUTATION_PATTERNS)
        interrogative = text.strip().endswith("?") or bool(re.match(
            r"^(what|how|which|who|whose|name|list)\b", text.strip()))
        if mutations or not interrogative or not apps:
            return None
        return f"interrogative=yes; mutation=none; apps={','.join(apps)}"
    mutations = _patterns(text, APPWORLD_MUTATION_PATTERNS)
    has_auth = bool(re.search(APPWORLD_AUTH_PATTERN, text))
    if not mutations and not has_auth:
        return None
    return (f"mutation={','.join(mutations) if mutations else 'none'}; "
            f"auth_language={'yes' if has_auth else 'no'}; "
            f"apps={','.join(apps) if apps else 'none'}")


def _bfcl_tools(task):
    return list(task.get("path") or [])


def _bfcl_rule(scenario, task):
    tools = _bfcl_tools(task)
    classes = list(task.get("involved_classes") or [])
    mutation_tools = sorted(set(tools) & BFCL_MUTATION_TOOLS)
    if scenario == "discovery":
        namespace_tools = sorted(set(tools) & BFCL_DISCOVERY_TOOLS)
        filesystem = "GorillaFileSystem" in classes
        if not filesystem and not namespace_tools:
            return None
        return (f"filesystem_namespace={'yes' if filesystem else 'no'}; "
                f"discovery_tools={','.join(namespace_tools) if namespace_tools else 'none'}; "
                f"classes={','.join(classes)}")
    if scenario == "read":
        read_tools = [tool for tool in tools if tool not in BFCL_MUTATION_TOOLS]
        if len(read_tools) < 2:
            return None
        return (f"read_or_state_tools={len(read_tools)} "
                f"({','.join(sorted(set(read_tools)))}); "
                f"mutation_tools={','.join(mutation_tools) if mutation_tools else 'none'}")
    if not mutation_tools:
        return None
    return (f"mutation_tools={','.join(mutation_tools)}; "
            f"classes={','.join(classes)}")


def _tau2_issue_type(task_id):
    match = re.match(r"^\[([a-z_]+)\]", task_id)
    return match.group(1) if match else "unknown"


def _tau2_action_names(task):
    criteria = task.get("evaluation_criteria") or {}
    actions = criteria.get("actions") or []
    return sorted({action.get("name") for action in actions if action.get("name")})


def _tau2_assertions(task):
    criteria = task.get("evaluation_criteria") or {}
    assertions = criteria.get("env_assertions") or []
    return sorted({item.get("func_name") for item in assertions if item.get("func_name")})


def _tau2_rule(scenario, task):
    task_id = task["id"]
    issue = _tau2_issue_type(task_id)
    ticket = task.get("ticket") or ""
    actions = _tau2_action_names(task)
    assertions = _tau2_assertions(task)
    if scenario == "discovery":
        has_customer_lookup = "Customer name:" in ticket and "phone number:" in ticket
        if issue != "service_issue" or not has_customer_lookup:
            return None
        return (f"issue_type={issue}; customer_lookup=yes; "
                f"assertions={','.join(assertions)}; actions={','.join(actions) if actions else 'none'}")
    if scenario == "read":
        if issue != "mobile_data_issue":
            return None
        return (f"issue_type={issue}; status_assertions={','.join(assertions)}; "
                f"actions={','.join(actions) if actions else 'none'}")
    if issue != "mms_issue":
        return None
    return (f"issue_type={issue}; write_recovery_actions={','.join(actions)}; "
            f"assertions={','.join(assertions)}")


RULES = {
    ("appworld", "discovery"): lambda task: _appworld_rule("discovery", task),
    ("appworld", "read"): lambda task: _appworld_rule("read", task),
    ("appworld", "auth_operation"): lambda task: _appworld_rule("auth_operation", task),
    ("bfcl", "discovery"): lambda task: _bfcl_rule("discovery", task),
    ("bfcl", "read"): lambda task: _bfcl_rule("read", task),
    ("bfcl", "auth_operation"): lambda task: _bfcl_rule("auth_operation", task),
    ("tau2", "discovery"): lambda task: _tau2_rule("discovery", task),
    ("tau2", "read"): lambda task: _tau2_rule("read", task),
    ("tau2", "auth_operation"): lambda task: _tau2_rule("auth_operation", task),
}


def _selection_reason(seed, dataset, scenario, split, task, evidence, rank, pool_size,
                      selected_index, fallback=None):
    task_id = task["id"]
    digest = _hash_key(seed, dataset, task_id)
    reason = (f"{dataset}.{scenario}.v1: {evidence}; "
              f"sha256({seed}:{dataset}:{task_id})={digest[:12]} "
              f"rank {rank}/{pool_size} among structurally eligible unused {split} tasks; "
              f"selected {selected_index}/2")
    if fallback:
        reason += f"; fallback: {fallback}"
    return reason


def _select_split(dataset, scenario, split, tasks, seed, used):
    rule = RULES[(dataset, scenario)]
    eligible = []
    for task in tasks:
        evidence = rule(task)
        if evidence:
            eligible.append((task, evidence))
    eligible.sort(key=lambda item: (_hash_key(seed, dataset, item[0]["id"]), item[0]["id"]))
    available = [item for item in eligible if item[0]["id"] not in used]

    picked = []
    for index, (task, evidence) in enumerate(available[:2], 1):
        picked.append((task, _selection_reason(
            seed, dataset, scenario, split, task, evidence, index, len(available), index)))

    if len(picked) < 2:
        picked_ids = {item[0]["id"] for item in picked}
        fallback_pool = [
            task for task in tasks
            if task["id"] not in used and task["id"] not in picked_ids
        ]
        fallback_pool.sort(key=lambda task: (_hash_key(seed, dataset, task["id"]), task["id"]))
        note = (f"only {len(available)} structurally eligible unused {split} task(s) "
                f"remained (<2); filled by hash order from remaining {split} tasks")
        for index, task in enumerate(fallback_pool[:2 - len(picked)], 1):
            picked.append((task, _selection_reason(
                seed, dataset, scenario, split, task, "structural rule not matched",
                index, len(fallback_pool), len(picked) + 1, fallback=note)))

    if len(picked) != 2:
        raise RuntimeError(f"cannot fill {dataset}/{scenario}/{split}: {len(picked)} cases")
    return picked


def _load_inputs():
    manifest_path = DATA / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    seed = manifest["seed"]
    task_hashes = {}
    for dataset, filename in TASK_FILES.items():
        path = DATA / filename
        digest = _sha256_bytes(path.read_bytes())
        declared = manifest["datasets"][dataset]["task_file_sha256"]
        if digest != declared:
            raise SystemExit(f"{filename} sha256 {digest} does not match manifest {declared}")
        task_hashes[dataset] = digest
    return manifest, _sha256_bytes(manifest_bytes), seed, task_hashes


def build_document():
    manifest, manifest_digest, seed, task_hashes = _load_inputs()
    document = {
        "version": VERSION,
        "seed": seed,
        "generated_at": GENERATED_AT,
        "manifest_sha256": manifest_digest,
        "cells": [],
        "cases": [],
    }

    selections = {}
    for dataset in DATASETS:
        tasks = _read_json(DATA / TASK_FILES[dataset])
        by_split = {
            split: [task for task in tasks if task["id"] in set(manifest["datasets"][dataset][split])]
            for split in SPLITS
        }
        used = set()
        for scenario in SCENARIO_PRIORITY[dataset]:
            for split in SPLITS:
                picks = _select_split(dataset, scenario, split, by_split[split], seed, used)
                selections[(dataset, scenario, split)] = picks
                used.update(task["id"] for task, _ in picks)

    for dataset in DATASETS:
        for scenario in SCENARIOS:
            cell_cases = []
            for split in SPLITS:
                for index, (task, reason) in enumerate(
                        selections[(dataset, scenario, split)], 1):
                    case = {
                        "case_id": f"{dataset}-{scenario}-{split}-{index}",
                        "dataset": dataset,
                        "task_id": task["id"],
                        "scenario": scenario,
                        "split": split,
                        "augmented": True,
                        "augmentation": AUGMENTATION_TEXT,
                        "source_task_id": task["id"],
                        "expected_modules": EXPECTED_MODULES[dataset][scenario],
                        "selection_reason": reason,
                        "augmentation_rule": AUGMENTATION_RULES[(dataset, scenario)],
                        "task_file_sha256": task_hashes[dataset],
                    }
                    cell_cases.append(case)
                    document["cases"].append(case)
            document["cells"].append({
                "cell_id": f"{dataset}-{scenario}",
                "dataset": dataset,
                "scenario": scenario,
                "expected_modules": EXPECTED_MODULES[dataset][scenario],
                "counts": {"total": 4, "dev": 2, "heldout": 2},
                "case_ids": [case["case_id"] for case in cell_cases],
                "augmentation_rule": AUGMENTATION_RULES[(dataset, scenario)],
            })

    if len(document["cases"]) != 36:
        raise RuntimeError(f"expected 36 cases, built {len(document['cases'])}")
    case_ids = [case["case_id"] for case in document["cases"]]
    if len(case_ids) != len(set(case_ids)):
        raise RuntimeError("duplicate case_id")
    return document


def serialize(document):
    return json.dumps(document, sort_keys=True, indent=2, ensure_ascii=False) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = serialize(build_document())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
    print(f"wrote {args.output} ({len(json.loads(payload)['cases'])} cases)")


if __name__ == "__main__":
    main()
