"""Render the 36-case x 7-arm smoke matrix report from result files.

The ``*.result.json`` files under the results root are the only source of
numbers; ``matrix-summary.jsonl`` is deliberately ignored. Rows produced by a
different executable hash are listed as stale/excluded and never enter a table
or the Ours verdict.
"""
import argparse
import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import statistics
import sys

from exp_core import runner
from exp_core.settings import ROOT, DATA, RESULTS as RESULTS_ROOT

RESULTS = RESULTS_ROOT / "smoke"
SCENARIOS = DATA / "scenarios-36.json"
ARM_ORDER = ("bare", "rw_openai", "rw_fastmcp", "rw_gateway", "rw_mehan",
             "rw_merchant", "ours")
CASE_FIELDS = ("dataset", "task_id", "scenario", "split", "augmented")
TERMINAL = ("completed", "error")
CSV_COLUMNS = (
    "case_id", "dataset", "task_id", "scenario", "split", "augmented", "arm",
    "method", "status", "task_success", "active_time_s", "task_time_s",
    "harness_queue_wait_s", "input_tokens", "output_tokens",
    "cached_input_tokens", "logical_tool_calls", "mcp_requests",
    "request_body_bytes", "response_body_bytes", "M1_hits", "M1_lookups",
    "M2_hits", "M2_eligible", "M3_hits", "M3_lookups", "M4_hits",
    "M4_failure_hits", "retries", "publish_events", "publish_bytes",
    "published_results_used", "context_dedup_saved_bytes", "server_cache_hits",
    "model_calls", "model_attempts", "model_process_failures", "model_retries",
    "run_id", "error", "budget_exhausted", "policy_sha256")
RUNTIME_METRICS = ("M1_hits", "M1_lookups", "M2_hits", "M2_eligible", "M3_hits",
                   "M3_lookups", "M4_hits", "M4_failure_hits", "retries")
SUM_METRICS = ("input_tokens", "output_tokens", "mcp_requests",
               "logical_tool_calls", "M1_hits", "M2_hits", "M3_hits",
               "M4_hits", "publish_bytes")


def load_scenarios(path=SCENARIOS):
    """Pre-registered case metadata, in manifest order."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = document.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"{path} has no cases")
    seen = set()
    for case in cases:
        case_id = case.get("case_id")
        if not case_id or case_id in seen:
            raise ValueError(f"{path}: empty or duplicate case_id {case_id!r}")
        for field in CASE_FIELDS:
            if field not in case:
                raise ValueError(f"{case_id}: missing field {field}")
        seen.add(case_id)
    return cases


def load_rows(results_root=RESULTS):
    """Every ``*.result.json`` row with its path and mtime, oldest first."""
    rows = []
    root = Path(results_root)
    if not root.is_dir():
        return rows
    for path in root.rglob("*.result.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            row = dict(payload) if isinstance(payload, dict) else {
                "_parse_error": "result file is not a JSON object"}
        except (OSError, ValueError) as error:
            row = {"_parse_error": str(error)}
        row["_path"] = str(path)
        try:
            row["_mtime"] = path.stat().st_mtime
        except OSError:
            row["_mtime"] = 0.0
        rows.append(row)
    rows.sort(key=lambda row: (row["_mtime"], row["_path"]))
    return rows


def select_current(rows, code_sha256):
    """Split rows into current-hash rows, stale rows and unreadable files."""
    current, stale, unreadable = [], [], []
    for row in rows:
        if row.get("_parse_error"):
            unreadable.append(row)
        elif row.get("code_sha256") == code_sha256:
            current.append(row)
        else:
            stale.append(row)
    return current, stale, unreadable


def row_arm(row):
    return row.get("arm") or row.get("method")


def metric(row, key):
    """Read one metric, resolving the nested runtime/server-cache locations."""
    if key in RUNTIME_METRICS:
        runtime = row.get("runtime")
        return runtime.get(key) if isinstance(runtime, dict) else None
    if key == "logical_tool_calls":
        value = row.get("logical_tool_calls")
        if value is None:
            runtime = row.get("runtime")
            value = runtime.get("logical_calls") if isinstance(runtime, dict) else None
        return value
    if key == "server_cache_hits":
        cache = row.get("server_cache")
        return cache.get("hits") if isinstance(cache, dict) else None
    return row.get(key)


def _numbers(values):
    return [float(value) for value in values
            if isinstance(value, (int, float)) and not isinstance(value, bool)]


def aggregate(rows):
    """Descriptive aggregate for one row group; never treats errors as success."""
    completed = [row for row in rows if row.get("status") == "completed"]
    active = _numbers(row.get("active_time_s") for row in completed)
    successes = sum(1 for row in rows if row.get("task_success") is True)
    totals = {}
    for key in SUM_METRICS:
        values = _numbers(metric(row, key) for row in rows)
        totals[key] = (sum(values), len(values))
    return {
        "n": len(rows),
        "completed": len(completed),
        "error": sum(1 for row in rows if row.get("status") == "error"),
        "other": sum(1 for row in rows if row.get("status") not in TERMINAL),
        "success": successes,
        "success_rate": successes / len(rows) if rows else None,
        "n_active": len(active),
        "mean_active": statistics.fmean(active) if active else None,
        "median_active": statistics.median(active) if active else None,
        "totals": totals,
    }


def _delta_pct(ours, other):
    if not other:
        return None
    return (ours - other) / other * 100.0


def paired(rows, arm, other):
    """Paired comparison on case_ids both arms completed."""
    def completed_by(name):
        table = {}
        for row in rows:
            if row_arm(row) != name or row.get("status") != "completed":
                continue
            table.setdefault(row.get("case_id"), row)
        return table

    left, right = completed_by(arm), completed_by(other)
    shared = sorted(set(left) & set(right), key=str)

    def active(table):
        return _numbers(table[case_id].get("active_time_s") for case_id in shared)

    def values(table, key):
        return _numbers(metric(table[case_id], key) for case_id in shared)

    left_active, right_active = active(left), active(right)
    left_input, right_input = values(left, "input_tokens"), values(right, "input_tokens")
    left_mcp, right_mcp = values(left, "mcp_requests"), values(right, "mcp_requests")
    ours_input, other_input = sum(left_input), sum(right_input)
    ours_mcp, other_mcp = sum(left_mcp), sum(right_mcp)
    ours_mean = statistics.fmean(left_active) if left_active else None
    other_mean = statistics.fmean(right_active) if right_active else None
    return {
        "arm": other,
        "n": len(shared),
        "case_ids": shared,
        "ours_mean_active": ours_mean,
        "other_mean_active": other_mean,
        "delta_active": (ours_mean - other_mean
                         if ours_mean is not None and other_mean is not None else None),
        "n_active_ours": len(left_active),
        "n_active_other": len(right_active),
        "ours_input": ours_input,
        "other_input": other_input,
        "n_input_ours": len(left_input),
        "n_input_other": len(right_input),
        "delta_input_pct": _delta_pct(ours_input, other_input),
        "ours_mcp": ours_mcp,
        "other_mcp": other_mcp,
        "n_mcp_ours": len(left_mcp),
        "n_mcp_other": len(right_mcp),
        "delta_mcp_pct": _delta_pct(ours_mcp, other_mcp),
        "ours_success": sum(1 for case_id in shared
                            if left[case_id].get("task_success") is True),
        "other_success": sum(1 for case_id in shared
                             if right[case_id].get("task_success") is True),
    }


def _num(value, digits=2):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _int(value):
    if value is None:
        return "-"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _pct(value):
    return f"{value:.1%}" if value is not None else "-"


def _cell(value):
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _table(headers, records):
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(_cell(value) for value in record) + " |"
                 for record in records)
    return lines


def _sum_cell(summary, key):
    total, count = summary["totals"][key]
    return f"{_int(total)} (n={count})"


def _mean_cell(summary, key):
    return f"{_num(summary[key])} (n={summary['n_active']})"


def _rate_cell(summary):
    return f"{_pct(summary['success_rate'])} (n={summary['n']})"


class Report:
    def __init__(self, *, code_sha256, generated_at, cases, case_index, arm_list,
                 rows, stale, unreadable, excluded_by_filter, unknown_rows,
                 unknown_arm_rows, missing, duplicates, integrity_errors,
                 filters):
        self.code_sha256 = code_sha256
        self.generated_at = generated_at
        self.cases = cases
        self.case_index = case_index
        self.arm_list = arm_list
        self.rows = rows
        self.stale = stale
        self.unreadable = unreadable
        self.excluded_by_filter = excluded_by_filter
        self.unknown_rows = unknown_rows
        self.unknown_arm_rows = unknown_arm_rows
        self.missing = missing
        self.duplicates = duplicates
        self.integrity_errors = integrity_errors
        self.filters = filters
        self.expected = [(case["case_id"], arm) for case in cases
                         for arm in arm_list]
        self.counts = {"completed": 0, "error": 0, "other": 0}
        index = {(row.get("case_id"), row_arm(row)): row for row in rows}
        for identity in self.expected:
            row = index.get(identity)
            if row is None:
                continue
            status = row.get("status")
            if status == "completed":
                self.counts["completed"] += 1
            elif status == "error":
                self.counts["error"] += 1
            else:
                self.counts["other"] += 1
        models = sorted({row["model"] for row in rows if row.get("model")})
        self.model = "、".join(models)

    def known(self, row):
        return row.get("case_id") in self.case_index

    def meta(self, row):
        case = self.case_index.get(row.get("case_id"))
        if case is not None:
            return {field: case.get(field) for field in CASE_FIELDS}
        return {field: row.get(field) for field in CASE_FIELDS}

    def rows_for(self, arm, pool=None):
        return [row for row in (self.rows if pool is None else pool)
                if row_arm(row) == arm]

    def strongest_other(self):
        candidates = []
        for arm in self.arm_list:
            if arm == "ours":
                continue
            summary = aggregate(self.rows_for(arm))
            if summary["success_rate"] is None:
                continue
            candidates.append((summary["success_rate"], -ARM_ORDER.index(arm),
                               arm, summary))
        if not candidates:
            return None, None
        best = max(candidates, key=lambda item: (item[0], item[1]))
        return best[2], best[3]

    def verdict(self):
        labels = {
            "a": "正确率不低于最强对照 arm",
            "b": "配对 case 上 active_time_s 均值更低",
            "c": "相同配对 case 上 input_tokens 合计更低",
            "d": "相同配对 case 上 MCP 请求合计更少",
        }
        ours = aggregate(self.rows_for("ours"))
        strongest, strongest_summary = self.strongest_other()
        pairs = paired(self.rows, "ours", strongest) if strongest else None
        if ours["completed"] == 0:
            items = [{"key": key, "label": labels[key], "met": False,
                      "observed": "Ours 无已完成行，无数据"}
                     for key in "abcd"]
            return {"strongest": strongest, "items": items, "met": False,
                    "no_data": True, "pairs": pairs}
        items = []
        if strongest is None:
            observed_a = "无其他 arm 的当前结果"
            met_a = False
        else:
            met_a = (ours["success_rate"] is not None
                     and strongest_summary["success_rate"] is not None
                     and ours["success_rate"] >= strongest_summary["success_rate"])
            observed_a = (f"Ours {_pct(ours['success_rate'])}（n={ours['n']}） vs "
                          f"{strongest} {_pct(strongest_summary['success_rate'])}"
                          f"（n={strongest_summary['n']}）")
        items.append({"key": "a", "label": labels["a"], "met": met_a,
                      "observed": observed_a})

        if pairs is None or pairs["n"] == 0:
            met_b, observed_b = False, "无共同完成的配对 case"
        elif pairs["ours_mean_active"] is None or pairs["other_mean_active"] is None:
            met_b, observed_b = False, "配对行缺少 active_time_s"
        else:
            met_b = pairs["ours_mean_active"] < pairs["other_mean_active"]
            observed_b = (f"Ours {pairs['ours_mean_active']:.2f} s"
                          f"（n={pairs['n_active_ours']}） vs {strongest} "
                          f"{pairs['other_mean_active']:.2f} s"
                          f"（n={pairs['n_active_other']}），Δ "
                          f"{pairs['delta_active']:+.2f} s；配对数 {pairs['n']}")
        items.append({"key": "b", "label": labels["b"], "met": met_b,
                      "observed": observed_b})

        input_complete = (pairs is not None and pairs["n"] > 0
                          and pairs["n_input_ours"] == pairs["n"]
                          and pairs["n_input_other"] == pairs["n"])
        if not input_complete:
            met_c, observed_c = False, "配对 case 缺少 input_tokens"
        else:
            met_c = pairs["ours_input"] < pairs["other_input"]
            observed_c = (f"Ours {_int(pairs['ours_input'])}"
                          f"（n={pairs['n_input_ours']}） vs {strongest} "
                          f"{_int(pairs['other_input'])}"
                          f"（n={pairs['n_input_other']}），Δ "
                          f"{pairs['delta_input_pct']:+.1f}%")
        items.append({"key": "c", "label": labels["c"], "met": met_c,
                      "observed": observed_c})

        mcp_complete = (pairs is not None and pairs["n"] > 0
                        and pairs["n_mcp_ours"] == pairs["n"]
                        and pairs["n_mcp_other"] == pairs["n"])
        if not mcp_complete:
            met_d, observed_d = False, "配对 case 缺少 mcp_requests"
        else:
            met_d = pairs["ours_mcp"] < pairs["other_mcp"]
            observed_d = (f"Ours {_int(pairs['ours_mcp'])}"
                          f"（n={pairs['n_mcp_ours']}） vs {strongest} "
                          f"{_int(pairs['other_mcp'])}"
                          f"（n={pairs['n_mcp_other']}），Δ "
                          f"{pairs['delta_mcp_pct']:+.1f}%")
        items.append({"key": "d", "label": labels["d"], "met": met_d,
                      "observed": observed_d})
        return {"strongest": strongest, "items": items,
                "met": all(item["met"] for item in items),
                "no_data": False, "pairs": pairs}

    def csv_records(self):
        records = []
        for row in self.rows:
            meta = self.meta(row)
            record = {}
            for column in CSV_COLUMNS:
                if column in CASE_FIELDS:
                    value = meta[column]
                elif column in RUNTIME_METRICS or column in (
                        "logical_tool_calls", "server_cache_hits"):
                    value = metric(row, column)
                elif column == "arm":
                    value = row_arm(row)
                elif column == "method":
                    value = row.get("method") or row_arm(row)
                else:
                    value = row.get(column)
                record[column] = value
            records.append(record)
        return records

    def csv_text(self):
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=list(CSV_COLUMNS),
                                lineterminator="\n")
        writer.writeheader()
        for record in self.csv_records():
            writer.writerow({key: "" if record.get(key) is None
                             else record.get(key) for key in CSV_COLUMNS})
        return buffer.getvalue()

    def markdown(self):
        lines = ["# 36-case MCP 缓存 smoke 矩阵报告", ""]
        lines += self._header_lines()
        lines += ["", "## 完整性审计", ""] + self._integrity_lines()
        lines += ["", "## 分 arm 汇总", "",
                  "各 arm 的 case 覆盖可能不同，本表为未配对描述统计，"
                  "不得直接当作同 case 集合上的对比。", ""]
        lines += self._arm_table(self.rows)
        lines += ["", "## Ours 与各 arm 的配对比较", "",
                  "仅在 Ours 与该 arm 都完成（status=completed）的 case 上比较；"
                  "这是结论判定唯一使用的表。", ""]
        lines += self._paired_table()
        lines += ["", "## 数据集 × 场景", "",
                  "每个 cell 内各 arm 覆盖的 case 数可能不同，本表未配对，"
                  "跨 arm 差异仅为描述性。", ""]
        lines += self._cell_table()
        lines += ["", "## 原始 vs 增广", ""]
        lines += self._split_lines()
        lines += ["", "## 失败 / 超时 / 预算耗尽", ""]
        lines += self._failure_lines()
        lines += ["", "## 逐 case 明细", ""]
        lines += self._detail_lines()
        lines += ["", "## Ours 结论（spec R7）", ""]
        lines += self._verdict_lines()
        lines += ["", "## 附录：过期/排除行（code_sha256 不匹配）", ""]
        lines += self._stale_lines()
        lines += ["", "## 附录：完整性错误", ""]
        lines += self._error_lines()
        return "\n".join(lines) + "\n"

    def _header_lines(self):
        covered = len({row.get("case_id") for row in self.rows
                       if self.known(row)})
        lines = [
            f"- 当前结果行数：{len(self.rows)}",
            f"- code_sha256：`{self.code_sha256}`",
            f"- 模型：{self.model or '无当前结果'}",
            f"- 覆盖 case：{covered} / {len(self.cases)}（预注册清单过滤后的范围）",
            f"- 生成时间（UTC）：{self.generated_at}",
            "- 数字来源：本报告全部数字仅由结果目录下的 `*.result.json` 逐行计算；"
            "未使用 `matrix-summary.jsonl`，未引用任何历史实验汇总。",
        ]
        if self.filters:
            lines.append("- 过滤条件：" + "；".join(self.filters))
        if self.excluded_by_filter:
            lines.append(f"- 按过滤条件排除的当前行：{self.excluded_by_filter}")
        return lines

    def _integrity_lines(self):
        lines = [
            f"- 期望身份：{len(self.cases)}（case） × {len(self.arm_list)}（arm）"
            f" = {len(self.expected)}",
            f"- 已完成：{self.counts['completed']}；错误：{self.counts['error']}；"
            f"非终态：{self.counts['other']}；缺失：{len(self.missing)}",
            f"- 重复身份：{self.duplicates}；过期/排除行：{len(self.stale)}；"
            f"未知 case 行：{len(self.unknown_rows)}；"
            f"未知 arm 行：{len(self.unknown_arm_rows)}",
            f"- 完整性错误：{len(self.integrity_errors)}",
        ]
        if self.missing:
            lines += ["", "缺失身份（case_id / arm）：", ""]
            lines += [f"- `{case_id}` / `{arm}`" for case_id, arm in self.missing]
        else:
            lines += ["", "无缺失身份。"]
        return lines

    def _arm_table(self, pool):
        headers = ["arm", "n", "完成", "错误", "成功", "成功率（全部尝试）",
                   "active_time 均值 s", "active_time 中位数 s", "input_tokens",
                   "output_tokens", "MCP 请求", "logical 调用", "M1 命中",
                   "M2 命中", "M3 命中", "M4 命中", "publish_bytes"]
        records = []
        for arm in self.arm_list:
            summary = aggregate([row for row in pool if row_arm(row) == arm])
            records.append([
                arm, summary["n"], summary["completed"], summary["error"],
                summary["success"], _rate_cell(summary),
                _mean_cell(summary, "mean_active"),
                _mean_cell(summary, "median_active"),
                _sum_cell(summary, "input_tokens"),
                _sum_cell(summary, "output_tokens"),
                _sum_cell(summary, "mcp_requests"),
                _sum_cell(summary, "logical_tool_calls"),
                _sum_cell(summary, "M1_hits"),
                _sum_cell(summary, "M2_hits"),
                _sum_cell(summary, "M3_hits"),
                _sum_cell(summary, "M4_hits"),
                _sum_cell(summary, "publish_bytes"),
            ])
        return _table(headers, records)

    def _paired_table(self):
        headers = ["对照 arm", "配对数", "Ours 均值 s", "对照均值 s", "Δ 均值 s",
                   "Ours input tokens", "对照 input tokens", "Δ input %",
                   "Ours MCP 请求", "对照 MCP 请求", "Δ MCP %",
                   "Ours 成功", "对照 成功"]
        records = []
        for arm in self.arm_list:
            if arm == "ours":
                continue
            pair = paired(self.rows, "ours", arm)
            records.append([
                arm, pair["n"],
                f"{_num(pair['ours_mean_active'])} (n={pair['n_active_ours']})",
                f"{_num(pair['other_mean_active'])} (n={pair['n_active_other']})",
                _num(pair["delta_active"], 2),
                f"{_int(pair['ours_input'])} (n={pair['n_input_ours']})",
                f"{_int(pair['other_input'])} (n={pair['n_input_other']})",
                f"{pair['delta_input_pct']:+.1f}%"
                if pair["delta_input_pct"] is not None else "-",
                f"{_int(pair['ours_mcp'])} (n={pair['n_mcp_ours']})",
                f"{_int(pair['other_mcp'])} (n={pair['n_mcp_other']})",
                f"{pair['delta_mcp_pct']:+.1f}%"
                if pair["delta_mcp_pct"] is not None else "-",
                pair["ours_success"], pair["other_success"],
            ])
        return _table(headers, records)

    def _cell_table(self):
        headers = ["dataset", "scenario", "arm", "n", "成功", "成功率",
                   "active_time 均值 s", "input_tokens", "MCP 请求"]
        cells = []
        for case in self.cases:
            key = (case["dataset"], case["scenario"])
            if key not in cells:
                cells.append(key)
        records = []
        for dataset, scenario in cells:
            pool = [row for row in self.rows
                    if self.known(row)
                    and self.meta(row)["dataset"] == dataset
                    and self.meta(row)["scenario"] == scenario]
            for arm in self.arm_list:
                summary = aggregate([row for row in pool if row_arm(row) == arm])
                records.append([
                    dataset, scenario, arm, summary["n"], summary["success"],
                    _rate_cell(summary), _mean_cell(summary, "mean_active"),
                    _sum_cell(summary, "input_tokens"),
                    _sum_cell(summary, "mcp_requests"),
                ])
        return _table(headers, records)

    def _split_lines(self):
        lines = []
        for title, value in (("原始 case（augmented=false）", False),
                             ("增广 case（augmented=true）", True)):
            pool = [row for row in self.rows
                    if self.known(row) and self.meta(row)["augmented"] is value]
            declared = sum(1 for case in self.cases
                           if case.get("augmented") is value)
            lines += [f"### {title}", "",
                      f"预注册范围内该类 case 数：{declared}；当前结果行数：{len(pool)}。",
                      ""]
            lines += self._arm_table(pool)
            lines.append("")
        return lines

    @staticmethod
    def _is_failure(row):
        return (row.get("status") == "error"
                or row.get("budget_exhausted") is True
                or row.get("task_success") is None)

    def _failure_lines(self):
        rows = [row for row in self.rows if self._is_failure(row)]
        if not rows:
            return ["无失败/超时/预算耗尽行。"]
        failed = sum(1 for row in rows if row.get("status") == "error")
        exhausted = sum(1 for row in rows if row.get("budget_exhausted") is True)
        unknown = sum(1 for row in rows if row.get("task_success") is None)
        headers = ["case_id", "arm", "status", "budget_exhausted",
                   "task_success", "run_id", "error"]
        records = [[row.get("case_id"), row_arm(row), row.get("status"),
                    row.get("budget_exhausted"), row.get("task_success"),
                    row.get("run_id"), row.get("error")] for row in rows]
        return ([f"共 {len(rows)} 行：error {failed} 行；budget_exhausted=true "
                 f"{exhausted} 行；task_success 为空 {unknown} 行。", ""]
                + _table(headers, records))

    def _detail_lines(self):
        if not self.rows:
            return ["无当前结果行。"]
        headers = ["case_id", "dataset", "scenario", "split", "augmented", "arm",
                   "success", "active_time_s", "input_tokens", "output_tokens",
                   "mcp_requests", "M1_hits", "M2_hits", "M3_hits", "M4_hits",
                   "status", "run_id"]
        records = []
        for row in self.rows:
            meta = self.meta(row)
            success = row.get("task_success")
            records.append([
                row.get("case_id"), meta["dataset"], meta["scenario"],
                meta["split"],
                "-" if meta["augmented"] is None
                else ("是" if meta["augmented"] else "否"),
                row_arm(row),
                "-" if success is None else ("是" if success else "否"),
                _num(row.get("active_time_s")), _int(metric(row, "input_tokens")),
                _int(metric(row, "output_tokens")),
                _int(metric(row, "mcp_requests")),
                _int(metric(row, "M1_hits")), _int(metric(row, "M2_hits")),
                _int(metric(row, "M3_hits")), _int(metric(row, "M4_hits")),
                row.get("status"), row.get("run_id"),
            ])
        return _table(headers, records)

    def _verdict_lines(self):
        verdict = self.verdict()
        lines = [
            "- 判定规则（spec R7）：(a) 正确率不低于最强对照 arm；(b) 配对 case 上 "
            "active_time_s 均值更低；(c) 相同配对 case 上 input_tokens 合计更低；"
            "(d) 相同配对 case 上 MCP 请求合计更少。任一未达成即总体未达成，"
            "不以其他指标替代。",
        ]
        if verdict["strongest"]:
            lines.append(f"- 最强对照 arm（其他 arm 中成功率最高者）："
                         f"`{verdict['strongest']}`")
        else:
            lines.append("- 最强对照 arm：无可用对照（无其他 arm 的当前结果）")
        for item in verdict["items"]:
            state = "达成" if item["met"] else "未达成"
            lines.append(f"- ({item['key']}) {item['label']}：{state} — "
                         f"{item['observed']}")
        lines.append(f"- 总体结论：{'达成' if verdict['met'] else '未达成'}")
        if verdict.get("no_data"):
            lines.append("- 说明：Ours 无已完成行，判定为未达成/无数据。")
        lines += ["", "### 与 rw_merchant 的配对比较（最强传统组合，单独列出）", ""]
        if "rw_merchant" not in self.arm_list:
            lines.append("- 本次过滤未包含 rw_merchant，无法列出配对比较。")
            return lines
        pair = paired(self.rows, "ours", "rw_merchant")
        if pair["n"] == 0:
            lines.append("- 无 Ours 与 rw_merchant 都完成的配对 case。")
            return lines
        if pair["delta_input_pct"] is not None:
            input_line = (f"- input_tokens 合计：Ours {_int(pair['ours_input'])}"
                          f"（n={pair['n_input_ours']}） vs rw_merchant "
                          f"{_int(pair['other_input'])}"
                          f"（n={pair['n_input_other']}），Δ "
                          f"{pair['delta_input_pct']:+.1f}%")
        else:
            input_line = "- input_tokens 合计：-"
        if pair["delta_mcp_pct"] is not None:
            mcp_line = (f"- MCP 请求合计：Ours {_int(pair['ours_mcp'])}"
                        f"（n={pair['n_mcp_ours']}） vs rw_merchant "
                        f"{_int(pair['other_mcp'])}"
                        f"（n={pair['n_mcp_other']}），Δ "
                        f"{pair['delta_mcp_pct']:+.1f}%")
        else:
            mcp_line = "- MCP 请求合计：-"
        lines += [
            f"- 配对数：{pair['n']}",
            f"- active_time_s 均值：Ours {_num(pair['ours_mean_active'])} s"
            f"（n={pair['n_active_ours']}） vs rw_merchant "
            f"{_num(pair['other_mean_active'])} s（n={pair['n_active_other']}），"
            f"Δ {_num(pair['delta_active'])} s",
            input_line,
            mcp_line,
            f"- 成功数：Ours {pair['ours_success']} / rw_merchant "
            f"{pair['other_success']}（配对数 {pair['n']}）",
        ]
        return lines

    def _stale_lines(self):
        if not self.stale:
            return ["无过期/排除行。"]
        headers = ["case_id", "arm", "code_sha256", "run_id"]
        records = [[row.get("case_id"), row_arm(row), row.get("code_sha256"),
                    row.get("run_id")]
                   for row in sorted(self.stale,
                                     key=lambda row: (str(row.get("case_id")),
                                                      str(row_arm(row)),
                                                      str(row.get("run_id"))))]
        return ([f"共 {len(records)} 行；这些行不进入任何表与结论。", ""]
                + _table(headers, records))

    def _error_lines(self):
        if not self.integrity_errors:
            return ["无完整性错误。"]
        return [f"- {error}" for error in self.integrity_errors]


def build_report(results_root=RESULTS, scenarios=SCENARIOS, code_sha256=None,
                 cases=None, arms=None, generated_at=None):
    case_list = load_scenarios(scenarios)
    case_index = {case["case_id"]: case for case in case_list}
    filters = []
    if cases:
        unknown = sorted(set(cases) - set(case_index))
        if unknown:
            raise ValueError("unknown case_id: " + ", ".join(unknown))
        selected = set(cases)
        case_list = [case for case in case_list if case["case_id"] in selected]
        filters.append("cases=" + ",".join(sorted(selected)))
    if arms:
        unknown = sorted(set(arms) - set(ARM_ORDER))
        if unknown:
            raise ValueError("unknown arm: " + ", ".join(unknown))
        arm_list = [arm for arm in ARM_ORDER if arm in set(arms)]
        filters.append("arm=" + ",".join(arm_list))
    else:
        arm_list = list(ARM_ORDER)

    digest = code_sha256 or runner.code_sha256(ROOT)
    rows = load_rows(results_root)
    current, stale, unreadable = select_current(rows, digest)
    integrity_errors = [
        f"无法解析结果文件 {row['_path']}: {row['_parse_error']}"
        for row in unreadable]
    unknown_rows = [row for row in current
                    if row.get("case_id") not in case_index]
    unknown_arm_rows = [row for row in current if row_arm(row) not in ARM_ORDER]
    if unknown_rows:
        integrity_errors.append(
            f"{len(unknown_rows)} 行使用未预注册的 case_id："
            + ", ".join(sorted({str(row.get('case_id')) for row in unknown_rows}))
            + "；这些行不进入数据集×场景与原始/增广表")
    if unknown_arm_rows:
        integrity_errors.append(
            f"{len(unknown_arm_rows)} 行使用未知 arm："
            + ", ".join(sorted({str(row_arm(row)) for row in unknown_arm_rows})))

    selected_cases = {case["case_id"] for case in case_list} if cases else None
    scoped, excluded = [], 0
    for row in current:
        if selected_cases is not None and row.get("case_id") not in selected_cases:
            excluded += 1
            continue
        if row_arm(row) not in arm_list:
            excluded += 1
            continue
        scoped.append(row)

    groups = {}
    for row in scoped:
        groups.setdefault((row.get("case_id"), row_arm(row)), []).append(row)
    selected_rows, duplicates = [], 0
    for key in sorted(groups, key=lambda item: (str(item[0]), str(item[1]))):
        group = groups[key]
        selected_rows.append(group[0])
        if len(group) > 1:
            duplicates += len(group) - 1
            identities = {(row.get("policy_sha256"), row.get("code_sha256"),
                           row.get("case_id"), row_arm(row)) for row in group}
            if len(identities) == 1:
                integrity_errors.append(
                    f"重复身份 {key[0]} / {key[1]}：{len(group)} 个结果文件，"
                    f"使用最早文件 {group[0].get('run_id')}")
            else:
                integrity_errors.append(
                    f"同一 case/arm 存在多个 policy_sha256：{key[0]} / {key[1]}，"
                    f"共 {len(group)} 个结果文件，使用最早文件 "
                    f"{group[0].get('run_id')}")
    selected_rows.sort(key=lambda row: (str(row.get("case_id")),
                                        str(row_arm(row))))
    if duplicates:
        integrity_errors.append(
            f"共 {duplicates} 个重复结果文件被忽略（每个 case/arm 只保留最早一个）")

    expected = [(case["case_id"], arm) for case in case_list for arm in arm_list]
    present = {(row.get("case_id"), row_arm(row)) for row in selected_rows}
    missing = [identity for identity in expected if identity not in present]

    return Report(
        code_sha256=digest,
        generated_at=generated_at or datetime.now(timezone.utc).isoformat(
            timespec="seconds"),
        cases=case_list,
        case_index=case_index,
        arm_list=arm_list,
        rows=selected_rows,
        stale=stale,
        unreadable=unreadable,
        excluded_by_filter=excluded,
        unknown_rows=unknown_rows,
        unknown_arm_rows=unknown_arm_rows,
        missing=missing,
        duplicates=duplicates,
        integrity_errors=integrity_errors,
        filters=filters,
    )


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=RESULTS,
                        help="root scanned recursively for *.result.json")
    parser.add_argument("--scenarios", type=Path, default=SCENARIOS,
                        help="pre-registered case manifest")
    parser.add_argument("--out", type=Path, default=None,
                        help="markdown output (default: <results-root>/report.md)")
    parser.add_argument("--csv", type=Path, default=None,
                        help="csv output (default: <results-root>/summary.csv)")
    parser.add_argument("--code-sha256", default=None,
                        help="only rows with this executable hash are current")
    parser.add_argument("--cases", action="append", default=None,
                        help="case_id filter for pilot reports; repeatable")
    parser.add_argument("--arm", action="append", choices=list(ARM_ORDER),
                        default=None, help="arm filter; repeatable")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = build_report(
            results_root=args.results_root, scenarios=args.scenarios,
            code_sha256=args.code_sha256, cases=args.cases, arms=args.arm)
    except (OSError, ValueError, KeyError) as error:
        print(f"report generation failed: {error}", file=sys.stderr)
        return 2
    out_path = args.out or Path(args.results_root) / "report.md"
    csv_path = args.csv or Path(args.results_root) / "summary.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report.markdown(), encoding="utf-8")
    csv_path.write_text(report.csv_text(), encoding="utf-8")
    print(f"report: {len(report.rows)} current rows, "
          f"{len(report.stale)} stale, {len(report.missing)} missing -> "
          f"{out_path} / {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
