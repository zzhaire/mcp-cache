from exp_core.policies import make_policy

POLICY = make_policy(
    "ours", "本文方法",
    "M1–M4＋决策前有效性引用与操作状态提供；开发策略可显式覆盖。",
    (), publish_read_results=True, publish_budget_bytes=8192,
    publish_operations=True, context_dedup=True)
INFO = {
    "name": "ours", "method_id": "ours",
    "reproduction_status": "prototype",
    "source": "paper.md, Design",
    "scope": "本文原型；策略改动与搜索仅用于开发配置",
}
