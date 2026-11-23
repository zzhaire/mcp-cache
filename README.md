# mcp-cache

面向多步智能体的远程 MCP 通信与运行时协同优化：实验代码。

这个仓库目前包含两部分：

- **可控的 MCP 实验服务器（lab）**：`exp_core/lab/`、`exp_core/adapters/server.py`。一个可以注入故障、外部变化、登录失效和目录变更的 MCP 服务器，带每个工具的契约（读写类别、新鲜度分级、读写的资源、是否支持操作键）。说明见 `docs/lab-server.md`，极端场景见 `docs/lab-stress-cases.md`。
- **M1–M4 缓存调度器**：`exp_core/scheduler.py`、`exp_core/scheduler_policy.py`、`ours/`，参数在 `configs/scheduler-presets.json` 和 `ours/scheduler.schema.json`。`default` 预设与设计稿一致。说明见 `docs/m1-m4-scheduler.md`。

其余代码（`exp_core/runner.py`、`exp_core/matrix.py` 等）是驱动一次运行和汇总结果所需的框架。

## 运行

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
cp configs/environment.example.json configs/environment.local.json   # 按本机路径修改
./run lab-check                    # 不调用模型的机制检查
./run lab-check --suite all --server   # 同时走真实 MCP 服务器并核对
python -m unittest discover -s tests
```

## 没有包含的内容

- 取自或派生自 BFCL、τ²-bench、AppWorld 的任务数据。相关代码在 `data/recon/` 和 `exp_core/lab/domains/`，需要自行获取原数据集后用 `./run recon-build` 生成；依赖这些数据的测试在没有数据时会失败或跳过。
- 运行结果和本机配置。
