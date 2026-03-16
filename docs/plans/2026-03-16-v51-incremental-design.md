# AgentForge v5.1 增量改进设计

## 目标

在 v5.0 已能端到端运行的基础上，修复关键正确性 bug 并补齐发布生态。

## 改动清单

### 1. Direction-aware 比较（正确性 bug）

**问题：** 系统在 minimize 方向下行为完全错误：
- `select_winners` 按 score 降序排，选最差实验当赢家
- `is_done` 在 score > target 时就停止（minimize 应该是 <=）
- `best.score` 初始值 0.0，minimize 时应为 inf

**方案：** 在 `config.py` 加 `is_better()` / `best_initial_score()` 辅助函数。
修改 5 处比较逻辑，传入 direction 参数。

**改动文件：**
- `agentforge/config.py` — 加 2 个函数
- `agentforge/state.py` — `create_initial` 接受 direction，BestResult 初始值
- `agentforge/orchestrator.py` — 5 处比较逻辑

### 2. Publish + Validate（发布生态）

**新建文件：**
- `agentforge/validate.py` ~80 行 — 校验 challenge 完整性
- `agentforge/publish.py` ~60 行 — 发布流程（validate → 确认 → 提示 push）

**改动文件：**
- `agentforge/cli.py` — 加 `validate`、`publish` 子命令

### 3. Data 管理（数据准备）

**新建文件：**
- `agentforge/data.py` ~60 行 — 数据检查、下载、校验、锁定

**改动文件：**
- `agentforge/orchestrator.py` — 在 `_init_or_resume` 中调用 `prepare_data`
- `agentforge/config.py` — ChallengeConfig 加可选 `data` 段

## 不做的事

- 不重写 scorer.py、hardware.py、monitor.py（已经够用）
- 不加模板系统（auto-setup 已经替代了手动模板）
- 不加日志轮转（短期运行不需要）
- 不加归一化分数（保持原始 metric 值更直观）
