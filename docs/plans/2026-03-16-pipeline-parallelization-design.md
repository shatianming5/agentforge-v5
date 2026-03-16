# Pipeline 并行化设计

## 背景

当前 AgentForge 的 Phase 1 (策略生成) 和 Phase 2 (训练评分) 是串行的：一个 Codex 调用生成所有策略分支，然后才启动并行训练。用户希望将整个流程改为流水线式：策略生成后，每个策略独立走「Codex 实现 → 训练 → 评分」全流程，最后汇总选 winner。

## 需求

1. Phase 1 精简为只生成策略规格列表（不写代码）
2. 每个策略启动独立 Codex 实例并行实现代码
3. 每个策略训练完成后立即评分（流水线式）
4. CLI 端 Rich Live 实时进度表格
5. 结构化事件接口供外部程序消费
6. 无并发约束，N 个策略全部并行

## 架构：线程池 Pipeline

```
Phase 1 (精简 Codex)
  → 只生成 StrategySpec[] (name + description + approach)
  ↓
PipelineOrchestrator
  ├── Worker-0 (Thread) → Codex实现 → worktree → train → score
  ├── Worker-1 (Thread) → Codex实现 → worktree → train → score
  └── Worker-N (Thread) → Codex实现 → worktree → train → score
  ↓
EventBus (queue.Queue)
  ├──→ LiveProgressDisplay (Rich Live 表格)
  └──→ 结构化事件回调 (JSON)
  ↓
Barrier: 全部完成后汇总选 winner
```

## 数据结构

### StrategySpec (新增)

```python
@dataclass
class StrategySpec:
    name: str           # "cosine_lr_decay"
    description: str    # "使用余弦退火学习率调度"
    approach: str       # 具体做法描述
    category: str       # "optim" | "arch" | "data" | "reg"
    risk: str           # "high" | "low"
    estimated_train_command: str  # 预估的训练命令
```

### PipelineEvent

```python
@dataclass
class PipelineEvent:
    worker_index: int
    strategy_name: str
    phase: str          # "implementing" | "training" | "scoring" | "done" | "failed"
    timestamp: float
    progress: dict | None   # {"epoch": 3, "loss": 0.45, "eta_seconds": 120}
    score: float | None
    error: str | None
    log_tail: str | None
```

## PipelineWorker 生命周期

```
PENDING → IMPLEMENTING → TRAINING → SCORING → DONE / FAILED
```

1. **IMPLEMENTING**: 构建单策略 prompt → CodexCLI.run() → Codex 创建分支并提交代码
2. **TRAINING**: ExperimentSetup.create() 创建 worktree → Popen() 启动训练 → 循环监控
3. **SCORING**: Scorer.score() 评分
4. **DONE/FAILED**: 发送最终事件

## EventBus

- `queue.Queue` 实现，线程安全
- 生产者：N 个 PipelineWorker
- 消费者线程分发给所有订阅者
- 订阅者：LiveProgressDisplay (Rich) + 用户自定义回调

## CLI 进度展示

Rich Live 表格实时刷新：

```
┌────┬──────────────────┬──────────────┬──────────┬────────┐
│ #  │ Strategy         │ Phase        │ Progress │ Score  │
├────┼──────────────────┼──────────────┼──────────┼────────┤
│ 0  │ cosine_lr_decay  │ training     │ ep 5/20  │ —      │
│ 1  │ mixup_augment    │ implementing │ Codex... │ —      │
│ 2  │ wider_backbone   │ scoring      │ bench... │ —      │
│ 3  │ label_smoothing  │ done         │ ✓        │ 0.8234 │
└────┴──────────────────┴──────────────┴──────────┴────────┘
```

无 Rich 时降级为定时 print。

## 文件改动

### 新增

| 文件 | 职责 |
|------|------|
| `agentforge/pipeline.py` | PipelineWorker, PipelineOrchestrator, EventBus, PipelineEvent |

### 修改

| 文件 | 改动 |
|------|------|
| `agentforge/agent.py` | 新增 StrategySpec; PromptBuilder.build_spec_only(); OutputParser.parse_specs(); CodexCLI.implement_strategy() |
| `agentforge/orchestrator.py` | _run_round() 重写为使用 PipelineOrchestrator |
| `agentforge/display.py` | 新增 LiveProgressDisplay (Rich Live 表格) |
| `agentforge/monitor.py` | 提取 SingleExperimentMonitor 供 Worker 使用 |

### 不变

| 文件 | 原因 |
|------|------|
| `experiment.py` | ExperimentSetup.create() 原封复用 |
| `scorer.py` | Scorer.score() 原封复用 |
| `stream.py` | stream_run() 原封复用 |

## 向后兼容

- ParallelRunner 保留不删除
- PipelineOrchestrator.run() 返回 list[StrategyResult]，与现有接口一致
- SessionState 结构不变
