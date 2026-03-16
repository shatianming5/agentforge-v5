# AgentForge

One command. Your code gets better.

```bash
pip install agentforge
agentforge run
# 10 rounds → val_loss 1.88 → 1.79
```

AgentForge is an automated optimization framework that uses AI coding agents (Codex CLI) to iteratively improve your codebase. It analyzes your project, generates optimization strategies, runs experiments in parallel, and keeps the best results.

Works on ML training, inference optimization, algorithm tuning — any repo with a measurable metric.

## How It Works

```
┌─────────────────────────────────────────────────┐
│  Phase 0: Auto-Setup (first run only)           │
│  Codex analyzes repo → generates challenge.yaml │
│  + benchmark.py + test_suite.py                 │
└──────────────────┬──────────────────────────────┘
                   ▼
┌─────────────────────────────────────────────────┐
│  Phase 1: Strategy Development                  │
│  Codex reads code → proposes N strategies →     │
│  creates git branches with changes              │
└──────────────────┬──────────────────────────────┘
                   ▼
┌─────────────────────────────────────────────────┐
│  Phase 2: Parallel Evaluation                   │
│  N experiments run in parallel (git worktree)   │
│  → test suite → benchmark → score               │
└──────────────────┬──────────────────────────────┘
                   ▼
          Best strategies survive.
          Repeat for up to 25 rounds.
```

## Quickstart

### Option A: With an existing challenge

```bash
pip install agentforge
cd your-project/
agentforge run challenge.yaml
```

### Option B: Auto-setup (zero config)

```bash
pip install agentforge
cd your-project/
agentforge run
# AgentForge will:
#   1. Analyze your project with Codex (read-only)
#   2. Generate challenge.yaml, benchmark.py, test_suite.py
#   3. Ask you to confirm each file
#   4. Start optimizing
```

### Prerequisites

- Python 3.11+
- [Codex CLI](https://github.com/openai/codex) installed and logged in
- Git repository

## challenge.yaml

The challenge file defines what to optimize:

```yaml
challenge:
  name: "nanoGPT shakespeare_char"
  description: "Minimize validation loss on Shakespeare character-level LM"

target:
  metric: val_loss
  value: 1.8
  direction: minimize      # or "maximize"

tests:
  smoke: python3 test_suite.py
  full: python3 test_suite.py
  benchmark: python3 benchmark.py

constraints:
  writable:
    - train.py
    - model.py
    - config/
  read_only:
    - data/
    - README.md

data:                      # optional
  path: data/
  source: auto
```

- **target**: the metric to optimize, extracted from `results/benchmark.json`
- **tests.benchmark**: must produce `results/benchmark.json` with `{"<metric>": <value>}`
- **tests.full**: regression test suite, must pass before scoring
- **constraints**: AgentForge physically locks `read_only` files (chmod)

## CLI Commands

```bash
agentforge run [challenge.yaml]   # Start optimization (auto-setup if no config)
agentforge status                  # Show session status, best score, budget
agentforge stop                    # Graceful stop after current round
agentforge logs --follow           # Tail daemon logs
agentforge hint "try dropout 0.1"  # Inject hint for next Agent session
agentforge skip                    # Skip current phase
agentforge replan                  # Force strategic reset next round
agentforge resume                  # Resume paused session
agentforge export                  # Export best solution as git patch
agentforge validate                # Check challenge.yaml completeness
agentforge publish                 # Validate + prepare for publishing
```

## Real-Time Progress

AgentForge shows a live progress table during optimization:

```
──────── AgentForge v5.1 | Agent: Codex | Hardware: cpu, 8 cores | N=2 ─────────
─────────────────────────────────── Round 1 ────────────────────────────────────
  Phase 1: Agent 生成策略... done (12.9m, 2 strategies)
  Phase 2: 训练完成 (1.1m)
┏━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━┓
┃ #    ┃ Strategy                  ┃      Score ┃ Status       ┃
┡━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━┩
│ 0    │ rmsnorm_swiglu_fast_decay │     1.8500 │ winner       │
│ 1    │ lion_high_lr_cosine       │          — │ oom          │
└──────┴───────────────────────────┴────────────┴──────────────┘
  Best so far: 1.8500 (round 1)
─────────────────────────────────── RESULTS ────────────────────────────────────
  Status:     completed
  Best score: 1.850000
  Rounds:     1
  Time:       14.0m
```

## Architecture

```
agentforge/
├── orchestrator.py    # Main loop: init → round → done
├── agent.py           # Codex CLI interaction + prompt building
├── analyzer.py        # Project analysis (auto-setup Phase 0)
├── generator.py       # Generate challenge.yaml / benchmark.py / test_suite.py
├── confirm.py         # Interactive file confirmation
├── experiment.py      # Git worktree isolation + env setup
├── runner.py          # Parallel experiment launcher
├── scorer.py          # Test suite + benchmark scoring
├── monitor.py         # OOM / NaN / timeout / straggler detection
├── display.py         # Rich terminal output
├── state.py           # Session state (JSON persistence)
├── config.py          # challenge.yaml parser
├── hardware.py        # GPU/CPU detection + N computation
├── sandbox.py         # Read-only file protection (chmod)
├── cleanup.py         # Worktree cleanup + disk management
├── repair.py          # All-fail recovery (venv rebuild / rollback)
├── strategy.py        # Strategy validation (diversity, risk)
├── anti_oscillation.py # Plateau detection + seed management
├── data.py            # Dataset download + checksum + lock
├── validate.py        # Challenge completeness checks
├── publish.py         # Publish workflow
├── daemon.py          # Background daemon management
└── cli.py             # Click CLI entry point
```

## Key Design Decisions

- **Git worktree** for experiment isolation (fast, shared .git, low disk)
- **Subprocess-based scoring** (no CUDA context pollution between experiments)
- **Physical chmod** on read-only files (Agent can't accidentally modify data)
- **Anti-oscillation** (plateau detection, deterministic seeds, strategy tabu list)
- **Self-repair** (all-fail → diagnose → rebuild venv or rollback to best commit)
- **Direction-aware** comparisons throughout (minimize/maximize)
- **Graceful degradation** (rich → plain print, GPU → CPU, Codex fail → branch detection)

## Development

```bash
git clone https://github.com/shatianming5/agentforge-v5.git
cd agentforge-v5
pip install -e ".[dev]"
pytest tests/ -v
```

153 tests covering all modules.

## License

MIT
