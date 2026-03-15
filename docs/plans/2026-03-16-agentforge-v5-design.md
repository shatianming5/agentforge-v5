# AgentForge v5.0 Implementation Design

## Architecture: Monolithic Modular (Python)

Single package `agentforge/`, split by responsibility. Modules communicate via dataclasses. No global state.

## Module Map

```
agentforge/
├── state.py          # SessionState dataclass + atomic JSON read/write
├── hardware.py       # HardwareDetector + compute_N
├── agent.py          # CodexCLI wrapper + PromptBuilder + OutputParser
├── sandbox.py        # chmod read-only, venv isolation, timeout
├── strategy.py       # StrategyValidator (fingerprint dedup, category diversity)
├── experiment.py     # ExperimentSetup (git clone --shared, venv, env vars)
├── runner.py         # ParallelRunner (subprocess.Popen per experiment)
├── monitor.py        # Monitor thread (timeout, VRAM trend, NaN, disk, straggler)
├── scorer.py         # Scorer (benchmark + test suite in fresh subprocess)
├── cleanup.py        # Trial cleanup, GPU reset, disk GC, loser branch deletion
├── repair.py         # SelfRepair (all-fail diagnosis, env rebuild, rollback)
├── anti_oscillation.py # Plateau detection, fingerprint overlap, seed policy
├── daemon.py         # Daemon (fork, PID file, signal handlers, credential snapshot)
├── cli.py            # Click CLI (run/status/stop/hint/skip/replan/resume/export/logs)
└── orchestrator.py   # Main loop: Phase1 → cleanup → Phase2 → update → save
```

## Dependency Flow (top-down, no cycles)

```
cli → daemon → orchestrator
                    ↓
      ┌─────────────┼─────────────┐
      ↓             ↓             ↓
  agent.py     runner.py     cleanup.py
  (Phase 1)    (Phase 2)     (between)
      ↓             ↓
  sandbox.py    monitor.py
  strategy.py   scorer.py
                experiment.py
      ↓             ↓
      └──── state.py ┘
             hardware.py
```

## Data Model

5 core dataclasses: HardwareInfo, StrategyResult, RoundResult, Budget, SessionState.
All fields explicitly typed. No dict[str, Any].

## Orchestrator

Synchronous round loop: develop → cleanup → train → score → select → save.
One state file write per round. Crash recovery = resume from last completed round.

## Phase 1: Agent Session

CodexCLI is a thin wrapper (subprocess.run). PromptBuilder generates structured prompt from state+config. OutputParser extracts JSON summary. StrategyValidator checks fingerprint dedup + category diversity.

## Phase 2: Parallel Training

ParallelRunner spawns N processes. Monitor thread checks 5 conditions (timeout, VRAM trend, NaN, disk, straggler). Scorer runs benchmark in fresh subprocess.

## ExperimentSetup

git clone --shared (hardlinks, <1s). gc.auto=0 at session start. Isolated venv per experiment. CPU affinity via taskset.

## Self-Repair

All-fail diagnosis: environmental → rebuild venv + retry. Code-related → new Agent session with error logs. Final fallback → rollback to best commit.

## Anti-Oscillation

Taboo list (never compressed). Fingerprint dedup (>70% overlap → reject). Category diversity (≥3 categories). Mandatory exploration (≥2/N high-risk). Plateau handling (3 rounds stagnant → force different approaches).

## CLI + Daemon

Click-based CLI. Daemon via os.fork + setsid + PID file. SIGTERM → graceful stop. Credential snapshot at startup.

## Agent Backend

Codex CLI only (initial). Future: add Claude Code by splitting agent.py into agent/codex.py + agent/claude_code.py + agent/base.py.

## Key Decisions

- No plugin system (YAGNI — only one agent backend)
- No microservices (Spec philosophy: 1.5 components)
- Dataclass communication (no implicit state sharing)
- Atomic state file (write-tmp-fsync-rename)
- Monitor is minimal (Agent's trial catches 60% of failures)
