from __future__ import annotations
from pathlib import Path
from click.testing import CliRunner
from agentforge.cli import cli
from agentforge.state import SessionState, HardwareInfo, StateFile


def _setup_state(tmp_path):
    state = SessionState.create_initial(
        session_id="af-test", repo_url=str(tmp_path),
        hardware=HardwareInfo("cpu", "", 0, 4, 16, 50),
        N=1, gpus_per_experiment=0, rounds_max=10, gpu_hours_max=0,
    )
    state.best.score = 0.75
    state.current_round = 3
    sf = StateFile(tmp_path / ".agentforge" / "state.json")
    sf.save(state)


class TestCLI:
    def test_status(self, tmp_path):
        _setup_state(tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, ["status", "--workdir", str(tmp_path)])
        assert result.exit_code == 0
        assert "af-test" in result.output
        assert "0.75" in result.output

    def test_status_no_session(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(cli, ["status", "--workdir", str(tmp_path)])
        assert "No session" in result.output

    def test_hint(self, tmp_path):
        _setup_state(tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, ["hint", "try cosine annealing", "--workdir", str(tmp_path)])
        assert result.exit_code == 0
        sf = StateFile(tmp_path / ".agentforge" / "state.json")
        state = sf.load()
        assert "try cosine annealing" in state.hints_pending

    def test_replan(self, tmp_path):
        _setup_state(tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, ["replan", "--workdir", str(tmp_path)])
        assert result.exit_code == 0
        sf = StateFile(tmp_path / ".agentforge" / "state.json")
        state = sf.load()
        assert any("STRATEGIC RESET" in h for h in state.hints_pending)

    def test_export_no_commit(self, tmp_path):
        _setup_state(tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, ["export", "--workdir", str(tmp_path)])
        assert "No best commit" in result.output

    def test_resume(self, tmp_path):
        _setup_state(tmp_path)
        sf = StateFile(tmp_path / ".agentforge" / "state.json")
        state = sf.load()
        state.status = "paused"
        sf.save(state)
        runner = CliRunner()
        result = runner.invoke(cli, ["resume", "--workdir", str(tmp_path)])
        assert result.exit_code == 0
        state = sf.load()
        assert state.status == "running"

    def test_logs_no_file(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(cli, ["logs", "--workdir", str(tmp_path)])
        assert "No logs" in result.output

    def test_logs_show(self, tmp_path):
        log_dir = tmp_path / ".agentforge"
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "daemon.log").write_text("line1\nline2\n")
        runner = CliRunner()
        result = runner.invoke(cli, ["logs", "--workdir", str(tmp_path)])
        assert result.exit_code == 0
        assert "line1" in result.output
        assert "line2" in result.output

    def test_skip_no_daemon(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(cli, ["skip", "--workdir", str(tmp_path)])
        assert "No running daemon" in result.output

    def test_stop_no_daemon(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(cli, ["stop", "--workdir", str(tmp_path)])
        assert result.exit_code == 0

    def test_hint_no_session(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(cli, ["hint", "test", "--workdir", str(tmp_path)])
        assert "No session" in result.output

    def test_resume_no_session(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(cli, ["resume", "--workdir", str(tmp_path)])
        assert "No session" in result.output

    def test_replan_no_session(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(cli, ["replan", "--workdir", str(tmp_path)])
        assert "No session" in result.output

    def test_export_with_commit(self, tmp_path):
        _setup_state(tmp_path)
        sf = StateFile(tmp_path / ".agentforge" / "state.json")
        state = sf.load()
        state.best.commit = "abc123"
        state.best.score = 0.92
        sf.save(state)
        runner = CliRunner()
        result = runner.invoke(cli, ["export", "--workdir", str(tmp_path)])
        assert result.exit_code == 0
        assert "0.92" in result.output
        assert "abc123" in result.output

    def test_cli_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "AgentForge" in result.output


def test_run_without_config_path(tmp_path):
    """不传 config_path 也能启动（触发 auto-setup）。"""
    from click.testing import CliRunner
    from agentforge.cli import cli

    runner = CliRunner()
    result = runner.invoke(cli, ["run", "--workdir", str(tmp_path)])
    # 不应该报 "Missing argument" 错误
    assert "Missing argument" not in (result.output or "")
