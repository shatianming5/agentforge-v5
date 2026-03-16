from __future__ import annotations
import os
import signal
from pathlib import Path
import click
from agentforge.daemon import Daemon
from agentforge.state import StateFile


@click.group()
def cli():
    """AgentForge v5.0 — The Agent writes it, runs it, fixes it, ships it."""
    pass


def _get_workdir(workdir):
    return Path(workdir) if workdir else Path.cwd()


def _get_state_file(workdir):
    return StateFile(workdir / ".agentforge" / "state.json")


@cli.command()
@click.argument("config_path", type=click.Path(exists=True), required=False, default=None)
@click.option("--workdir", type=click.Path(), default=None)
def run(config_path, workdir):
    """Start the optimization daemon. If config_path is omitted, auto-setup runs first."""
    wd = _get_workdir(workdir)
    config = Path(config_path) if config_path else None
    daemon = Daemon(config_path=config, workdir=wd)
    daemon.start()


@cli.command()
@click.option("--workdir", type=click.Path(), default=None)
def status(workdir):
    """Show current session status."""
    wd = _get_workdir(workdir)
    sf = _get_state_file(wd)
    if not sf.exists():
        click.echo("No session found.")
        return
    state = sf.load()
    click.echo(f"Session:  {state.session_id}")
    click.echo(f"Status:   {state.status}")
    click.echo(f"Round:    {state.current_round}")
    click.echo(f"Best:     {state.best.score} (round {state.best.round})")
    click.echo(f"N:        {state.N}")
    click.echo(f"Hardware: {state.hardware.num_gpus}x {state.hardware.gpu_model or 'CPU'}")
    click.echo(f"Budget:   {state.budget.rounds_used}/{state.budget.rounds_max} rounds, "
               f"{state.budget.gpu_hours_used:.1f}/{state.budget.gpu_hours_max} GPU-hours")


@cli.command()
@click.option("--workdir", type=click.Path(), default=None)
def stop(workdir):
    """Stop the daemon gracefully."""
    wd = _get_workdir(workdir)
    daemon = Daemon(config_path=None, workdir=wd)
    daemon.stop()


@cli.command()
@click.argument("message")
@click.option("--workdir", type=click.Path(), default=None)
def hint(message, workdir):
    """Inject a hint into the next Agent session."""
    wd = _get_workdir(workdir)
    sf = _get_state_file(wd)
    if not sf.exists():
        click.echo("No session found.")
        return
    state = sf.load()
    state.hints_pending.append(message)
    sf.save(state)
    click.echo(f"Hint added: {message}")


@cli.command()
@click.option("--workdir", type=click.Path(), default=None)
def skip(workdir):
    """Skip the current phase (Phase 1: stop Agent; Phase 2: stop training, score partials)."""
    wd = _get_workdir(workdir)
    daemon = Daemon(config_path=None, workdir=wd)
    pid = daemon.read_pid()
    if not pid:
        click.echo("No running daemon found.")
        return
    # Send SIGUSR1 to trigger phase skip in daemon
    try:
        os.kill(pid, signal.SIGUSR1)
        # Also write skip flag to state for phase-aware handling
        sf = _get_state_file(wd)
        if sf.exists():
            state = sf.load()
            state.hints_pending.append("__SKIP_CURRENT_PHASE__")
            sf.save(state)
        click.echo("Skip signal sent. Current phase will be interrupted.")
    except ProcessLookupError:
        click.echo("Daemon not running.")


@cli.command()
@click.option("--workdir", type=click.Path(), default=None)
def replan(workdir):
    """Force strategic reset in next round."""
    wd = _get_workdir(workdir)
    sf = _get_state_file(wd)
    if not sf.exists():
        click.echo("No session found.")
        return
    state = sf.load()
    state.hints_pending.append("HUMAN REQUESTED STRATEGIC RESET. Try completely new approaches.")
    sf.save(state)
    click.echo("Replan flag set for next round.")


@cli.command()
@click.option("--workdir", type=click.Path(), default=None)
def resume(workdir):
    """Resume from last completed round."""
    wd = _get_workdir(workdir)
    sf = _get_state_file(wd)
    if not sf.exists():
        click.echo("No session to resume.")
        return
    state = sf.load()
    state.status = "running"
    sf.save(state)
    click.echo("Session status set to running.")


@cli.command()
@click.option("--follow", is_flag=True)
@click.option("--workdir", type=click.Path(), default=None)
def logs(follow, workdir):
    """View daemon logs."""
    wd = _get_workdir(workdir)
    log_path = wd / ".agentforge" / "daemon.log"
    if not log_path.exists():
        click.echo("No logs found.")
        return
    if follow:
        os.execlp("tail", "tail", "-f", str(log_path))
    else:
        click.echo(log_path.read_text())


@cli.command(name="export")
@click.option("--workdir", type=click.Path(), default=None)
def export_best(workdir):
    """Export best solution as git patch."""
    wd = _get_workdir(workdir)
    sf = _get_state_file(wd)
    if not sf.exists():
        click.echo("No session found.")
        return
    state = sf.load()
    if not state.best.commit:
        click.echo("No best commit recorded.")
        return
    click.echo(f"Best: score={state.best.score}, commit={state.best.commit}")
    click.echo(f"To create a patch: git format-patch {state.best.commit}^..{state.best.commit}")
