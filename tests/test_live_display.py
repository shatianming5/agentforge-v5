# tests/test_live_display.py
import time
from agentforge.display import LiveProgressDisplay
from agentforge.pipeline import PipelineEvent


def test_live_display_update_state():
    display = LiveProgressDisplay(num_workers=3, use_rich=False)

    display.handle_event(PipelineEvent(
        worker_index=0, strategy_name="cosine_lr",
        phase="implementing", timestamp=time.time(),
    ))
    assert display._states[0]["phase"] == "implementing"
    assert display._states[0]["strategy"] == "cosine_lr"

    display.handle_event(PipelineEvent(
        worker_index=0, strategy_name="cosine_lr",
        phase="training", timestamp=time.time(),
        progress={"elapsed_s": 30},
    ))
    assert display._states[0]["phase"] == "training"

    display.handle_event(PipelineEvent(
        worker_index=0, strategy_name="cosine_lr",
        phase="done", timestamp=time.time(),
        score=0.85,
    ))
    assert display._states[0]["phase"] == "done"
    assert display._states[0]["score"] == 0.85


def test_live_display_plain_render():
    display = LiveProgressDisplay(num_workers=2, use_rich=False)

    display.handle_event(PipelineEvent(
        worker_index=0, strategy_name="strategy_a",
        phase="training", timestamp=time.time(),
        log_tail="step 5 loss=0.3",
    ))
    display.handle_event(PipelineEvent(
        worker_index=1, strategy_name="strategy_b",
        phase="done", timestamp=time.time(), score=0.9,
    ))

    text = display.render_plain()
    assert "strategy_a" in text
    assert "training" in text
    assert "strategy_b" in text
    assert "0.9" in text
