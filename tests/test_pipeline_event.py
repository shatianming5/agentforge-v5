# tests/test_pipeline_event.py
import time
import threading
from agentforge.pipeline import PipelineEvent, EventBus


def test_pipeline_event_creation():
    evt = PipelineEvent(
        worker_index=0, strategy_name="cosine_lr",
        phase="training", timestamp=time.time(),
    )
    assert evt.phase == "training"
    assert evt.score is None
    assert evt.error is None


def test_event_to_dict():
    evt = PipelineEvent(
        worker_index=1, strategy_name="mixup",
        phase="done", timestamp=1234567890.0,
        score=0.85,
    )
    d = evt.to_dict()
    assert d["worker_index"] == 1
    assert d["phase"] == "done"
    assert d["score"] == 0.85


def test_eventbus_emit_and_subscribe():
    bus = EventBus()
    received = []
    bus.subscribe(lambda e: received.append(e))

    consumer = threading.Thread(target=bus.run_consumer, daemon=True)
    consumer.start()

    evt = PipelineEvent(
        worker_index=0, strategy_name="test",
        phase="implementing", timestamp=time.time(),
    )
    bus.emit(evt)
    bus.shutdown()
    consumer.join(timeout=2)

    assert len(received) == 1
    assert received[0].strategy_name == "test"


def test_eventbus_multiple_subscribers():
    bus = EventBus()
    received_a = []
    received_b = []
    bus.subscribe(lambda e: received_a.append(e))
    bus.subscribe(lambda e: received_b.append(e))

    consumer = threading.Thread(target=bus.run_consumer, daemon=True)
    consumer.start()

    bus.emit(PipelineEvent(
        worker_index=0, strategy_name="s1",
        phase="done", timestamp=time.time(), score=0.9,
    ))
    bus.shutdown()
    consumer.join(timeout=2)

    assert len(received_a) == 1
    assert len(received_b) == 1
