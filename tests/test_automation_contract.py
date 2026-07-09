"""Verify automation tools are registered."""


def test_automation_tool_count():
    """8 automation tools should be registered as module-level functions."""
    from mcp_server.tools import automation

    expected = [
        'get_clip_automation',
        'set_clip_automation',
        'clear_clip_automation',
        'apply_automation_shape',
        'apply_automation_recipe',
        'get_automation_recipes',
        'generate_automation_curve',
        'analyze_for_automation',
    ]
    for name in expected:
        assert hasattr(automation, name), f"Missing tool: {name}"
        assert callable(getattr(automation, name)), f"Not callable: {name}"


# -- P2-10: set_clip_automation clamps fixed-range mixer envelope values ------

class _FakeAbleton:
    def __init__(self):
        self.last = None

    def send_command(self, name, params=None):
        self.last = (name, params)
        return {"ok": True}


class _FakeCtx:
    def __init__(self, ableton):
        self.lifespan_context = {"ableton": ableton}


def _call_set_clip_automation(parameter_type, points, **kwargs):
    from mcp_server.tools import automation
    ableton = _FakeAbleton()
    ctx = _FakeCtx(ableton)
    automation.set_clip_automation(
        ctx,
        track_index=0,
        clip_index=0,
        parameter_type=parameter_type,
        points=points,
        **kwargs,
    )
    return ableton.last[1]["points"]


def test_set_clip_automation_clamps_volume():
    """Volume above 1.0 / below 0.0 is clamped to the 0.0-1.0 range."""
    out = _call_set_clip_automation(
        "volume",
        [{"time": 0.0, "value": 1.5}, {"time": 1.0, "value": -0.3}],
    )
    assert out[0]["value"] == 1.0
    assert out[1]["value"] == 0.0


def test_set_clip_automation_clamps_send():
    out = _call_set_clip_automation(
        "send",
        [{"time": 0.0, "value": 2.0}],
        send_index=0,
    )
    assert out[0]["value"] == 1.0


def test_set_clip_automation_clamps_panning():
    """Panning is clamped to -1.0..1.0, not 0.0-1.0."""
    out = _call_set_clip_automation(
        "panning",
        [{"time": 0.0, "value": -1.7}, {"time": 1.0, "value": 1.4}],
    )
    assert out[0]["value"] == -1.0
    assert out[1]["value"] == 1.0


def test_set_clip_automation_device_unclamped():
    """Device params keep their native range (caller scales them)."""
    out = _call_set_clip_automation(
        "device",
        [{"time": 0.0, "value": 135.0}],
        device_index=0,
        parameter_index=2,
    )
    assert out[0]["value"] == 135.0
