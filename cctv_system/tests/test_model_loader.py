"""Tests for ModelLoader graceful-degradation behaviour.

These run without torch/ultralytics installed: every model should fail to load
but the loader must degrade gracefully (plug-ins -> built-in mode, core -> off)
rather than raising.
"""

import textwrap

from pipelines.model_loader import CORE_MODELS, PLUGIN_MODELS, ModelLoader


def _write_config(tmp_path, body):
    cfg = tmp_path / "pipeline.yaml"
    cfg.write_text(textwrap.dedent(body))
    return str(cfg)


def test_loader_degrades_when_weights_absent(tmp_path):
    cfg = _write_config(
        tmp_path,
        """
        device: cpu
        models:
          base: models/does_not_exist.pt
          helmet: models/helmet/missing.pt
          seatbelt: models/seatbelt/missing.pt
          triple_rider: null
          red_light: null
          wrong_side: null
        thresholds: {}
        """,
    )
    loader = ModelLoader(config_path=cfg, device="cpu")
    models = loader.load_all()

    # Every logical model is present in the result mapping.
    for name in (*CORE_MODELS, *PLUGIN_MODELS):
        assert name in models

    # Optional plug-ins with null weights are in built-in mode (not available).
    for name in PLUGIN_MODELS:
        assert not models[name].available
        assert models[name].source == "built-in"

    # Core models are unavailable (no ultralytics / weights) but did not crash.
    for name in CORE_MODELS:
        assert not models[name].available


def test_is_available_lazy_loads(tmp_path):
    cfg = _write_config(
        tmp_path,
        """
        device: cpu
        models:
          triple_rider: null
        """,
    )
    loader = ModelLoader(config_path=cfg, device="cpu")
    # No explicit load_all(); is_available should lazily resolve.
    assert loader.is_available("triple_rider") is False
    assert loader.get("triple_rider").source == "built-in"
