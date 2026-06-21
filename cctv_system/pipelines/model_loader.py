"""Load and cache all detection models for the violation pipeline.

The loader manages six *logical* models:

================  =========================================================
Name              Role
================  =========================================================
``base``          YOLO11n base detector (vehicles / persons / traffic light)
``helmet``        Fine-tuned helmet detector (``helmet_absent``/``present``)
``seatbelt``      Fine-tuned seatbelt detector
``triple_rider``  OPTIONAL plug-in; built-in logic used if absent
``red_light``     OPTIONAL plug-in; built-in logic used if absent
``wrong_side``    OPTIONAL plug-in; built-in logic used if absent
================  =========================================================

Each model is loaded once and cached. CUDA failures fall back to CPU on a
per-model basis so a single bad weight file cannot take down the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from .common import load_yaml, resolve_path, select_device, setup_logging

logger = setup_logging()

# Logical models and whether they are mandatory (core) or optional plug-ins.
CORE_MODELS = ("base", "helmet", "seatbelt")
# Optional plug-ins: violation models + a license-plate detector that localises
# plates within vehicle crops so OCR runs on tight plate regions (not whole cars).
PLUGIN_MODELS = ("triple_rider", "red_light", "wrong_side", "plate_detector")


@dataclass
class LoadedModel:
    """A loaded model plus metadata about where/how it loaded.

    Attributes
    ----------
    name : str
        Logical model name (e.g. ``"helmet"``).
    model : Any
        The underlying Ultralytics ``YOLO`` object, or ``None`` if unavailable.
    source : str
        Path the weights were loaded from (or ``"built-in"`` / ``"missing"``).
    device : str
        Device the model resides on (``"cuda:0"`` / ``"cpu"``).
    available : bool
        Whether the model actually loaded.
    """

    name: str
    model: Any = None
    source: str = "missing"
    device: str = "cpu"
    available: bool = False


@dataclass
class ModelLoader:
    """Load, cache and expose the pipeline's models.

    Parameters
    ----------
    config_path : str, optional
        Path to ``configs/pipeline.yaml``. Relative paths resolve against the
        package root.
    device : str, optional
        Override the device for all models (``"auto"``/``"cuda"``/``"cpu"``).
        If ``None``, the value from the config is used.

    Examples
    --------
    >>> loader = ModelLoader()
    >>> models = loader.load_all()
    >>> base = loader.get("base").model
    """

    config_path: str = "configs/pipeline.yaml"
    device: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)
    models: Dict[str, LoadedModel] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.config = load_yaml(self.config_path)
        req_device = self.device or self.config.get("device", "auto")
        self.device = select_device(req_device)
        logger.info("ModelLoader using device: %s", self.device)

    # ------------------------------------------------------------------ public
    def load_all(self) -> Dict[str, LoadedModel]:
        """Load every configured model (core + plug-ins) and cache them.

        Returns
        -------
        dict[str, LoadedModel]
            Mapping of logical name -> :class:`LoadedModel`.
        """
        paths: Dict[str, Any] = self.config.get("models", {})
        for name in CORE_MODELS:
            self.models[name] = self._load_one(name, paths.get(name), optional=False)
        for name in PLUGIN_MODELS:
            self.models[name] = self._load_one(name, paths.get(name), optional=True)
        self._log_summary()
        return self.models

    def get(self, name: str) -> LoadedModel:
        """Return a cached :class:`LoadedModel`, loading lazily if needed.

        Parameters
        ----------
        name : str
            Logical model name.

        Returns
        -------
        LoadedModel
        """
        if name not in self.models:
            paths: Dict[str, Any] = self.config.get("models", {})
            optional = name in PLUGIN_MODELS
            self.models[name] = self._load_one(name, paths.get(name), optional=optional)
        return self.models[name]

    def is_available(self, name: str) -> bool:
        """Whether a model loaded successfully (so a detector can use it)."""
        return self.get(name).available

    # ----------------------------------------------------------------- private
    def _load_one(self, name: str, weight_path: Optional[str], optional: bool) -> LoadedModel:
        """Load a single Ultralytics model with CUDA->CPU fallback.

        Missing optional weights are not an error: the pipeline uses built-in
        logic in that case. Missing *core* weights (other than the auto-download
        base) produce a warning and a disabled model.
        """
        # No path configured -> plug-in runs in built-in mode (or core disabled).
        if not weight_path:
            if optional:
                logger.info("[%s] no weights configured -> built-in logic mode.", name)
                return LoadedModel(name=name, source="built-in", available=False)
            if name == "base":
                weight_path = "models/yolo11n.pt"  # will auto-download below
            else:
                logger.warning("[%s] no weights configured and none available.", name)
                return LoadedModel(name=name, source="missing", available=False)

        resolved = resolve_path(weight_path)
        assert resolved is not None

        # The base model auto-downloads from Ultralytics by short name if the
        # local file is absent; everything else must already exist on disk.
        if not resolved.exists():
            if name == "base":
                logger.info("[base] %s not found locally; will auto-download yolo11n.", resolved)
                resolved = Path("yolo11n.pt")  # Ultralytics resolves this remotely
            elif optional:
                logger.info("[%s] weights '%s' absent -> built-in logic mode.", name, resolved)
                return LoadedModel(name=name, source="built-in", available=False)
            else:
                logger.warning("[%s] weights '%s' not found -> model disabled.", name, resolved)
                return LoadedModel(name=name, source="missing", available=False)

        return self._instantiate(name, str(resolved))

    def _instantiate(self, name: str, source: str) -> LoadedModel:
        """Create the Ultralytics model object, with device fallback."""
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - runtime dependency
            logger.error("[%s] ultralytics not installed: %s", name, exc)
            return LoadedModel(name=name, source=source, available=False)

        for device in self._device_candidates():
            try:
                model = YOLO(source)
                model.to(device)
                logger.info("[%s] loaded from '%s' on %s.", name, source, device)
                return LoadedModel(
                    name=name, model=model, source=source, device=device, available=True
                )
            except Exception as exc:  # noqa: BLE001 - report and try next device
                logger.warning("[%s] load on %s failed: %s", name, device, exc)
                continue

        logger.error("[%s] failed to load on any device.", name)
        return LoadedModel(name=name, source=source, available=False)

    def _device_candidates(self) -> list[str]:
        """Device order to try: the chosen device first, then CPU fallback."""
        if self.device and self.device != "cpu":
            return [self.device, "cpu"]
        return ["cpu"]

    def _log_summary(self) -> None:
        """Emit a one-line summary of what loaded and from where."""
        logger.info("---- Model load summary ----")
        for name in (*CORE_MODELS, *PLUGIN_MODELS):
            lm = self.models.get(name)
            if lm is None:
                continue
            status = "OK" if lm.available else ("built-in" if lm.source == "built-in" else "OFF")
            logger.info("  %-13s : %-8s (%s)", name, status, lm.source)
        logger.info("----------------------------")
