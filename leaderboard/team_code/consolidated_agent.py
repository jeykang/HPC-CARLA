"""
Universal, modular agent wrapper for HPC-CARLA.

This module implements a four-stage pipeline around arbitrary CARLA
Leaderboard agents (InterFuser, LAV, TCP, ...):

    Stage 1  (Sensor specification):
        - Build sensor specs from a YAML config or the underlying agent's
          own `sensors()` definition.
        - Create per-sensor output directories for standardized dataset
          collection.

    Stage 2  (Data collection):
        - Save raw sensor streams from CARLA into the standardized
          directory layout, irrespective of the underlying agent.

    Stage 3  (Agent inference):
        - Dynamically import and instantiate the real driving agent as
          specified in the YAML config.
        - Delegate control decisions to its original `run_step`.

    Stage 4  (Control post-processing):
        - Apply optional global safety / sanity constraints to the
          returned `carla.VehicleControl`.

The intent is to keep the wrapper maximally agent-agnostic while
preserving near-original behaviour for InterFuser, LAV, TCP, etc.
"""

import os
import json
import time
import datetime
import pathlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import importlib
import importlib.util

import numpy as np
import cv2
import yaml

import carla
from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track

# Weather presets — mirrors auto_pilot.py WEATHERS / WEATHERS_IDS exactly.
# Index 0–13 match leaderboard training splits; 14–20 are night variants.
_WEATHERS = {
    "ClearNoon":       carla.WeatherParameters.ClearNoon,
    "ClearSunset":     carla.WeatherParameters.ClearSunset,
    "CloudyNoon":      carla.WeatherParameters.CloudyNoon,
    "CloudySunset":    carla.WeatherParameters.CloudySunset,
    "WetNoon":         carla.WeatherParameters.WetNoon,
    "WetSunset":       carla.WeatherParameters.WetSunset,
    "MidRainyNoon":    carla.WeatherParameters.MidRainyNoon,
    "MidRainSunset":   carla.WeatherParameters.MidRainSunset,
    "WetCloudyNoon":   carla.WeatherParameters.WetCloudyNoon,
    "WetCloudySunset": carla.WeatherParameters.WetCloudySunset,
    "HardRainNoon":    carla.WeatherParameters.HardRainNoon,
    "HardRainSunset":  carla.WeatherParameters.HardRainSunset,
    "SoftRainNoon":    carla.WeatherParameters.SoftRainNoon,
    "SoftRainSunset":  carla.WeatherParameters.SoftRainSunset,
    "ClearNight":      carla.WeatherParameters(5.0,   0.0,  0.0,  10.0, -1.0, -90.0, 60.0, 75.0,  1.0,   0.0),
    "CloudyNight":     carla.WeatherParameters(60.0,  0.0,  0.0,  10.0, -1.0, -90.0, 60.0, 0.75,  0.1,   0.0),
    "WetNight":        carla.WeatherParameters(5.0,   0.0, 50.0,  10.0, -1.0, -90.0, 60.0, 75.0,  1.0,  60.0),
    "WetCloudyNight":  carla.WeatherParameters(60.0,  0.0, 50.0,  10.0, -1.0, -90.0, 60.0, 0.75,  0.1,  60.0),
    "SoftRainNight":   carla.WeatherParameters(60.0, 30.0, 50.0,  30.0, -1.0, -90.0, 60.0, 0.75,  0.1,  60.0),
    "MidRainyNight":   carla.WeatherParameters(80.0, 60.0, 60.0,  60.0, -1.0, -90.0, 60.0, 0.75,  0.1,  80.0),
    "HardRainNight":   carla.WeatherParameters(100.0,100.0, 90.0, 100.0, -1.0, -90.0,100.0, 0.75,  0.1, 100.0),
}
_WEATHER_IDS = list(_WEATHERS)

# Pipeline support (used only when config contains `pipeline:`).
# Keep this import resilient across environments and Python versions.
PipelineEngine = None  # type: ignore
try:
    # Preferred when imported as team_code.consolidated_agent
    from .pipeline_engine import PipelineEngine  # type: ignore
except Exception:  # pragma: no cover
    try:
        # Fallback when team_code is directly on PYTHONPATH
        from team_code.pipeline_engine import PipelineEngine  # type: ignore
    except Exception:
        PipelineEngine = None  # type: ignore


# -------------------------------------------------------------------------
# Helper utilities
# -------------------------------------------------------------------------


def get_entry_point() -> str:
    """Entrypoint name expected by the CARLA Leaderboard."""
    return "ConsolidatedAgent"


def _dynamic_import(module_path: str, class_name: str):
    """
    Import a class dynamically from a module path.

    Parameters
    ----------
    module_path : str
        Dotted Python module path, e.g. "team_code.interfuser_agent_orig".
    class_name : str
        Name of the class inside that module.

    Returns
    -------
    type
        The imported class object.
    """
    module = importlib.import_module(module_path)
    try:
        cls = getattr(module, class_name)
    except AttributeError as exc:
        raise ImportError(
            f"Class '{class_name}' not found in module '{module_path}'"
        ) from exc
    return cls


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _now_string() -> str:
    now = datetime.datetime.now()
    return "_".join(f"{x:02d}" for x in (now.month, now.day, now.hour, now.minute, now.second))


# -------------------------------------------------------------------------
# ConsolidatedAgent
# -------------------------------------------------------------------------


class ConsolidatedAgent(AutonomousAgent):
    """
    Agent-agnostic wrapper with a four-stage modular pipeline.

    The wrapper is configured by a YAML file whose path is provided either
    via the Leaderboard `--agent-config` argument or via the AGENT_CONFIG
    environment variable.

    Expected YAML schema (minimal):

        agent:
          name: interfuser          # For directory naming / metadata
          module: team_code.interfuser_agent_orig
          class_name: InterfuserAgent
          config_file: path/to/interfuser_conf.py   # Passed to agent.setup(...)

        # Optional: explicit sensor definitions. If omitted, the wrapper
        # will fall back to `agent.sensors()`.
        sensors:
          - id: rgb
            type: sensor.camera.rgb
            x: 1.5
            y: 0.0
            z: 2.0
            roll: 0.0
            pitch: 0.0
            yaw: 0.0
            width: 800
            height: 600
            fov: 90

        # Optional: external config exclusively for the wrapper or downstream
        # consumers. The format is intentionally generic.
        external_config:
          format: python    # or "yaml"
          path: path/to/config.py
          class_name: GlobalConfig
          attribute: null

        # Optional data collection toggle
        collect_data: true

    The intent is that adding a new agent is done by:
        1) placing its original code somewhere on PYTHONPATH,
        2) creating a YAML file pointing to its module/class/config,
        3) (optionally) defining a sensor list and external_config.
    """

    # The Leaderboard expects each agent to define this
    track = Track.SENSORS

    def __init__(self, path_to_conf_file: str = "") -> None:
        # NOTE: The CARLA Leaderboard instantiates agents as:
        #   agent = AgentClass(args.agent_config)
        # and the base `AutonomousAgent.__init__` immediately calls `setup()`.
        # Therefore, we must:
        #   1) accept `path_to_conf_file` here,
        #   2) initialize our own fields BEFORE calling `super().__init__`,
        #      because `setup()` relies on them.

        # Runtime config (from YAML)
        self._config: Optional[Dict[str, Any]] = None
        self._config_path: Optional[str] = None

        # Optional "external_config" loaded by the wrapper
        self.external_config: Optional[Any] = None

        # Underlying "real" agent (InterFuser/LAV/TCP/...)
        self._inner_agent: Optional[Any] = None

        # Optional config-defined pipeline for *new* agents
        self._pipeline: Optional[Any] = None

        # Last output control (for pipeline warmup / frame skipping)
        self._pipeline_last_control: Optional[carla.VehicleControl] = None

        # Data collection
        self.collect_data: bool = False
        self.save_root: Optional[str] = None
        self.save_path: Optional[str] = None
        self.sensor_data_paths: Dict[str, str] = {}
        self._global_step: int = -1

        # Per-run collection counters (used for coverage plots)
        self._frames_saved_by_sensor: Dict[str, int] = {}
        self._data_collection_started_at: Optional[str] = None

        # Cached sensor list built in Stage 1
        self._sensor_list: Optional[List[Dict[str, Any]]] = None

        # Cache sensor types by id (used for stable dataset folder naming)
        self._sensor_type_by_id: Dict[str, str] = {}

        # Optional extension hooks (all no-op unless configured)
        self._extensions_loaded: bool = False
        self._extensions: List[Any] = []

        # Weather: applied once on the first run_step call (world is ready by then).
        # _init() is commented out in the leaderboard evaluator, so run_step is
        # the earliest reliable point to call world.set_weather().
        self._weather_applied: bool = False

        super().__init__(path_to_conf_file)

    # ------------------------------------------------------------------
    # Configuration / setup
    # ------------------------------------------------------------------

    def setup(self, path_to_conf_file: str) -> None:
        """
        Leaderboard-standard setup entrypoint.

        The Leaderboard calls:
            agent = ConsolidatedAgent()
            agent.setup(path_to_conf_file)

        In our case, `path_to_conf_file` is expected to be the YAML file
        described above. If not provided, we fall back to AGENT_CONFIG.
        """
        # Ensure AGENT_CONFIG is available for `sensors()`-before-setup cases.
        if path_to_conf_file:
            os.environ.setdefault("AGENT_CONFIG", path_to_conf_file)

        # Stage 0: load wrapper config & initialize inner agent / dataset.
        self._ensure_config_loaded(path_hint=path_to_conf_file)
        self._ensure_extensions_loaded()
        self._ensure_pipeline_or_inner_loaded()
        self._initialize_data_collection()

        # Allow extensions to do any late initialization.
        self._call_extension_hook("on_setup")

    # ------------------------------------------------------------------
    # Stage 1 – Sensor specification
    # ------------------------------------------------------------------

    def sensors(self) -> List[Dict[str, Any]]:
        """
        Stage 1: build the list of CARLA sensors.

        Priority:
            1. Use explicit `sensors:` from the YAML config if present.
            2. Otherwise, query the underlying agent's `sensors()` method.
        """
        if self._sensor_list is not None:
            return self._sensor_list

        # Make sure we have a config (needed even for sensors)
        self._ensure_config_loaded(path_hint=None)
        self._ensure_extensions_loaded()
        self._ensure_pipeline_or_inner_loaded()

        sensor_specs = self._config.get("sensors", None)

        if sensor_specs is None and self._pipeline is None and hasattr(self._inner_agent, "sensors"):
            # Directly reuse underlying agent's sensors definition
            sensors = self._inner_agent.sensors()
        else:
            # Build sensors from YAML's generic list
            sensors = self._build_sensors_from_config(sensor_specs or [])

        if not isinstance(sensors, list):
            raise TypeError(f"Agent sensors() must return a list, got {type(sensors)!r}")

        # Allow extensions to modify the sensor list (opt-in).
        sensors = self._call_extension_hook_chain("on_sensors", sensors) or sensors
        if not isinstance(sensors, list):
            raise TypeError(f"Extension on_sensors must return a list, got {type(sensors)!r}")

        # Cache sensor types by id for stable folder naming later.
        try:
            self._sensor_type_by_id = {
                s.get("id"): s.get("type")
                for s in sensors
                if isinstance(s, dict) and s.get("id") and s.get("type")
            }
        except Exception:
            self._sensor_type_by_id = {}

        # Attach per-sensor output directories
        self._setup_sensor_directories(sensors)

        self._sensor_list = sensors
        return sensors

    def _build_sensors_from_config(self, sensor_specs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert YAML sensor specs into the Leaderboard-compatible list.

        YAML entries are assumed to look like:

            - id: rgb
              type: sensor.camera.rgb
              x: 1.5
              y: 0.0
              z: 2.0
              roll: 0.0
              pitch: 0.0
              yaw: 0.0
              width: 800
              height: 600
              fov: 90
              sensor_tick: 0.05

        Only a subset is strictly required; reasonable defaults are used.
        """
        sensors: List[Dict[str, Any]] = []

        for spec in sensor_specs:
            if "type" not in spec or "id" not in spec:
                raise ValueError(f"Invalid sensor spec (missing type/id): {spec}")

            sensor: Dict[str, Any] = {
                "type": spec["type"],
                "id": spec["id"],
            }

            # Positional parameters (if provided)
            for key in ["x", "y", "z", "roll", "pitch", "yaw"]:
                if key in spec:
                    sensor[key] = spec[key]

            # Sampling / tick
            if "sensor_tick" in spec:
                sensor["sensor_tick"] = spec["sensor_tick"]

            # Camera- or LiDAR-specific parameters
            if "camera" in spec["type"]:
                sensor["width"] = spec.get("width", 800)
                sensor["height"] = spec.get("height", 600)
                sensor["fov"] = spec.get("fov", 90)
            elif "lidar" in spec["type"]:
                sensor["channels"] = spec.get("channels", 32)
                sensor["range"] = spec.get("range", 50.0)
                sensor["points_per_second"] = spec.get("points_per_second", 100000)
                sensor["rotation_frequency"] = spec.get("rotation_frequency", 10.0)
                sensor["upper_fov"] = spec.get("upper_fov", 10.0)
                sensor["lower_fov"] = spec.get("lower_fov", -30.0)

            sensors.append(sensor)

        return sensors

    def _setup_sensor_directories(self, sensors: List[Dict[str, Any]]) -> None:
        """
        Create per-sensor directories under `self.save_path`.

        This is part of Stage 1 logically, but depends on dataset path
        which is initialized once the YAML is available.
        """
        if not self.collect_data or self.save_path is None:
            return

        for sensor in sensors:
            sensor_id = sensor["id"]
            folder_name = self._get_sensor_folder_name(sensor["type"], sensor_id)
            sensor_path = os.path.join(self.save_path, folder_name)
            os.makedirs(sensor_path, exist_ok=True)
            self.sensor_data_paths[sensor_id] = sensor_path

    @staticmethod
    def _get_sensor_folder_name(sensor_type: str, sensor_id: str) -> str:
        """
        Map a sensor (type, id) to a folder name.

        This is intentionally simple but stable; if you have an existing
        dataset layout, you can adjust this mapping to match.
        """
        if "camera" in sensor_type:
            return f"rgb_{sensor_id}"
        if "lidar" in sensor_type:
            return f"lidar_{sensor_id}"
        if "gnss" in sensor_type or "gps" in sensor_id.lower():
            return "gps"
        if "imu" in sensor_type:
            return "imu"
        if "speedometer" in sensor_type or sensor_id.lower() == "speed":
            return "measurements"
        return sensor_id

    # ------------------------------------------------------------------
    # Stage 2 – Data collection
    # ------------------------------------------------------------------

    def run_step(self, input_data: Dict[str, Tuple[int, Any]], timestamp: float) -> carla.VehicleControl:
        """
        The main control loop called by the Leaderboard.

        Stage 2: Save raw sensor data.
        Stage 3: Delegate to underlying agent.
        Stage 4: Apply global control post-processing.
        """
        self._global_step += 1

        # Apply weather on the first step — the world is guaranteed to be ready
        # by this point (CarlaDataProvider.set_world() has been called), whereas
        # setup() runs before the world is loaded and _init() is commented out
        # in the leaderboard evaluator.
        if not self._weather_applied:
            self._weather_applied = True
            try:
                from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
                world = CarlaDataProvider.get_world()
                if world is not None:
                    idx_str = (
                        os.environ.get("WEATHER_INDEX")
                        or os.environ.get("WEATHER")
                        or os.environ.get("WEATHERS")
                        or ""
                    ).strip()
                    if idx_str:
                        idx = int(idx_str)
                        if 0 <= idx < len(_WEATHER_IDS):
                            world.set_weather(_WEATHERS[_WEATHER_IDS[idx]])
            except Exception:
                pass

        # Ensure full initialization (for the first call when run_step is
        # invoked before setup() for any reason).
        self._ensure_config_loaded(path_hint=None)
        self._ensure_extensions_loaded()
        self._ensure_pipeline_or_inner_loaded()
        self._initialize_data_collection()

        # Stage 2: standardized dataset output
        if self.collect_data:
            self._save_sensor_data(input_data, timestamp)

        # Optional: allow extensions to transform the input before inference.
        input_data = self._call_extension_hook_chain("on_before_run_step", input_data, timestamp) or input_data
        if not isinstance(input_data, dict):
            raise TypeError(
                f"Extension on_before_run_step must return a dict, got {type(input_data)!r}"
            )

        # Stage 3: inference
        if self._pipeline is not None:
            ctx = {
                "agent": self,
                "input_data": input_data,
                "timestamp": timestamp,
                "global_step": self._global_step,
                "last_control": self._pipeline_last_control,
                "config": self._config,
                "external_config": self.external_config,
            }
            ctx = self._pipeline.run(ctx)
            control = ctx.get("control")
        else:
            control = self._inner_agent.run_step(input_data, timestamp)

        control = self._coerce_control(control)
        if self._pipeline is not None:
            self._pipeline_last_control = control

        # Optional: allow extensions to postprocess control with access to inputs.
        control = (
            self._call_extension_hook_chain("on_after_run_step", control, input_data, timestamp)
            or control
        )

        # Stage 4: global safety / sanity adjustments
        control = self._postprocess_control(control)

        return control

    def set_global_plan(self, global_plan_gps, global_plan_world_coord) -> None:
        """Propagate the global plan to both wrapper and inner agent.

        Many agents (notably TCP) expect `self._global_plan` to be set via
        the Leaderboard calling `set_global_plan(...)` on the agent instance.
        Since we delegate `run_step` to an inner agent object, we must forward
        the plan as well.
        """
        super().set_global_plan(global_plan_gps, global_plan_world_coord)

        self._ensure_config_loaded(path_hint=None)

        # Pipeline mode: RoutePlannerNextCommand reads _global_plan directly from
        # context['agent'] (= this object), which super() already updated above.
        if self._config is not None and self._config.get("pipeline") is not None:
            return

        self._ensure_inner_agent_loaded()

        if hasattr(self._inner_agent, "set_global_plan"):
            self._inner_agent.set_global_plan(global_plan_gps, global_plan_world_coord)
        else:
            # Best-effort: mirror the downsampled plan into the inner agent
            try:
                setattr(self._inner_agent, "_global_plan", getattr(self, "_global_plan", None))
                setattr(
                    self._inner_agent,
                    "_global_plan_world_coord",
                    getattr(self, "_global_plan_world_coord", None),
                )
            except Exception:
                pass

    def destroy(self) -> None:
        """Write a small run summary (best-effort) and forward destroy()."""
        try:
            # Best-effort run summary for plotting / joining dataset to job logs.
            if self.collect_data and self.save_path is not None:
                try:
                    summary = {
                        "run_id": os.environ.get("HPC_CARLA_RUN_ID") or os.environ.get("SLURM_JOB_ID"),
                        "job_id": os.environ.get("HPC_CARLA_JOB_ID"),
                        "run_tag": os.environ.get("HPC_CARLA_RUN_TAG"),
                        "node": os.environ.get("SLURMD_NODENAME") or os.uname().nodename,
                        "gpu_id": os.environ.get("GPU_ID"),
                        "global_steps": int(self._global_step),
                        "frames_saved_by_sensor": dict(self._frames_saved_by_sensor),
                        "data_collection_started_at": self._data_collection_started_at,
                        "data_collection_ended_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    }
                    out_path = os.path.join(self.save_path, "run_summary.json")
                    with open(out_path, "w") as f:
                        json.dump(summary, f, indent=2)
                except Exception:
                    pass

            self._call_extension_hook("on_destroy")
            if self._inner_agent is not None and hasattr(self._inner_agent, "destroy"):
                self._inner_agent.destroy()
        finally:
            return

    def _coerce_control(self, control) -> carla.VehicleControl:
        """Coerce pipeline/agent output into carla.VehicleControl.

        Supported inputs:
        - carla.VehicleControl
        - dict with keys:
            - {steer, throttle, brake} (preferred)
            - {steer, acc} where acc>=0 -> throttle, acc<0 -> brake
            - {steer, throttle} (brake defaults to 0)
        - tuple/list length 3 (order configurable via YAML):
            pipeline_control_tuple_order: [steer, throttle, brake] (default)
        """
        if isinstance(control, carla.VehicleControl):
            return control
        if isinstance(control, (tuple, list)) and len(control) == 3:
            order = None
            try:
                if isinstance(self._config, dict):
                    order = self._config.get("pipeline_control_tuple_order")
            except Exception:
                order = None

            if not order:
                order = ["steer", "throttle", "brake"]
            if not isinstance(order, list) or len(order) != 3:
                raise ValueError(
                    "pipeline_control_tuple_order must be a list of 3 strings, e.g. [steer, throttle, brake]"
                )

            as_dict = {str(order[i]): control[i] for i in range(3)}
            return self._coerce_control(as_dict)
        if isinstance(control, dict):
            steer = float(control.get("steer", 0.0))

            # Preferred explicit keys
            if "throttle" in control or "brake" in control:
                throttle = float(control.get("throttle", 0.0))
                brake = float(control.get("brake", 0.0))
                return carla.VehicleControl(steer=steer, throttle=throttle, brake=brake)

            # Alternative common convention: acceleration + steer
            if "acc" in control:
                acc = float(control.get("acc", 0.0))
                if acc >= 0.0:
                    return carla.VehicleControl(steer=steer, throttle=min(1.0, acc), brake=0.0)
                return carla.VehicleControl(steer=steer, throttle=0.0, brake=min(1.0, abs(acc)))

            # Last-resort: only steer
            return carla.VehicleControl(steer=steer, throttle=0.0, brake=0.0)
        if control is None:
            return carla.VehicleControl()
        raise TypeError(f"Unsupported control type: {type(control)!r}")

    def _initialize_data_collection(self) -> None:
        """
        Initialize dataset output directories and metadata.

        The base directory is chosen by the following priority:

            1. HPC_CARLA_DATASET_ROOT
            2. SAVE_PATH
            3. current working directory (fallback; mainly for debugging)

        The final layout is:

            <root> / <agent_name> / <weather> / <route_stem> / ...

        where <agent_name> is taken from YAML's `agent.name`, <weather>
        and <route_stem> come from environment variables typically set
        by the orchestrator.
        """
        if self.save_path is not None:
            # Already initialized
            return

        self.collect_data = bool(self._config.get("collect_data", True))

        # Global override to disable data collection (useful on low-storage nodes).
        # Keeps evaluation metrics/checkpoints intact.
        # Accepted env vars:
        # - HPC_CARLA_COLLECT_DATA=0/1
        # - COLLECT_DATA=0/1
        # - DISABLE_DATA_COLLECTION=1
        def _env_bool(name: str) -> Optional[bool]:
            raw = os.environ.get(name)
            if raw is None:
                return None
            v = str(raw).strip().lower()
            if v in ("1", "true", "yes", "y", "on"):
                return True
            if v in ("0", "false", "no", "n", "off"):
                return False
            return None

        if _env_bool("DISABLE_DATA_COLLECTION") is True:
            self.collect_data = False
        else:
            forced = _env_bool("HPC_CARLA_COLLECT_DATA")
            if forced is None:
                forced = _env_bool("COLLECT_DATA")
            if forced is not None:
                self.collect_data = bool(forced)

        if not self.collect_data:
            return

        if self._data_collection_started_at is None:
            try:
                self._data_collection_started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
            except Exception:
                self._data_collection_started_at = datetime.datetime.now().isoformat()

        project_root = os.environ.get("PROJECT_ROOT") or os.getcwd()

        # Prefer the repo's dedicated dataset directory by default.
        # Priority:
        #   1) HPC_CARLA_DATASET_ROOT (explicit override)
        #   2) DATASET_DIR (docker-compose sets this to /workspace/dataset)
        #   3) <PROJECT_ROOT>/dataset if it exists
        #   4) SAVE_PATH (legacy)
        #   5) cwd
        dataset_dir = os.environ.get("DATASET_DIR")
        default_repo_dataset = os.path.join(project_root, "dataset")
        base_root = (
            os.environ.get("HPC_CARLA_DATASET_ROOT")
            or dataset_dir
            or (default_repo_dataset if os.path.isdir(default_repo_dataset) else None)
            or os.environ.get("SAVE_PATH")
            or os.getcwd()
        )
        self.save_root = base_root

        agent_cfg = self._config.get("agent", {}) or {}
        legacy_agent_cfg = self._config.get("agent_config", {}) or {}
        agent_name = (
            os.environ.get("HPC_CARLA_AGENT_NAME")
            or agent_cfg.get("name")
            or agent_cfg.get("id")
            or self._config.get("model_type")
            or legacy_agent_cfg.get("agent_class")
            or "agent"
        )

        routes_path = os.environ.get("ROUTES", "route_unknown.xml")
        route_stem = Path(routes_path).stem

        # Weather name or index – depends on orchestration.
        weather = (
            os.environ.get("WEATHERS")
            or os.environ.get("WEATHER")
            or os.environ.get("WEATHER_INDEX")
            or "unknown"
        )
        # Normalize a bare integer index to "weather_N" to match manage_continuous labels.
        try:
            weather = f"weather_{int(weather)}"
        except (ValueError, TypeError):
            pass

        # If the orchestrator (manage_continuous.py) pre-computed the full leaf
        # path, honour it directly so sensor data and leaderboard checkpoint land
        # in the same directory tree (weather_N/map_NN/route_name/).
        explicit_save_path = os.environ.get("SAVE_PATH")
        if explicit_save_path:
            self.save_path = explicit_save_path
            run_tag = os.environ.get("HPC_CARLA_RUN_TAG") or Path(explicit_save_path).name
        else:
            run_tag = os.environ.get("HPC_CARLA_RUN_TAG") or _now_string()
            self.save_path = os.path.join(self.save_root, agent_name, weather, f"{route_stem}_{run_tag}")
        os.makedirs(self.save_path, exist_ok=True)

        # Initialize metadata file
        meta = {
            "agent_name": agent_name,
            "agent_module": agent_cfg.get("module") or legacy_agent_cfg.get("agent_file"),
            "agent_class": agent_cfg.get("class_name") or legacy_agent_cfg.get("agent_class"),
            "route": routes_path,
            "weather": weather,
            "created_at": datetime.datetime.now().isoformat(),
            "run_id": os.environ.get("HPC_CARLA_RUN_ID") or os.environ.get("SLURM_JOB_ID"),
            "job_id": os.environ.get("HPC_CARLA_JOB_ID"),
            "run_tag": run_tag,
            "node": os.environ.get("SLURMD_NODENAME") or os.uname().nodename,
            "gpu_id": os.environ.get("GPU_ID"),
        }
        meta_path = os.path.join(self.save_path, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    def _save_sensor_data(self, input_data: Dict[str, Tuple[int, Any]], timestamp: float) -> None:
        """
        Save all sensor streams for the current step.

        `input_data` uses the Leaderboard convention:
            input_data[sensor_id] = (frame_number, raw_data)

        where raw_data is:
            - a numpy array for camera and lidar,
            - a dict for the speedometer,
            - a CARLA measurement object for GNSS/IMU, etc.
        """
        if not self.collect_data or self.save_path is None:
            return

        for sensor_id, (frame, raw) in input_data.items():
            if sensor_id not in self.sensor_data_paths:
                # This can happen if the sensor list was overridden by the
                # agent; in that case, we lazily create a directory.
                sensor_type = self._sensor_type_by_id.get(sensor_id, "")
                folder_name = self._get_sensor_folder_name(sensor_type, sensor_id)
                sensor_path = os.path.join(self.save_path, folder_name)
                os.makedirs(sensor_path, exist_ok=True)
                self.sensor_data_paths[sensor_id] = sensor_path

            sensor_dir = self.sensor_data_paths[sensor_id]
            self._save_single_sensor(sensor_id, sensor_dir, frame, raw, timestamp)

            # Track per-sensor frame counts for lightweight summaries.
            try:
                self._frames_saved_by_sensor[sensor_id] = int(self._frames_saved_by_sensor.get(sensor_id, 0)) + 1
            except Exception:
                pass

            # Optional: extensions can save additional representations.
            self._call_extension_hook("on_save_sensor", sensor_id, sensor_dir, frame, raw, timestamp)

    def _save_single_sensor(
        self,
        sensor_id: str,
        sensor_dir: str,
        frame: int,
        raw: Any,
        timestamp: float,
    ) -> None:
        """
        Save a single sensor observation.

        - RGB: PNG
        - LIDAR: NPY (float32)
        - Speedometer / dict-like: JSON
        - GNSS/IMU: JSON (position / compass / accel/gyro)
        - Fallback: NPY via numpy.array(raw)
        """
        # Use frame number in filename for easier alignment; timestamp is
        # kept in a small JSON sidecar if needed.
        base_name = f"{frame:06d}"

        # Camera-like: numpy array with shape (H, W, C)
        if isinstance(raw, np.ndarray) and raw.ndim == 3:
            # CARLA gives BGRA; drop alpha. cv2.imwrite expects BGR — no conversion needed.
            img_bgr = raw[:, :, :3]
            out_path = os.path.join(sensor_dir, base_name + ".png")
            cv2.imwrite(out_path, img_bgr)

        # LIDAR: we assume an (N, 3 or 4) numpy array
        elif isinstance(raw, np.ndarray) and raw.ndim == 2:
            out_path = os.path.join(sensor_dir, base_name + ".npy")
            np.save(out_path, raw.astype(np.float32))

        # Dict-like measurements (speedometer etc.)
        elif isinstance(raw, dict):
            out_path = os.path.join(sensor_dir, base_name + ".json")
            payload = dict(raw)
            payload["_timestamp"] = float(timestamp)
            with open(out_path, "w") as f:
                json.dump(payload, f)

        # CARLA GNSS / IMU types
        elif isinstance(raw, carla.GnssMeasurement):
            payload = {
                "lat": raw.latitude,
                "lon": raw.longitude,
                "alt": raw.altitude,
                "_timestamp": float(timestamp),
            }
            out_path = os.path.join(sensor_dir, base_name + ".json")
            with open(out_path, "w") as f:
                json.dump(payload, f)

        elif isinstance(raw, carla.IMUMeasurement):
            payload = {
                "accel": [raw.accelerometer.x, raw.accelerometer.y, raw.accelerometer.z],
                "gyro": [raw.gyroscope.x, raw.gyroscope.y, raw.gyroscope.z],
                "compass": raw.compass,
                "_timestamp": float(timestamp),
            }
            out_path = os.path.join(sensor_dir, base_name + ".json")
            with open(out_path, "w") as f:
                json.dump(payload, f)

        # Fallback: try to cast to numpy and save
        else:
            try:
                arr = np.array(raw)
                out_path = os.path.join(sensor_dir, base_name + ".npy")
                np.save(out_path, arr)
            except Exception:
                # Last resort: write repr() for debugging
                out_path = os.path.join(sensor_dir, base_name + ".txt")
                with open(out_path, "w") as f:
                    f.write(repr(raw))

    # ------------------------------------------------------------------
    # Stage 3 – Inner agent loading and inference
    # ------------------------------------------------------------------

    def _ensure_config_loaded(self, path_hint: Optional[str]) -> None:
        """
        Ensure the wrapper YAML config is loaded.

        Priority:
            1. Already loaded -> return.
            2. `path_hint` (from Leaderboard setup()) if provided.
            3. AGENT_CONFIG env var.
        """
        if self._config is not None:
            return

        cfg_path = path_hint or os.environ.get("AGENT_CONFIG")
        if cfg_path is None:
            raise RuntimeError(
                "ConsolidatedAgent config not set. Provide a YAML path either "
                "via Leaderboard --agent-config or AGENT_CONFIG env var."
            )

        self._config_path = cfg_path
        self._config = _load_yaml(cfg_path)

        # Optional external config block (for advanced use).
        ext_cfg_spec = self._config.get("external_config")
        if ext_cfg_spec:
            self.external_config = self._load_external_config(ext_cfg_spec)
        else:
            self.external_config = None

    # ------------------------------------------------------------------
    # Extension hooks (optional, opt-in)
    # ------------------------------------------------------------------

    def _ensure_extensions_loaded(self) -> None:
        """Load optional extensions from config.

        This keeps compatibility by defaulting to no extensions.

        YAML schema:

            extensions:
              - module: some_pkg.some_mod
                class_name: SomeExtension
                args: { ... }

        Each extension may implement any of these methods (all optional):
            - on_setup(self)
            - on_sensors(self, sensors) -> sensors
            - on_before_run_step(self, input_data, timestamp) -> input_data
            - on_after_run_step(self, control, input_data, timestamp) -> control
            - on_save_sensor(self, sensor_id, sensor_dir, frame, raw, timestamp)
            - on_destroy(self)
        """
        if self._extensions_loaded:
            return
        self._extensions_loaded = True

        if self._config is None:
            return

        specs = self._config.get("extensions") or []
        if not specs:
            return

        if not isinstance(specs, list):
            raise ValueError("'extensions' must be a list")

        loaded: List[Any] = []
        for spec in specs:
            if not isinstance(spec, dict):
                raise ValueError(f"Invalid extension spec (expected dict): {spec!r}")
            module_path = spec.get("module")
            class_name = spec.get("class_name") or spec.get("class")
            args = spec.get("args") or {}
            if not module_path or not class_name:
                raise ValueError(f"Extension spec must include module and class_name: {spec!r}")
            if not isinstance(args, dict):
                raise ValueError(f"Extension args must be a dict: {spec!r}")

            ExtClass = _dynamic_import(module_path, class_name)
            try:
                ext = ExtClass(**args)
            except TypeError:
                # Allow extensions with a no-arg constructor.
                ext = ExtClass()
                for k, v in args.items():
                    setattr(ext, k, v)
            loaded.append(ext)

        self._extensions = loaded

    def _call_extension_hook(self, hook_name: str, *args):
        if not self._extensions:
            return None
        for ext in self._extensions:
            fn = getattr(ext, hook_name, None)
            if callable(fn):
                try:
                    fn(self, *args)
                except TypeError:
                    # Back-compat with extensions that don't accept agent.
                    fn(*args)
        return None

    def _call_extension_hook_chain(self, hook_name: str, value, *args):
        """Call hook sequentially where each hook can transform `value`."""
        if not self._extensions:
            return value
        out = value
        for ext in self._extensions:
            fn = getattr(ext, hook_name, None)
            if not callable(fn):
                continue
            try:
                res = fn(self, out, *args)
            except TypeError:
                res = fn(out, *args)
            if res is not None:
                out = res
        return out

    def _ensure_inner_agent_loaded(self) -> None:
        """
        Dynamically import and set up the underlying driving agent.

        The agent is loaded exactly once and then reused for the entire
        episode. Its own `setup()` is called with the `config_file`
        provided in the YAML (if any); otherwise we pass the wrapper
        YAML path as a fallback.
        """
        if self._inner_agent is not None:
            return

        if self._config is None:
            self._ensure_config_loaded(path_hint=None)

        # Support two schemas:
        #   (A) New schema:
        #       agent:
        #         module: team_code.interfuser_agent_orig
        #         class_name: InterfuserAgent
        #         config_file: /path/to/conf
        #   (B) Existing HPC-CARLA schema:
        #       agent_config:
        #         agent_file: /workspace/leaderboard/team_code/interfuser_agent.py
        #         agent_class: InterfuserAgent
        #         config_path: /workspace/leaderboard/team_code/interfuser/interfuser_config.py

        agent_cfg = self._config.get("agent") or self._config.get("agent_config") or {}

        module_path = agent_cfg.get("module")
        class_name = agent_cfg.get("class_name") or agent_cfg.get("class")
        agent_file = agent_cfg.get("agent_file")
        agent_class = agent_cfg.get("agent_class")

        if module_path and class_name:
            AgentClass = _dynamic_import(module_path, class_name)
            inner_conf = agent_cfg.get("config_file") or self._config_path
        elif agent_file and agent_class:
            agent_file = self._resolve_agent_file_path(agent_file)
            module = self._import_module_from_path(agent_file)
            try:
                AgentClass = getattr(module, agent_class)
            except AttributeError as exc:
                raise ImportError(
                    f"Class '{agent_class}' not found in agent_file '{agent_file}'"
                ) from exc
            inner_conf = agent_cfg.get("config_path") or self._config_path
        else:
            raise ValueError(
                "YAML must define either (agent.module + agent.class_name) or "
                "(agent_config.agent_file + agent_config.agent_class)."
            )

        # Let the inner agent perform its own configuration. We keep this
        # very close to original semantics to preserve leaderboard-level
        # behaviour.
        # Some provided configs historically point at a non-existent 'config.py'
        # while shipping 'interfuser_config.py'. If the requested config path
        # doesn't exist, try a small, safe fallback.
        inner_conf = self._resolve_config_path(inner_conf)

        # Instantiate the inner agent.
        # Prefer passing the config path first since most Leaderboard agents
        # inherit AutonomousAgent and require it in __init__.
        initialized_with_conf = True
        try:
            self._inner_agent = AgentClass(inner_conf)
        except TypeError:
            initialized_with_conf = False
            self._inner_agent = AgentClass()
            if hasattr(self._inner_agent, "setup"):
                self._inner_agent.setup(inner_conf)

        # If the underlying agent defines its own track, respect it; fall
        # back to SENSORS otherwise.
        if hasattr(self._inner_agent, "track"):
            self.track = self._inner_agent.track
        else:
            self.track = Track.SENSORS

        # Optionally pass the wrapper's external_config object if the
        # inner agent knows how to use it.
        if self.external_config is not None:
            setattr(self._inner_agent, "external_config", self.external_config)

    def _ensure_pipeline_or_inner_loaded(self) -> None:
        """Choose pipeline mode (new agents) or legacy inner-agent proxy mode."""
        if self._pipeline is not None or self._inner_agent is not None:
            return

        if self._config is None:
            self._ensure_config_loaded(path_hint=None)

        pipeline_specs = self._config.get("pipeline")

        # Avoid ambiguous configs: either you define a pipeline (new agent)
        # or you point at a legacy agent implementation.
        if pipeline_specs is not None:
            if self._config.get("agent") is not None or self._config.get("agent_config") is not None:
                raise ValueError(
                    "Config must use either 'pipeline' (composed agent) OR 'agent/agent_config' (legacy), not both."
                )

        # If 'pipeline' is present, we treat this config as a composed/new agent.
        # Otherwise we preserve existing behavior and load the legacy agent.
        if pipeline_specs is not None:
            if PipelineEngine is None:
                raise RuntimeError(
                    "Pipeline mode requested but PipelineEngine is unavailable. "
                    "This usually indicates a Python version/import-path mismatch."
                )
            self._pipeline = PipelineEngine(pipeline_specs)
            self._pipeline.setup(self, self._config)
            return

        self._ensure_inner_agent_loaded()

    # ------------------------------------------------------------------
    # Stage 4 – Global control post-processing
    # ------------------------------------------------------------------

    @staticmethod
    def _postprocess_control(control: Optional[carla.VehicleControl]) -> carla.VehicleControl:
        """
        Apply global safety/sanity adjustments to the returned control.

        The idea is to preserve the underlying agent's behaviour as much
        as possible while guarding against NaNs and out-of-range values.
        """
        if control is None:
            control = carla.VehicleControl()

        # Clamp basic ranges
        control.steer = float(np.clip(control.steer, -1.0, 1.0))
        control.throttle = float(np.clip(control.throttle, 0.0, 1.0))
        control.brake = float(np.clip(control.brake, 0.0, 1.0))

        # Ensure automatic gear
        control.manual_gear_shift = False

        # If both throttle and brake are positive, prefer brake.
        if control.throttle > 0.0 and control.brake > 0.0:
            control.throttle = 0.0

        return control

    # ------------------------------------------------------------------
    # External config loading helpers
    # ------------------------------------------------------------------

    def _load_external_config(self, spec: Dict[str, Any]) -> Any:
        """
        Load an optional external configuration object as described in the YAML.

        Supported formats:

            format: "python"
                path: path/to/module_or_file.py
                class_name: GlobalConfig
                attribute: optional attribute name to extract from the class
                           instance (e.g., "cfg" if you want obj.cfg)

            format: "yaml"
                path: path/to/file.yaml
                (returns a dict)
        """
        if not spec:
            return None

        # Shorthand: allow external_config: /path/to/file.py (or .yaml)
        # Used by existing configs in this repo.
        if isinstance(spec, str):
            path = self._resolve_config_path(spec)
            if path.endswith((".yaml", ".yml")):
                return _load_yaml(path)
            if path.endswith(".py"):
                return self._import_module_from_path(path)
            # Fallback: try import as module path
            return importlib.import_module(path)

        fmt = spec.get("format") or spec.get("type") or "python"
        path = spec.get("path")
        if not path:
            raise ValueError("external_config must specify a 'path'.")

        path = self._resolve_config_path(path)

        if fmt.lower() in ("yaml", "yml"):
            return _load_yaml(path)

        if fmt.lower() == "python":
            class_name = spec.get("class_name")
            attribute = spec.get("attribute")
            if not class_name:
                raise ValueError("external_config.python requires 'class_name'.")

            # Load module from an arbitrary file path if needed
            if path.endswith(".py") and not self._is_module_path(path):
                module_name = Path(path).stem + "_extcfg"
                spec_obj = importlib.util.spec_from_file_location(module_name, path)
                if spec_obj is None or spec_obj.loader is None:
                    raise ImportError(f"Cannot load external config module from {path}")
                module = importlib.util.module_from_spec(spec_obj)
                spec_obj.loader.exec_module(module)
            else:
                module = importlib.import_module(path)

            cfg_cls = getattr(module, class_name)
            cfg_obj = cfg_cls()

            if attribute:
                return getattr(cfg_obj, attribute)
            return cfg_obj

        raise ValueError(f"Unsupported external_config format: {fmt!r}")

    def _import_module_from_path(self, path: str):
        """Import either a Python file path or a dotted module path."""
        # If it looks like a file path (or exists), load from file.
        if path.endswith(".py"):
            if not os.path.exists(path):
                raise ImportError(f"Python file not found: {path}")

            module_name = Path(path).stem + "_dynmod"
            spec_obj = importlib.util.spec_from_file_location(module_name, path)
            if spec_obj is None or spec_obj.loader is None:
                raise ImportError(f"Cannot load module from {path}")
            module = importlib.util.module_from_spec(spec_obj)
            spec_obj.loader.exec_module(module)
            return module
        return importlib.import_module(path)

    def _resolve_config_path(self, path: Optional[str]) -> Optional[str]:
        """Best-effort fixups for config paths inside /workspace mounts."""
        if not path:
            return path
        if os.path.exists(path):
            return path

        # Common InterFuser config typo in provided YAMLs.
        base_dir = os.path.dirname(path)
        if os.path.basename(path) == "config.py":
            candidate = os.path.join(base_dir, "interfuser_config.py")
            if os.path.exists(candidate):
                return candidate

        return path

    def _resolve_agent_file_path(self, path: str) -> str:
        """Resolve legacy agent_file paths like /team_code/foo_agent.py -> /team_code/foo/foo_agent.py."""
        if os.path.exists(path):
            return path

        base_dir = os.path.dirname(path)
        base_name = os.path.basename(path)

        if base_name.endswith("_agent.py"):
            prefix = base_name[: -len("_agent.py")]
            candidate = os.path.join(base_dir, prefix, base_name)
            if os.path.exists(candidate):
                return candidate

        return path

    @staticmethod
    def _is_module_path(path: str) -> bool:
        """
        Heuristic: treat strings without directory separators as module paths.
        """
        return ("/" not in path) and ("\\" not in path)
