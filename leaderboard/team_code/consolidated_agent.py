#!/usr/bin/env python3
"""
Universal Consolidated Agent for CARLA Leaderboard 1.0
Supports any agent configuration format and model architecture
"""

import os
import sys
import yaml
import torch
import importlib
import json
import numpy as np
from PIL import Image
from pathlib import Path
import cv2
from collections import deque
from leaderboard.autoagents.autonomous_agent import AutonomousAgent
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
import carla

def get_entry_point():
    """Required by CARLA Leaderboard."""
    return "ConsolidatedAgent"

class ConsolidatedAgent(AutonomousAgent):
    """
    Universal agent that handles any CARLA Leaderboard 1.0 compatible agent.
    Supports multiple configuration formats, model architectures, and loading methods.
    """
    
    DEFAULT_SENSORS = [
        {'type': 'sensor.camera.rgb', 'x': 1.3, 'y': 0.0, 'z': 2.3, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
         'width': 900, 'height': 256, 'fov': 100, "enable_postprocess_effects": True,
        "gamma": 2.2,
        "exposure_mode": "histogram", 'id': 'rgb_front'},
        {'type': 'sensor.camera.rgb', 'x': 0.0, 'y': 0.0, 'z': 50.0, 'roll': 0.0, 'pitch': -90.0, 'yaw': 0.0,
         'width': 512, 'height': 512, 'fov': 110, "enable_postprocess_effects": True,
        "gamma": 2.2,
        "exposure_mode": "histogram", 'id': 'bev'},
        {'type': 'sensor.camera.semantic_segmentation', 'x': 1.3, 'y': 0.0, 'z': 2.3, 
         'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0, 'width': 900, 'height': 256, 'fov': 100, 'id': 'semantic_front'},
        {'type': 'sensor.camera.depth', 'x': 1.3, 'y': 0.0, 'z': 2.3, 
         'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0, 'width': 900, 'height': 256, 'fov': 100, 'id': 'depth_front'},
        {'type': 'sensor.lidar.ray_cast', 'x': 0.0, 'y': 0.0, 'z': 2.5, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
         'id': 'lidar', 'channels': 64, 'range': 100, 'points_per_second': 1000000, 
         'rotation_frequency': 20, 'upper_fov': 10, 'lower_fov': -30},
        {'type': 'sensor.other.imu', 'x': 0.0, 'y': 0.0, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0, 'id': 'imu'},
        {'type': 'sensor.other.gnss', 'x': 0.0, 'y': 0.0, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0, 'id': 'gps'},
        {'type': 'sensor.speedometer', 'id': 'speed'}
    ]

    # ---- NEW: weather helper presets and application ----
    _WEATHER_PRESETS = [
        "ClearNoon","CloudyNoon","WetNoon","WetCloudyNoon",
        "MidRainyNoon","HardRainNoon","SoftRainNoon",
        "ClearSunset","CloudySunset","WetSunset","WetCloudySunset",
        "MidRainSunset","HardRainSunset","SoftRainSunset",
    ]
    
    _ENFORCE_W_N_TICKS = int(os.environ.get("WEATHER_ENFORCE_TICKS", "120"))  # ~6s @20Hz, ~2s @60Hz
    _ENFORCE_EVERY_N   = int(os.environ.get("WEATHER_ENFORCE_EVERY", "10"))   # re-apply cadence during warmup

    def _resolve_weather_from_env(self):
        """
        Resolve a carla.WeatherParameters from env:
          - WEATHER_PRESET (e.g., 'ClearNoon')
          - or WEATHER_INDEX (0–13 or 1–14)
        Returns (name, WeatherParameters) or (None, None) if not specified/invalid.
        """
        preset_env = os.environ.get('WEATHER_PRESET', '').strip()
        index_env = os.environ.get('WEATHER_INDEX', '').strip()

        name = None
        wp = None

        # Prefer explicit preset name if provided
        if preset_env:
            if hasattr(carla.WeatherParameters, preset_env):
                name = preset_env
                wp = getattr(carla.WeatherParameters, name)
            else:
                print(f"ConsolidatedAgent: WARNING - WEATHER_PRESET '{preset_env}' not found in carla.WeatherParameters")

        # Fallback to index
        if wp is None and index_env:
            try:
                idx = int(index_env)
                # accept 1–14 or 0–13
                #if 1 <= idx <= 14:
                #    idx -= 1
                idx = max(0, min(idx, len(self._WEATHER_PRESETS) - 1))
                name = self._WEATHER_PRESETS[idx]
                wp = getattr(carla.WeatherParameters, name)
            except Exception:
                print(f"ConsolidatedAgent: WARNING - Invalid WEATHER_INDEX '{index_env}'")

        return name, wp
    
    def _force_weather_if_needed(self, world, frame):
        if self._target_weather is None or world is None:
            return
        must_force = (frame <= self._weather_force_until) \
                    or (frame - self._last_weather_apply) >= self._weather_refresh_every
        if must_force:
            try:
                # Only call set_weather if different (cheap check)
                if world.get_weather() != self._target_weather:
                    world.set_weather(self._target_weather)
                self._last_weather_apply = frame
            except Exception as e:
                print(f"ConsolidatedAgent: weather force skipped: {e}")



    def _resolve_weather_from_env(self):
        """
        Resolve a carla.WeatherParameters from env:
        - WEATHER_PRESET (e.g., 'ClearNoon')
        - or WEATHER_INDEX (0–13 or 1–14)
        Returns (name, WeatherParameters) or (None, None).
        """
        preset_env = os.environ.get('WEATHER_PRESET', '').strip()
        index_env  = os.environ.get('WEATHER_INDEX', '').strip()

        name, wp = None, None
        if preset_env:
            if hasattr(carla.WeatherParameters, preset_env):
                name = preset_env
                wp   = getattr(carla.WeatherParameters, name)

        if wp is None and index_env:
            try:
                idx = int(index_env)
                #if 1 <= idx <= len(self._WEATHER_PRESETS):  # accept 1-based
                #    idx -= 1
                idx  = max(0, min(idx, len(self._WEATHER_PRESETS)-1))
                name = self._WEATHER_PRESETS[idx]
                wp   = getattr(carla.WeatherParameters, name)
            except Exception:
                print(f"[ConsolidatedAgent] Invalid WEATHER_INDEX='{index_env}'")

        return name, wp

    def _apply_weather(self, world, name, wp):
        try:
            world.set_weather(wp)
            self._weather_applied = {"name": name}
            print(f"[ConsolidatedAgent] Weather applied: {name}")
            return True
        except Exception as e:
            print(f"[ConsolidatedAgent] Failed to set weather '{name}': {e}")
            return False

    def _apply_weather_from_env(self):
        """
        Apply the resolved weather to the current CARLA world if available.
        Stores summary in self._weather_applied for later metadata logging.
        """
        name, wp = self._resolve_weather_from_env()
        if not wp:
            return  # nothing to do

        world = None
        try:
            world = CarlaDataProvider.get_world()
        except Exception as e:
            print(f"ConsolidatedAgent: get_world() not ready yet: {e}")

        # If DataProvider world isn't ready (rare), attempt direct client
        if world is None:
            try:
                host = os.environ.get("CARLA_HOST", os.environ.get("HOST", "127.0.0.1"))
                port = int(os.environ.get("CARLA_PORT", os.environ.get("PORT", "2000")))
                client = carla.Client(host, port)
                client.set_timeout(5.0)
                world = client.get_world()
            except Exception as e:
                print(f"ConsolidatedAgent: Unable to acquire world via client to set weather: {e}")

        if world is not None:
            try:
                world.set_weather(wp)
                self._weather_applied = {
                    "index": os.environ.get('WEATHER_INDEX', ''),
                    "preset": name,
                }
                print(f"ConsolidatedAgent: Applied weather -> {name} (WEATHER_INDEX={os.environ.get('WEATHER_INDEX','')})")
            except Exception as e:
                print(f"ConsolidatedAgent: Failed to apply weather '{name}': {e}")
        else:
            print("ConsolidatedAgent: WARNING - No CARLA world available; cannot apply weather")

    # -------------------------------------------------------------------------

    def setup(self, path_to_config_yaml):
        self._weather_enforced_once = False
        print(f"ConsolidatedAgent: Loading configuration from {path_to_config_yaml}")
        with open(path_to_config_yaml, 'r') as f:
            self.config = yaml.safe_load(f)
        
        if 'external_config' in self.config:
            self._load_external_config()
        
        self.model_path = self.config.get('model_path', 'none')
        self.model_type = self.config.get('model_type', 'generic')
        self.model_config = self.config.get('model_config', {})
        
        if 'agent_config' in self.config:
            self._extract_from_agent()
        
        if 'lidar_model_dir' in self.model_config or 'uniplanner_dir' in self.model_config:
            self._convert_lav_config()

        self._config_path = path_to_config_yaml
        
        self.sensor_config = self.config.get('sensors', self.DEFAULT_SENSORS)
        if self.sensor_config == 'default':
            self.sensor_config = self.DEFAULT_SENSORS
        
        self.control_config = self.config.get('control', {
            'target_speed': 30.0,
            'brake_threshold': 0.5,
            'steer_damping': 0.3
        })
        
        print(f"ConsolidatedAgent: Configuration loaded:")
        print(f"  Model path: {self.model_path}")
        print(f"  Model type: {self.model_type}")
        if 'model_components' in self.config:
            print(f"  Model components: {len(self.config['model_components'])} components")
        print(f"  Sensors: {len(self.sensor_config)} configured")

        # Resolve desired weather from env and apply once
        self._desired_weather = self._resolve_weather_from_env()
        self._weather_enforce_until = 0
        self._last_weather_apply_frame = -999999

        if self._desired_weather[1] is not None:
            world = CarlaDataProvider.get_world()
            if world:
                if self._apply_weather(world, *self._desired_weather):
                    # Keep re-applying for a short warmup window, since LB/ScenarioRunner
                    # may overwrite weather once scenarios start.
                    frame = world.get_snapshot().frame if world.get_snapshot() else 0
                    self._weather_enforce_until = frame + _ENFORCE_W_N_TICKS
                    self._last_weather_apply_frame = frame

        
        self._load_model()
        self._initialize_data_collection()
        
        self.input_buffer = {}
        self.waypoint_buffer = deque(maxlen=50)
        self.traffic_light_buffer = deque(maxlen=10)
        self.stop_sign_buffer = deque(maxlen=10)
        
        self.prev_steer = 0.0
        self.prev_brake = 0.0
        self.frame_count = 0
        self.last_speed = 0.0
        
        print(f"ConsolidatedAgent: Setup complete")
    
    def _load_external_config(self):
        external_config_path = Path(self.config['external_config'])
        print(f"ConsolidatedAgent: Loading external config from {external_config_path}")
        if not external_config_path.exists():
            print(f"Warning: External config not found: {external_config_path}")
            return
        ext = external_config_path.suffix.lower()
        if ext == '.py':
            self._load_python_config(external_config_path)
        elif ext in ['.yaml', '.yml']:
            with open(external_config_path, 'r') as f:
                external_config = yaml.safe_load(f)
            self._merge_configs(self.config, external_config)
        elif ext == '.json':
            with open(external_config_path, 'r') as f:
                external_config = json.load(f)
            self._merge_configs(self.config, external_config)
        else:
            print(f"Warning: Unknown config format: {ext}")
    
    def _load_python_config(self, config_path):
        print(f"ConsolidatedAgent: Loading Python config from {config_path}")
        sys.path.insert(0, str(config_path.parent))
        module_name = config_path.stem
        try:
            config_module = importlib.import_module(module_name)
            external_config = {}
            for attr_name in dir(config_module):
                if 'config' in attr_name.lower() and not attr_name.startswith('_'):
                    attr = getattr(config_module, attr_name)
                    if isinstance(attr, type):
                        cfg = attr()
                        for k in dir(cfg):
                            if not k.startswith('_'):
                                v = getattr(cfg, k)
                                if not callable(v):
                                    external_config[k] = v
                    elif isinstance(attr, dict):
                        external_config.update(attr)
                    elif hasattr(attr, '__dict__'):
                        for k, v in attr.__dict__.items():
                            if not k.startswith('_'):
                                external_config[k] = v
            for key in dir(config_module):
                if not key.startswith('_') and key.isupper():
                    v = getattr(config_module, key)
                    if not callable(v):
                        external_config[key] = v
            if hasattr(config_module, 'get_config'):
                external_config.update(config_module.get_config())
            elif hasattr(config_module, 'make_config'):
                external_config.update(config_module.make_config())
            if hasattr(config_module, 'GlobalConfig'):
                GC = config_module.GlobalConfig
                if hasattr(GC, 'model_path'):
                    self.config['model_path'] = GC.model_path
                if hasattr(GC, 'record_frame_rate'):
                    self.config['frame_rate'] = GC.record_frame_rate
                for attr in dir(GC):
                    if not attr.startswith('_'):
                        v = getattr(GC, attr)
                        if not callable(v):
                            external_config[attr] = v
            self.config.setdefault('model_config', {}).update(external_config)
            print(f"  Extracted {len(external_config)} configuration values from Python config")
        except Exception as e:
            print(f"Error loading Python config: {e}")
            import traceback; traceback.print_exc()
    
    def _extract_from_agent(self):
        agent_config = self.config['agent_config']
        print(f"ConsolidatedAgent: Extracting from agent using config: {agent_config}")
        agent_file = agent_config.get('agent_file')
        agent_class_name = agent_config.get('agent_class')
        config_path = agent_config.get('config_path')
        if not agent_file or not agent_class_name:
            print("Warning: agent_file and agent_class required for agent extraction")
            return
        agent_path = Path(agent_file); sys.path.insert(0, str(agent_path.parent))
        try:
            module_name = agent_path.stem
            agent_module = importlib.import_module(module_name)
            agent_class = getattr(agent_module, agent_class_name)
            agent = agent_class(config_path) if config_path else agent_class()
            if hasattr(agent, 'setup') and config_path:
                agent.setup(config_path)
            self._extract_models_from_agent(agent)
            if hasattr(agent, 'config'):
                self.config.setdefault('model_config', {})
                if hasattr(agent.config, '__dict__'):
                    self.config['model_config'].update(agent.config.__dict__)
                elif isinstance(agent.config, dict):
                    self.config['model_config'].update(agent.config)
            print(f"  Successfully extracted model and config from {agent_class_name}")
        except Exception as e:
            print(f"Error extracting from agent: {e}")
            import traceback; traceback.print_exc()
    
    def _extract_models_from_agent(self, agent):
        model_attrs = ['model', 'net', 'network', 'backbone', 'policy', 'actor', 'planner', 'controller']
        for attr in model_attrs:
            if hasattr(agent, attr):
                model = getattr(agent, attr)
                if isinstance(model, torch.nn.Module):
                    self.model = model
                    print(f"  Extracted model from agent.{attr}")
                    return
        multi = {
            'lidar_model': ['lidar_model','lidar_net','lidar_encoder'],
            'bev_model': ['bev_model','bev_net','bev_encoder'],
            'seg_model': ['seg_model','segmentation','seg_net'],
            'uniplanner': ['uniplanner','planner','planning_model'],
            'controller': ['controller','control_model','control_net'],
        }
        self.model_components = {}
        for cname, attrs in multi.items():
            for attr in attrs:
                if hasattr(agent, attr):
                    m = getattr(agent, attr)
                    if isinstance(m, torch.nn.Module):
                        self.model_components[cname] = m
                        print(f"  Extracted {cname} from agent.{attr}")
                        break
        if self.model_components:
            if 'lav' in self.model_type.lower():
                self._create_lav_wrapper()
            else:
                self.model = self.model_components
    
    def _merge_configs(self, base_config, external_config):
        for key, value in external_config.items():
            if key not in base_config:
                base_config[key] = value
            elif isinstance(value, dict) and isinstance(base_config[key], dict):
                self._merge_configs(base_config[key], value)
    
    def _convert_lav_config(self):
        if 'model_components' not in self.config:
            self.config['model_components'] = {}
        mapping = {
            'lidar_model_dir': ('lidar_model', 'checkpoint'),
            'uniplanner_dir': ('uniplanner', 'checkpoint'),
            'bra_model_dir': ('bra_model', 'checkpoint'),
            'bra_model_trace_dir': ('bra_model_trace', 'trace'),
            'seg_model_dir': ('seg_model', 'checkpoint'),
            'seg_model_trace_dir': ('seg_model_trace', 'trace'),
            'bev_model_dir': ('bev_model', 'checkpoint'),
        }
        for k, (cname, ctype) in mapping.items():
            if k in self.model_config:
                path = self.model_config[k]
                if path and os.path.exists(path):
                    self.config['model_components'][cname] = {'path': path, 'type': ctype}
        if self.model_type == 'generic':
            self.model_type = 'lav'
        print(f"ConsolidatedAgent: Converted LAV config - found {len(self.config['model_components'])} components")
    
    def _load_model(self):
        if 'model_components' in self.config:
            self._load_multi_component_model()
        elif self.model_path and self.model_path != 'none':
            print(f"ConsolidatedAgent: Loading model from {self.model_path}")
            if self.model_type == 'interfuser':
                self._load_interfuser_model()
            elif self.model_type == 'lav':
                self._load_lav_model()
            elif self.model_type == 'transfuser':
                self._load_transfuser_model()
            else:
                self._load_generic_model()
            if hasattr(self, 'model') and hasattr(self.model, 'eval'):
                self.model.eval()
                print(f"ConsolidatedAgent: Model loaded and set to eval mode")
        else:
            print("ConsolidatedAgent: No model specified, using rule-based control")
            self.model = None
    
    def _load_multi_component_model(self):
        print("ConsolidatedAgent: Loading multi-component model architecture")
        self.model_components = {}
        components_config = self.config.get('model_components', {})
        for cname, info in components_config.items():
            cpath = info.get('path'); ctype = info.get('type', 'checkpoint')
            if not cpath or not os.path.exists(cpath):
                print(f"Warning: Component {cname} path not found: {cpath}")
                continue
            print(f"  Loading {cname} from {cpath}")
            try:
                if ctype == 'trace':
                    self.model_components[cname] = torch.jit.load(cpath)
                elif ctype == 'checkpoint':
                    ckpt = torch.load(cpath, map_location='cpu')
                    if isinstance(ckpt, torch.nn.Module):
                        self.model_components[cname] = ckpt
                    elif 'model' in ckpt:
                        self.model_components[cname] = ckpt['model']
                    elif 'state_dict' in ckpt:
                        mci = info.get('model_class')
                        if mci:
                            module = importlib.import_module(mci['module'])
                            model_class = getattr(module, mci['name'])
                            model = model_class(**mci.get('args', {}))
                            model.load_state_dict(ckpt['state_dict'])
                            self.model_components[cname] = model
                        else:
                            self.model_components[cname] = ckpt  # raw sd for later
                    else:
                        self.model_components[cname] = ckpt
                if hasattr(self.model_components[cname], 'eval'):
                    self.model_components[cname].eval()
                    if hasattr(self.model_components[cname], 'to'):
                        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                        self.model_components[cname] = self.model_components[cname].to(device)
                print(f"    ✓ {cname} loaded successfully")
            except Exception as e:
                print(f"    ✗ Failed to load {cname}: {e}")
        if self.model_type == 'lav':
            self._create_lav_wrapper()
        else:
            self.model = self.model_components
        print(f"ConsolidatedAgent: Loaded {len(self.model_components)} model components")
    
    def _load_generic_model(self):
        """Safer generic loader that won't call .to() on a dict."""
        try:
            ckpt = None
            try:
                ckpt = torch.load(self.model_path, map_location='cpu', weights_only=True)  # PyTorch >=2.0
            except TypeError:
                ckpt = torch.load(self.model_path, map_location='cpu')
            if 'model_class' in self.model_config:
                module_path = self.model_config['model_module']
                class_name = self.model_config['model_class']
                module = importlib.import_module(module_path)
                model_class = getattr(module, class_name)
                model_args = self.model_config.get('model_args', {})
                self.model = model_class(**model_args)
                sd = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
                self.model.load_state_dict(sd)
            else:
                # If it's a pure state_dict but we don't know class, bail out gracefully
                if isinstance(ckpt, dict) and 'state_dict' in ckpt and not isinstance(ckpt.get('model'), torch.nn.Module):
                    raise RuntimeError("Checkpoint contains a state_dict but no model_class was provided.")
                self.model = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if hasattr(self.model, 'to'):
                self.model = self.model.to(self.device)
        except Exception as e:
            print(f"Warning: Could not load model as PyTorch: {e}")
            print("Will use simple rule-based control as fallback")
            self.model = None
    
    def _load_interfuser_model(self):
        """Load InterFuser baseline and state_dict safely (no wandb required)."""
        try:
            os.environ.setdefault("WANDB_MODE", "disabled")
            try:
                from team_code.interfuser.interfuser.timm.models.interfuser import interfuser_baseline
            except ImportError:
                # fallback import path if repo layout differs
                from interfuser.timm.models.interfuser import interfuser_baseline
            self.model = interfuser_baseline()
            try:
                ckpt = torch.load(self.model_path, map_location='cpu', weights_only=True)
            except TypeError:
                ckpt = torch.load(self.model_path, map_location='cpu')
            # Extract state dict
            if isinstance(ckpt, dict) and 'state_dict' in ckpt:
                sd = ckpt['state_dict']
            else:
                sd = ckpt
            # Strip potential "module." prefixes
            if isinstance(sd, dict):
                sd = {k.replace('module.', ''): v for k, v in sd.items()}
            missing, unexpected = self.model.load_state_dict(sd, strict=False)
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = self.model.to(self.device).eval()
            print(f"ConsolidatedAgent: InterFuser model loaded (missing={len(missing)}, unexpected={len(unexpected)})")
        except Exception as e:
            print(f"Error loading InterFuser model: {e}")
            print("Trying fallback to generic model loading...")
            self._load_generic_model()
    
    def _load_lav_model(self):
        if 'model_components' in self.config:
            self._load_multi_component_model(); return
        try:
            ckpt = torch.load(self.model_path, map_location='cpu')
            if 'model' in ckpt:
                self.model = ckpt['model']
            elif 'state_dict' in ckpt:
                from lav_model import LAV
                self.model = LAV(**self.model_config.get('model_args', {}))
                self.model.load_state_dict(ckpt['state_dict'])
            else:
                self.model = ckpt
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if hasattr(self.model, 'to'):
                self.model = self.model.to(self.device)
        except Exception as e:
            print(f"Error loading LAV model: {e}")
            self._load_generic_model()
    
    def _create_lav_wrapper(self):
        class LAVModelWrapper:
            def __init__(self, components, device):
                self.components = components
                self.device = device
                self.lidar_model = components.get('lidar_model')
                self.uniplanner = components.get('uniplanner')
                self.bra_model = components.get('bra_model')
                self.seg_model = components.get('seg_model')
                self.bev_model = components.get('bev_model')
            def __call__(self, inputs):
                outputs = {}
                if self.bev_model and 'bev' in inputs:
                    outputs['bev_features'] = self.bev_model(inputs['bev'])
                if self.lidar_model and 'lidar' in inputs:
                    outputs['lidar_features'] = self.lidar_model(inputs['lidar'])
                if self.seg_model and 'rgb_front' in inputs:
                    outputs['segmentation'] = self.seg_model(inputs['rgb_front'])
                if self.uniplanner:
                    planner_input = {}
                    if 'bev_features' in outputs: planner_input['bev'] = outputs['bev_features']
                    if 'lidar_features' in outputs: planner_input['lidar'] = outputs['lidar_features']
                    if 'measurements' in inputs: planner_input['measurements'] = inputs['measurements']
                    outputs.update(self.uniplanner(planner_input))
                if self.bra_model and 'bev_features' in outputs:
                    outputs['behavior'] = self.bra_model(outputs['bev_features'])
                return outputs
            def eval(self):
                for c in self.components.values():
                    if hasattr(c, 'eval'): c.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = LAVModelWrapper(self.model_components, self.device)
    
    def _load_transfuser_model(self):
        try:
            from transfuser_model import TransFuser
            self.model = TransFuser(self.model_config)
            ckpt = torch.load(self.model_path, map_location='cpu')
            self.model.load_state_dict(ckpt['state_dict'])
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = self.model.to(self.device)
        except Exception as e:
            print(f"Error loading TransFuser model: {e}")
            self._load_generic_model()
    
    def sensors(self):
        sensors = []
        for sensor_spec in self.sensor_config:
            sensor = {'type': sensor_spec['type'], 'id': sensor_spec['id']}
            for key in ['x','y','z','roll','pitch','yaw']:
                if key in sensor_spec:
                    sensor[key] = sensor_spec[key]
            if 'camera' in sensor_spec['type']:
                sensor['width'] = sensor_spec.get('width', 800)
                sensor['height'] = sensor_spec.get('height', 600)
                sensor['fov'] = sensor_spec.get('fov', 90)
            elif 'lidar' in sensor_spec['type']:
                sensor['channels'] = sensor_spec.get('channels', 32)
                sensor['range'] = sensor_spec.get('range', 50)
                sensor['points_per_second'] = sensor_spec.get('points_per_second', 100000)
                sensor['rotation_frequency'] = sensor_spec.get('rotation_frequency', 10)
                sensor['upper_fov'] = sensor_spec.get('upper_fov', 10)
                sensor['lower_fov'] = sensor_spec.get('lower_fov', -30)
            sensors.append(sensor)
        self._setup_sensor_directories(sensors)
        return sensors
    
    def _setup_sensor_directories(self, sensors):
        for sensor in sensors:
            sensor_id = sensor['id']
            folder_name = self._get_sensor_folder_name(sensor['type'], sensor_id)
            sensor_path = os.path.join(self.save_path, folder_name)
            os.makedirs(sensor_path, exist_ok=True)
            self.sensor_data_paths[sensor_id] = sensor_path
    
    def run_step(self, input_data, timestamp):
        # Enforce / heal weather
        if getattr(self, "_desired_weather", (None, None))[1] is not None:
            world = CarlaDataProvider.get_world()
            if world:
                snap = world.get_snapshot()
                frame = snap.frame if snap else 0
                desired_name, desired_wp = self._desired_weather

                # During warm-up, re-apply every N frames
                if frame <= self._weather_enforce_until:
                    if frame - self._last_weather_apply_frame >= _ENFORCE_EVERY_N:
                        self._apply_weather(world, desired_name, desired_wp)
                        self._last_weather_apply_frame = frame
                else:
                    # After warm-up, heal if someone toggled it later
                    current = world.get_weather()
                    # Compare by fields because WeatherParameters has no __eq__
                    def _as_tuple(w):
                        return (w.cloudiness, w.precipitation, w.precipitation_deposits,
                                w.wind_intensity, w.sun_azimuth_angle, w.sun_altitude_angle,
                                w.fog_density, w.fog_distance, w.wetness, w.fog_falloff)
                    if _as_tuple(current) != _as_tuple(desired_wp):
                        self._apply_weather(world, desired_name, desired_wp)
                        self._last_weather_apply_frame = frame


        try:
            self._save_sensor_data(input_data, timestamp)
        except Exception as e:
            print(f"Warning: Data saving failed: {e}")
        processed = self._process_sensor_data(input_data)
        if 'speed' in processed:
            self.last_speed = processed['speed']
        if self.model is not None or (hasattr(self, 'model_components') and self.model_components):
            control = self._model_inference(processed, timestamp)
        else:
            control = self._rule_based_control(processed, timestamp)
        control = self._postprocess_control(control)
        self.frame_count += 1
        return control
    
    # --- processing helpers (unchanged from your version) ---
    def _process_sensor_data(self, input_data):
        processed = {}
        for sensor_id, sensor_data in input_data.items():
            data = sensor_data[1] if (isinstance(sensor_data, tuple) and len(sensor_data) == 2) else sensor_data
            sid = sensor_id.lower()
            if 'rgb' in sid:
                processed[sensor_id] = self._process_rgb_image(data)
            elif 'semantic' in sid:
                processed[sensor_id] = self._process_semantic_image(data)
            elif 'depth' in sid:
                processed[sensor_id] = self._process_depth_image(data)
            elif 'lidar' in sid:
                processed[sensor_id] = self._process_lidar(data)
            elif 'imu' in sid:
                processed[sensor_id] = self._process_imu(data)
            elif 'gps' in sid or 'gnss' in sid:
                processed[sensor_id] = self._process_gps(data)
            elif 'speed' in sid:
                processed[sensor_id] = self._process_speed(data)
            else:
                processed[sensor_id] = data
        return processed
    
    def _process_rgb_image(self, data):
        if hasattr(data, 'raw_data'):
            array = np.frombuffer(data.raw_data, dtype=np.uint8).reshape((data.height, data.width, 4))[:, :, :3]
        else:
            array = np.array(data)
            if len(array.shape) == 3 and array.shape[2] == 4:
                array = array[:, :, :3]
        tensor = torch.from_numpy(array).float() / 255.0
        return tensor.permute(2, 0, 1)
    
    def _process_semantic_image(self, data):
        if hasattr(data, 'raw_data'):
            array = np.frombuffer(data.raw_data, dtype=np.uint8).reshape((data.height, data.width, 4))
            array = array[:, :, 2]
        else:
            array = np.array(data)
        return torch.from_numpy(array).long()
    
    def _process_depth_image(self, data):
        if hasattr(data, 'raw_data'):
            array = np.frombuffer(data.raw_data, dtype=np.uint8).reshape((data.height, data.width, 4))
            normalized = (array[:, :, 2] + array[:, :, 1] * 256.0 + array[:, :, 0] * 256.0 * 256.0) / (256.0 * 256.0 * 256.0 - 1.0)
            depth = normalized * 1000.0
        else:
            depth = np.array(data)
        return torch.from_numpy(depth).float()
    
    def _process_lidar(self, data):
        if hasattr(data, 'raw_data'):
            points = np.frombuffer(data.raw_data, dtype=np.float32).reshape((-1, 4))
        else:
            points = np.array(data)
        return torch.from_numpy(points).float()
    
    def _process_imu(self, data):
        if hasattr(data, 'accelerometer'):
            imu_dict = {
                'accelerometer': np.array([data.accelerometer.x, data.accelerometer.y, data.accelerometer.z]),
                'gyroscope': np.array([data.gyroscope.x, data.gyroscope.y, data.gyroscope.z]),
                'compass': data.compass
            }
        else:
            imu_dict = data
        return imu_dict
    
    def _process_gps(self, data):
        if hasattr(data, 'latitude'):
            gps_dict = {'lat': data.latitude, 'lon': data.longitude, 'alt': getattr(data, 'altitude', 0.0)}
        else:
            gps_dict = data
        return gps_dict
    
    def _process_speed(self, data):
        if hasattr(data, 'speed'):
            speed = data.speed
        elif isinstance(data, dict):
            speed = float(data.get('speed', 0.0))
        else:
            try:
                speed = float(data)
            except Exception:
                speed = 0.0
        return speed
    
    def _model_inference(self, processed_data, timestamp):
        try:
            with torch.no_grad():
                if self.model_type == 'interfuser':
                    model_input = self._prepare_interfuser_input(processed_data)
                elif self.model_type == 'lav':
                    model_input = self._prepare_lav_input(processed_data)
                elif self.model_type == 'transfuser':
                    model_input = self._prepare_transfuser_input(processed_data)
                else:
                    model_input = self._prepare_generic_input(processed_data)
                output = self.model(model_input)
                control = self._output_to_control(output)
                return control
        except Exception as e:
            print(f"Model inference failed: {e}")
            return self._rule_based_control(processed_data, timestamp)
    
    def _prepare_generic_input(self, processed_data):
        images = []
        for _, v in processed_data.items():
            if isinstance(v, torch.Tensor) and v.dim() >= 2:
                if v.dim() == 2: v = v.unsqueeze(0)
                images.append(v)
        if images:
            return torch.stack(images).unsqueeze(0).to(self.device)
        else:
            return torch.zeros(1, 3, 256, 256).to(self.device)
    
    def _prepare_interfuser_input(self, processed_data):
        inputs = {}
        if 'rgb' in processed_data:
            rgb_tensor = processed_data['rgb'].unsqueeze(0).to(self.device)
            inputs['rgb'] = rgb_tensor
            _, _, h, w = rgb_tensor.shape
            ch, cw = 128, 128
            if h >= ch and w >= cw:
                sh = (h - ch)//2; sw = (w - cw)//2
                inputs['rgb_center'] = rgb_tensor[:, :, sh:sh+ch, sw:sw+cw]
            else:
                inputs['rgb_center'] = rgb_tensor
        if 'rgb_left' in processed_data:
            inputs['rgb_left'] = processed_data['rgb_left'].unsqueeze(0).to(self.device)
        if 'rgb_right' in processed_data:
            inputs['rgb_right'] = processed_data['rgb_right'].unsqueeze(0).to(self.device)
        if 'lidar' in processed_data:
            lidar = processed_data['lidar']
            if lidar.dim() == 2 and lidar.shape[1] == 4:
                try:
                    from team_code.utils import lidar_to_histogram_features
                    lidar_np = lidar.cpu().numpy()
                    lidar_hist = lidar_to_histogram_features(lidar_np[:, :3], crop=224)
                    inputs['lidar'] = torch.from_numpy(lidar_hist).float().to(self.device).unsqueeze(0)
                except ImportError:
                    print("Warning: Could not import lidar_to_histogram_features, using dummy lidar")
                    inputs['lidar'] = torch.zeros(1, 224, 224).to(self.device).unsqueeze(0)
            else:
                inputs['lidar'] = lidar.unsqueeze(0).to(self.device)
        else:
            inputs['lidar'] = torch.zeros(1, 224, 224).to(self.device).unsqueeze(0)
        measurements = torch.zeros(1, 7).to(self.device)
        if 'speed' in processed_data:
            measurements[0, 6] = processed_data['speed'] / 40.0
        if hasattr(self, '_command') and self._command is not None:
            cmd_idx = self._command - 1
            if 0 <= cmd_idx < 6:
                measurements[0, cmd_idx] = 1.0
        else:
            measurements[0, 3] = 1.0
        inputs['measurements'] = measurements
        inputs['target_point'] = torch.zeros(1, 2).to(self.device)
        return inputs
    
    def _prepare_lav_input(self, processed_data):
        inputs = {}
        if 'bev' in processed_data:
            inputs['bev'] = processed_data['bev'].unsqueeze(0).to(self.device)
        if 'semantic_bev' in processed_data:
            inputs['semantic_bev'] = processed_data['semantic_bev'].unsqueeze(0).to(self.device)
        for key in ['RGB_1','rgb_front','rgb']:
            if key in processed_data:
                inputs['rgb_front'] = processed_data[key].unsqueeze(0).to(self.device); break
        for key in ['RGB_0','rgb_left_side','rgb_left']:
            if key in processed_data:
                inputs['rgb_left_side'] = processed_data[key].unsqueeze(0).to(self.device); break
        for key in ['RGB_2','rgb_right_side','rgb_right']:
            if key in processed_data:
                inputs['rgb_right_side'] = processed_data[key].unsqueeze(0).to(self.device); break
        if 'rgb_rear' in processed_data:
            inputs['rgb_rear'] = processed_data['rgb_rear'].unsqueeze(0).to(self.device)
        for key in ['LIDAR','lidar']:
            if key in processed_data:
                inputs['lidar'] = processed_data[key].unsqueeze(0).to(self.device); break
        measurements = torch.zeros(1, 10).to(self.device)
        for key in ['speed','EGO']:
            if key in processed_data:
                measurements[0,0] = processed_data[key] / 40.0; break
        for key in ['gps','GPS']:
            if key in processed_data and isinstance(processed_data[key], dict):
                gps = processed_data[key]; measurements[0,1] = gps.get('lat', 0.0); measurements[0,2] = gps.get('lon', 0.0); break
        inputs['measurements'] = measurements
        return inputs

    def _output_to_control(self, output):
        control = carla.VehicleControl()
        if isinstance(output, (list, tuple)) and len(output) == 6 and torch.is_tensor(output[1]):
            _, pred_waypoints, *_ = output
            wps = pred_waypoints.detach().cpu().numpy()
            control = self._waypoints_to_control(wps, control)
            cur = getattr(self, "last_speed", 0.0)
            tgt = getattr(self, "control_config", {}).get("target_speed", 30.0)
            if cur < tgt: 
                control.throttle, control.brake = 0.7, 0.0
            else:
                control.throttle, control.brake = 0.0, 0.3
            return control
        if isinstance(output, dict):
            if 'control' in output:
                ctrl = output['control']
                control.steer = float(ctrl.get('steer', 0.0)); control.throttle = float(ctrl.get('throttle', 0.0)); control.brake = float(ctrl.get('brake', 0.0))
            elif 'steer' in output:
                control.steer = float(output.get('steer', 0.0)); control.throttle = float(output.get('throttle', 0.0)); control.brake = float(output.get('brake', 0.0))
            elif 'action' in output:
                action = output['action']; 
                if isinstance(action, torch.Tensor): action = action.cpu().numpy().flatten()
                control.steer = float(action[0]) if len(action)>0 else 0.0
                control.throttle = float(action[1]) if len(action)>1 else 0.0
                control.brake = float(action[2]) if len(action)>2 else 0.0
            elif 'waypoints' in output:
                waypoints = output['waypoints']
                if isinstance(waypoints, torch.Tensor): waypoints = waypoints.cpu().numpy()
                control = self._waypoints_to_control(waypoints, control)
                if 'speed' in output:
                    target_speed = float(output['speed'])
                    current_speed = self.last_speed if hasattr(self, 'last_speed') else 0.0
                    if current_speed < target_speed: control.throttle, control.brake = 0.7, 0.0
                    else: control.throttle, control.brake = 0.0, 0.3
            else:
                for key in ['pred_control','controls','output']:
                    if key in output: return self._output_to_control(output[key])
                control.steer = 0.0; control.throttle = 0.3; control.brake = 0.0
        elif isinstance(output, (list, tuple)):
            control.steer = float(output[0]) if len(output)>0 else 0.0
            control.throttle = float(output[1]) if len(output)>1 else 0.0
            control.brake = float(output[2]) if len(output)>2 else 0.0
        elif isinstance(output, torch.Tensor):
            arr = output.cpu().numpy().flatten()
            control.steer = float(arr[0]) if len(arr)>0 else 0.0
            control.throttle = float(arr[1]) if len(arr)>1 else 0.0
            control.brake = float(arr[2]) if len(arr)>2 else 0.0
        control.steer = float(np.clip(control.steer, -1.0, 1.0))
        control.throttle = float(np.clip(control.throttle, 0.0, 1.0))
        control.brake = float(np.clip(control.brake, 0.0, 1.0))
        control.hand_brake = False; control.manual_gear_shift = False
        return control
    
    def _waypoints_to_control(self, waypoints, control):
        if len(waypoints.shape) == 3:
            waypoints = waypoints[0]
        if len(waypoints) < 2:
            return control
        dx = waypoints[1,0] - waypoints[0,0]
        dy = waypoints[1,1] - waypoints[0,1]
        angle = np.arctan2(dy, dx)
        control.steer = np.clip(angle / 0.7, -1.0, 1.0)
        return control
    
    def _rule_based_control(self, processed_data, timestamp):
        control = carla.VehicleControl()
        current_speed = processed_data.get('speed', 0.0)
        target_speed = self.control_config['target_speed']
        if current_speed < target_speed: control.throttle, control.brake = 0.7, 0.0
        else: control.throttle, control.brake = 0.0, 0.3
        control.steer = self._compute_steer_from_waypoints()
        control.hand_brake = False; control.manual_gear_shift = False
        return control
    
    def _compute_steer_from_waypoints(self):
        if not hasattr(self, '_global_plan_world_coord') or not self._global_plan_world_coord:
            return 0.0
        try:
            ego = CarlaDataProvider.get_hero_actor()
            if ego is None: return 0.0
            transform = ego.get_transform()
            location = transform.location; forward_vec = transform.get_forward_vector(); right_vec = transform.get_right_vector()
            min_dist = float('inf'); target = None
            for wp in self._global_plan_world_coord:
                if isinstance(wp, tuple) and len(wp) >= 2:
                    wp_loc = wp[0]
                    if hasattr(wp_loc, 'location'):
                        wx, wy = wp_loc.location.x, wp_loc.location.y
                    else:
                        wx, wy = wp_loc[0], wp_loc[1]
                    dist = np.sqrt((wx - location.x)**2 + (wy - location.y)**2)
                    if 2.0 < dist < min_dist:
                        min_dist = dist; target = (wx, wy)
            if not target: return 0.0
            dx = target[0] - location.x; dy = target[1] - location.y
            dot_forward = dx * forward_vec.x + dy * forward_vec.y
            dot_right = dx * right_vec.x + dy * right_vec.y
            angle = np.arctan2(dot_right, dot_forward)
            steer = np.clip(angle / 0.7, -1.0, 1.0)
            return float(steer)
        except Exception as e:
            print(f"Error computing steering: {e}")
            return 0.0
    
    def _postprocess_control(self, control):
        if hasattr(control, 'steer'):
            damping = self.control_config.get('steer_damping', 0.3)
            control.steer = (1 - damping) * control.steer + damping * self.prev_steer
            self.prev_steer = control.steer
        if hasattr(control, 'brake') and hasattr(control, 'throttle'):
            if control.brake > self.control_config.get('brake_threshold', 0.5):
                control.throttle = 0.0
        return control
    
    def _get_local_waypoints(self, num_waypoints=10):
        if not hasattr(self, '_global_plan_world_coord'):
            return torch.zeros(1, num_waypoints, 2)
        try:
            ego = CarlaDataProvider.get_hero_actor()
            if ego is None:
                return torch.zeros(1, num_waypoints, 2)
            transform = ego.get_transform()
            waypoints = []
            for wp in self._global_plan_world_coord[:num_waypoints]:
                if isinstance(wp, tuple) and len(wp) >= 2:
                    if hasattr(wp[0], 'location'):
                        wp_loc = wp[0].location
                        local_x = wp_loc.x - transform.location.x
                        local_y = wp_loc.y - transform.location.y
                    else:
                        local_x = wp[0] - transform.location.x
                        local_y = wp[1] - transform.location.y
                    waypoints.append([local_x, local_y])
            while len(waypoints) < num_waypoints:
                waypoints.append([0.0, 0.0])
            return torch.tensor(waypoints[:num_waypoints]).unsqueeze(0).float()
        except Exception as e:
            print(f"Error getting local waypoints: {e}")
            return torch.zeros(1, num_waypoints, 2)
    
    def _initialize_data_collection(self):
        """
        Initialize data collection infrastructure, saving to:
        {DATASET_DIR}/{agent}/weather_{idx}/map_{NN}/{route}
        """
        base_dir = os.environ.get('DATASET_DIR') or os.path.join(os.environ.get('WORKSPACE_DIR', '/workspace'), 'dataset')

        agent_name = os.environ.get('AGENT_NAME') or (Path(getattr(self, '_config_path','')).stem or 'unknown_agent')
        weather_idx = os.environ.get('WEATHER_INDEX', '0')
        routes_file = os.environ.get('ROUTES_FILE', '')
        route_name = os.environ.get('ROUTE_NAME') or (Path(routes_file).stem if routes_file else 'route_unknown')

        town_num = os.environ.get('TOWN_NUM', '')
        if not town_num:
            import re
            m = re.search(r'town(\d+)', routes_file, flags=re.IGNORECASE)
            town_num = m.group(1) if m else 'unknown'

        # Labels as requested
        try:
            weather_label = f"weather_{int(weather_idx)}"
        except Exception:
            weather_label = f"weather_{str(weather_idx)}"
        try:
            map_label = f"map_{int(town_num):02d}"
        except Exception:
            map_label = f"map_{town_num}"

        self.save_path = os.path.join(str(base_dir), str(agent_name), weather_label, map_label, str(route_name))
        self.save_path = os.path.expandvars(os.environ.get('SAVE_PATH', self.save_path))

        self.frame_counter = 0
        self.sensor_data_paths = {}
        os.makedirs(self.save_path, exist_ok=True)

        self.metadata = {
            'model_type': getattr(self, 'model_type', 'generic'),
            'model_path': getattr(self, 'model_path', ''),
            'config': getattr(self, 'config', {}),
            'save_path': self.save_path,
            'timestamp': self._get_timestamp(),
            'frames': []
        }

        # ---- NEW: record applied weather in metadata, if any ----
        if hasattr(self, '_weather_applied'):
            self.metadata['weather'] = dict(self._weather_applied)
        else:
            # still store intent if provided
            self.metadata['weather'] = {
                'index': os.environ.get('WEATHER_INDEX', ''),
                'preset': os.environ.get('WEATHER_PRESET', '')
            }

        print("ConsolidatedAgent: Data collection initialized")
        print(f"  Save path: {self.save_path}")

    def _get_timestamp(self):
        from datetime import datetime
        return datetime.now().isoformat()
    
    def _get_sensor_folder_name(self, sensor_type, sensor_id):
        if 'camera.rgb' in sensor_type:
            return sensor_id
        elif 'camera.semantic' in sensor_type:
            return f"semantic_{sensor_id}"
        elif 'camera.depth' in sensor_type:
            return f"depth_{sensor_id}"
        elif 'lidar' in sensor_type:
            return "lidar"
        elif 'radar' in sensor_type:
            return f"radar_{sensor_id}"
        elif 'imu' in sensor_type:
            return "imu"
        elif 'gnss' in sensor_type:
            return "gps"
        elif 'speedometer' in sensor_type:
            return "speed"
        else:
            return sensor_id.replace(' ', '_')
    
    def _save_sensor_data(self, input_data, timestamp):
        frame_data = {'frame': self.frame_counter, 'timestamp': timestamp, 'sensors': {}}
        for sensor_id, sensor_data in input_data.items():
            sensor_path = self.sensor_data_paths.get(sensor_id)
            if sensor_path is None:
                continue
            try:
                filename = self._process_and_save_sensor(sensor_id, sensor_data, sensor_path)
                if filename:
                    frame_data['sensors'][sensor_id] = filename
            except Exception as e:
                print(f"Warning: Failed to save data for sensor {sensor_id}: {e}")
        self.metadata['frames'].append(frame_data)
        if self.frame_counter % 50 == 0:
            self._save_metadata()
        self.frame_counter += 1
    
    def _process_and_save_sensor(self, sensor_id, sensor_data, sensor_path):
        data = sensor_data[1] if (isinstance(sensor_data, tuple) and len(sensor_data) == 2) else sensor_data
        sid = sensor_id.lower()
        if hasattr(data, 'raw_data'):
            return self._save_raw_sensor_data(sid, data, sensor_path)
        elif isinstance(data, np.ndarray):
            return self._save_numpy_data(sid, data, sensor_path)
        elif isinstance(data, dict):
            return self._save_dict_data(data, sensor_path)
        elif hasattr(data, '__dict__'):
            return self._save_object_data(sid, data, sensor_path)
        else:
            try:
                filename = f"{self.frame_counter:04d}.json"
                with open(os.path.join(sensor_path, filename), 'w') as f:
                    json.dump({'value': str(data)}, f, indent=2)
                return filename
            except:
                return None
    
    def _save_raw_sensor_data(self, sid, data, sensor_path):
        if 'rgb' in sid or 'bev' in sid:
            arr = np.frombuffer(data.raw_data, dtype=np.uint8).reshape((data.height, data.width, 4))[:, :, :3]
            Image.fromarray(arr).save(os.path.join(sensor_path, f"{self.frame_counter:04d}.png"))
            return f"{self.frame_counter:04d}.png"
        elif 'semantic' in sid:
            arr = np.frombuffer(data.raw_data, dtype=np.uint8).reshape((data.height, data.width, 4))
            img = Image.fromarray(arr[:, :, 2], mode='L')
            img.save(os.path.join(sensor_path, f"{self.frame_counter:04d}.png"))
            return f"{self.frame_counter:04d}.png"
        elif 'depth' in sid:
            arr = np.frombuffer(data.raw_data, dtype=np.uint8).reshape((data.height, data.width, 4))
            normalized = (arr[:, :, 2] + arr[:, :, 1]*256.0 + arr[:, :, 0]*(256.0**2)) / ((256.0**3) - 1.0)
            depth_m = normalized * 1000.0
            np.save(os.path.join(sensor_path, f"{self.frame_counter:04d}.npy"), depth_m, allow_pickle=True)
            return f"{self.frame_counter:04d}.npy"
        elif 'lidar' in sid:
            points = np.frombuffer(data.raw_data, dtype=np.float32).reshape((-1, 4))
            np.save(os.path.join(sensor_path, f"{self.frame_counter:04d}.npy"), points, allow_pickle=True)
            return f"{self.frame_counter:04d}.npy"
        return None
    
    def _save_numpy_data(self, sid, data, sensor_path):
        if len(data.shape) == 3 and data.shape[2] in [3, 4]:
            if data.shape[2] == 4: data = data[:, :, :3]
            Image.fromarray(data).save(os.path.join(sensor_path, f"{self.frame_counter:04d}.png"))
            return f"{self.frame_counter:04d}.png"
        filename = f"{self.frame_counter:04d}.npy"
        np.save(os.path.join(sensor_path, filename), data, allow_pickle=True)
        return filename
    
    def _save_dict_data(self, data, sensor_path):
        filename = f"{self.frame_counter:04d}.json"
        with open(os.path.join(sensor_path, filename), 'w') as f:
            json.dump(data, f, indent=2)
        return filename
    
    def _save_object_data(self, sid, data, sensor_path):
        extracted = {}
        if 'gnss' in sid or 'gps' in sid:
            if hasattr(data, 'latitude'):
                extracted = {'lat': data.latitude, 'lon': data.longitude, 'alt': getattr(data, 'altitude', 0.0)}
        elif 'imu' in sid:
            if hasattr(data, 'accelerometer'):
                extracted = {
                    'accelerometer': [data.accelerometer.x, data.accelerometer.y, data.accelerometer.z],
                    'gyroscope': [data.gyroscope.x, data.gyroscope.y, data.gyroscope.z],
                    'compass': data.compass
                }
        elif 'speed' in sid:
            extracted = {'speed': getattr(data, 'speed', float(data))}
        if extracted:
            filename = f"{self.frame_counter:04d}.json"
            with open(os.path.join(sensor_path, filename), 'w') as f:
                json.dump(extracted, f, indent=2)
            return filename
        return None
    
    def _save_metadata(self):
        try:
            metadata_path = os.path.join(self.save_path, 'metadata.json')
            with open(metadata_path, 'w') as f:
                meta = dict(self.metadata)
                if 'frames' in meta and len(meta['frames']) > 100:
                    meta['frames'] = meta['frames'][-100:]
                meta["applied_weather"] = getattr(self, "_weather_applied", {})
                json.dump(meta, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save metadata: {e}")
    
    def destroy(self):
        try:
            metadata_path = os.path.join(self.save_path, 'metadata.json')
            self.metadata['summary'] = {
                'total_frames': self.frame_counter,
                'sensors_used': list(self.sensor_data_paths.keys()),
                'completion_time': self._get_timestamp()
            }
            with open(metadata_path, 'w') as f:
                json.dump(self.metadata, f, indent=2)
            print(f"ConsolidatedAgent: Data collection complete")
            print(f"  Total frames: {self.frame_counter}")
            print(f"  Save location: {self.save_path}")
        except Exception as e:
            print(f"Warning: Failed to save final metadata: {e}")
        if hasattr(self, 'model'):
            del self.model
