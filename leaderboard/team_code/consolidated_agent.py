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
<<<<<<< HEAD
<<<<<<< HEAD
=======
import carla
>>>>>>> 5c6ba0f (trying to set up rsync)
=======
import carla
>>>>>>> 8a07600 (fixing issues w/ model loading)

def get_entry_point():
    """Required by CARLA Leaderboard."""
    return "ConsolidatedAgent"

class ConsolidatedAgent(AutonomousAgent):
    """
    Universal agent that handles any CARLA Leaderboard 1.0 compatible agent.
    Supports multiple configuration formats, model architectures, and loading methods.
    """
    
    # Default sensor configuration
    DEFAULT_SENSORS = [
        # RGB cameras
        {'type': 'sensor.camera.rgb', 'x': 1.3, 'y': 0.0, 'z': 2.3, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
         'width': 900, 'height': 256, 'fov': 100, 'id': 'rgb_front'},
        {'type': 'sensor.camera.rgb', 'x': 0.0, 'y': 0.0, 'z': 50.0, 'roll': 0.0, 'pitch': -90.0, 'yaw': 0.0,
         'width': 512, 'height': 512, 'fov': 110, 'id': 'bev'},
        
        # Semantic segmentation cameras
        {'type': 'sensor.camera.semantic_segmentation', 'x': 1.3, 'y': 0.0, 'z': 2.3, 
         'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0, 'width': 900, 'height': 256, 'fov': 100, 'id': 'semantic_front'},
        
        # Depth camera
        {'type': 'sensor.camera.depth', 'x': 1.3, 'y': 0.0, 'z': 2.3, 
         'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0, 'width': 900, 'height': 256, 'fov': 100, 'id': 'depth_front'},
        
        # LiDAR
        {'type': 'sensor.lidar.ray_cast', 'x': 0.0, 'y': 0.0, 'z': 2.5, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
         'id': 'lidar', 'channels': 64, 'range': 100, 'points_per_second': 1000000, 
         'rotation_frequency': 20, 'upper_fov': 10, 'lower_fov': -30},
        
        # IMU
        {'type': 'sensor.other.imu', 'x': 0.0, 'y': 0.0, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0, 'id': 'imu'},
        
        # GPS
        {'type': 'sensor.other.gnss', 'x': 0.0, 'y': 0.0, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0, 'id': 'gps'},
        
        # Speedometer
        {'type': 'sensor.speedometer', 'id': 'speed'}
    ]
    
    def setup(self, path_to_config_yaml):
        """Load configuration and model."""
        print(f"ConsolidatedAgent: Loading configuration from {path_to_config_yaml}")
        
        with open(path_to_config_yaml, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Check if we should load an external config file (can be .py, .yaml, .json, etc.)
        if 'external_config' in self.config:
            self._load_external_config()
        
        # Extract configuration
        self.model_path = self.config.get('model_path', 'none')
        self.model_type = self.config.get('model_type', 'generic')
        self.model_config = self.config.get('model_config', {})
        
        # Check if we should extract model from an existing agent
        if 'agent_config' in self.config:
            self._extract_from_agent()
        
        # Check if model components are specified directly in model_config
        if 'lidar_model_dir' in self.model_config or 'uniplanner_dir' in self.model_config:
            # Convert LAV-style config to our format
            self._convert_lav_config()
        
        # Sensor configuration - use provided or default
        self.sensor_config = self.config.get('sensors', self.DEFAULT_SENSORS)
        if self.sensor_config == 'default':
            self.sensor_config = self.DEFAULT_SENSORS
        
        # Control parameters
        self.control_config = self.config.get('control', {
            'target_speed': 30.0,  # km/h
            'brake_threshold': 0.5,
            'steer_damping': 0.3
        })
        
        print(f"ConsolidatedAgent: Configuration loaded:")
        print(f"  Model path: {self.model_path}")
        print(f"  Model type: {self.model_type}")
        if 'model_components' in self.config:
            print(f"  Model components: {len(self.config['model_components'])} components")
        print(f"  Sensors: {len(self.sensor_config)} configured")
        
        # Load the model
        self._load_model()
        
        # Initialize data collection
        self._initialize_data_collection()
        
        # Initialize processing buffers
        self.input_buffer = {}
        self.waypoint_buffer = deque(maxlen=50)
        self.traffic_light_buffer = deque(maxlen=10)
        self.stop_sign_buffer = deque(maxlen=10)
        
        # Control state
        self.prev_steer = 0.0
        self.prev_brake = 0.0
        self.frame_count = 0
        self.last_speed = 0.0
        
        print(f"ConsolidatedAgent: Setup complete")
    
    def _load_external_config(self):
        """Load external configuration file (supports .py, .yaml, .json)."""
        external_config_path = self.config['external_config']
        print(f"ConsolidatedAgent: Loading external config from {external_config_path}")
        
        external_config_path = Path(external_config_path)
        
        if not external_config_path.exists():
            print(f"Warning: External config not found: {external_config_path}")
            return
        
        # Determine config type by extension
        ext = external_config_path.suffix.lower()
        
        if ext == '.py':
            # Python config file (like InterFuser)
            self._load_python_config(external_config_path)
        elif ext in ['.yaml', '.yml']:
            # YAML config file
            with open(external_config_path, 'r') as f:
                external_config = yaml.safe_load(f)
            self._merge_configs(self.config, external_config)
        elif ext == '.json':
            # JSON config file
            with open(external_config_path, 'r') as f:
                external_config = json.load(f)
            self._merge_configs(self.config, external_config)
        else:
            print(f"Warning: Unknown config format: {ext}")
    
    def _load_python_config(self, config_path):
        """Load configuration from a Python file."""
        print(f"ConsolidatedAgent: Loading Python config from {config_path}")
        
        # Add directory to path
        sys.path.insert(0, str(config_path.parent))
        
        # Import the config module
        module_name = config_path.stem
        try:
            config_module = importlib.import_module(module_name)
            
            # Extract configuration - try different common patterns
            external_config = {}
            
            # Pattern 1: GlobalConfig class or similar
            for attr_name in dir(config_module):
                if 'config' in attr_name.lower() and not attr_name.startswith('_'):
                    attr = getattr(config_module, attr_name)
                    if isinstance(attr, type):
                        # It's a class, instantiate it or extract class variables
                        config_obj = attr()
                        for key in dir(config_obj):
                            if not key.startswith('_'):
                                value = getattr(config_obj, key)
                                if not callable(value):
                                    external_config[key] = value
                    elif isinstance(attr, dict):
                        # Direct dictionary
                        external_config.update(attr)
                    elif hasattr(attr, '__dict__'):
                        # Object with attributes
                        for key, value in attr.__dict__.items():
                            if not key.startswith('_'):
                                external_config[key] = value
            
            # Pattern 2: Direct module attributes
            for key in dir(config_module):
                if not key.startswith('_') and key.isupper():
                    # Uppercase attributes are often config values
                    value = getattr(config_module, key)
                    if not callable(value):
                        external_config[key] = value
            
            # Pattern 3: get_config() or similar function
            if hasattr(config_module, 'get_config'):
                external_config.update(config_module.get_config())
            elif hasattr(config_module, 'make_config'):
                external_config.update(config_module.make_config())
            
            # Special handling for InterFuser-style config
            if hasattr(config_module, 'GlobalConfig'):
                config_class = config_module.GlobalConfig
                # Extract model paths and parameters
                if hasattr(config_class, 'model_path'):
                    self.config['model_path'] = config_class.model_path
                if hasattr(config_class, 'record_frame_rate'):
                    self.config['frame_rate'] = config_class.record_frame_rate
                    
                # Extract all attributes
                for attr in dir(config_class):
                    if not attr.startswith('_'):
                        value = getattr(config_class, attr)
                        if not callable(value):
                            external_config[attr] = value
            
            # Merge the extracted config
            if 'model_config' not in self.config:
                self.config['model_config'] = {}
            self.config['model_config'].update(external_config)
            
            print(f"  Extracted {len(external_config)} configuration values from Python config")
            
        except Exception as e:
            print(f"Error loading Python config: {e}")
            import traceback
            traceback.print_exc()
    
    def _extract_from_agent(self):
        """Extract model and configuration from an existing agent implementation."""
        agent_config = self.config['agent_config']
        print(f"ConsolidatedAgent: Extracting from agent using config: {agent_config}")
        
        agent_file = agent_config.get('agent_file')
        agent_class_name = agent_config.get('agent_class')
        config_path = agent_config.get('config_path')
        
        if not agent_file or not agent_class_name:
            print("Warning: agent_file and agent_class required for agent extraction")
            return
        
        # Add agent directory to path
        agent_path = Path(agent_file)
        sys.path.insert(0, str(agent_path.parent))
        
        try:
            # Import agent module
            module_name = agent_path.stem
            agent_module = importlib.import_module(module_name)
            
            # Get agent class
            agent_class = getattr(agent_module, agent_class_name)
            
            # Instantiate agent
            if config_path:
                agent = agent_class(config_path)
                if hasattr(agent, 'setup'):
                    agent.setup(config_path)
            else:
                agent = agent_class()
            
            # Extract model(s)
            self._extract_models_from_agent(agent)
            
            # Extract configuration
            if hasattr(agent, 'config'):
                if 'model_config' not in self.config:
                    self.config['model_config'] = {}
                    
                if hasattr(agent.config, '__dict__'):
                    self.config['model_config'].update(agent.config.__dict__)
                elif isinstance(agent.config, dict):
                    self.config['model_config'].update(agent.config)
            
            print(f"  Successfully extracted model and config from {agent_class_name}")
            
        except Exception as e:
            print(f"Error extracting from agent: {e}")
            import traceback
            traceback.print_exc()
    
    def _extract_models_from_agent(self, agent):
        """Extract model(s) from an instantiated agent."""
        # Common model attribute names
        model_attrs = ['model', 'net', 'network', 'backbone', 'policy', 
                      'actor', 'planner', 'controller']
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 8a07600 (fixing issues w/ model loading)
        
        # Check for single model
        for attr in model_attrs:
            if hasattr(agent, attr):
                model = getattr(agent, attr)
                if isinstance(model, torch.nn.Module):
                    self.model = model
                    print(f"  Extracted model from agent.{attr}")
                    return
        
        # Check for multiple models (like LAV)
        multi_model_attrs = {
            'lidar_model': ['lidar_model', 'lidar_net', 'lidar_encoder'],
            'bev_model': ['bev_model', 'bev_net', 'bev_encoder'],
            'seg_model': ['seg_model', 'segmentation', 'seg_net'],
            'uniplanner': ['uniplanner', 'planner', 'planning_model'],
            'controller': ['controller', 'control_model', 'control_net'],
        }
        
        self.model_components = {}
        for component_name, possible_attrs in multi_model_attrs.items():
            for attr in possible_attrs:
                if hasattr(agent, attr):
                    model = getattr(agent, attr)
                    if isinstance(model, torch.nn.Module):
                        self.model_components[component_name] = model
                        print(f"  Extracted {component_name} from agent.{attr}")
                        break
        
        if self.model_components:
            # Create wrapper based on agent type
            if 'lav' in self.model_type.lower():
                self._create_lav_wrapper()
            else:
                self.model = self.model_components
    
    def _merge_configs(self, base_config, external_config):
        """Merge external config into base config."""
        for key, value in external_config.items():
            if key not in base_config:
                base_config[key] = value
            elif isinstance(value, dict) and isinstance(base_config[key], dict):
                # Recursively merge dictionaries
                self._merge_configs(base_config[key], value)
            # else: keep base_config value (base takes precedence for existing keys)
    
    def _convert_lav_config(self):
        """Convert LAV-style config with individual model paths to our format."""
        if 'model_components' not in self.config:
            self.config['model_components'] = {}
        
        # Map LAV config keys to our component names
        lav_component_mapping = {
            'lidar_model_dir': ('lidar_model', 'checkpoint'),
            'uniplanner_dir': ('uniplanner', 'checkpoint'),
            'bra_model_dir': ('bra_model', 'checkpoint'),
            'bra_model_trace_dir': ('bra_model_trace', 'trace'),
            'seg_model_dir': ('seg_model', 'checkpoint'),
            'seg_model_trace_dir': ('seg_model_trace', 'trace'),
            'bev_model_dir': ('bev_model', 'checkpoint'),
        }
        
        for config_key, (component_name, component_type) in lav_component_mapping.items():
            if config_key in self.model_config:
                path = self.model_config[config_key]
                if path and os.path.exists(path):
                    self.config['model_components'][component_name] = {
                        'path': path,
                        'type': component_type
                    }
        
        # Set model type to LAV if not already set
        if self.model_type == 'generic':
            self.model_type = 'lav'
        
        print(f"ConsolidatedAgent: Converted LAV config - found {len(self.config['model_components'])} components")
    
    def _load_model(self):
        """Load the neural network model(s) based on type."""
        # Check if we have multiple model components
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
            
            # Set model to eval mode
            if hasattr(self, 'model') and hasattr(self.model, 'eval'):
                self.model.eval()
                print(f"ConsolidatedAgent: Model loaded and set to eval mode")
        else:
            print("ConsolidatedAgent: No model specified, using rule-based control")
            self.model = None
    
    def _load_multi_component_model(self):
        """Load multiple model components for complex architectures."""
        print("ConsolidatedAgent: Loading multi-component model architecture")
        
        self.model_components = {}
        components_config = self.config.get('model_components', {})
        
        # Load each component
        for component_name, component_info in components_config.items():
            component_path = component_info.get('path')
            component_type = component_info.get('type', 'checkpoint')
            
            if not component_path or not os.path.exists(component_path):
                print(f"Warning: Component {component_name} path not found: {component_path}")
                continue
            
            print(f"  Loading {component_name} from {component_path}")
            
            try:
                if component_type == 'trace':
                    # Load JIT traced model
                    self.model_components[component_name] = torch.jit.load(component_path)
                elif component_type == 'checkpoint':
                    # Load regular checkpoint
                    checkpoint = torch.load(component_path, map_location='cpu')
                    
                    # Handle different checkpoint formats
                    if isinstance(checkpoint, torch.nn.Module):
                        self.model_components[component_name] = checkpoint
                    elif 'model' in checkpoint:
                        self.model_components[component_name] = checkpoint['model']
                    elif 'state_dict' in checkpoint:
                        # Need to instantiate model first - use component config
                        model_class_info = component_info.get('model_class')
                        if model_class_info:
                            module = importlib.import_module(model_class_info['module'])
                            model_class = getattr(module, model_class_info['name'])
                            model_args = model_class_info.get('args', {})
                            model = model_class(**model_args)
                            model.load_state_dict(checkpoint['state_dict'])
                            self.model_components[component_name] = model
                        else:
                            self.model_components[component_name] = checkpoint
                    else:
                        self.model_components[component_name] = checkpoint
                        
                # Move to device and set to eval
                if hasattr(self.model_components[component_name], 'eval'):
                    self.model_components[component_name].eval()
                    if hasattr(self.model_components[component_name], 'to'):
                        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                        self.model_components[component_name] = self.model_components[component_name].to(device)
                        
                print(f"    ✓ {component_name} loaded successfully")
                
            except Exception as e:
                print(f"    ✗ Failed to load {component_name}: {e}")
                
        # Create wrapper model for compatibility
        if self.model_type == 'lav':
            self._create_lav_wrapper()
        else:
            # Generic wrapper
            self.model = self.model_components
        
        print(f"ConsolidatedAgent: Loaded {len(self.model_components)} model components")
    
    def _load_generic_model(self):
        """Load a generic PyTorch model."""
        try:
            # Try to load as PyTorch checkpoint
            checkpoint = torch.load(self.model_path, map_location='cpu')
            
            # Try to instantiate model from config
            if 'model_class' in self.model_config:
                module_path = self.model_config['model_module']
                class_name = self.model_config['model_class']
                
                module = importlib.import_module(module_path)
                model_class = getattr(module, class_name)
                
                # Instantiate model with config
                model_args = self.model_config.get('model_args', {})
                self.model = model_class(**model_args)
                
                # Load weights
                if 'state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
            else:
                # Direct model object in checkpoint
                self.model = checkpoint['model'] if 'model' in checkpoint else checkpoint
            
            # Move to GPU if available
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = self.model.to(self.device)
            
        except Exception as e:
            print(f"Warning: Could not load model as PyTorch: {e}")
            print("Will use simple rule-based control as fallback")
            self.model = None
    
    def _load_interfuser_model(self):
        """Load InterFuser-specific model."""
        try:
<<<<<<< HEAD
            # InterFuser uses specific model structure
            from interfuser.timm.models import create_model
            
            # Create model architecture
            self.model = create_model(
                self.model_config.get('architecture', 'interfuser_baseline'),
                pretrained=False,
                **self.model_config.get('model_args', {})
            )
            
            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location='cpu')
            self.model.load_state_dict(checkpoint['state_dict'])
            
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = self.model.to(self.device)
            
        except Exception as e:
            print(f"Error loading InterFuser model: {e}")
            self._load_generic_model()
    
    def _load_lav_model(self):
        """Load LAV-specific model."""
        # Check if using multi-component architecture (typical for LAV)
        if 'model_components' in self.config:
            self._load_multi_component_model()
            return
            
        # Single model fallback
        try:
            checkpoint = torch.load(self.model_path, map_location='cpu')
            
            # Extract model from checkpoint
            if 'model' in checkpoint:
                self.model = checkpoint['model']
            elif 'state_dict' in checkpoint:
                # Need to instantiate model first
                from lav_model import LAV  # Assuming LAV model class exists
                self.model = LAV(**self.model_config.get('model_args', {}))
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                self.model = checkpoint
=======
            # Use the correct import path
            from team_code.interfuser.interfuser.timm.models.interfuser import interfuser_baseline
            
            # Create model directly
            self.model = interfuser_baseline()
            
            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location='cpu')
            
            # InterFuser saves the model under 'state_dict' key
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                # Try loading directly if it's just the state dict
                self.model.load_state_dict(checkpoint)
>>>>>>> 8a07600 (fixing issues w/ model loading)
            
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = self.model.to(self.device)
            
<<<<<<< HEAD
        except Exception as e:
            print(f"Error loading LAV model: {e}")
            self._load_generic_model()
    
    def _create_lav_wrapper(self):
        """Create a wrapper for LAV's multi-component architecture."""
        class LAVModelWrapper:
            def __init__(self, components, device):
                self.components = components
                self.device = device
                
                # Extract individual models
                self.lidar_model = components.get('lidar_model')
                self.uniplanner = components.get('uniplanner')
                self.bra_model = components.get('bra_model')
                self.seg_model = components.get('seg_model')
                self.bev_model = components.get('bev_model')
                
            def __call__(self, inputs):
                """Forward pass through LAV pipeline."""
                # This is a simplified version - actual LAV may have more complex pipeline
                outputs = {}
                
                # Process BEV if available
                if self.bev_model and 'bev' in inputs:
                    bev_features = self.bev_model(inputs['bev'])
                    outputs['bev_features'] = bev_features
                
                # Process LiDAR if available
                if self.lidar_model and 'lidar' in inputs:
                    lidar_features = self.lidar_model(inputs['lidar'])
                    outputs['lidar_features'] = lidar_features
                
                # Segmentation
                if self.seg_model and 'rgb_front' in inputs:
                    seg_output = self.seg_model(inputs['rgb_front'])
                    outputs['segmentation'] = seg_output
                
                # Planning with uniplanner
                if self.uniplanner:
                    # Combine features for planning
                    planner_input = {}
                    if 'bev_features' in outputs:
                        planner_input['bev'] = outputs['bev_features']
                    if 'lidar_features' in outputs:
                        planner_input['lidar'] = outputs['lidar_features']
                    if 'measurements' in inputs:
                        planner_input['measurements'] = inputs['measurements']
                    
                    control_output = self.uniplanner(planner_input)
                    outputs.update(control_output)
                
                # Behavior prediction/refinement
                if self.bra_model and 'bev_features' in outputs:
                    bra_output = self.bra_model(outputs['bev_features'])
                    outputs['behavior'] = bra_output
                
                return outputs
            
            def eval(self):
                """Set all components to eval mode."""
                for component in self.components.values():
                    if hasattr(component, 'eval'):
                        component.eval()
                        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = LAVModelWrapper(self.model_components, self.device)
    
    def _load_transfuser_model(self):
        """Load TransFuser-specific model."""
        try:
            # TransFuser model loading
            from transfuser_model import TransFuser
            
            self.model = TransFuser(self.model_config)
            checkpoint = torch.load(self.model_path, map_location='cpu')
            self.model.load_state_dict(checkpoint['state_dict'])
            
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = self.model.to(self.device)
            
        except Exception as e:
            print(f"Error loading TransFuser model: {e}")
            self._load_generic_model()
    
    def sensors(self):
        """Return sensor configuration."""
        sensors = []
        
        for sensor_spec in self.sensor_config:
            sensor = {
                'type': sensor_spec['type'],
                'id': sensor_spec['id']
            }
            
            # Add positional parameters if available
            for key in ['x', 'y', 'z', 'roll', 'pitch', 'yaw']:
                if key in sensor_spec:
                    sensor[key] = sensor_spec[key]
            
            # Add sensor-specific parameters
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
        
        # Create save directories
        self._setup_sensor_directories(sensors)
        
        return sensors
    
    def _setup_sensor_directories(self, sensors):
        """Create directories for sensor data storage."""
        for sensor in sensors:
            sensor_id = sensor['id']
            folder_name = self._get_sensor_folder_name(sensor['type'], sensor_id)
            sensor_path = os.path.join(self.save_path, folder_name)
            os.makedirs(sensor_path, exist_ok=True)
            self.sensor_data_paths[sensor_id] = sensor_path
    
    def run_step(self, input_data, timestamp):
        """Process sensor data and return control commands."""
        # Save raw sensor data
        try:
            self._save_sensor_data(input_data, timestamp)
        except Exception as e:
            print(f"Warning: Data saving failed: {e}")
        
        # Process sensor data
        processed_data = self._process_sensor_data(input_data)
        
        # Track speed for control decisions
        if 'speed' in processed_data:
            self.last_speed = processed_data['speed']
        
        # Get control commands
        if self.model is not None or (hasattr(self, 'model_components') and self.model_components):
            control = self._model_inference(processed_data, timestamp)
        else:
            control = self._rule_based_control(processed_data, timestamp)
        
        # Apply control post-processing
        control = self._postprocess_control(control)
        
        self.frame_count += 1
        
        return control
    
    def _process_sensor_data(self, input_data):
        """Process raw sensor data into model-ready format."""
        processed = {}
        
        for sensor_id, sensor_data in input_data.items():
            # Extract actual data from tuple format
            if isinstance(sensor_data, tuple) and len(sensor_data) == 2:
                _, data = sensor_data
            else:
                data = sensor_data
            
            sensor_id_lower = sensor_id.lower()
            
            # Process different sensor types
            if 'rgb' in sensor_id_lower:
                processed[sensor_id] = self._process_rgb_image(data)
            elif 'semantic' in sensor_id_lower:
                processed[sensor_id] = self._process_semantic_image(data)
            elif 'depth' in sensor_id_lower:
                processed[sensor_id] = self._process_depth_image(data)
            elif 'lidar' in sensor_id_lower:
                processed[sensor_id] = self._process_lidar(data)
            elif 'imu' in sensor_id_lower:
                processed[sensor_id] = self._process_imu(data)
            elif 'gps' in sensor_id_lower or 'gnss' in sensor_id_lower:
                processed[sensor_id] = self._process_gps(data)
            elif 'speed' in sensor_id_lower:
                processed[sensor_id] = self._process_speed(data)
            else:
                processed[sensor_id] = data
        
        return processed
    
    def _process_rgb_image(self, data):
        """Process RGB camera data."""
        if hasattr(data, 'raw_data'):
            array = np.frombuffer(data.raw_data, dtype=np.uint8)
            array = array.reshape((data.height, data.width, 4))
            array = array[:, :, :3]  # Remove alpha channel
        else:
            array = np.array(data)
        
        # Convert to tensor
        tensor = torch.from_numpy(array).float() / 255.0
        tensor = tensor.permute(2, 0, 1)  # HWC to CHW
        
        return tensor
    
    def _process_semantic_image(self, data):
        """Process semantic segmentation data."""
        if hasattr(data, 'raw_data'):
            array = np.frombuffer(data.raw_data, dtype=np.uint8)
            array = array.reshape((data.height, data.width, 4))
            array = array[:, :, 2]  # Red channel contains class IDs
        else:
            array = np.array(data)
        
        tensor = torch.from_numpy(array).long()
        return tensor
    
    def _process_depth_image(self, data):
        """Process depth camera data."""
        if hasattr(data, 'raw_data'):
            array = np.frombuffer(data.raw_data, dtype=np.uint8)
            array = array.reshape((data.height, data.width, 4))
            
            # Convert to depth values
            normalized = (array[:, :, 2] + 
                         array[:, :, 1] * 256.0 + 
                         array[:, :, 0] * 256.0 * 256.0) / (256.0 * 256.0 * 256.0 - 1.0)
            depth = normalized * 1000.0
        else:
            depth = np.array(data)
        
        tensor = torch.from_numpy(depth).float()
        return tensor
    
    def _process_lidar(self, data):
        """Process LiDAR point cloud."""
        if hasattr(data, 'raw_data'):
            points = np.frombuffer(data.raw_data, dtype=np.float32)
            points = points.reshape((-1, 4))
        else:
            points = np.array(data)
        
        # Convert to tensor
        tensor = torch.from_numpy(points).float()
        return tensor
    
    def _process_imu(self, data):
        """Process IMU data."""
        if hasattr(data, 'accelerometer'):
            imu_dict = {
                'accelerometer': np.array([data.accelerometer.x, 
                                          data.accelerometer.y, 
                                          data.accelerometer.z]),
                'gyroscope': np.array([data.gyroscope.x, 
                                      data.gyroscope.y, 
                                      data.gyroscope.z]),
                'compass': data.compass
            }
        else:
            imu_dict = data
        
        return imu_dict
    
    def _process_gps(self, data):
        """Process GPS data."""
        if hasattr(data, 'latitude'):
            gps_dict = {
                'lat': data.latitude,
                'lon': data.longitude,
                'alt': getattr(data, 'altitude', 0.0)
            }
        else:
            gps_dict = data
        
        return gps_dict
    
    def _process_speed(self, data):
        """Process speed data."""
        if hasattr(data, 'speed'):
            # Object with speed attribute
            speed = data.speed
        elif isinstance(data, dict):
            # Dictionary format - this is what CARLA speedometer returns
            speed = float(data.get('speed', 0.0))
        else:
            # Try direct conversion as fallback
            try:
                speed = float(data)
            except (TypeError, ValueError) as e:
                print(f"Warning: Could not convert speed data to float: {type(data)} - {data}")
                speed = 0.0
    
        return speed
    
    def _model_inference(self, processed_data, timestamp):
        """Run model inference to get control commands."""
        from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
        
        try:
            with torch.no_grad():
                # Prepare model inputs based on model type
                if self.model_type == 'interfuser':
                    model_input = self._prepare_interfuser_input(processed_data)
                elif self.model_type == 'lav':
                    model_input = self._prepare_lav_input(processed_data)
                elif self.model_type == 'transfuser':
                    model_input = self._prepare_transfuser_input(processed_data)
                else:
                    model_input = self._prepare_generic_input(processed_data)
                
                # Run inference
                output = self.model(model_input)
                
                # Convert output to control commands
                control = self._output_to_control(output)
                
                return control
                
        except Exception as e:
            print(f"Model inference failed: {e}")
            # Fallback to rule-based control
            return self._rule_based_control(processed_data, timestamp)
    
    def _prepare_generic_input(self, processed_data):
        """Prepare generic model input."""
        # Stack all image tensors if available
        images = []
        for key, value in processed_data.items():
            if isinstance(value, torch.Tensor) and value.dim() >= 2:
                if value.dim() == 2:
                    value = value.unsqueeze(0)
                images.append(value)
        
        if images:
            # Batch dimension
            return torch.stack(images).unsqueeze(0).to(self.device)
        else:
            # Return dummy input
            return torch.zeros(1, 3, 256, 256).to(self.device)
    
    def _prepare_interfuser_input(self, processed_data):
        """Prepare InterFuser-specific input."""
        # InterFuser expects multiple camera views and measurements
        inputs = {}
        
        # Camera inputs
        if 'rgb_front' in processed_data:
            inputs['rgb'] = processed_data['rgb_front'].unsqueeze(0).to(self.device)
        
        # Measurements (speed, etc.)
        measurements = torch.zeros(1, 10).to(self.device)  # Placeholder
        if 'speed' in processed_data:
            measurements[0, 0] = processed_data['speed'] / 40.0  # Normalize
        
        inputs['measurements'] = measurements
        
        # Waypoints (if available from global plan)
        if hasattr(self, 'global_plan_world_coord'):
            waypoints = self._get_local_waypoints()
            inputs['waypoints'] = waypoints.to(self.device)
        
        return inputs
    
    def _prepare_lav_input(self, processed_data):
        """Prepare LAV-specific input."""
        # LAV uses BEV and front camera with multiple model components
        inputs = {}
        
        if 'bev' in processed_data:
            inputs['bev'] = processed_data['bev'].unsqueeze(0).to(self.device)
        if 'semantic_bev' in processed_data:
            inputs['semantic_bev'] = processed_data['semantic_bev'].unsqueeze(0).to(self.device)
        if 'rgb_front' in processed_data:
            inputs['rgb_front'] = processed_data['rgb_front'].unsqueeze(0).to(self.device)
        
        # Additional cameras for 360 coverage
        for key in ['rgb_left_side', 'rgb_right_side', 'rgb_rear']:
            if key in processed_data:
                inputs[key] = processed_data[key].unsqueeze(0).to(self.device)
        
        # LiDAR data if available
        if 'lidar' in processed_data:
            inputs['lidar'] = processed_data['lidar'].unsqueeze(0).to(self.device)
        
        # Measurements
        measurements = torch.zeros(1, 10).to(self.device)
        if 'speed' in processed_data:
            measurements[0, 0] = processed_data['speed'] / 40.0  # Normalize
        if 'gps' in processed_data:
            gps = processed_data['gps']
            if isinstance(gps, dict):
                measurements[0, 1] = gps.get('lat', 0.0)
                measurements[0, 2] = gps.get('lon', 0.0)
        
        inputs['measurements'] = measurements
        
        # Add waypoints if available
        if hasattr(self, '_global_plan_world_coord'):
            waypoints = self._get_local_waypoints()
            inputs['waypoints'] = waypoints.to(self.device)
        
        return inputs
    
    def _prepare_transfuser_input(self, processed_data):
        """Prepare TransFuser-specific input."""
        # TransFuser uses image and LiDAR
        inputs = {}
        
        if 'rgb_front' in processed_data:
            inputs['image'] = processed_data['rgb_front'].unsqueeze(0).to(self.device)
        if 'lidar' in processed_data:
            inputs['lidar'] = processed_data['lidar'].unsqueeze(0).to(self.device)
        
        return inputs
    
    def _output_to_control(self, output):
        """Convert model output to CARLA vehicle control."""
        from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
        
        control = CarlaDataProvider.get_world().get_blueprint_library().find('controller.ai.walker').make_control()
        
        # Handle different output formats
        if isinstance(output, dict):
            # Dictionary output - check for various key patterns
            if 'control' in output:
                # Nested control dict
                ctrl = output['control']
                control.steer = float(ctrl.get('steer', 0.0))
                control.throttle = float(ctrl.get('throttle', 0.0))
                control.brake = float(ctrl.get('brake', 0.0))
            elif 'steer' in output:
                # Direct control values
                control.steer = float(output.get('steer', 0.0))
                control.throttle = float(output.get('throttle', 0.0))
                control.brake = float(output.get('brake', 0.0))
            elif 'action' in output:
                # Action vector
                action = output['action']
                if isinstance(action, torch.Tensor):
                    action = action.cpu().numpy().flatten()
                control.steer = float(action[0]) if len(action) > 0 else 0.0
                control.throttle = float(action[1]) if len(action) > 1 else 0.0
                control.brake = float(action[2]) if len(action) > 2 else 0.0
            elif 'waypoints' in output:
                # Waypoint-based control - need to compute control from waypoints
                waypoints = output['waypoints']
                if isinstance(waypoints, torch.Tensor):
                    waypoints = waypoints.cpu().numpy()
                control = self._waypoints_to_control(waypoints, control)
                
                # Check if speed is also predicted
                if 'speed' in output:
                    target_speed = float(output['speed'])
                    current_speed = self.last_speed if hasattr(self, 'last_speed') else 0.0
                    if current_speed < target_speed:
                        control.throttle = 0.7
                        control.brake = 0.0
                    else:
                        control.throttle = 0.0
                        control.brake = 0.3
            else:
                # Unknown dictionary format, try to extract something useful
                for key in ['pred_control', 'controls', 'output']:
                    if key in output:
                        return self._output_to_control(output[key])
                # Fallback to defaults
                control.steer = 0.0
                control.throttle = 0.3
                control.brake = 0.0
                
        elif isinstance(output, (list, tuple)):
            # List/tuple output [steer, throttle, brake]
            control.steer = float(output[0]) if len(output) > 0 else 0.0
            control.throttle = float(output[1]) if len(output) > 1 else 0.0
            control.brake = float(output[2]) if len(output) > 2 else 0.0
        elif isinstance(output, torch.Tensor):
            # Tensor output
            output = output.cpu().numpy().flatten()
            control.steer = float(output[0]) if len(output) > 0 else 0.0
            control.throttle = float(output[1]) if len(output) > 1 else 0.0
            control.brake = float(output[2]) if len(output) > 2 else 0.0
        else:
            # Unknown format, use defaults
            control.steer = 0.0
            control.throttle = 0.3
            control.brake = 0.0
        
        # Clamp values
        control.steer = np.clip(control.steer, -1.0, 1.0)
        control.throttle = np.clip(control.throttle, 0.0, 1.0)
        control.brake = np.clip(control.brake, 0.0, 1.0)
        
        control.hand_brake = False
        control.manual_gear_shift = False
        
        return control
    
    def _waypoints_to_control(self, waypoints, control):
        """Convert predicted waypoints to control commands."""
        if len(waypoints.shape) == 3:
            waypoints = waypoints[0]  # Remove batch dimension
        
        if len(waypoints) < 2:
            return control
        
        # Use first few waypoints for steering
        if len(waypoints) >= 2:
            # Calculate angle to second waypoint
            dx = waypoints[1, 0] - waypoints[0, 0] if len(waypoints) > 1 else waypoints[0, 0]
            dy = waypoints[1, 1] - waypoints[0, 1] if len(waypoints) > 1 else waypoints[0, 1]
            
            angle = np.arctan2(dy, dx)
            control.steer = np.clip(angle / 0.7, -1.0, 1.0)  # 0.7 rad ~ 40 degrees max
        
        return control
    
    def _rule_based_control(self, processed_data, timestamp):
        """Simple rule-based control as fallback."""
        from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
        
        control = CarlaDataProvider.get_world().get_blueprint_library().find('controller.ai.walker').make_control()
        
        # Get current speed
        current_speed = processed_data.get('speed', 0.0)
        target_speed = self.control_config['target_speed']
        
        # Simple speed control
        if current_speed < target_speed:
            control.throttle = 0.7
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = 0.3
        
        # Simple steering (try to follow waypoints if available)
        control.steer = self._compute_steer_from_waypoints()
        
        control.hand_brake = False
        control.manual_gear_shift = False
        
        return control
    
    def _compute_steer_from_waypoints(self):
        """Compute steering angle from waypoints."""
        if not hasattr(self, '_global_plan_world_coord') or not self._global_plan_world_coord:
            return 0.0
        
        try:
            from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
            
            # Get ego vehicle
            ego_vehicle = CarlaDataProvider.get_hero_actor()
            if ego_vehicle is None:
                return 0.0
            
            # Get current transform
            transform = ego_vehicle.get_transform()
            location = transform.location
            rotation = transform.rotation
            
            # Find nearest waypoint
            min_dist = float('inf')
            target_waypoint = None
            
            for i, waypoint in enumerate(self._global_plan_world_coord):
                if isinstance(waypoint, tuple):
                    if len(waypoint) >= 2:
                        wp_loc = waypoint[0] if isinstance(waypoint[0], object) else waypoint
                        if hasattr(wp_loc, 'location'):
                            wp_x, wp_y = wp_loc.location.x, wp_loc.location.y
                        else:
                            wp_x, wp_y = wp_loc[0], wp_loc[1]
                    else:
                        continue
                else:
                    continue
                
                dist = np.sqrt((wp_x - location.x)**2 + (wp_y - location.y)**2)
                
                # Look for waypoint ahead
                if dist < min_dist and dist > 2.0:  # At least 2m ahead
                    min_dist = dist
                    target_waypoint = (wp_x, wp_y)
            
            if target_waypoint is None:
                return 0.0
            
            # Calculate steering angle
            dx = target_waypoint[0] - location.x
            dy = target_waypoint[1] - location.y
            
            # Convert to vehicle coordinate system
            forward_vec = transform.get_forward_vector()
            right_vec = transform.get_right_vector()
            
            dot_forward = dx * forward_vec.x + dy * forward_vec.y
            dot_right = dx * right_vec.x + dy * right_vec.y
            
            # Calculate angle
            angle = np.arctan2(dot_right, dot_forward)
            
            # Convert to steering command (-1 to 1)
            steer = np.clip(angle / 0.7, -1.0, 1.0)  # 0.7 rad ~ 40 degrees max
            
            return float(steer)
            
        except Exception as e:
            print(f"Error computing steering: {e}")
            return 0.0
    
    def _postprocess_control(self, control):
        """Apply control post-processing for smoothness and safety."""
        # Smooth steering
        if hasattr(control, 'steer'):
            damping = self.control_config.get('steer_damping', 0.3)
            control.steer = (1 - damping) * control.steer + damping * self.prev_steer
            self.prev_steer = control.steer
        
        # Prevent throttle and brake at same time
        if hasattr(control, 'brake') and hasattr(control, 'throttle'):
            if control.brake > self.control_config.get('brake_threshold', 0.5):
                control.throttle = 0.0
        
        return control
    
    def _get_local_waypoints(self, num_waypoints=10):
        """Get local waypoints relative to ego vehicle."""
        if not hasattr(self, '_global_plan_world_coord'):
            return torch.zeros(1, num_waypoints, 2)
        
=======
        
        # Check for single model
        for attr in model_attrs:
            if hasattr(agent, attr):
                model = getattr(agent, attr)
                if isinstance(model, torch.nn.Module):
                    self.model = model
                    print(f"  Extracted model from agent.{attr}")
                    return
        
        # Check for multiple models (like LAV)
        multi_model_attrs = {
            'lidar_model': ['lidar_model', 'lidar_net', 'lidar_encoder'],
            'bev_model': ['bev_model', 'bev_net', 'bev_encoder'],
            'seg_model': ['seg_model', 'segmentation', 'seg_net'],
            'uniplanner': ['uniplanner', 'planner', 'planning_model'],
            'controller': ['controller', 'control_model', 'control_net'],
        }
        
        self.model_components = {}
        for component_name, possible_attrs in multi_model_attrs.items():
            for attr in possible_attrs:
                if hasattr(agent, attr):
                    model = getattr(agent, attr)
                    if isinstance(model, torch.nn.Module):
                        self.model_components[component_name] = model
                        print(f"  Extracted {component_name} from agent.{attr}")
                        break
        
        if self.model_components:
            # Create wrapper based on agent type
            if 'lav' in self.model_type.lower():
                self._create_lav_wrapper()
            else:
                self.model = self.model_components
    
    def _merge_configs(self, base_config, external_config):
        """Merge external config into base config."""
        for key, value in external_config.items():
            if key not in base_config:
                base_config[key] = value
            elif isinstance(value, dict) and isinstance(base_config[key], dict):
                # Recursively merge dictionaries
                self._merge_configs(base_config[key], value)
            # else: keep base_config value (base takes precedence for existing keys)
    
    def _convert_lav_config(self):
        """Convert LAV-style config with individual model paths to our format."""
        if 'model_components' not in self.config:
            self.config['model_components'] = {}
        
        # Map LAV config keys to our component names
        lav_component_mapping = {
            'lidar_model_dir': ('lidar_model', 'checkpoint'),
            'uniplanner_dir': ('uniplanner', 'checkpoint'),
            'bra_model_dir': ('bra_model', 'checkpoint'),
            'bra_model_trace_dir': ('bra_model_trace', 'trace'),
            'seg_model_dir': ('seg_model', 'checkpoint'),
            'seg_model_trace_dir': ('seg_model_trace', 'trace'),
            'bev_model_dir': ('bev_model', 'checkpoint'),
        }
        
        for config_key, (component_name, component_type) in lav_component_mapping.items():
            if config_key in self.model_config:
                path = self.model_config[config_key]
                if path and os.path.exists(path):
                    self.config['model_components'][component_name] = {
                        'path': path,
                        'type': component_type
                    }
        
        # Set model type to LAV if not already set
        if self.model_type == 'generic':
            self.model_type = 'lav'
        
        print(f"ConsolidatedAgent: Converted LAV config - found {len(self.config['model_components'])} components")
    
    def _load_model(self):
        """Load the neural network model(s) based on type."""
        # Check if we have multiple model components
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
            
            # Set model to eval mode
            if hasattr(self, 'model') and hasattr(self.model, 'eval'):
                self.model.eval()
                print(f"ConsolidatedAgent: Model loaded and set to eval mode")
        else:
            print("ConsolidatedAgent: No model specified, using rule-based control")
            self.model = None
    
    def _load_multi_component_model(self):
        """Load multiple model components for complex architectures."""
        print("ConsolidatedAgent: Loading multi-component model architecture")
        
        self.model_components = {}
        components_config = self.config.get('model_components', {})
        
        # Load each component
        for component_name, component_info in components_config.items():
            component_path = component_info.get('path')
            component_type = component_info.get('type', 'checkpoint')
            
            if not component_path or not os.path.exists(component_path):
                print(f"Warning: Component {component_name} path not found: {component_path}")
                continue
            
            print(f"  Loading {component_name} from {component_path}")
            
            try:
                if component_type == 'trace':
                    # Load JIT traced model
                    self.model_components[component_name] = torch.jit.load(component_path)
                elif component_type == 'checkpoint':
                    # Load regular checkpoint
                    checkpoint = torch.load(component_path, map_location='cpu')
                    
                    # Handle different checkpoint formats
                    if isinstance(checkpoint, torch.nn.Module):
                        self.model_components[component_name] = checkpoint
                    elif 'model' in checkpoint:
                        self.model_components[component_name] = checkpoint['model']
                    elif 'state_dict' in checkpoint:
                        # Need to instantiate model first - use component config
                        model_class_info = component_info.get('model_class')
                        if model_class_info:
                            module = importlib.import_module(model_class_info['module'])
                            model_class = getattr(module, model_class_info['name'])
                            model_args = model_class_info.get('args', {})
                            model = model_class(**model_args)
                            model.load_state_dict(checkpoint['state_dict'])
                            self.model_components[component_name] = model
                        else:
                            self.model_components[component_name] = checkpoint
                    else:
                        self.model_components[component_name] = checkpoint
                        
                # Move to device and set to eval
                if hasattr(self.model_components[component_name], 'eval'):
                    self.model_components[component_name].eval()
                    if hasattr(self.model_components[component_name], 'to'):
                        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                        self.model_components[component_name] = self.model_components[component_name].to(device)
                        
                print(f"    ✓ {component_name} loaded successfully")
                
            except Exception as e:
                print(f"    ✗ Failed to load {component_name}: {e}")
                
        # Create wrapper model for compatibility
        if self.model_type == 'lav':
            self._create_lav_wrapper()
        else:
            # Generic wrapper
            self.model = self.model_components
        
        print(f"ConsolidatedAgent: Loaded {len(self.model_components)} model components")
    
    def _load_generic_model(self):
        """Load a generic PyTorch model."""
        try:
            # Try to load as PyTorch checkpoint
            checkpoint = torch.load(self.model_path, map_location='cpu')
            
            # Try to instantiate model from config
            if 'model_class' in self.model_config:
                module_path = self.model_config['model_module']
                class_name = self.model_config['model_class']
                
                module = importlib.import_module(module_path)
                model_class = getattr(module, class_name)
                
                # Instantiate model with config
                model_args = self.model_config.get('model_args', {})
                self.model = model_class(**model_args)
                
                # Load weights
                if 'state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
            else:
                # Direct model object in checkpoint
                self.model = checkpoint['model'] if 'model' in checkpoint else checkpoint
            
            # Move to GPU if available
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = self.model.to(self.device)
            
        except Exception as e:
            print(f"Warning: Could not load model as PyTorch: {e}")
            print("Will use simple rule-based control as fallback")
            self.model = None
    
    def _load_interfuser_model(self):
        """Load InterFuser-specific model."""
        try:
            # InterFuser uses specific model structure
            from interfuser.timm.models import create_model
            
            # Create model architecture
            self.model = create_model(
                self.model_config.get('architecture', 'interfuser_baseline'),
                pretrained=False,
                **self.model_config.get('model_args', {})
            )
            
            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location='cpu')
            self.model.load_state_dict(checkpoint['state_dict'])
            
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = self.model.to(self.device)
            
        except Exception as e:
            print(f"Error loading InterFuser model: {e}")
=======
            print(f"ConsolidatedAgent: InterFuser model loaded successfully")
            
        except ImportError as e:
            print(f"Error importing InterFuser modules: {e}")
            print("Trying fallback to generic model loading...")
            self._load_generic_model()
        except Exception as e:
            print(f"Error loading InterFuser model: {e}")
            print("Trying fallback to generic model loading...")
>>>>>>> 8a07600 (fixing issues w/ model loading)
            self._load_generic_model()
    
    def _load_lav_model(self):
        """Load LAV-specific model."""
        # Check if using multi-component architecture (typical for LAV)
        if 'model_components' in self.config:
            self._load_multi_component_model()
            return
            
        # Single model fallback
        try:
            checkpoint = torch.load(self.model_path, map_location='cpu')
            
            # Extract model from checkpoint
            if 'model' in checkpoint:
                self.model = checkpoint['model']
            elif 'state_dict' in checkpoint:
                # Need to instantiate model first
                from lav_model import LAV  # Assuming LAV model class exists
                self.model = LAV(**self.model_config.get('model_args', {}))
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                self.model = checkpoint
            
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = self.model.to(self.device)
            
        except Exception as e:
            print(f"Error loading LAV model: {e}")
            self._load_generic_model()
    
    def _create_lav_wrapper(self):
        """Create a wrapper for LAV's multi-component architecture."""
        class LAVModelWrapper:
            def __init__(self, components, device):
                self.components = components
                self.device = device
                
                # Extract individual models
                self.lidar_model = components.get('lidar_model')
                self.uniplanner = components.get('uniplanner')
                self.bra_model = components.get('bra_model')
                self.seg_model = components.get('seg_model')
                self.bev_model = components.get('bev_model')
                
            def __call__(self, inputs):
                """Forward pass through LAV pipeline."""
                # This is a simplified version - actual LAV may have more complex pipeline
                outputs = {}
                
                # Process BEV if available
                if self.bev_model and 'bev' in inputs:
                    bev_features = self.bev_model(inputs['bev'])
                    outputs['bev_features'] = bev_features
                
                # Process LiDAR if available
                if self.lidar_model and 'lidar' in inputs:
                    lidar_features = self.lidar_model(inputs['lidar'])
                    outputs['lidar_features'] = lidar_features
                
                # Segmentation
                if self.seg_model and 'rgb_front' in inputs:
                    seg_output = self.seg_model(inputs['rgb_front'])
                    outputs['segmentation'] = seg_output
                
                # Planning with uniplanner
                if self.uniplanner:
                    # Combine features for planning
                    planner_input = {}
                    if 'bev_features' in outputs:
                        planner_input['bev'] = outputs['bev_features']
                    if 'lidar_features' in outputs:
                        planner_input['lidar'] = outputs['lidar_features']
                    if 'measurements' in inputs:
                        planner_input['measurements'] = inputs['measurements']
                    
                    control_output = self.uniplanner(planner_input)
                    outputs.update(control_output)
                
                # Behavior prediction/refinement
                if self.bra_model and 'bev_features' in outputs:
                    bra_output = self.bra_model(outputs['bev_features'])
                    outputs['behavior'] = bra_output
                
                return outputs
            
            def eval(self):
                """Set all components to eval mode."""
                for component in self.components.values():
                    if hasattr(component, 'eval'):
                        component.eval()
                        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = LAVModelWrapper(self.model_components, self.device)
    
    def _load_transfuser_model(self):
        """Load TransFuser-specific model."""
        try:
            # TransFuser model loading
            from transfuser_model import TransFuser
            
            self.model = TransFuser(self.model_config)
            checkpoint = torch.load(self.model_path, map_location='cpu')
            self.model.load_state_dict(checkpoint['state_dict'])
            
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = self.model.to(self.device)
            
        except Exception as e:
            print(f"Error loading TransFuser model: {e}")
            self._load_generic_model()
    
    def sensors(self):
        """Return sensor configuration."""
        sensors = []
        
        for sensor_spec in self.sensor_config:
            sensor = {
                'type': sensor_spec['type'],
                'id': sensor_spec['id']
            }
            
            # Add positional parameters if available
            for key in ['x', 'y', 'z', 'roll', 'pitch', 'yaw']:
                if key in sensor_spec:
                    sensor[key] = sensor_spec[key]
            
            # Add sensor-specific parameters
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
        
        # Create save directories
        self._setup_sensor_directories(sensors)
        
        return sensors
    
    def _setup_sensor_directories(self, sensors):
        """Create directories for sensor data storage."""
        for sensor in sensors:
            sensor_id = sensor['id']
            folder_name = self._get_sensor_folder_name(sensor['type'], sensor_id)
            sensor_path = os.path.join(self.save_path, folder_name)
            os.makedirs(sensor_path, exist_ok=True)
            self.sensor_data_paths[sensor_id] = sensor_path
    
    def run_step(self, input_data, timestamp):
        """Process sensor data and return control commands."""
        # Save raw sensor data
        try:
            self._save_sensor_data(input_data, timestamp)
        except Exception as e:
            print(f"Warning: Data saving failed: {e}")
        
        # Process sensor data
        processed_data = self._process_sensor_data(input_data)
        
        # Track speed for control decisions
        if 'speed' in processed_data:
            self.last_speed = processed_data['speed']
        
        # Get control commands
        if self.model is not None or (hasattr(self, 'model_components') and self.model_components):
            control = self._model_inference(processed_data, timestamp)
        else:
            control = self._rule_based_control(processed_data, timestamp)
        
        # Apply control post-processing
        control = self._postprocess_control(control)
        
        self.frame_count += 1
        
        return control
    
    def _process_sensor_data(self, input_data):
        """Process raw sensor data into model-ready format."""
        processed = {}
        
        for sensor_id, sensor_data in input_data.items():
            # Extract actual data from tuple format
            if isinstance(sensor_data, tuple) and len(sensor_data) == 2:
                _, data = sensor_data
            else:
                data = sensor_data
            
            sensor_id_lower = sensor_id.lower()
            
            # Process different sensor types
            if 'rgb' in sensor_id_lower:
                processed[sensor_id] = self._process_rgb_image(data)
            elif 'semantic' in sensor_id_lower:
                processed[sensor_id] = self._process_semantic_image(data)
            elif 'depth' in sensor_id_lower:
                processed[sensor_id] = self._process_depth_image(data)
            elif 'lidar' in sensor_id_lower:
                processed[sensor_id] = self._process_lidar(data)
            elif 'imu' in sensor_id_lower:
                processed[sensor_id] = self._process_imu(data)
            elif 'gps' in sensor_id_lower or 'gnss' in sensor_id_lower:
                processed[sensor_id] = self._process_gps(data)
            elif 'speed' in sensor_id_lower:
                processed[sensor_id] = self._process_speed(data)
            else:
                processed[sensor_id] = data
        
        return processed
    
    def _process_rgb_image(self, data):
        """Process RGB camera data."""
        if hasattr(data, 'raw_data'):
            array = np.frombuffer(data.raw_data, dtype=np.uint8)
            array = array.reshape((data.height, data.width, 4))
            array = array[:, :, :3]  # Remove alpha channel
        else:
            array = np.array(data)
        
        # Convert to tensor
        tensor = torch.from_numpy(array).float() / 255.0
        tensor = tensor.permute(2, 0, 1)  # HWC to CHW
        
        return tensor
    
    def _process_semantic_image(self, data):
        """Process semantic segmentation data."""
        if hasattr(data, 'raw_data'):
            array = np.frombuffer(data.raw_data, dtype=np.uint8)
            array = array.reshape((data.height, data.width, 4))
            array = array[:, :, 2]  # Red channel contains class IDs
        else:
            array = np.array(data)
        
        tensor = torch.from_numpy(array).long()
        return tensor
    
    def _process_depth_image(self, data):
        """Process depth camera data."""
        if hasattr(data, 'raw_data'):
            array = np.frombuffer(data.raw_data, dtype=np.uint8)
            array = array.reshape((data.height, data.width, 4))
            
            # Convert to depth values
            normalized = (array[:, :, 2] + 
                         array[:, :, 1] * 256.0 + 
                         array[:, :, 0] * 256.0 * 256.0) / (256.0 * 256.0 * 256.0 - 1.0)
            depth = normalized * 1000.0
        else:
            depth = np.array(data)
        
        tensor = torch.from_numpy(depth).float()
        return tensor
    
    def _process_lidar(self, data):
        """Process LiDAR point cloud."""
        if hasattr(data, 'raw_data'):
            points = np.frombuffer(data.raw_data, dtype=np.float32)
            points = points.reshape((-1, 4))
        else:
            points = np.array(data)
        
        # Convert to tensor
        tensor = torch.from_numpy(points).float()
        return tensor
    
    def _process_imu(self, data):
        """Process IMU data."""
        if hasattr(data, 'accelerometer'):
            imu_dict = {
                'accelerometer': np.array([data.accelerometer.x, 
                                          data.accelerometer.y, 
                                          data.accelerometer.z]),
                'gyroscope': np.array([data.gyroscope.x, 
                                      data.gyroscope.y, 
                                      data.gyroscope.z]),
                'compass': data.compass
            }
        else:
            imu_dict = data
        
        return imu_dict
    
    def _process_gps(self, data):
        """Process GPS data."""
        if hasattr(data, 'latitude'):
            gps_dict = {
                'lat': data.latitude,
                'lon': data.longitude,
                'alt': getattr(data, 'altitude', 0.0)
            }
        else:
            gps_dict = data
        
        return gps_dict
    
    def _process_speed(self, data):
        """Process speed data."""
        if hasattr(data, 'speed'):
            # Object with speed attribute
            speed = data.speed
        elif isinstance(data, dict):
            # Dictionary format - this is what CARLA speedometer returns
            speed = float(data.get('speed', 0.0))
        else:
            # Try direct conversion as fallback
            try:
                speed = float(data)
            except (TypeError, ValueError) as e:
                print(f"Warning: Could not convert speed data to float: {type(data)} - {data}")
                speed = 0.0
    
        return speed
    
    def _model_inference(self, processed_data, timestamp):
        """Run model inference to get control commands."""
        from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
        
        try:
            with torch.no_grad():
                # Prepare model inputs based on model type
                if self.model_type == 'interfuser':
                    model_input = self._prepare_interfuser_input(processed_data)
                elif self.model_type == 'lav':
                    model_input = self._prepare_lav_input(processed_data)
                elif self.model_type == 'transfuser':
                    model_input = self._prepare_transfuser_input(processed_data)
                else:
                    model_input = self._prepare_generic_input(processed_data)
                
                # Run inference
                output = self.model(model_input)
                
                # Convert output to control commands
                control = self._output_to_control(output)
                
                return control
                
        except Exception as e:
            print(f"Model inference failed: {e}")
            # Fallback to rule-based control
            return self._rule_based_control(processed_data, timestamp)
    
    def _prepare_generic_input(self, processed_data):
        """Prepare generic model input."""
        # Stack all image tensors if available
        images = []
        for key, value in processed_data.items():
            if isinstance(value, torch.Tensor) and value.dim() >= 2:
                if value.dim() == 2:
                    value = value.unsqueeze(0)
                images.append(value)
        
        if images:
            # Batch dimension
            return torch.stack(images).unsqueeze(0).to(self.device)
        else:
            # Return dummy input
            return torch.zeros(1, 3, 256, 256).to(self.device)
    
    def _prepare_interfuser_input(self, processed_data):
        """Prepare InterFuser-specific input."""
        # InterFuser expects multiple camera views and measurements
        inputs = {}
        
        # Camera inputs
        if 'rgb_front' in processed_data:
            inputs['rgb'] = processed_data['rgb_front'].unsqueeze(0).to(self.device)
        
        # Measurements (speed, etc.)
        measurements = torch.zeros(1, 10).to(self.device)  # Placeholder
        if 'speed' in processed_data:
            measurements[0, 0] = processed_data['speed'] / 40.0  # Normalize
        
        inputs['measurements'] = measurements
        
        # Waypoints (if available from global plan)
        if hasattr(self, 'global_plan_world_coord'):
            waypoints = self._get_local_waypoints()
            inputs['waypoints'] = waypoints.to(self.device)
        
        return inputs
    
    def _prepare_lav_input(self, processed_data):
        """Prepare LAV-specific input."""
        # LAV uses BEV and front camera with multiple model components
        inputs = {}
        
        if 'bev' in processed_data:
            inputs['bev'] = processed_data['bev'].unsqueeze(0).to(self.device)
        if 'semantic_bev' in processed_data:
            inputs['semantic_bev'] = processed_data['semantic_bev'].unsqueeze(0).to(self.device)
        if 'rgb_front' in processed_data:
            inputs['rgb_front'] = processed_data['rgb_front'].unsqueeze(0).to(self.device)
        
        # Additional cameras for 360 coverage
        for key in ['rgb_left_side', 'rgb_right_side', 'rgb_rear']:
            if key in processed_data:
                inputs[key] = processed_data[key].unsqueeze(0).to(self.device)
        
        # LiDAR data if available
        if 'lidar' in processed_data:
            inputs['lidar'] = processed_data['lidar'].unsqueeze(0).to(self.device)
        
        # Measurements
        measurements = torch.zeros(1, 10).to(self.device)
        if 'speed' in processed_data:
            measurements[0, 0] = processed_data['speed'] / 40.0  # Normalize
        if 'gps' in processed_data:
            gps = processed_data['gps']
            if isinstance(gps, dict):
                measurements[0, 1] = gps.get('lat', 0.0)
                measurements[0, 2] = gps.get('lon', 0.0)
        
        inputs['measurements'] = measurements
        
        # Add waypoints if available
        if hasattr(self, '_global_plan_world_coord'):
            waypoints = self._get_local_waypoints()
            inputs['waypoints'] = waypoints.to(self.device)
        
        return inputs
    
    def _prepare_transfuser_input(self, processed_data):
        """Prepare TransFuser-specific input."""
        # TransFuser uses image and LiDAR
        inputs = {}
        
        if 'rgb_front' in processed_data:
            inputs['image'] = processed_data['rgb_front'].unsqueeze(0).to(self.device)
        if 'lidar' in processed_data:
            inputs['lidar'] = processed_data['lidar'].unsqueeze(0).to(self.device)
        
        return inputs
    
    def _output_to_control(self, output):
        """Convert model output to CARLA vehicle control."""
        from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
        
        #control = CarlaDataProvider.get_world().get_blueprint_library().find('controller.ai.walker').make_control()
        control = carla.VehicleControl()

        # Handle different output formats
        if isinstance(output, dict):
            # Dictionary output - check for various key patterns
            if 'control' in output:
                # Nested control dict
                ctrl = output['control']
                control.steer = float(ctrl.get('steer', 0.0))
                control.throttle = float(ctrl.get('throttle', 0.0))
                control.brake = float(ctrl.get('brake', 0.0))
            elif 'steer' in output:
                # Direct control values
                control.steer = float(output.get('steer', 0.0))
                control.throttle = float(output.get('throttle', 0.0))
                control.brake = float(output.get('brake', 0.0))
            elif 'action' in output:
                # Action vector
                action = output['action']
                if isinstance(action, torch.Tensor):
                    action = action.cpu().numpy().flatten()
                control.steer = float(action[0]) if len(action) > 0 else 0.0
                control.throttle = float(action[1]) if len(action) > 1 else 0.0
                control.brake = float(action[2]) if len(action) > 2 else 0.0
            elif 'waypoints' in output:
                # Waypoint-based control - need to compute control from waypoints
                waypoints = output['waypoints']
                if isinstance(waypoints, torch.Tensor):
                    waypoints = waypoints.cpu().numpy()
                control = self._waypoints_to_control(waypoints, control)
                
                # Check if speed is also predicted
                if 'speed' in output:
                    target_speed = float(output['speed'])
                    current_speed = self.last_speed if hasattr(self, 'last_speed') else 0.0
                    if current_speed < target_speed:
                        control.throttle = 0.7
                        control.brake = 0.0
                    else:
                        control.throttle = 0.0
                        control.brake = 0.3
            else:
                # Unknown dictionary format, try to extract something useful
                for key in ['pred_control', 'controls', 'output']:
                    if key in output:
                        return self._output_to_control(output[key])
                # Fallback to defaults
                control.steer = 0.0
                control.throttle = 0.3
                control.brake = 0.0
                
        elif isinstance(output, (list, tuple)):
            # List/tuple output [steer, throttle, brake]
            control.steer = float(output[0]) if len(output) > 0 else 0.0
            control.throttle = float(output[1]) if len(output) > 1 else 0.0
            control.brake = float(output[2]) if len(output) > 2 else 0.0
        elif isinstance(output, torch.Tensor):
            # Tensor output
            output = output.cpu().numpy().flatten()
            control.steer = float(output[0]) if len(output) > 0 else 0.0
            control.throttle = float(output[1]) if len(output) > 1 else 0.0
            control.brake = float(output[2]) if len(output) > 2 else 0.0
        else:
            # Unknown format, use defaults
            control.steer = 0.0
            control.throttle = 0.3
            control.brake = 0.0
        
        # Clamp values
        control.steer = np.clip(control.steer, -1.0, 1.0)
        control.throttle = np.clip(control.throttle, 0.0, 1.0)
        control.brake = np.clip(control.brake, 0.0, 1.0)
        
        control.hand_brake = False
        control.manual_gear_shift = False
        
        return control
    
    def _waypoints_to_control(self, waypoints, control):
        """Convert predicted waypoints to control commands."""
        if len(waypoints.shape) == 3:
            waypoints = waypoints[0]  # Remove batch dimension
        
        if len(waypoints) < 2:
            return control
        
        # Use first few waypoints for steering
        if len(waypoints) >= 2:
            # Calculate angle to second waypoint
            dx = waypoints[1, 0] - waypoints[0, 0] if len(waypoints) > 1 else waypoints[0, 0]
            dy = waypoints[1, 1] - waypoints[0, 1] if len(waypoints) > 1 else waypoints[0, 1]
            
            angle = np.arctan2(dy, dx)
            control.steer = np.clip(angle / 0.7, -1.0, 1.0)  # 0.7 rad ~ 40 degrees max
        
        return control
    
    def _rule_based_control(self, processed_data, timestamp):
        """Simple rule-based control as fallback."""
        from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
        
        #control = CarlaDataProvider.get_world().get_blueprint_library().find('controller.ai.walker').make_control()
        control = carla.VehicleControl()

        # Get current speed
        current_speed = processed_data.get('speed', 0.0)
        target_speed = self.control_config['target_speed']
        
        # Simple speed control
        if current_speed < target_speed:
            control.throttle = 0.7
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = 0.3
        
        # Simple steering (try to follow waypoints if available)
        control.steer = self._compute_steer_from_waypoints()
        
        control.hand_brake = False
        control.manual_gear_shift = False
        
        return control
    
    def _compute_steer_from_waypoints(self):
        """Compute steering angle from waypoints."""
        if not hasattr(self, '_global_plan_world_coord') or not self._global_plan_world_coord:
            return 0.0
        
        try:
            from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
            
            # Get ego vehicle
            ego_vehicle = CarlaDataProvider.get_hero_actor()
            if ego_vehicle is None:
                return 0.0
            
            # Get current transform
            transform = ego_vehicle.get_transform()
            location = transform.location
            rotation = transform.rotation
            
            # Find nearest waypoint
            min_dist = float('inf')
            target_waypoint = None
            
            for i, waypoint in enumerate(self._global_plan_world_coord):
                if isinstance(waypoint, tuple):
                    if len(waypoint) >= 2:
                        wp_loc = waypoint[0] if isinstance(waypoint[0], object) else waypoint
                        if hasattr(wp_loc, 'location'):
                            wp_x, wp_y = wp_loc.location.x, wp_loc.location.y
                        else:
                            wp_x, wp_y = wp_loc[0], wp_loc[1]
                    else:
                        continue
                else:
                    continue
                
                dist = np.sqrt((wp_x - location.x)**2 + (wp_y - location.y)**2)
                
                # Look for waypoint ahead
                if dist < min_dist and dist > 2.0:  # At least 2m ahead
                    min_dist = dist
                    target_waypoint = (wp_x, wp_y)
            
            if target_waypoint is None:
                return 0.0
            
            # Calculate steering angle
            dx = target_waypoint[0] - location.x
            dy = target_waypoint[1] - location.y
            
            # Convert to vehicle coordinate system
            forward_vec = transform.get_forward_vector()
            right_vec = transform.get_right_vector()
            
            dot_forward = dx * forward_vec.x + dy * forward_vec.y
            dot_right = dx * right_vec.x + dy * right_vec.y
            
            # Calculate angle
            angle = np.arctan2(dot_right, dot_forward)
            
            # Convert to steering command (-1 to 1)
            steer = np.clip(angle / 0.7, -1.0, 1.0)  # 0.7 rad ~ 40 degrees max
            
            return float(steer)
            
        except Exception as e:
            print(f"Error computing steering: {e}")
            return 0.0
    
    def _postprocess_control(self, control):
        """Apply control post-processing for smoothness and safety."""
        # Smooth steering
        if hasattr(control, 'steer'):
            damping = self.control_config.get('steer_damping', 0.3)
            control.steer = (1 - damping) * control.steer + damping * self.prev_steer
            self.prev_steer = control.steer
        
        # Prevent throttle and brake at same time
        if hasattr(control, 'brake') and hasattr(control, 'throttle'):
            if control.brake > self.control_config.get('brake_threshold', 0.5):
                control.throttle = 0.0
        
        return control
    
    def _get_local_waypoints(self, num_waypoints=10):
        """Get local waypoints relative to ego vehicle."""
        if not hasattr(self, '_global_plan_world_coord'):
            return torch.zeros(1, num_waypoints, 2)
        
<<<<<<< HEAD
>>>>>>> 5c6ba0f (trying to set up rsync)
=======
>>>>>>> 8a07600 (fixing issues w/ model loading)
        try:
            from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
            ego_vehicle = CarlaDataProvider.get_hero_actor()
            
            if ego_vehicle is None:
                return torch.zeros(1, num_waypoints, 2)
            
            transform = ego_vehicle.get_transform()
            
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
            
            # Pad if necessary
            while len(waypoints) < num_waypoints:
                waypoints.append([0.0, 0.0])
            
            return torch.tensor(waypoints[:num_waypoints]).unsqueeze(0).float()
            
        except Exception as e:
            print(f"Error getting local waypoints: {e}")
            return torch.zeros(1, num_waypoints, 2)
    
    def _initialize_data_collection(self):
        """Initialize data collection infrastructure."""
        default_save_path = os.path.join(
            os.environ.get('WORKSPACE_DIR', '/workspace'),
            'dataset',
            'consolidated'
        )
        self.save_path = os.environ.get('SAVE_PATH', default_save_path)
        self.save_path = os.path.expandvars(self.save_path)
        
        self.frame_counter = 0
        self.sensor_data_paths = {}
        
        os.makedirs(self.save_path, exist_ok=True)
        
        self.metadata = {
            'model_type': self.model_type,
            'model_path': self.model_path,
            'config': self.config,
            'save_path': self.save_path,
            'timestamp': self._get_timestamp(),
            'frames': []
        }
        
        print(f"ConsolidatedAgent: Data collection initialized")
        print(f"  Save path: {self.save_path}")
    
    def _get_timestamp(self):
        """Get current timestamp in ISO format."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def _get_sensor_folder_name(self, sensor_type, sensor_id):
        """Generate folder name for sensor data."""
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
        """Save all sensor data for current frame."""
        frame_data = {
            'frame': self.frame_counter,
            'timestamp': timestamp,
            'sensors': {}
        }
        
        for sensor_id, sensor_data in input_data.items():
            sensor_path = self.sensor_data_paths.get(sensor_id)
            if sensor_path is None:
                continue
            
            try:
                filename = self._process_and_save_sensor(
                    sensor_id, sensor_data, sensor_path
                )
                if filename:
                    frame_data['sensors'][sensor_id] = filename
            except Exception as e:
                print(f"Warning: Failed to save data for sensor {sensor_id}: {e}")
        
        self.metadata['frames'].append(frame_data)
        
        # Periodic metadata save
        if self.frame_counter % 50 == 0:
            self._save_metadata()
        
        self.frame_counter += 1
    
    def _process_and_save_sensor(self, sensor_id, sensor_data, sensor_path):
        """Process and save individual sensor data."""
        # Extract actual data from CARLA tuple format
        if isinstance(sensor_data, tuple) and len(sensor_data) == 2:
            _, actual_data = sensor_data
        else:
            actual_data = sensor_data
        
        sensor_id_lower = sensor_id.lower()
        
        if hasattr(actual_data, 'raw_data'):
            return self._save_raw_sensor_data(sensor_id_lower, actual_data, sensor_path)
        elif isinstance(actual_data, np.ndarray):
            return self._save_numpy_data(sensor_id_lower, actual_data, sensor_path)
        elif isinstance(actual_data, dict):
            return self._save_dict_data(actual_data, sensor_path)
        elif hasattr(actual_data, '__dict__'):
            return self._save_object_data(sensor_id_lower, actual_data, sensor_path)
        else:
            # Fallback for unknown types
            try:
                data = {'value': str(actual_data)}
                filename = f"{self.frame_counter:04d}.json"
                filepath = os.path.join(sensor_path, filename)
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2)
                return filename
            except:
                return None
    
    def _save_raw_sensor_data(self, sensor_id_lower, data, sensor_path):
        """Save sensor data with raw_data attribute."""
        if 'rgb' in sensor_id_lower or 'bev' in sensor_id_lower:
            image_array = np.frombuffer(data.raw_data, dtype=np.uint8)
            image_array = image_array.reshape((data.height, data.width, 4))
            image_array = image_array[:, :, :3]  # Remove alpha
            
            image = Image.fromarray(image_array)
            filename = f"{self.frame_counter:04d}.png"
            filepath = os.path.join(sensor_path, filename)
            image.save(filepath)
            return filename
            
        elif 'semantic' in sensor_id_lower:
            image_array = np.frombuffer(data.raw_data, dtype=np.uint8)
            image_array = image_array.reshape((data.height, data.width, 4))
            
            semantic_image = image_array[:, :, 2]  # Red channel for tags
            image = Image.fromarray(semantic_image, mode='L')
            
            filename = f"{self.frame_counter:04d}.png"
            filepath = os.path.join(sensor_path, filename)
            image.save(filepath)
            return filename
            
        elif 'depth' in sensor_id_lower:
            image_array = np.frombuffer(data.raw_data, dtype=np.uint8)
            image_array = image_array.reshape((data.height, data.width, 4))
            
            # Convert BGRA to depth values
            normalized_depth = (image_array[:, :, 2] + 
                              image_array[:, :, 1] * 256.0 + 
                              image_array[:, :, 0] * 256.0 * 256.0) / (256.0 * 256.0 * 256.0 - 1.0)
            depth_meters = normalized_depth * 1000.0
            
            filename = f"{self.frame_counter:04d}.npy"
            filepath = os.path.join(sensor_path, filename)
            np.save(filepath, depth_meters)
            return filename
            
        elif 'lidar' in sensor_id_lower:
            points = np.frombuffer(data.raw_data, dtype=np.float32)
            points = points.reshape((-1, 4))  # x, y, z, intensity
            
            filename = f"{self.frame_counter:04d}.npy"
            filepath = os.path.join(sensor_path, filename)
            np.save(filepath, points, allow_pickle=True)
            return filename
        
        return None
    
    def _save_numpy_data(self, sensor_id_lower, data, sensor_path):
        """Save numpy array data."""
        if len(data.shape) == 3 and data.shape[2] in [3, 4]:
            if data.shape[2] == 4:
                data = data[:, :, :3]
            image = Image.fromarray(data)
            filename = f"{self.frame_counter:04d}.png"
            filepath = os.path.join(sensor_path, filename)
            image.save(filepath)
        else:
            filename = f"{self.frame_counter:04d}.npy"
            filepath = os.path.join(sensor_path, filename)
            np.save(filepath, data, allow_pickle=True)
        return filename
    
    def _save_dict_data(self, data, sensor_path):
        """Save dictionary data as JSON."""
        filename = f"{self.frame_counter:04d}.json"
        filepath = os.path.join(sensor_path, filename)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        return filename
    
    def _save_object_data(self, sensor_id_lower, data, sensor_path):
        """Save object data (GNSS, IMU, etc.)."""
        extracted_data = {}
        
        if 'gnss' in sensor_id_lower or 'gps' in sensor_id_lower:
            if hasattr(data, 'latitude'):
                extracted_data = {
                    'lat': data.latitude,
                    'lon': data.longitude,
                    'alt': getattr(data, 'altitude', 0.0)
                }
        elif 'imu' in sensor_id_lower:
            if hasattr(data, 'accelerometer'):
                extracted_data = {
                    'accelerometer': [data.accelerometer.x, 
                                    data.accelerometer.y, 
                                    data.accelerometer.z],
                    'gyroscope': [data.gyroscope.x, 
                                data.gyroscope.y, 
                                data.gyroscope.z],
                    'compass': data.compass
                }
        elif 'speed' in sensor_id_lower:
            if hasattr(data, 'speed'):
                extracted_data = {'speed': data.speed}
            else:
                extracted_data = {'speed': float(data)}
        
        if extracted_data:
            filename = f"{self.frame_counter:04d}.json"
            filepath = os.path.join(sensor_path, filename)
            with open(filepath, 'w') as f:
                json.dump(extracted_data, f, indent=2)
            return filename
        
        return None
    
    def _save_metadata(self):
        """Save metadata to disk."""
        try:
            metadata_path = os.path.join(self.save_path, 'metadata.json')
            with open(metadata_path, 'w') as f:
                metadata_copy = dict(self.metadata)
                # Limit frames in periodic saves
                if 'frames' in metadata_copy and len(metadata_copy['frames']) > 100:
                    metadata_copy['frames'] = metadata_copy['frames'][-100:]
                json.dump(metadata_copy, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save metadata: {e}")
    
    def destroy(self):
        """Clean up and save final metadata."""
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
        
        # Clean up model
        if hasattr(self, 'model'):
            del self.model