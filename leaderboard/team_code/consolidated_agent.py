import os
import sys
import yaml
import importlib
import json
import numpy as np
from PIL import Image
from pathlib import Path
from leaderboard.autoagents.autonomous_agent import AutonomousAgent

def get_entry_point():
    """Required by CARLA Leaderboard."""
    return "ConsolidatedAgent"

class ConsolidatedAgent(AutonomousAgent):
    """
    Universal agent wrapper that dynamically loads any CARLA 0.9.10 compatible agent
    from a YAML configuration file and automatically handles data collection.
    """
    
    def setup(self, path_to_config_yaml):
        """Load agent configuration and instantiate the specified agent."""
        print(f"ConsolidatedAgent: Loading configuration from {path_to_config_yaml}")
        
        with open(path_to_config_yaml, 'r') as f:
            self.agent_config = yaml.safe_load(f)

        agent_file_path = self.agent_config['agent_file']
        agent_class_name = self.agent_config['agent_class']
        agent_specific_config = self.agent_config['config_path']

        print(f"ConsolidatedAgent: Configuration loaded:")
        print(f"  Agent file: {agent_file_path}")
        print(f"  Agent class: {agent_class_name}")
        print(f"  Agent config: {agent_specific_config}")

        module_path = self._convert_path_to_module(agent_file_path)
        
        try:
            print(f"ConsolidatedAgent: Importing module '{module_path}'")
            agent_module = importlib.import_module(module_path)
        except ImportError as e:
            print(f"FATAL: Could not import module '{module_path}'.")
            print(f"  Error: {e}")
            print(f"  Original path: {agent_file_path}")
            print(f"  Check the 'agent_file' path in your YAML config")
            print(f"  Current Python path includes:")
            for p in sys.path[:5]:
                print(f"    - {p}")
            raise e

        if not hasattr(agent_module, agent_class_name):
            print(f"FATAL: Module '{module_path}' does not have class '{agent_class_name}'")
            print(f"  Available classes: {[x for x in dir(agent_module) if not x.startswith('_')]}")
            raise AttributeError(f"Class {agent_class_name} not found in {module_path}")
        
        agent_class = getattr(agent_module, agent_class_name)
        
        print(f"ConsolidatedAgent: Instantiating {agent_class_name}...")
        self.agent_instance = agent_class(agent_specific_config)

        print(f"ConsolidatedAgent: Setting up the loaded agent instance...")
        self.agent_instance.setup(agent_specific_config)

        print(f"ConsolidatedAgent: Successfully loaded agent '{agent_class_name}'")
        print(f"ConsolidatedAgent: Data saving is ENABLED")
        
        self._initialize_data_collection()

    def _convert_path_to_module(self, agent_file_path):
        """Convert file path to Python module path."""
        agent_file_path = os.path.normpath(agent_file_path)
        
        root_paths = [
            '/workspace/',
            os.environ.get('WORKSPACE_DIR', '/workspace') + '/',
            os.environ.get('PROJECT_ROOT', '') + '/',
            os.getcwd() + '/',
        ]
        
        module_path = agent_file_path
        
        for root in root_paths:
            if root and agent_file_path.startswith(root):
                module_path = agent_file_path[len(root):]
                break
        
        if module_path.endswith('.py'):
            module_path = module_path[:-3]
        
        module_path = module_path.replace('/', '.').replace('\\', '.')
        
        # Try common prefixes that might need removal
        prefixes_to_remove = ['leaderboard.team_code.', 'team_code.']
        
        for prefix in prefixes_to_remove:
            if module_path.startswith(prefix):
                test_path = module_path[len(prefix):]
                try:
                    importlib.util.find_spec(test_path)
                    module_path = test_path
                    break
                except (ImportError, ModuleNotFoundError):
                    pass
        
        return module_path

    def _initialize_data_collection(self):
        """Initialize data collection infrastructure."""
        default_save_path = os.path.join(
            os.environ.get('WORKSPACE_DIR', '/workspace'),
            'dataset',
            'default'
        )
        self.save_path = os.environ.get('SAVE_PATH', default_save_path)
        self.save_path = os.path.expandvars(self.save_path)
        
        self.frame_counter = 0
        self.sensor_data_paths = {}
        
        os.makedirs(self.save_path, exist_ok=True)
        
        self.metadata = {
            'agent_type': self.agent_config.get('agent_class', 'unknown'),
            'config': self.agent_config,
            'save_path': self.save_path,
            'timestamp': self._get_timestamp(),
            'frames': []
        }
        
        print(f"ConsolidatedAgent: Data collection initialized")
        print(f"  Save path: {self.save_path}")
        print(f"  Agent type: {self.metadata['agent_type']}")

    def _get_timestamp(self):
        """Get current timestamp in ISO format."""
        from datetime import datetime
        return datetime.now().isoformat()

    def sensors(self):
        """Get sensor configuration from the loaded agent and create save directories."""
        if not hasattr(self, 'agent_instance'):
            # Leaderboard calls sensors() before setup()
            config_path = os.environ.get('TEAM_CONFIG')
            if not config_path:
                raise RuntimeError("TEAM_CONFIG environment variable not set")
            self.setup(config_path)

        sensors = self.agent_instance.sensors()
        
        for sensor in sensors:
            sensor_type = sensor['type']
            sensor_id = sensor['id']
            sensor_id_lower = sensor_id.lower()
            
            folder_name = self._get_sensor_folder_name(sensor_type, sensor_id_lower)
            sensor_path = os.path.join(self.save_path, folder_name)
            os.makedirs(sensor_path, exist_ok=True)
            self.sensor_data_paths[sensor_id] = sensor_path
            
        print(f"ConsolidatedAgent: Created directories for {len(sensors)} sensors")
        return sensors

    def _get_sensor_folder_name(self, sensor_type, sensor_id_lower):
        """Generate folder name for sensor data."""
        if 'camera.rgb' in sensor_type:
            return sensor_id_lower
        elif 'camera.semantic' in sensor_type:
            return f"semantic_{sensor_id_lower}"
        elif 'camera.depth' in sensor_type:
            return f"depth_{sensor_id_lower}"
        elif 'lidar' in sensor_type:
            return "lidar"
        elif 'radar' in sensor_type:
            return f"radar_{sensor_id_lower}"
        elif 'imu' in sensor_type:
            return "imu"
        elif 'gnss' in sensor_type:
            return "gps"
        elif 'speedometer' in sensor_type:
            return "speed"
        else:
            return sensor_id_lower.replace(' ', '_')

    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        """Set global plan for both wrapper and loaded agent."""
        super(ConsolidatedAgent, self).set_global_plan(global_plan_gps, global_plan_world_coord)

        # Save plan to metadata (handle both dict and tuple formats)
        if hasattr(self, 'metadata'):
            try:
                gps_list = []
                for waypoint in global_plan_gps:
                    if isinstance(waypoint, dict):
                        gps_list.append((float(waypoint['lat']), float(waypoint['lon'])))
                    elif isinstance(waypoint, (tuple, list)):
                        gps_list.append((float(waypoint[0]), float(waypoint[1])))
                    else:
                        continue
                
                world_list = []
                for coord in global_plan_world_coord:
                    if isinstance(coord, dict):
                        x = float(coord.get('x', 0))
                        y = float(coord.get('y', 0))
                        z = float(coord.get('z', 0))
                        world_list.append((x, y, z))
                    elif isinstance(coord, (tuple, list)):
                        if len(coord) >= 3:
                            world_list.append((float(coord[0]), float(coord[1]), float(coord[2])))
                        elif len(coord) == 2:
                            world_list.append((float(coord[0]), float(coord[1])))
                    else:
                        continue
                
                self.metadata['global_plan'] = {
                    'gps': gps_list,
                    'world_coord': world_list
                }
                self._save_metadata()
            except Exception as e:
                print(f"Warning: Failed to save global plan to metadata: {e}")

        if hasattr(self, 'agent_instance') and self.agent_instance:
            self.agent_instance.set_global_plan(global_plan_gps, global_plan_world_coord)

    def _save_metadata(self):
        """Save metadata to disk."""
        try:
            metadata_path = os.path.join(self.save_path, 'metadata.json')
            with open(metadata_path, 'w') as f:
                metadata_copy = dict(self.metadata)
                # Limit frames in periodic saves to avoid huge files
                if 'frames' in metadata_copy and len(metadata_copy['frames']) > 100:
                    metadata_copy['frames'] = metadata_copy['frames'][-100:]
                json.dump(metadata_copy, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save metadata: {e}")

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

    def _process_and_save_sensor(self, sensor_id, sensor_data, sensor_path):
        """Process and save individual sensor data."""
        # Extract actual data from CARLA tuple format
        if isinstance(sensor_data, tuple) and len(sensor_data) == 2:
            actual_timestamp, actual_data = sensor_data
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
        """Save sensor data with raw_data attribute (cameras, lidar)."""
        if 'rgb' in sensor_id_lower or 'tel_rgb' in sensor_id_lower or 'bev' in sensor_id_lower:
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

    def run_step(self, input_data, timestamp):
        """Execute one step: save data then run agent."""
        # Save data before running agent (ensures capture even if agent crashes)
        try:
            self._save_sensor_data(input_data, timestamp)
        except Exception as e:
            print(f"Warning: Data saving failed for frame {self.frame_counter}: {e}")
        
        self.frame_counter += 1
        return self.agent_instance.run_step(input_data, timestamp)

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
            print(f"  Metadata saved: {metadata_path}")
        except Exception as e:
            print(f"Warning: Failed to save final metadata: {e}")
        
        if hasattr(self, 'agent_instance') and self.agent_instance:
            try:
                self.agent_instance.destroy()
            except Exception as e:
                print(f"Warning: Agent cleanup failed: {e}")
