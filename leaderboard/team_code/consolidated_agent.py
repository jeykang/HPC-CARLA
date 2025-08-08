import os
import yaml
import importlib
import importlib.util
import json
import numpy as np
from PIL import Image
from pathlib import Path
from leaderboard.autoagents.autonomous_agent import AutonomousAgent

def get_entry_point():
    return "ConsolidatedAgent"

class ConsolidatedAgent(AutonomousAgent):
    """Generic loader + automatic data collection for CARLA 0.9.10 agents."""

    def setup(self, path_to_config_yaml):
        print(f"ConsolidatedAgent: Loading configuration from {path_to_config_yaml}")
        with open(path_to_config_yaml, 'r') as f:
            self.agent_config = yaml.safe_load(f)

        agent_file_path = str(self.agent_config['agent_file'])
        agent_class_name = self.agent_config['agent_class']
        agent_specific_config = self.agent_config['config_path']

        print("ConsolidatedAgent: Configuration loaded:")
        print(f"  Agent file: {agent_file_path}")
        print(f"  Agent class: {agent_class_name}")
        print(f"  Agent config: {agent_specific_config}")

        module = self._load_agent_module(agent_file_path)
        if not hasattr(module, agent_class_name):
            available = [x for x in dir(module) if not x.startswith('_')]
            raise AttributeError(f"Class {agent_class_name} not found in loaded module. Available: {available}")

        agent_class = getattr(module, agent_class_name)
        print(f"ConsolidatedAgent: Instantiating {agent_class_name}...")
        self.agent_instance = agent_class(agent_specific_config)
        print("ConsolidatedAgent: Setting up the loaded agent instance...")
        self.agent_instance.setup(agent_specific_config)

        print(f"ConsolidatedAgent: Successfully loaded agent '{agent_class_name}'")
        print("ConsolidatedAgent: Data saving is ENABLED (automatic for all sensors)")

        self._initialize_data_collection()

    def _load_agent_module(self, agent_file_path: str):
        p = Path(os.path.expandvars(os.path.expanduser(agent_file_path)))
        if p.exists() and p.is_file():
            spec = importlib.util.spec_from_file_location(p.stem, str(p))
            if spec is None or spec.loader is None:
                raise ImportError(f"Failed to load module from file: {p}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            print(f"ConsolidatedAgent: Imported from file path '{p}'")
            return module
        # Treat as module path (e.g., "leaderboard.team_code.my_agent")
        print(f"ConsolidatedAgent: Importing module '{agent_file_path}'")
        return importlib.import_module(agent_file_path)

    def _initialize_data_collection(self):
        dataset_root = os.environ.get('DATASET_ROOT', None)
        cfg_save = self.agent_config.get('save_path')
        default_root = str(Path.cwd() / 'dataset')
        fallback = os.path.join(dataset_root or default_root, 'default')
        self.save_path = os.environ.get('SAVE_PATH', cfg_save or fallback)

        self.frame_counter = 0
        self.sensor_data_paths = {}
        Path(self.save_path).mkdir(parents=True, exist_ok=True)
        self.metadata = {
            'agent_type': self.agent_config.get('agent_class', 'unknown'),
            'config': self.agent_config,
            'frames': []
        }
        print(f"ConsolidatedAgent: Data collection initialized. Save path: {self.save_path}")

    def sensors(self):
        if not hasattr(self, 'agent_instance'):
            self.setup(os.environ.get('TEAM_CONFIG'))

        sensors = self.agent_instance.sensors()
        for sensor in sensors:
            sensor_type = sensor['type']
            sensor_id = sensor['id']
            sensor_id_lower = sensor_id.lower()

            if 'camera.rgb' in sensor_type:
                folder_name = sensor_id_lower
            elif 'camera.semantic' in sensor_type:
                folder_name = f"semantic_{sensor_id_lower}"
            elif 'camera.depth' in sensor_type:
                folder_name = f"depth_{sensor_id_lower}"
            elif 'lidar' in sensor_type:
                folder_name = "lidar"
            elif 'radar' in sensor_type:
                folder_name = f"radar_{sensor_id_lower}"
            elif 'imu' in sensor_type:
                folder_name = "imu"
            elif 'gnss' in sensor_type:
                folder_name = "gps"
            elif 'speedometer' in sensor_type:
                folder_name = "speed"
            else:
                folder_name = sensor_id_lower.replace(' ', '_')

            sensor_path = os.path.join(self.save_path, folder_name)
            os.makedirs(sensor_path, exist_ok=True)
            self.sensor_data_paths[sensor_id] = sensor_path

        return sensors

    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        super(ConsolidatedAgent, self).set_global_plan(global_plan_gps, global_plan_world_coord)
        if hasattr(self, 'agent_instance') and self.agent_instance:
            self.agent_instance.set_global_plan(global_plan_gps, global_plan_world_coord)

    def _save_sensor_data(self, input_data, timestamp):
        frame_data = {'frame': self.frame_counter, 'timestamp': timestamp, 'sensors': {}}

        for sensor_id, sensor_data in input_data.items():
            sensor_path = self.sensor_data_paths.get(sensor_id)
            if sensor_path is None:
                continue

            try:
                actual_data = sensor_data[1] if isinstance(sensor_data, tuple) and len(sensor_data) == 2 else sensor_data
                sensor_id_lower = sensor_id.lower()

                if hasattr(actual_data, 'raw_data'):
                    if 'rgb' in sensor_id_lower or 'tel_rgb' in sensor_id_lower or 'bev' in sensor_id_lower:
                        image_array = np.frombuffer(actual_data.raw_data, dtype=np.uint8).reshape((actual_data.height, actual_data.width, 4))[:, :, :3]
                        filename = f"{self.frame_counter:04d}.png"
                        Image.fromarray(image_array).save(os.path.join(sensor_path, filename))
                        frame_data['sensors'][sensor_id] = filename

                    elif 'semantic' in sensor_id_lower:
                        image_array = np.frombuffer(actual_data.raw_data, dtype=np.uint8).reshape((actual_data.height, actual_data.width, 4))
                        semantic_image = image_array[:, :, 2]
                        filename = f"{self.frame_counter:04d}.png"
                        Image.fromarray(semantic_image, mode='L').save(os.path.join(sensor_path, filename))
                        frame_data['sensors'][sensor_id] = filename

                    elif 'depth' in sensor_id_lower:
                        image_array = np.frombuffer(actual_data.raw_data, dtype=np.uint8).reshape((actual_data.height, actual_data.width, 4))
                        normalized_depth = (
                            image_array[:, :, 2] +
                            image_array[:, :, 1] * 256.0 +
                            image_array[:, :, 0] * 256.0 * 256.0
                        ) / (256.0 * 256.0 * 256.0 - 1.0)
                        depth_meters = normalized_depth * 1000.0
                        filename = f"{self.frame_counter:04d}.npy"
                        np.save(os.path.join(sensor_path, filename), depth_meters)
                        frame_data['sensors'][sensor_id] = filename

                    elif 'lidar' in sensor_id_lower:
                        points = np.frombuffer(actual_data.raw_data, dtype=np.float32).reshape((-1, 4))
                        filename = f"{self.frame_counter:04d}.npy"
                        np.save(os.path.join(sensor_path, filename), points, allow_pickle=True)
                        frame_data['sensors'][sensor_id] = filename

                elif isinstance(actual_data, np.ndarray):
                    if len(actual_data.shape) == 3 and actual_data.shape[2] in [3, 4]:
                        if actual_data.shape[2] == 4:
                            actual_data = actual_data[:, :, :3]
                        filename = f"{self.frame_counter:04d}.png"
                        Image.fromarray(actual_data).save(os.path.join(sensor_path, filename))
                        frame_data['sensors'][sensor_id] = filename
                    else:
                        filename = f"{self.frame_counter:04d}.npy"
                        np.save(os.path.join(sensor_path, filename), actual_data, allow_pickle=True)
                        frame_data['sensors'][sensor_id] = filename

                elif isinstance(actual_data, dict):
                    filename = f"{self.frame_counter:04d}.json"
                    with open(os.path.join(sensor_path, filename), 'w') as f:
                        json.dump(actual_data, f, indent=2)
                    frame_data['sensors'][sensor_id] = filename

                elif hasattr(actual_data, '__dict__'):
                    data = {}
                    if 'gnss' in sensor_id_lower or 'gps' in sensor_id_lower:
                        if hasattr(actual_data, 'latitude'):
                            data = {'lat': actual_data.latitude, 'lon': actual_data.longitude, 'alt': actual_data.altitude}
                        elif isinstance(actual_data, (list, np.ndarray)):
                            data = {'lat': actual_data[0], 'lon': actual_data[1]}
                            if len(actual_data) > 2:
                                data['alt'] = actual_data[2]
                    elif 'imu' in sensor_id_lower:
                        if hasattr(actual_data, 'accelerometer'):
                            data = {
                                'accelerometer': [actual_data.accelerometer.x, actual_data.accelerometer.y, actual_data.accelerometer.z],
                                'gyroscope': [actual_data.gyroscope.x, actual_data.gyroscope.y, actual_data.gyroscope.z],
                                'compass': actual_data.compass
                            }
                        elif isinstance(actual_data, (list, np.ndarray)):
                            data = {'compass': actual_data[-1]}
                    elif 'speed' in sensor_id_lower:
                        if hasattr(actual_data, 'speed'):
                            data = {'speed': actual_data.speed}
                        elif isinstance(actual_data, dict) and 'speed' in actual_data:
                            data = actual_data
                        else:
                            data = {'speed': float(actual_data)}
                    else:
                        if hasattr(actual_data, '__dict__'):
                            data = {k: v for k, v in actual_data.__dict__.items()
                                    if not k.startswith('_') and isinstance(v, (int, float, str, list, dict))}

                    if data:
                        filename = f"{self.frame_counter:04d}.json"
                        with open(os.path.join(sensor_path, filename), 'w') as f:
                            json.dump(data, f, indent=2)
                        frame_data['sensors'][sensor_id] = filename

            except Exception as e:
                print(f"Warning: Failed to save data for sensor {sensor_id}: {e}")

        self.metadata['frames'].append(frame_data)

        if self.frame_counter % 10 == 0 or self.frame_counter == 0:
            metadata_path = os.path.join(self.save_path, 'metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(self.metadata, f, indent=2)

    def run_step(self, input_data, timestamp):
        try:
            self._save_sensor_data(input_data, timestamp)
        except Exception as e:
            print(f"Warning: Data saving failed for frame {self.frame_counter}: {e}")

        self.frame_counter += 1
        return self.agent_instance.run_step(input_data, timestamp)

    def destroy(self):
        try:
            metadata_path = os.path.join(self.save_path, 'metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(self.metadata, f, indent=2)
            print(f"ConsolidatedAgent: Data collection complete. {self.frame_counter} frames saved to {self.save_path}")
        except Exception as e:
            print(f"Warning: Failed to save final metadata: {e}")

        if hasattr(self, 'agent_instance') and self.agent_instance:
            self.agent_instance.destroy()
