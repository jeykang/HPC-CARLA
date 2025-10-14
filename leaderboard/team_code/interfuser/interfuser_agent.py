import os
import json
import datetime
import pathlib
import time
import imp
import cv2
import carla
from collections import deque

import torch
import carla
import numpy as np
from PIL import Image
from easydict import EasyDict

from torchvision import transforms
from leaderboard.autoagents import autonomous_agent
# MODIFIED: Removed create_model and now directly import the model function
from team_code.interfuser.interfuser.timm.models.interfuser import interfuser_baseline
from team_code.utils import lidar_to_histogram_features, transform_2d_points
from team_code.planner import RoutePlanner
from team_code.interfuser.interfuser_controller import InterfuserController
from team_code.tracker import Tracker

import math
import yaml

SAVE_PATH = os.environ.get("SAVE_PATH", None)
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def get_entry_point():
    return "InterfuserAgent"


class Resize2FixedSize:
    def __init__(self, size):
        self.size = size

    def __call__(self, pil_img):
        pil_img = pil_img.resize(self.size)
        return pil_img


def create_carla_rgb_transform(
    input_size, need_scale=True, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
):

    if isinstance(input_size, (tuple, list)):
        img_size = input_size[-2:]
    else:
        img_size = input_size
    tfl = []

    if isinstance(input_size, (tuple, list)):
        input_size_num = input_size[-1]
    else:
        input_size_num = input_size

    if need_scale:
        if input_size_num == 112:
            tfl.append(Resize2FixedSize((170, 128)))
        elif input_size_num == 128:
            tfl.append(Resize2FixedSize((195, 146)))
        elif input_size_num == 224:
            tfl.append(Resize2FixedSize((341, 256)))
        elif input_size_num == 256:
            tfl.append(Resize2FixedSize((288, 288)))
        else:
            raise ValueError("Can't find proper crop size")
    tfl.append(transforms.CenterCrop(img_size))
    tfl.append(transforms.ToTensor())
    tfl.append(transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)))

    return transforms.Compose(tfl)


class InterfuserAgent(autonomous_agent.AutonomousAgent):
    def setup(self, path_to_conf_file):
        self.lidar_processed = list()
        self.track = autonomous_agent.Track.SENSORS
        self.step = -1
        self.wall_start = time.time()
        self.initialized = False
        self.rgb_front_transform = create_carla_rgb_transform(224)
        self.rgb_left_transform = create_carla_rgb_transform(128)
        self.rgb_right_transform = create_carla_rgb_transform(128)
        self.rgb_center_transform = create_carla_rgb_transform(128, need_scale=False)

        self.tracker = Tracker()

        self.input_buffer = {
            "rgb": deque(),
            "rgb_left": deque(),
            "rgb_right": deque(),
            "rgb_rear": deque(),
            "lidar": deque(),
            "gps": deque(),
            "thetas": deque(),
        }

        self.config = imp.load_source("MainModel", path_to_conf_file).GlobalConfig()
        self.skip_frames = self.config.skip_frames
        self.controller = InterfuserController(self.config)
        
        # MODIFIED: Directly instantiate the model, bypassing timm.create_model
        self.net = interfuser_baseline()
        path_to_model_file = self.config.model_path
        print('load model: %s' % path_to_model_file)
        self.net.load_state_dict(torch.load(path_to_model_file)["state_dict"])
        self.net.cuda()
        self.net.eval()
            
        self.softmax = torch.nn.Softmax(dim=1)
        self.traffic_meta_moving_avg = np.zeros((400, 7))
        self.momentum = self.config.momentum
        self.prev_lidar = None
        self.prev_control = None

        self.save_path = None
        if SAVE_PATH is not None:
            now = datetime.datetime.now()
            string = pathlib.Path(os.environ["ROUTES"]).stem + "_"
            string += "_".join(
                map(
                    lambda x: "%02d" % x,
                    (now.month, now.day, now.hour, now.minute, now.second),
                )
            )

            print(string)

            self.save_path = pathlib.Path(SAVE_PATH)
            self.save_path.mkdir(parents=True, exist_ok=True)

            # NEW: Create sub-directories for data saving
            (self.save_path / "rgb").mkdir(exist_ok=True)
            (self.save_path / "rgb_left").mkdir(exist_ok=True)
            (self.save_path / "rgb_right").mkdir(exist_ok=True)
            (self.save_path / "lidar").mkdir(exist_ok=True)
            (self.save_path / "measurements").mkdir(exist_ok=True)


    def _init(self):
        self._route_planner = RoutePlanner(4.0, 50.0)
        self._route_planner.set_route(self._global_plan, True)
        self.initialized = True

    def _get_position(self, tick_data):
        gps = tick_data["gps"]
        gps = (gps - self._route_planner.mean) * self._route_planner.scale
        return gps

    def sensors(self):
        return [
            {
                "type": "sensor.camera.rgb", "x": 1.3, "y": 0.0, "z": 2.3,
                "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
                "width": 800, "height": 600, "fov": 100, "id": "rgb",
            },
            {
                "type": "sensor.camera.rgb", "x": 1.3, "y": 0.0, "z": 2.3,
                "roll": 0.0, "pitch": 0.0, "yaw": -60.0,
                "width": 400, "height": 300, "fov": 100, "id": "rgb_left",
            },
            {
                "type": "sensor.camera.rgb", "x": 1.3, "y": 0.0, "z": 2.3,
                "roll": 0.0, "pitch": 0.0, "yaw": 60.0,
                "width": 400, "height": 300, "fov": 100, "id": "rgb_right",
            },
            {
                "type": "sensor.lidar.ray_cast", "x": 1.3, "y": 0.0, "z": 2.5,
                "roll": 0.0, "pitch": 0.0, "yaw": -90.0, "id": "lidar",
            },
            {
                "type": "sensor.other.imu", "x": 0.0, "y": 0.0, "z": 0.0,
                "roll": 0.0, "pitch": 0.0, "yaw": 0.0, "sensor_tick": 0.05, "id": "imu",
            },
            {
                "type": "sensor.other.gnss", "x": 0.0, "y": 0.0, "z": 0.0,
                "roll": 0.0, "pitch": 0.0, "yaw": 0.0, "sensor_tick": 0.01, "id": "gps",
            },
            {"type": "sensor.speedometer", "reading_frequency": 20, "id": "speed"},
        ]

    def tick(self, input_data):
        result = {}
        # Get sensor data
        result["rgb"] = cv2.cvtColor(input_data["rgb"][1][:, :, :3], cv2.COLOR_BGR2RGB)
        result["rgb_left"] = cv2.cvtColor(input_data["rgb_left"][1][:, :, :3], cv2.COLOR_BGR2RGB)
        result["rgb_right"] = cv2.cvtColor(input_data["rgb_right"][1][:, :, :3], cv2.COLOR_BGR2RGB)
        result["gps"] = input_data["gps"][1][:2]
        result["speed"] = input_data["speed"][1]["speed"]
        result["compass"] = input_data["imu"][1][-1]
        
        if math.isnan(result["compass"]):
            result["compass"] = 0.0

        # Process LiDAR
        lidar_data = input_data['lidar'][1]
        result['raw_lidar'] = lidar_data
        pos = self._get_position(result)
        lidar_unprocessed = lidar_data[:, :3]
        lidar_unprocessed[:, 1] *= -1
        full_lidar = transform_2d_points(
            lidar_unprocessed, np.pi / 2 - result["compass"], -pos[0], -pos[1],
            np.pi / 2 - result["compass"], -pos[0], -pos[1],
        )
        lidar_processed = lidar_to_histogram_features(full_lidar, crop=224)
        if self.step % 2 == 0 or self.step < 4:
            self.prev_lidar = lidar_processed
        result["lidar"] = self.prev_lidar

        # Process route and command
        result["gps"] = pos
        next_wp, next_cmd = self._route_planner.run_step(pos)
        result["next_command"] = next_cmd.value
        
        theta = result["compass"] + np.pi / 2
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        local_command_point = np.array([next_wp[0] - pos[0], next_wp[1] - pos[1]])
        result["target_point"] = R.T.dot(local_command_point)

        return result

    @torch.no_grad()
    def run_step(self, input_data, timestamp):
        if not self.initialized:
            self._init()

        self.step += 1
        if self.step % self.skip_frames != 0 and self.step > 4:
            return self.prev_control

        tick_data = self.tick(input_data)
        velocity = tick_data["speed"]
        command = tick_data["next_command"]

        # Prepare model inputs
        rgb = self.rgb_front_transform(Image.fromarray(tick_data["rgb"])).unsqueeze(0).cuda().float()
        rgb_left = self.rgb_left_transform(Image.fromarray(tick_data["rgb_left"])).unsqueeze(0).cuda().float()
        rgb_right = self.rgb_right_transform(Image.fromarray(tick_data["rgb_right"])).unsqueeze(0).cuda().float()
        rgb_center = self.rgb_center_transform(Image.fromarray(tick_data["rgb"])).unsqueeze(0).cuda().float()

        cmd_one_hot = [0, 0, 0, 0, 0, 0]
        cmd_one_hot[command - 1] = 1
        cmd_one_hot.append(velocity)
        mes = torch.from_numpy(np.array(cmd_one_hot)).float().unsqueeze(0).cuda()

        model_input = {
            "rgb": rgb, "rgb_left": rgb_left, "rgb_right": rgb_right, "rgb_center": rgb_center,
            "measurements": mes,
            "target_point": torch.from_numpy(tick_data["target_point"]).float().cuda().view(1, -1),
            "lidar": torch.from_numpy(tick_data["lidar"]).float().cuda().unsqueeze(0)
        }
        
        # Run model
        traffic_meta, pred_waypoints, is_junction, traffic_light_state, stop_sign, _ = self.net(model_input)
        
        # Process model outputs
        traffic_meta = traffic_meta.detach().cpu().numpy()[0]
        pred_waypoints = pred_waypoints.detach().cpu().numpy()[0]
        is_junction = self.softmax(is_junction).detach().cpu().numpy().reshape(-1)[0]
        traffic_light_state = self.softmax(traffic_light_state).detach().cpu().numpy().reshape(-1)[0]
        stop_sign = self.softmax(stop_sign).detach().cpu().numpy().reshape(-1)[0]

        if self.step % 2 == 0 or self.step < 4:
            traffic_meta = self.tracker.update_and_predict(traffic_meta.reshape(20, 20, -1), tick_data['gps'], tick_data['compass'], self.step // 2)
            traffic_meta = traffic_meta.reshape(400, -1)
            self.traffic_meta_moving_avg = (self.momentum * self.traffic_meta_moving_avg + (1 - self.momentum) * traffic_meta)
        traffic_meta = self.traffic_meta_moving_avg

        # Get control commands
        steer, throttle, brake, meta_infos = self.controller.run_step(
            velocity, pred_waypoints, is_junction, traffic_light_state, stop_sign, traffic_meta
        )

        if brake < 0.05: brake = 0.0
        if brake > 0.1: throttle = 0.0

        control = carla.VehicleControl(steer=float(steer), throttle=float(throttle), brake=float(brake))

        if self.step % 2 != 0 and self.step > 4:
            control = self.prev_control
        else:
            self.prev_control = control

        # NEW: Save data if SAVE_PATH is defined
        if SAVE_PATH is not None and self.step % self.skip_frames == 0:
            self.save(tick_data, control, pred_waypoints, meta_infos)

        return control

    def save(self, tick_data, control, pred_waypoints, meta_infos):
        """
        NEW: This method saves all the necessary sensor and measurement data.
        """
        """frame = self.step // self.skip_frames

        # Save images
        Image.fromarray(tick_data['rgb']).save(self.save_path / 'rgb' / f'{frame:04d}.png')
        Image.fromarray(tick_data['rgb_left']).save(self.save_path / 'rgb_left' / f'{frame:04d}.png')
        Image.fromarray(tick_data['rgb_right']).save(self.save_path / 'rgb_right' / f'{frame:04d}.png')

        # Save LiDAR
        np.save(self.save_path / 'lidar' / f'{frame:04d}.npy', tick_data['raw_lidar'], allow_pickle=True)"""
        pass

    def destroy(self):
        del self.net
