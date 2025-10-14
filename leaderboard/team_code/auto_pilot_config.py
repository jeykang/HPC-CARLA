# auto_pilot_config.py
import os

class GlobalConfig:
    """Configuration for auto_pilot data collection"""
    
    # Data collection settings
    rgb_only = False  # CRITICAL: Set to False to collect all sensor data
    save_skip_frames = 10
    waypoint_disturb = 0
    waypoint_disturb_seed = 2021
    destory_hazard_actors = True
    
    # Weather settings (None = use scenario weather)
    weather = None
    
    # You can also include the controller settings from interfuser_config.py
    turn_KP = 1.25
    turn_KI = 0.75
    turn_KD = 0.3
    turn_n = 40
    
    speed_KP = 5.0
    speed_KI = 0.5
    speed_KD = 1.0
    speed_n = 40
    
    max_throttle = 0.75
    brake_speed = 0.1
    brake_ratio = 1.1
    clip_delta = 0.35
    
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

# Create a dictionary that BaseAgent can use
config = {
    'rgb_only': False,
    'save_skip_frames': 10,
    'waypoint_disturb': 0,
    'waypoint_disturb_seed': 2021,
    'destory_hazard_actors': True,
    'weather': None
}