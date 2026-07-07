"""Minimal CILRS config.

Vendored (trimmed) from autonomousvision/transfuser @ cvpr2021 : cilrs/config.py.
Only the fields the network / inference path actually reads are kept; the training
dataset-path machinery from the original GlobalConfig is intentionally dropped
(it referenced a non-existent research-cluster path and is irrelevant to eval).

Original reference fields mirrored EXACTLY:
  seq_len=1, pred_len=0, ignore_sides=True, ignore_rear=True,
  input_resolution=256, scale=1, crop=256, max_throttle=0.75
(cilrs/config.py lines 5-26).
"""


class GlobalConfig:
    # Data / temporal
    seq_len = 1          # input timesteps (CILRS uses a single frame)
    pred_len = 0         # future waypoints predicted; not used by CILRS

    # Which cameras feed the encoder (defaults match the reference eval config)
    ignore_sides = True  # don't consider left/right cameras
    ignore_rear = True   # don't consider rear camera

    # Image pre-processing (see cilrs/data.py::scale_and_crop_image)
    input_resolution = 256
    scale = 1            # resize divisor before centre-crop (1 == no resize)
    crop = 256           # centre-crop side length

    # Control head
    max_throttle = 0.75  # upper limit on throttle signal value in dataset

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
