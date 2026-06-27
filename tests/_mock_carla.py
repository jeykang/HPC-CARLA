"""Permissive stand-in for the `carla` module so agent code imports off-cluster.

Installed into sys.modules by conftest when the real CARLA egg isn't importable
(i.e. outside the container). VehicleControl is real enough to hold/inspect the
control fields; everything else resolves to a no-op stub.
"""


class VehicleControl:
    def __init__(self, throttle=0.0, steer=0.0, brake=0.0, hand_brake=False,
                 reverse=False, manual_gear_shift=False, gear=0):
        self.throttle = float(throttle)
        self.steer = float(steer)
        self.brake = float(brake)
        self.hand_brake = hand_brake
        self.reverse = reverse
        self.manual_gear_shift = manual_gear_shift
        self.gear = gear

    def __repr__(self):
        return f"VehicleControl(throttle={self.throttle}, steer={self.steer}, brake={self.brake})"


class _Stub:
    """Catch-all: constructible, callable, and attribute-accessible."""
    def __init__(self, *a, **k):
        pass

    def __call__(self, *a, **k):
        return _Stub()

    def __getattr__(self, name):
        return _Stub()


def __getattr__(name):  # PEP 562: any other carla.X resolves to a stub class
    return _Stub
