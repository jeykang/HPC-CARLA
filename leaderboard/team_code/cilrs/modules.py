"""CILRS-specific pipeline glue.

The heavy lifting (encoding, command-branch selection, control mapping) lives in
the vendored network (`team_code.cilrs.model.CILRSInterface`) and is executed by
the generic `team_code.pipeline_modules.TorchModelRunner`. The only CILRS-specific
post-processing left is turning the model's raw {steer, throttle, brake} tensors
into the final control dict, applying the exact brake-gating from the reference
agent.

No torch import here — scalars are pulled out with float() — so this module stays
importable off-cluster (though the offline validator only AST-parses it).
"""
from typing import Any, Dict


def _scalar(v):
    """Extract a python float from a torch tensor / numpy value / sequence / scalar.

    A CILRS control output is a (1,) torch tensor, for which float(v) works
    directly; the remaining branches are defensive fallbacks.
    """
    try:
        return float(v)
    except (TypeError, ValueError):
        pass
    item = getattr(v, "item", None)          # torch tensor / numpy scalar
    if callable(item):
        return float(v.item())
    try:
        return float(v[0])                    # any 1-element sequence
    except Exception:
        import numpy as np                    # last resort (available at runtime)
        return float(np.asarray(v).reshape(-1)[0])


class CILRSControl:
    """Map the CILRS model output dict -> {steer, throttle, brake} control dict.

    Faithful reproduction of CILRSAgent.run_step() post-processing
    (cvpr2021 leaderboard/team_code/cilrs_agent.py lines 205-215):

        steer    = steer.squeeze(0).item()
        throttle = throttle.squeeze(0).item()
        brake    = brake.squeeze(0).item()
        if brake < 0.05: brake = 0.0
        if throttle > brake: brake = 0.0

    Reads:
      - context[pred_key] : dict with keys steer/throttle/brake (each a (1,) tensor)
    Writes:
      - context[out_key]  : {steer, throttle, brake}
    """

    def __init__(
        self,
        pred_key: str = "cilrs_pred",
        out_key: str = "control",
        steer_key: str = "steer",
        throttle_key: str = "throttle",
        brake_key: str = "brake",
        brake_threshold: float = 0.05,
    ):
        self.pred_key = pred_key
        self.out_key = out_key
        self.steer_key = steer_key
        self.throttle_key = throttle_key
        self.brake_key = brake_key
        self.brake_threshold = float(brake_threshold)

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        pred = context[self.pred_key]

        steer = _scalar(pred[self.steer_key])
        throttle = _scalar(pred[self.throttle_key])
        brake = _scalar(pred[self.brake_key])

        # Reference gating (order matters, mirrors run_step lines 209-210).
        if brake < self.brake_threshold:
            brake = 0.0
        if throttle > brake:
            brake = 0.0

        context[self.out_key] = {"steer": steer, "throttle": throttle, "brake": brake}
        return context
