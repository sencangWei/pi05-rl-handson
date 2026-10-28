"""Inverse of robosuite 1.4.1 relative OSC goals, not joint-action relabeling."""
import numpy as np
from scipy.spatial.transform import Rotation


def pose_delta(before_pos, before_rot, target_pos, target_rot):
    """World translation and world axis-angle (left-multiplied rotation)."""
    return np.r_[np.asarray(target_pos) - before_pos,
                 Rotation.from_matrix(np.asarray(target_rot) @ np.asarray(before_rot).T).as_rotvec()]


def inverse_scale(delta, input_min, input_max, output_min, output_max, gripper):
    """Return bounded action and unclipped six-axis demand for explicit saturation audit."""
    delta = np.asarray(delta, dtype=float)
    lo, hi, out_lo, out_hi = [np.broadcast_to(np.asarray(x, dtype=float), (6,))
                              for x in (input_min, input_max, output_min, output_max)]
    if delta.shape != (6,) or not np.isfinite(np.r_[delta, lo, hi, out_lo, out_hi, gripper]).all():
        raise ValueError("OSC needs six finite pose deltas and a finite gripper")
    if np.any(hi <= lo) or np.any(out_hi <= out_lo) or not -1 <= gripper <= 1:
        raise ValueError("Invalid action ranges or gripper")
    demand = (delta - (out_hi + out_lo) / 2) / ((out_hi - out_lo) / (hi - lo)) + (hi + lo) / 2
    return np.r_[np.clip(demand, lo, hi), gripper], demand


def acceptance(rows, reference, limits, hold_steps):
    """Outcome and tracking are separate gates; timeout/collision cannot count as recovery."""
    if len(rows) != len(reference) or len(rows) <= hold_steps:
        raise ValueError("Incomplete paired execution")
    motion = rows[:-hold_steps]
    position = float(np.percentile([r["position_error_m"] for r in motion], 95))
    rotation = float(np.percentile([r["rotation_error_rad"] for r in motion], 95))
    released = all(r["native_goal"] and not r["finger_contact"] for r in rows[-hold_steps:])
    peak = max(r["cabinet_penetration_m"] for r in rows)
    reference_peak = max(r["cabinet_penetration_m"] for r in reference)
    # This is a measured proxy, not proof of force closure or global collision freedom.
    ref_grasp = any(r["grasp"] for r in reference[:-hold_steps])
    grasp = any(r["grasp"] for r in motion) or not ref_grasp
    gates = dict(released_goal_hold=released, grasp_observed=grasp,
                 position_tracking=position <= limits["position_p95_m"],
                 rotation_tracking=rotation <= limits["rotation_p95_rad"],
                 cabinet_contact=peak <= reference_peak + limits["cabinet_extra_penetration_m"])
    return dict(passed=all(gates.values()), gates=gates, position_p95_m=position,
                rotation_p95_rad=rotation, cabinet_peak_penetration_m=peak,
                reference_cabinet_peak_penetration_m=reference_peak)
