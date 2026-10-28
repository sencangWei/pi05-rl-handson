"""Descriptive contact/lift events; these proxies do not establish force closure."""
import numpy as np


def runs(mask, minimum=5):
    """Return zero-based inclusive runs, without joining separate sequences."""
    result, start = [], None
    for i, flag in enumerate([*mask, False]):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if i - start >= minimum:
                result.append([start, i - 1])
            start = None
    return result


def summarize(rows, initial_bowl_z):
    if not rows:
        raise ValueError("Empty event sequence")
    grasp = np.array([r["grasp"] for r in rows], dtype=bool)
    lifted = np.array([r["bowl_pos"][2] - initial_bowl_z >= .03 for r in rows])
    # A close command or single-finger contact is deliberately insufficient.
    lift_runs = runs(grasp & lifted)
    return dict(
        controls=len(rows), grasp_controls=int(grasp.sum()),
        grasp_runs=runs(grasp), lifted_grasp_runs=lift_runs,
        lifted_contact_runs=runs([r["finger_contact"] and lift for r, lift in zip(rows, lifted)]),
        max_lift_m=max(r["bowl_pos"][2] - initial_bowl_z for r in rows),
        native_success_controls=sum(r["native_success"] for r in rows),
        released_goal_runs=runs([r["native_success"] and not r["finger_contact"] for r in rows]),
        min_bowl_plate_xy_m=min(float(np.linalg.norm(np.array(r["bowl_pos"][:2]) -
                                                   r["plate_pos"][:2])) for r in rows),
    )


def paired_divergence(parent, candidate):
    if any(not r["fidelity"]["passed"] for r in [*parent, *candidate]):
        raise ValueError("Replay fidelity failed; events cannot be compared")
    n = min(len(parent), len(candidate))
    distances = {k: np.linalg.norm(np.array([r[k] for r in parent[:n]]) -
                                  np.array([r[k] for r in candidate[:n]]), axis=1)
                 for k in ("eef_pos", "bowl_pos")}
    return {k: (found[0][0] + 1 if (found := runs(d > .01)) else None)
            for k, d in distances.items()}
