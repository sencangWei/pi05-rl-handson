"""Version-pinned, offline LIBERO/OSC snapshots; never execute Python pickle payloads.

Only restore archives produced locally by this project. Native MuJoCo data is a
version-specific binary, not an interchange format or a replacement for old logs.
Capture only at an OSC control boundary, outside a temporary joint jog.
"""
import hashlib
import importlib.metadata
import json
import random

import mujoco
import numpy as np

VERSIONS = {"mujoco": "3.8.1", "robosuite": "1.4.1", "numpy": "1.26.4"}
OBS_FIELDS = ("_time_since_last_sample", "_current_delay", "_current_observed_value", "_sampled")
BUFFERS = ("recent_qpos", "recent_actions", "recent_torques", "recent_ee_forcetorques",
           "recent_ee_pose", "recent_ee_vel", "recent_ee_vel_buffer", "recent_ee_acc")


def encode(value):
    if isinstance(value, np.ndarray):
        return {"array": value.tolist(), "dtype": value.dtype.str, "shape": list(value.shape)}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return {"tuple": [encode(x) for x in value]}
    if isinstance(value, list):
        return [encode(x) for x in value]
    if isinstance(value, dict):
        return {k: encode(v) for k, v in value.items()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported snapshot value: {type(value).__name__}")


def decode(value):
    if isinstance(value, list):
        return [decode(x) for x in value]
    if isinstance(value, dict):
        if set(value) == {"array", "dtype", "shape"}:
            return np.asarray(value["array"], dtype=value["dtype"]).reshape(value["shape"])
        if set(value) == {"tuple"}:
            return tuple(decode(x) for x in value["tuple"])
        return {k: decode(v) for k, v in value.items()}
    return value


def identity(env):
    versions = {k: importlib.metadata.version(k) for k in VERSIONS}
    if versions != VERSIONS:
        raise ValueError(f"Snapshot runtime mismatch: {versions}")
    robot = env.env.robots[0]
    if (len(env.env.robots) != 1 or robot.controller.name != "OSC_POSE" or
            robot.controller.interpolator_pos is not None or
            robot.controller.interpolator_ori is not None):
        raise ValueError("Only a single non-interpolated OSC_POSE boundary is supported")
    return dict(versions=versions,
                model_sha256=hashlib.sha256(env.sim.model.get_xml().encode()).hexdigest(),
                controller_type=type(robot.controller).__name__,
                control_timestep=env.env.control_timestep)


def capture(env, path):
    """No forward, observation refresh or controller update during capture."""
    base, robot = env.env, env.env.robots[0]
    contract = identity(env)
    # Every controller attribute except its live sim handle is data in the pinned
    # implementation. Reject new non-data attributes instead of silently omitting.
    controller = {k: v for k, v in vars(robot.controller).items() if k != "sim"}
    python = dict(controller=controller, gripper=robot.gripper.current_action,
                  torques=robot.torques,
                  buffers={k: vars(getattr(robot, k)) for k in BUFFERS},
                  counters={k: getattr(base, k) for k in ("cur_time", "timestep", "done", "_action_dim")},
                  obs_cache=base._obs_cache,
                  observables={k: {f: getattr(v, f) for f in OBS_FIELDS}
                               for k, v in base._observables.items()},
                  numpy_rng=np.random.get_state(), python_rng=random.getstate())
    data = np.frombuffer(env.sim.data._data.__getstate__(), dtype=np.uint8)
    sig = mujoco.mjtState.mjSTATE_INTEGRATION
    integration = np.empty(mujoco.mj_stateSize(env.sim.model._model, sig))
    mujoco.mj_getState(env.sim.model._model, env.sim.data._data, integration, sig)
    metadata = dict(format=1, identity=contract, native_sha256=hashlib.sha256(data).hexdigest(),
                    python=encode(python))
    text = json.dumps(metadata, allow_nan=False, separators=(",", ":"))
    # Exclusive create: an experiment must never overwrite its reference snapshot.
    with open(path, "xb") as f:
        np.savez_compressed(f, metadata=np.asarray(text), native=data,
                            integration=integration, legacy=env.get_sim_state().copy())


def restore(env, path, mode="full"):
    """Restore into existing handles without recomputing their derived phase."""
    if mode not in ("full", "legacy", "native_only", "integration_python"):
        raise ValueError(f"Unknown restore mode: {mode}")
    contract = identity(env)
    with np.load(path, allow_pickle=False) as z:
        metadata = json.loads(str(z["metadata"]))
        if metadata["format"] != 1 or metadata["identity"] != contract:
            raise ValueError("Snapshot model/runtime/controller identity mismatch")
        native = z["native"].tobytes()
        if hashlib.sha256(native).hexdigest() != metadata["native_sha256"]:
            raise ValueError("Snapshot native data checksum mismatch")
        state = decode(metadata["python"])
        if set(state["observables"]) != set(env.env._observables):
            raise ValueError("Snapshot observation schema mismatch")
        if mode == "legacy":
            env.regenerate_obs_from_state(z["legacy"])
            return
        if mode == "integration_python":
            mujoco.mj_setState(env.sim.model._model, env.sim.data._data,
                              z["integration"], mujoco.mjtState.mjSTATE_INTEGRATION)
            env.sim.forward()
        else:
            # __setstate__ must target an uninitialized object. Calling it on an
            # already initialized MjData silently retains that object's old data.
            restored = mujoco.MjData.__new__(mujoco.MjData)
            restored.__setstate__(native)
            mujoco.mj_copyData(env.sim.data._data, env.sim.model._model, restored)
    if mode == "native_only":
        return
    robot, base = env.env.robots[0], env.env
    if set(state["controller"]) != set(vars(robot.controller)) - {"sim"}:
        raise ValueError("Snapshot controller schema mismatch")
    for key, value in state["controller"].items():
        setattr(robot.controller, key, value)
    robot.gripper.current_action = state["gripper"]
    robot.torques = state["torques"]
    for key, value in state["buffers"].items():
        vars(getattr(robot, key)).update(value)
    for key, value in state["counters"].items():
        setattr(base, key, value)
    base._obs_cache = state["obs_cache"]
    for key, fields in state["observables"].items():
        for field, value in fields.items():
            setattr(base._observables[key], field, value)
    np.random.set_state(state["numpy_rng"])
    random.setstate(state["python_rng"])
