"""Record real learned-policy training rollouts; no synthetic action labels."""
from io import BytesIO
import json
from pathlib import Path

import numpy as np

from export_sft_for_ppo import sha256


def validate_training_entry(entry):
    if entry["split"] != "train" or entry["source_initial_id"] not in range(1, 7):
        raise ValueError("Only frozen training source IDs 1..6 may be recorded")


class TrainingRecorder:
    def __init__(self, directory, entry, episode, prompt):
        validate_training_entry(entry)
        self.directory = Path(directory)
        self.entry, self.episode, self.prompt = entry, episode, prompt
        self.rows, self.physics, self.actions = [], [], []

    def add(self, obs, action, physics, state):
        from PIL import Image
        state, action = np.asarray(state), np.asarray(action)
        if (state.shape != (8,) or action.shape != (7,) or not np.isfinite(state).all()
                or not np.isfinite(action).all()):
            raise ValueError("Invalid pre-action state or OSC action")
        # Robosuite clips OSC inputs and Panda uses sign(gripper). Preserve the raw
        # command for exact replay; bounded labels have the same actuator effect.
        row = dict(state=state.tolist(), actions=np.clip(action, -1, 1).tolist(), episode_index=self.episode,
                   task_index=8, frame_index=len(self.rows))
        for key, source in (("image", "agentview_image"), ("wrist_image", "robot0_eye_in_hand_image")):
            image = np.ascontiguousarray(obs[source][::-1, ::-1])
            if image.shape != (256, 256, 3) or image.dtype != np.uint8:
                raise ValueError("Invalid pre-action RGB")
            encoded = BytesIO()
            Image.fromarray(image).save(encoded, format="PNG")
            row[key] = {"bytes": encoded.getvalue(), "path": None}
        self.rows.append(row)
        self.physics.append(np.asarray(physics).copy())
        self.actions.append(action.copy())

    def save(self, final_physics):
        import pyarrow as pa
        import pyarrow.parquet as pq
        path = self.directory / (self.entry["id"] + ".parquet")
        pq.write_table(pa.Table.from_pylist(self.rows), path)
        physics = self.directory / (self.entry["id"] + "-physics.npz")
        np.savez_compressed(physics, states=np.asarray(self.physics + [final_physics]),
                            actions=np.asarray(self.actions))
        record = dict(path=path.name, sha256=sha256(path), frames=len(self.rows),
                      episode=self.episode, task_id=8, prompt=self.prompt, split="train",
                      entry=self.entry, physics=physics.name, physics_sha256=sha256(physics),
                      provenance="learned-policy autonomous successful rollout, NOT human demonstration",
                      temporal_policy="all original pre-action RGB/state and executed actions at 20Hz",
                      action_labels="OSC clipped to [-1,1], Panda sign preserved; NPZ retains raw model commands",
                      recorder_sha256=sha256(Path(__file__)),
                      independent_replay_verified=False)
        path.with_suffix(".record.json").write_text(json.dumps(record, indent=2) + "\n")
