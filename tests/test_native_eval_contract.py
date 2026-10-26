"""CPU contract checks; no model allocation, Ray, renderer, or GPU."""
import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


evaluate = load_file('native_evaluate', ROOT / 'scripts/evaluate_native_ppo.py')
repack = load_file('native_repack', ROOT / 'third_party/RLinf/rlinf/models/embodiment/openpi/transforms/env/default.py')


def payload():
    return {'observation/image': np.arange(224 * 224 * 3, dtype=np.uint8).reshape(224, 224, 3),
            'observation/wrist_image': np.full((224, 224, 3), 51, np.uint8),
            'observation/state': np.arange(8, dtype=np.float64), 'prompt': 'turn on the stove'}


def test_actual_native_repack_preserves_prepared_pixels_state_prompt():
    source = payload()
    result = repack.repack_env_obs(evaluate.native_observation(source), select_state=lambda x: x)
    for key in ('observation/image', 'observation/wrist_image', 'observation/state'):
        np.testing.assert_array_equal(result[key][0], source[key])
    assert result['prompt'] == [source['prompt']]


def test_initial_contract_rejects_dtype_and_content_changes():
    state = np.arange(79, dtype=np.float64)
    expected = {'initial_state_sha256': hashlib.sha256(state.tobytes()).hexdigest(),
                'initial_state_dtype': 'float64', 'initial_state_shape': [79]}
    assert evaluate.check_initial_state(state, expected) == expected['initial_state_sha256']
    with pytest.raises(ValueError):
        evaluate.check_initial_state(state.astype(np.float32), expected)
    state[1] += 1
    with pytest.raises(ValueError):
        evaluate.check_initial_state(state, expected)


def test_policy_noise_resets_per_episode_and_increments_per_call():
    import torch

    class Model:
        device = 'cpu'

        def __init__(self):
            self.noises = []

        def predict_action_batch(self, obs, *, mode, compute_values, noise):
            assert mode == 'eval' and compute_values is False
            assert obs['states'].shape == (1, 8)
            self.noises.append(noise.clone())
            return torch.zeros((1, 5, 7)), {}

    model = Model()
    policy = evaluate.NativePolicy(model, {})
    policy.infer(payload())
    policy.infer(payload())
    evaluate.NativePolicy(model, {}).infer(payload())
    for index, seed in enumerate([42, 43, 42]):
        expected = np.random.default_rng(seed).standard_normal((10, 32)).astype(np.float32)
        np.testing.assert_array_equal(model.noises[index][0].numpy(), expected)


def test_rejects_unprepared_images():
    source = payload()
    source['observation/image'] = np.zeros((256, 256, 3), np.uint8)
    with pytest.raises(ValueError):
        evaluate.native_observation(source)


def test_actual_batched_transforms_preserve_layout_prompt_and_decode_chunk():
    import torch

    apply = load_file('native_apply', ROOT / 'third_party/RLinf/rlinf/models/embodiment/openpi/transforms/apply.py')
    source = payload()
    repacked = repack.repack_env_obs(evaluate.native_observation(source), select_state=lambda x: x)

    def input_transform(sample):
        assert sample['prompt'] == source['prompt']
        np.testing.assert_array_equal(sample['observation/image'], source['observation/image'])
        return {'image': sample['observation/image'], 'state': sample['observation/state']}

    processed = apply.apply_input_transform(input_transform, repacked, transpose=False)
    assert processed['image'].shape == (1, 224, 224, 3)
    np.testing.assert_array_equal(processed['state'][0].numpy(), source['observation/state'])
    raw = torch.arange(320, dtype=torch.float32).reshape(1, 10, 32)

    def output_transform(sample):
        # Stand-in for native Unnormalize + LIBERO selection; assert numpy boundary.
        assert isinstance(sample['actions'], np.ndarray)
        return {'actions': sample['actions'][:, :7] * 2 + 3}

    decoded = apply.apply_output_transform(
        output_transform, {'actions': raw, 'state': processed['state']}, action_chunk=5)
    torch.testing.assert_close(decoded['actions'], raw[:, :5, :7] * 2 + 3)
