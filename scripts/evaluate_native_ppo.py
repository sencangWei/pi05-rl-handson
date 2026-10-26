"""Evaluate native RLinf SFT/PPO weights with the existing eight-episode simulator."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
APP = Path('/home/robot/voice-arm-sim')
APP_REVISION = 'b0334d6424d85de642ac06248ddfe2009c36ee45'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def native_observation(payload):
    """Payload images are already rotated/resized; native repacking only renames."""
    images = [np.asarray(payload[key]) for key in
              ('observation/image', 'observation/wrist_image')]
    if any(image.shape != (224, 224, 3) or image.dtype != np.uint8 for image in images):
        raise ValueError('Expected prepared uint8 224x224 RGB images')
    state = np.asarray(payload['observation/state'])
    if state.shape != (8,) or not np.isfinite(state).all():
        raise ValueError('Expected finite LIBERO state of shape (8,)')
    return {'main_images': images[0][None], 'wrist_images': images[1][None],
            'states': state[None], 'task_descriptions': [payload['prompt']]}


def check_initial_state(initial, expected):
    initial = np.asarray(initial)
    actual = hashlib.sha256(initial.tobytes()).hexdigest()
    if (actual != expected['initial_state_sha256']
            or str(initial.dtype) != expected['initial_state_dtype']
            or list(initial.shape) != expected['initial_state_shape']):
        raise ValueError('Initial state differs from recorded SFT episode')
    return actual


class NativePolicy:
    _ws = None

    def __init__(self, model, metadata):
        self.model, self.metadata, self.index = model, metadata, 0

    def get_server_metadata(self):
        return self.metadata

    def infer(self, payload):
        import torch
        noise = np.random.default_rng(42 + self.index).standard_normal((10, 32))
        self.index += 1
        noise = torch.as_tensor(noise, dtype=torch.float32, device=self.model.device)[None]
        actions, _ = self.model.predict_action_batch(
            native_observation(payload), mode='eval', compute_values=False, noise=noise)
        actions = actions.detach().float().cpu().numpy()[0]
        if actions.shape != (5, 7) or not np.isfinite(actions).all():
            raise ValueError(f'Unexpected decoded native actions: {actions.shape}')
        return {'actions': actions}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT / 'evidence/ppo-composed.json')
    parser.add_argument('--tasks', type=int, nargs='+', default=[0, 7, 8, 9])
    parser.add_argument('--initial-states', type=int, nargs='+', default=[0, 10])
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Refusing nonempty output directory')
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=APP, text=True).strip() != APP_REVISION:
        raise ValueError('Simulator source revision changed')
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MUJOCO_GL', 'egl')
    os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
    os.environ.setdefault('ROBOT_PLATFORM', 'LIBERO')
    os.environ.setdefault('LIBERO_TYPE', 'standard')
    os.environ.setdefault('NUMBA_CACHE_DIR', str(ROOT / '.local/numba'))
    sys.path[:0] = [str(APP), str(ROOT / 'third_party/RLinf'), str(ROOT / 'scripts')]
    from baseline import configure_libero
    configure_libero()
    from vla import run
    run.configure_libero = configure_libero
    from libero.libero import benchmark
    suite = benchmark.get_benchmark_dict()['libero_goal']()
    reference = json.loads((ROOT / 'artifacts/20261026-sft-simulation/sft-summary.json').read_text())
    reference = {(x['task_id'], x['initial_state']): x for x in reference}
    state_hashes = {}
    for task in args.tasks:
        for initial in args.initial_states:
            expected = reference[(task, initial)]  # Fail closed outside recorded contract.
            states = run.load_initial_states(suite.get_task(task))
            state_hashes[f'{task}:{initial}'] = check_initial_state(states[initial], expected)

    import torch
    from omegaconf import OmegaConf
    from rlinf.models import get_model
    cfg = OmegaConf.load(args.config)
    model_cfg = OmegaConf.merge(cfg.actor.model, cfg.rollout.model)
    if (model_cfg.openpi.action_horizon != 10 or model_cfg.openpi.discrete_state_input
            or not model_cfg.is_lora or model_cfg.lora_target_scope != 'all_linear'
            or model_cfg.precision != 'bf16'):
        raise ValueError('Unexpected native rollout model contract')
    model_cfg.num_action_chunks = 5
    model_cfg.openpi.action_chunk = 5
    # Match the original JAX evaluation sampler, not PPO's five-step rollout.
    model_cfg.num_steps = 10
    model_cfg.openpi.num_steps = 10
    torch.manual_seed(42)
    model = get_model(model_cfg)  # Factory injects PEFT and critic before strict native load.
    base_sha256 = sha256(Path(model_cfg.model_path) / 'model.safetensors')
    load_result = None
    adapter_update = None
    checkpoint_format = None
    if args.checkpoint:
        # Local training artifacts include Python/NumPy RNG for exact resume.
        saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
        if saved.get('format') == 'rlinf-fsdp1-single-rank-adapter-v1':
            from rlinf.hybrid_engines.fsdp.strategy.adapter_checkpoint import (
                read_adapter_model,
            )
            read_adapter_model(model, saved, base_sha256, allow_dtype_cast=True)
            checkpoint_format = saved['format']
            weights = saved['model']
        else:
            weights = saved
            load_result = model.load_state_dict(weights, strict=True)
            checkpoint_format = 'full-state-dict'
        adapter_names = [name for name, _ in model.named_parameters() if '.lora_B.' in name]
        if not adapter_names or not all(torch.isfinite(weights[name]).all() for name in adapter_names):
            raise ValueError('Missing or nonfinite native LoRA B tensors')
        adapter_update = {
            'lora_b_tensors': len(adapter_names),
            'nonzero_lora_b_tensors': sum(bool(weights[name].count_nonzero()) for name in adapter_names),
            'all_lora_b_finite': True,
        }
        del weights, saved
    model = model.to('cuda').eval()
    if model.num_steps != 10 or model.action_horizon != 10 or model.action_chunk != 5:
        raise ValueError('Unexpected native sampler configuration')
    source_files = [Path(__file__).resolve(), APP / 'vla/run.py']
    source_files += sorted((ROOT / 'third_party/RLinf/rlinf/models/embodiment/openpi').rglob('*.py'))
    source_files.append(ROOT / 'third_party/RLinf/rlinf/models/__init__.py')
    metadata = {
        'backend': 'local-pi05-libero-lora', 'runtime': 'native-rlinf-pytorch',
        'checkpoint': str(args.checkpoint.resolve()) if args.checkpoint else None,
        'checkpoint_sha256': sha256(args.checkpoint) if args.checkpoint else None,
        'checkpoint_format': checkpoint_format,
        'base_weights_sha256': base_sha256,
        'norm_stats_sha256': sha256(model_cfg.openpi_data.norm_stats_path),
        'image_size': 224, 'action_horizon': 10, 'action_chunk': 5,
        'num_steps': 10, 'discrete_state_input': False, 'precision': 'bf16',
        'seed_protocol': 'numpy.default_rng(42+call).standard_normal((10,32))',
        'adapter_update': adapter_update,
        'adapter_contract_validated': checkpoint_format == 'rlinf-fsdp1-single-rank-adapter-v1',
        'strict_load': None if load_result is None else {
            'missing_keys': load_result.missing_keys, 'unexpected_keys': load_result.unexpected_keys},
    }
    identity = {'metadata': metadata, 'initial_state_sha256': state_hashes,
                'model_config': OmegaConf.to_container(model_cfg, resolve=True),
                'source_sha256': {str(p): sha256(p) for p in source_files},
                'config_sha256': sha256(args.config), 'simulator_revision': APP_REVISION}
    (args.output / 'identity.json').write_text(json.dumps(identity, indent=2) + '\n')
    reports = []
    for task in args.tasks:
        for initial in args.initial_states:
            options = run.parse_args(['--suite', 'libero_goal', '--task-id', str(task),
                '--initial-state', str(initial), '--output',
                str(args.output / f'goal-{task:02d}-init-{initial:02d}'),
                '--seed', '42', '--max-steps', '300', '--replan-steps', '5'])
            report = run.rollout(options, policy_factory=lambda port: NativePolicy(model, metadata))
            if ('initial_state_sha256' in report
                    and report['initial_state_sha256'] != state_hashes[f'{task}:{initial}']):
                raise ValueError('Rollout initial-state hash changed after preflight')
            report['reward_return'] = float(report['result'] == 'success') - .001 * report['steps']
            reports.append(report)
            (args.output / 'summary.json').write_text(json.dumps(reports, indent=2) + '\n')
            print(json.dumps(report), flush=True)
    return int(any(r['result'] == 'exception' for r in reports))


if __name__ == '__main__':
    raise SystemExit(main())
