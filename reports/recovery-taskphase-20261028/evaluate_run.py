"""Run only the frozen development evaluation after successful bounded training."""
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time

ROOT = Path('/home/robot/pi05-rl-handson')
os.chdir(ROOT)


def properties(unit):
    raw = subprocess.check_output(['systemctl', '--user', 'show', unit,
        '-p', 'ActiveState', '-p', 'Result', '-p', 'ExecMainStatus'], text=True)
    return dict(line.split('=', 1) for line in raw.splitlines())


started = time.monotonic()
while properties('pi05-taskphase-train-20261028')['ActiveState'] in ('active', 'activating', 'deactivating'):
    if time.monotonic() - started > 3700:
        raise TimeoutError('Bounded training did not exit')
    time.sleep(5)
train = properties('pi05-taskphase-train-20261028')
status = json.loads((ROOT/'evidence/manual6_taskphase_2000_v1-status.json').read_text())
assert train['Result'] == 'success' and train['ExecMainStatus'] == '0', train
assert status['status'] == 'COMPLETE' and status['actual_updates'] == 2000
assert status['sampling_preview_exact_match'] is True
plan = json.loads((ROOT/'evidence/recovery-taskphase-28-execution-plan.json').read_text())
for key in ('source_sha256', 'parent_files', 'replay_manifest_sha256', 'human_manifest_sha256'):
    assert status[key] == plan[key], key
evaluation = plan['evaluation']
assert hashlib.sha256((ROOT/evaluation['manifest']).read_bytes()).hexdigest() == evaluation['manifest_sha256']
with socket.socket() as check:
    check.bind(('127.0.0.1', 9875))
checkpoint = ROOT/'checkpoints/pi05_recovery_replay_lora/manual6_taskphase_2000_v1/1999'
unit = 'pi05-taskphase-server-20261028'
log = ROOT/'logs/manual6_taskphase_2000_v1-server.log'
assert checkpoint.is_dir() and not log.exists()
subprocess.run(['systemd-run', '--user', '--unit='+unit,
    '--property=WorkingDirectory='+str(ROOT), '--property=MemoryMax=32G', '--property=MemorySwapMax=0',
    '--property=RuntimeMaxSec=1800', '--property=StandardOutput=append:'+str(log),
    '--property=StandardError=append:'+str(log), '--setenv=XLA_PYTHON_CLIENT_PREALLOCATE=false',
    '--setenv=OMP_NUM_THREADS=4', '--setenv=OPENBLAS_NUM_THREADS=4',
    str(ROOT/'.venv-train/bin/python'), '-u', 'scripts/serve_sft.py',
    '--checkpoint', str(checkpoint), '--port', '9875'], check=True)
try:
    wait_started = time.monotonic()
    while not (log.exists() and 'server listening on' in log.read_text()):
        assert properties(unit)['ActiveState'] == 'active', log.read_text() if log.exists() else unit
        if time.monotonic() - wait_started > 900:
            raise TimeoutError('Policy startup deadline')
        time.sleep(5)
    env = dict(os.environ, PYTHONPATH='/home/robot/voice-arm-sim/.local/openpi/third_party/libero',
        TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1', MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl',
        CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    with (ROOT/'logs/manual6_taskphase_2000_v1-evaluation.log').open('x') as output:
        subprocess.run([str(ROOT/'.venv-ppo/bin/python'), '-u', 'scripts/evaluate_recovery.py',
            '--port', '9875', '--expected-server-checkpoint', str(checkpoint),
            '--manifest', evaluation['manifest'], '--split', 'dev', '--max-steps', '300',
            '--noise-seed', '42', '--output', 'artifacts/20261028-taskphase/candidate'],
            env=env, stdout=output, stderr=subprocess.STDOUT, check=True)
finally:
    subprocess.run(['systemctl', '--user', 'stop', unit], check=True)
print('FROZEN_DEVELOPMENT_EVALUATION_COMPLETE', flush=True)
