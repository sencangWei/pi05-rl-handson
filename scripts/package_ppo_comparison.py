"""Package matched native SFT/PPO rollouts without altering playback speed."""
import argparse
import json
import shutil
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sft', type=Path, required=True)
    parser.add_argument('--ppo', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    before = json.loads((args.sft / 'summary.json').read_text())
    after = json.loads((args.ppo / 'summary.json').read_text())
    def key(row):
        return row['task_id'], row['initial_state']
    left, right = {key(r): r for r in before}, {key(r): r for r in after}
    if len(left) != 8 or left.keys() != right.keys() or len(after) != 8:
        raise ValueError('Expected eight matched unique scenes')
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for task, initial in sorted(left):
        a, b = left[task, initial], right[task, initial]
        for field in ['initial_state_sha256', 'seed', 'replan_steps', 'max_steps', 'prompt']:
            if a[field] != b[field]:
                raise ValueError(f'Comparison differs in {field}')
        stem = f'goal-{task:02d}-init-{initial:02d}'
        for label, source in [('native-sft', args.sft), ('ppo', args.ppo)]:
            shutil.copy2(source / stem / 'paired_cameras.mp4',
                         args.output / f'{label}-{stem}.mp4')
        # Both sources encode one simulation control frame per video frame at 10 fps.
        # Freeze the shorter result while the other continues; never stretch time.
        filters = (
            '[0:v]setpts=PTS-STARTPTS,tpad=stop_mode=clone:stop_duration=60,'
            f"drawtext=fontcolor=white:box=1:boxcolor=black@0.7:fontsize=16:text='SFT {a['result']} steps {a['steps']}'[l];"
            '[1:v]setpts=PTS-STARTPTS,tpad=stop_mode=clone:stop_duration=60,'
            f"drawtext=fontcolor=white:box=1:boxcolor=black@0.7:fontsize=16:text='PPO {b['result']} steps {b['steps']}'[r];"
            '[l][r]hstack=inputs=2[v]')
        destination = args.output / f'comparison-{stem}.mp4'
        subprocess.run(['ffmpeg', '-nostdin', '-y', '-v', 'error',
                        '-i', str(args.output / f'native-sft-{stem}.mp4'),
                        '-i', str(args.output / f'ppo-{stem}.mp4'),
                        '-filter_complex', filters, '-map', '[v]',
                        '-t', str((max(a['steps'], b['steps']) + 1) / 10),
                        '-c:v', 'libx264', '-threads', '2', '-crf', '23',
                        '-pix_fmt', 'yuv420p', str(destination)], check=True)
        subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-i', str(destination),
                        '-f', 'null', '-'], check=True)
        rows.append({'task_id': task, 'initial_state': initial,
                     'initial_state_sha256': a['initial_state_sha256'],
                     'sft_result': a['result'], 'sft_steps': a['steps'],
                     'ppo_result': b['result'], 'ppo_steps': b['steps'],
                     'sft_reward': a['reward_return'], 'ppo_reward': b['reward_return'],
                     'video': destination.name})
    (args.output / 'comparison.json').write_text(json.dumps(rows, indent=2) + '\n')
    for label, source in [('native-sft', args.sft), ('ppo', args.ppo)]:
        for name in ['summary', 'identity']:
            shutil.copy2(source / f'{name}.json', args.output / f'{label}-{name}.json')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
