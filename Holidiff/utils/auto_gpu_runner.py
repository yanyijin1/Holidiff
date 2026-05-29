import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# train.py already tees stdout/stderr into the default project logs/ directory.
# This helper should not create a second persistent log path.

ROOT = Path(__file__).resolve().parents[2]
TRAIN_PY = ROOT / 'Holidiff' / 'train.py'


def query_gpus():
    cmd = [
        'nvidia-smi',
        '--query-gpu=index,memory.used,memory.total,utilization.gpu',
        '--format=csv,noheader,nounits',
    ]
    out = subprocess.check_output(cmd, text=True)
    gpus = []
    for line in out.strip().splitlines():
        idx, mem_used, mem_total, util = [x.strip() for x in line.split(',')]
        gpus.append({
            'index': int(idx),
            'memory_used': int(mem_used),
            'memory_total': int(mem_total),
            'util': int(util),
        })
    return gpus


def select_gpu(prefer=None):
    gpus = query_gpus()
    if prefer is not None:
        for gpu in gpus:
            if gpu['index'] == prefer:
                return gpu['index']
    gpus.sort(key=lambda x: (x['memory_used'], x['util'], x['index']))
    return gpus[0]['index']


def build_command(config, version, extra_args):
    cmd = [
        sys.executable,
        str(TRAIN_PY),
        '--config', config,
        '--version', version,
    ]
    cmd.extend(extra_args)
    return cmd


def main():
    parser = argparse.ArgumentParser(
        description='Auto-pick a GPU and launch HoliDiff training/testing. Output is written by train.py to the default project logs/ directory.'
    )
    parser.add_argument('--config', required=True, help='YAML config path')
    parser.add_argument('--version', required=True, help='Run version name')
    parser.add_argument('--env', default='holiday', help='Conda env name (reserved for compatibility)')
    parser.add_argument('--gpu', type=int, default=None, help='Force GPU index')
    parser.add_argument('--wait-seconds', type=float, default=0.0, help='Optional delay before launch')
    args, extra_args = parser.parse_known_args()

    if args.wait_seconds > 0:
        time.sleep(args.wait_seconds)

    gpu_id = select_gpu(args.gpu)
    cmd = build_command(args.config, args.version, extra_args)
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    print(f'[auto_gpu] selected physical gpu={gpu_id}')
    print(f'[auto_gpu] version={args.version}')
    print(f'[auto_gpu] config={args.config}')
    print(f'[auto_gpu] command={" ".join(cmd)}')
    print('[auto_gpu] use the default project logs/ directory emitted by train.py; do not add a second launcher log redirect unless needed.')
    sys.stdout.flush()

    proc = subprocess.run(cmd, env=env, cwd=str(ROOT))
    raise SystemExit(proc.returncode)


if __name__ == '__main__':
    main()
