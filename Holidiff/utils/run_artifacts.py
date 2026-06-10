from __future__ import annotations

from pathlib import Path

from Holidiff.utils.display_name import get_display_name


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = Path('/root/autodl-tmp/STdiff_runs/logs')
RESULT_DIR = PROJECT_ROOT / 'results'
TEST_RESULT_DIR = PROJECT_ROOT / 'test_results'


def build_run_name(args, iteration: int = 0) -> str:
    version = str(getattr(args, 'version', '') or '').strip()
    if version:
        if getattr(args, 'itr', 1) > 1:
            return f'{version}_{iteration}'
        return version

    run_tag = getattr(args, 'des', 'Exp')
    model_width = getattr(args, 'network_profile', '') or getattr(args, 'd_model', 'na')
    display_model = get_display_name(getattr(args, 'model', 'Model'))
    return '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_net{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
        getattr(args, 'task_name', 'task'),
        getattr(args, 'model_id', 'model'),
        display_model,
        getattr(args, 'data', 'data'),
        getattr(args, 'features', 'M'),
        getattr(args, 'seq_len', 'na'),
        getattr(args, 'label_len', 'na'),
        getattr(args, 'pred_len', 'na'),
        model_width,
        getattr(args, 'd_model', 'na'),
        getattr(args, 'n_heads', 'na'),
        getattr(args, 'e_layers', 'na'),
        getattr(args, 'd_layers', 'na'),
        getattr(args, 'd_ff', 'na'),
        getattr(args, 'expand', 'na'),
        getattr(args, 'd_conv', 'na'),
        getattr(args, 'factor', 'na'),
        getattr(args, 'embed', 'na'),
        getattr(args, 'distil', 'na'),
        run_tag,
        iteration,
    )


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def log_path(run_name: str) -> Path:
    return ensure_dir(LOG_DIR) / f'{run_name}.log'


def checkpoint_dir(checkpoints_root: str | Path, run_name: str) -> Path:
    return ensure_dir(Path(checkpoints_root) / run_name)


def checkpoint_path(checkpoints_root: str | Path, run_name: str) -> Path:
    return checkpoint_dir(checkpoints_root, run_name) / 'checkpoint.pth'


def result_text_path(run_name: str) -> Path:
    return ensure_dir(RESULT_DIR) / f'{run_name}.txt'


def result_json_path(run_name: str) -> Path:
    return ensure_dir(RESULT_DIR) / f'{run_name}.json'


def test_result_dir(run_name: str) -> Path:
    return ensure_dir(TEST_RESULT_DIR / run_name)
