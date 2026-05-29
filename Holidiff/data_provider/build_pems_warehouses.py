from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Holidiff.data_provider.traffic_warehouse_loader import preprocess_pems_npz


def main():
    root = Path('/root/yanyijin/STdiff/Holidiff/data')
    jobs = [
        {
            'name': 'PEMS03',
            'npz_path': root / 'PEMS03' / 'PEMS03.npz',
            'adj_path': root / 'PEMS03' / 'PEMS03.csv',
            'node_txt_path': root / 'PEMS03' / 'PEMS03.txt',
            'out_dir': root / 'PEMS03' / 'warehouse',
        },
        {
            'name': 'PEMS04',
            'npz_path': root / 'PEMS04' / 'PEMS04.npz',
            'adj_path': root / 'PEMS04' / 'PEMS04.csv',
            'node_txt_path': None,
            'out_dir': root / 'PEMS04' / 'warehouse',
        },
        {
            'name': 'PEMS08',
            'npz_path': root / 'PEMS08' / 'PEMS08.npz',
            'adj_path': root / 'PEMS08' / 'PEMS08.csv',
            'node_txt_path': None,
            'out_dir': root / 'PEMS08' / 'warehouse',
        },
    ]

    for job in jobs:
        print(f"==== processing {job['name']} ====")
        preprocess_pems_npz(
            npz_path=str(job['npz_path']),
            out_dir=str(job['out_dir']),
            adj_path=str(job['adj_path']),
            node_txt_path=str(job['node_txt_path']) if job['node_txt_path'] is not None else None,
            array_key='data',
            channel_idx=0,
            freq='5min',
            start_time='2000-01-01 00:00:00',
            clean_name='clean.csv',
            input_len=96,
            pred_len=12,
        )
    print('ALL_DONE')


if __name__ == '__main__':
    main()
