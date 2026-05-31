from pathlib import Path
import re

BASE = Path("/root/yanyijin/STdiff/Holidiff/configs")
FUJIAN_DIR = BASE / "fujian30"
LOCAL_DIR = BASE / "local"

MODELS = {
    "dlinear": "DLinear",
    "patchtst": "PatchTST",
    "itransformer": "iTransformer",
    "timesnet": "TimesNet",
    "tsdiff": "TSDiff",
    "diffusion_ts": "DiffusionTS",
    "simdiff": "SimDiff",
}

DATASETS = {
    "fujian30": {
        "label": "Fujian30",
        "dir": FUJIAN_DIR,
        "data": "fujian30",
        "root_path": "/root/autodl-tmp/STdiff_data/fujian-30",
        "data_path": "fujian30_clean.csv",
        "adj_path": "/root/autodl-tmp/STdiff_data/fujian-30/adjacent_gantry.csv",
        "checkpoints": "/root/autodl-tmp/STdiff_runs/checkpoints/fujian30",
        "freq": "15min",
        "enc_in": 30,
        "num_workers": 0,
        "zero_as_missing": "true",
        "extreme_filter_threshold": None,
    },
    "pems08-local40": {
        "label": "PEMS08-Local40",
        "dir": LOCAL_DIR / "pems08-local40",
        "data": "traffic_warehouse",
        "root_path": "/root/autodl-tmp/STdiff_data/PEMS08-Local40/warehouse",
        "data_path": "clean.csv",
        "adj_path": "/root/autodl-tmp/STdiff_data/PEMS08-Local40/PEMS08-Local40.csv",
        "checkpoints": "/root/autodl-tmp/STdiff_runs/checkpoints/pems08-local40",
        "freq": "5min",
        "enc_in": 40,
        "num_workers": 4,
        "zero_as_missing": "false",
        "extreme_filter_threshold": "null",
    },
    "pems04-local50": {
        "label": "PEMS04-Local50",
        "dir": LOCAL_DIR / "pems04-local50",
        "data": "traffic_warehouse",
        "root_path": "/root/autodl-tmp/STdiff_data/PEMS04-Local50/warehouse",
        "data_path": "clean.csv",
        "adj_path": "/root/autodl-tmp/STdiff_data/PEMS04-Local50/PEMS04-Local50.csv",
        "checkpoints": "/root/autodl-tmp/STdiff_runs/checkpoints/pems04-local50",
        "freq": "5min",
        "enc_in": 50,
        "num_workers": 4,
        "zero_as_missing": "false",
        "extreme_filter_threshold": "null",
    },
    "pems07-local60": {
        "label": "PEMS07-Local60",
        "dir": LOCAL_DIR / "pems07-local60",
        "data": "traffic_warehouse",
        "root_path": "/root/autodl-tmp/STdiff_data/PEMS07-Local60/warehouse",
        "data_path": "clean.csv",
        "adj_path": "/root/autodl-tmp/STdiff_data/PEMS07-Local60/PEMS07-Local60.csv",
        "checkpoints": "/root/autodl-tmp/STdiff_runs/checkpoints/pems07-local60",
        "freq": "5min",
        "enc_in": 60,
        "num_workers": 4,
        "zero_as_missing": "false",
        "extreme_filter_threshold": "null",
    },
}

HORIZONS = [12, 24, 36]


def replace_line(text: str, key: str, value) -> str:
    return re.sub(rf"^{re.escape(key)}:.*$", f"{key}: {value}", text, flags=re.M)


def ensure_metadata_block(text: str, dataset: dict) -> str:
    block = (
        f"checkpoints: {dataset['checkpoints']}\n\n"
        f"time_col: time_slot\n"
        f"node_col: station_index\n"
        f"target_col: traffic_flow\n"
        f"holiday_col: is_holiday\n"
        f"adj_path: {dataset['adj_path']}\n"
    )
    return re.sub(r"checkpoints: .*?(?:\n\n|\n)", block, text, count=1, flags=re.S)


for dataset in DATASETS.values():
    for model_key, model_name in MODELS.items():
        template = (FUJIAN_DIR / f"{model_key}.yaml").read_text()
        for horizon in HORIZONS:
            text = template
            text = replace_line(text, "model_id", f"{dataset['label']}_96_{horizon}_{model_name}")
            text = replace_line(text, "data", dataset["data"])
            text = replace_line(text, "root_path", dataset["root_path"])
            text = replace_line(text, "data_path", dataset["data_path"])
            text = replace_line(text, "freq", dataset["freq"])
            text = replace_line(text, "checkpoints", dataset["checkpoints"])
            text = replace_line(text, "seq_len", 96)
            text = replace_line(text, "label_len", 48)
            text = replace_line(text, "pred_len", horizon)
            text = replace_line(text, "enc_in", dataset["enc_in"])
            text = replace_line(text, "dec_in", dataset["enc_in"])
            text = replace_line(text, "c_out", dataset["enc_in"])
            text = replace_line(text, "num_workers", dataset["num_workers"])
            text = replace_line(text, "train_epochs", 100)
            text = replace_line(text, "patience", 20)
            text = replace_line(text, "learning_rate", 0.0001)
            text = replace_line(text, "patch_len", 12)
            text = replace_line(text, "stride", 6)

            if re.search(r"^adj_path:", text, flags=re.M):
                text = replace_line(text, "adj_path", dataset["adj_path"])
            else:
                text = ensure_metadata_block(text, dataset)

            for key, value in [
                ("time_col", "time_slot"),
                ("node_col", "station_index"),
                ("target_col", "traffic_flow"),
                ("holiday_col", "is_holiday"),
            ]:
                if re.search(rf"^{key}:", text, flags=re.M):
                    text = replace_line(text, key, value)

            if re.search(r"^zero_as_missing:", text, flags=re.M):
                text = replace_line(text, "zero_as_missing", dataset["zero_as_missing"])
            else:
                text = text.rstrip() + f"\nzero_as_missing: {dataset['zero_as_missing']}\n"

            if dataset["extreme_filter_threshold"] is not None:
                if re.search(r"^extreme_filter_threshold:", text, flags=re.M):
                    text = replace_line(text, "extreme_filter_threshold", dataset["extreme_filter_threshold"])
                else:
                    text = text.rstrip() + f"\nextreme_filter_threshold: {dataset['extreme_filter_threshold']}\n"

            if model_key == "tsdiff":
                text = replace_line(text, "tsdiff_seq_length", 96 + horizon)
            if model_key == "diffusion_ts":
                text = replace_line(text, "diffusion_ts_seq_length", 96 + horizon)
                text = replace_line(text, "diffusion_ts_feature_size", dataset["enc_in"])
            if model_key == "simdiff":
                text = replace_line(text, "patch_len", 12)
                text = replace_line(text, "stride", 6)

            output_path = dataset["dir"] / f"{model_key}_h{horizon}.yaml"
            output_path.write_text(text)

print("generated all dataset/model/horizon yaml files")
