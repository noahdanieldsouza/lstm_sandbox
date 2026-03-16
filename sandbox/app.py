from pathlib import Path
from typing import List

import streamlit as st

from core import build_executor_command, run_experiment, validate_dataset, write_split_config
from profiles import PROFILES


st.set_page_config(page_title="Lake Forecast Sandbox", layout="wide")

st.title("Lake Forecast Sandbox")
st.caption("Dataset swap + hyperparameter experiments for ecologists")

workspace_root = Path(__file__).resolve().parents[1]
data_dir = workspace_root / "data"
config_dir = workspace_root / "config"

with st.sidebar:
    st.header("Run Setup")
    profile_name = st.selectbox("Runtime profile", list(PROFILES.keys()), index=0)
    profile = PROFILES[profile_name]
    task_name = st.selectbox("Task", ["pretrain", "finetune", "zeroshot"], index=0)
    model_type = st.selectbox("Model", ["lstm", "transformer"], index=0)

st.subheader("1) Dataset")
available_csvs = sorted([p.name for p in data_dir.glob("*.csv")])
source_filename = st.selectbox("Choose dataset", available_csvs, index=0 if available_csvs else None)

col1, col2 = st.columns(2)
with col1:
    target_cols = st.multiselect(
        "Target columns",
        options=[],
        default=[],
        help="Load a dataset first, then use quick presets below.",
    )
with col2:
    quick_target = st.selectbox("Quick target preset", ["poc", "poc,DO_filled,secchi_m"], index=1)
    quick_targets = [c.strip() for c in quick_target.split(",")]

if source_filename:
    csv_path = data_dir / source_filename
    import pandas as pd

    df = pd.read_csv(csv_path, nrows=1)
    all_cols: List[str] = list(df.columns)
    feature_defaults = [c for c in quick_targets if c in all_cols]

    st.write(f"Detected columns: {len(all_cols)}")
    target_cols = st.multiselect("Target columns", options=all_cols, default=feature_defaults)
    masked_columns = st.multiselect("Exclude from input features", options=all_cols, default=feature_defaults)

    report = validate_dataset(csv_path, target_cols)
    if report.ok:
        st.success(f"Dataset looks valid ({report.n_rows} rows).")
    else:
        st.error("Dataset validation failed.")
        st.write(
            {
                "missing_columns": report.missing_columns,
                "parseable_datetime": report.parseable_datetime,
                "numeric_targets": report.numeric_targets,
            }
        )

st.subheader("2) Splits")
s1, s2, s3 = st.columns(3)
with s1:
    train_frac = st.slider("Train fraction", 0.1, 0.9, 0.6, 0.05)
with s2:
    val_frac = st.slider("Validation fraction", 0.05, 0.6, 0.2, 0.05)
with s3:
    test_frac = st.slider("Test fraction", 0.05, 0.6, 0.2, 0.05)

split_total = train_frac + val_frac + test_frac
if abs(split_total - 1.0) > 1e-6:
    st.warning(f"Fractions currently sum to {split_total:.2f}; they should sum to 1.0.")

st.subheader("3) Hyperparameters")
h1, h2, h3, h4 = st.columns(4)
with h1:
    max_epochs = st.number_input("Max epochs", 1, 1000, int(profile["max_epochs"]))
    batch_size = st.number_input("Batch size", 1, 2048, int(profile["batch_size"]))
with h2:
    lookback_window = st.number_input("Lookback window", 1, 512, int(profile["lookback_window"]))
    horizon_window = st.number_input("Horizon window", 1, 128, int(profile["horizon_window"]))
with h3:
    label_len = st.number_input("Label length", 1, 512, int(profile["label_len"]))
    learning_rate = st.number_input("Learning rate", 1e-6, 1.0, float(profile["learning_rate"]), format="%.6f")
with h4:
    weight_decay = st.number_input("Weight decay", 0.0, 1.0, float(profile["weight_decay"]), format="%.6f")
    dropout = st.number_input("Dropout", 0.0, 0.9, float(profile["dropout"]), format="%.3f")

with st.expander("Advanced"):
    c_out = st.number_input("Output size (c_out)", 1, 16, max(1, len(target_cols) if source_filename else 1))
    num_layers = st.number_input("LSTM layers", 1, 12, int(profile["num_layers"]))
    hidden_feature_size = st.number_input("LSTM hidden size", 4, 2048, int(profile["hidden_feature_size"]))
    ntrials = st.number_input("Trials", 1, 20, int(profile["ntrials"]))
    device = st.text_input("Device id", value="0", help="Use 0 for first GPU, or any value on CPU-only systems")
    run_name = st.text_input("Run name", value="sandbox")
    project_name = st.text_input("Project name", value="ecology_sandbox")
    wandb_mode = st.selectbox("WandB mode", ["disabled", "offline", "online"], index=["disabled", "offline", "online"].index(profile["wandb_mode"]))

st.subheader("4) Run")
ckpt_path = ""
if task_name in {"finetune", "zeroshot"}:
    ckpt_path = st.text_input(
        "Checkpoint path",
        value="",
        help="Required for finetune and zeroshot",
    )

run_clicked = st.button("Start Experiment", type="primary")

if run_clicked:
    if not source_filename:
        st.error("Please select a dataset.")
        st.stop()
    if not target_cols:
        st.error("Please select at least one target column.")
        st.stop()
    if abs(split_total - 1.0) > 1e-6:
        st.error("Train/val/test fractions must sum to 1.0.")
        st.stop()

    lake_name = Path(source_filename).stem
    sandbox_cfg = config_dir / "config_sandbox.json"
    write_split_config(sandbox_cfg, train_frac, val_frac, test_frac, lake_name)

    if task_name in {"finetune", "zeroshot"} and not ckpt_path:
        st.error("Checkpoint path is required for finetune and zeroshot.")
        st.stop()

    cmd_params = {
        "project_name": project_name,
        "run_name": run_name,
        "task_name": task_name,
        "config_name": sandbox_cfg.name,
        "config_base": str(config_dir),
        "data_path": str(data_dir),
        "source_filename": source_filename,
        "model_type": model_type,
        "lookback_window": lookback_window,
        "horizon_window": horizon_window,
        "label_len": label_len,
        "max_epochs": max_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "dropout": dropout,
        "num_layers": num_layers,
        "hidden_feature_size": hidden_feature_size,
        "ntrials": ntrials,
        "device": device,
        "c_out": c_out,
        "wandb_mode": wandb_mode,
        "save_code": "False",
    }

    if ckpt_path:
        cmd_params["ckpt_path"] = ckpt_path

    # Expand list-like args into repeated CLI tokens for argparse nargs='+'
    expanded_params = {}
    for k, v in cmd_params.items():
        if k in {"target_cols", "masked_columns"}:
            continue
        expanded_params[k] = v

    cmd = build_executor_command(workspace_root, expanded_params)
    if target_cols:
        cmd.extend(["--target_cols", *target_cols])
    if masked_columns:
        cmd.extend(["--masked_columns", *masked_columns])

    st.code(" ".join(cmd))

    output_box = st.empty()
    lines = []
    try:
        for line in run_experiment(cmd, cwd=workspace_root / "Transfer_Learning"):
            lines.append(line)
            output_box.code("\n".join(lines[-250:]))
        st.success("Experiment completed successfully.")
    except Exception as exc:
        st.error(str(exc))
