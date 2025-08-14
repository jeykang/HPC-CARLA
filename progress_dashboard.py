"""Live dashboard showing the latest data from each sensor during CARLA collection.

This Streamlit app scans the dataset directory for the newest files from each sensor
and displays them side‑by‑side. Each sensor's feed appears in its own pane so the
page resembles an autonomous-driving heads-up display while jobs are actively saving
data to the mounted filesystem.

The dataset is expected to follow the structure::

    ./dataset/agent-*/weather-*/<route>/<sensor>/<frame>.<ext>

All sensor folders under any route are searched and the latest ``.png`` or ``.npy``
in each is shown on the page. ``.npy`` files (e.g., LiDAR point clouds or depth
maps) are visualized alongside camera images.
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(
    page_title="CARLA Data Progress", page_icon="\U0001f4c8", layout="wide"
)


def gather_latest_sensor_files(data_dir: Path) -> dict[str, Path]:
    """Return mapping of ``agent/weather/route/sensor`` to its newest data file."""

    latest: dict[str, tuple[Path, float]] = {}
    for file_path in data_dir.rglob("*"):
        if file_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".npy"}:
            continue
        try:
            rel = file_path.relative_to(data_dir)
        except ValueError:
            continue
        parts = rel.parts
        if len(parts) < 5:
            continue
        agent, weather, route, sensor = parts[:4]
        label = f"{agent}/{weather}/{route}/{sensor}"
        try:
            mtime = file_path.stat().st_mtime
        except OSError:
            continue
        current = latest.get(label)
        if current is None or mtime > current[1]:
            latest[label] = (file_path, mtime)

    return {label: path for label, (path, _) in latest.items()}


def display_sensor_file(sensor: str, file_path: Path, data_dir: Path) -> None:
    """Render a sensor file in the Streamlit UI."""

    try:
        if file_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            img = Image.open(file_path)
            img.thumbnail((640, 480), Image.Resampling.LANCZOS)
            st.image(img, caption=str(file_path.relative_to(data_dir)))
        elif file_path.suffix.lower() == ".npy":
            arr = np.load(file_path)
            sensor_lower = sensor.lower()
            if "lidar" in sensor_lower and arr.ndim == 2 and arr.shape[1] >= 2:
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.scatter(arr[:, 0], arr[:, 1], s=0.5, c=arr[:, 2], cmap="viridis")
                ax.set_axis_off()
                ax.set_aspect("equal", adjustable="box")
                st.pyplot(fig)
            else:
                st.image(arr, caption=str(file_path.relative_to(data_dir)), clamp=True)
        else:
            st.write(str(file_path.relative_to(data_dir)))
    except Exception:
        st.write(str(file_path.relative_to(data_dir)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live dashboard for CARLA data collection"
    )
    default_dir = os.environ.get("CARLA_DATA_DIR", "./dataset")
    parser.add_argument(
        "--data-dir", default=default_dir, help="Path to dataset directory"
    )
    args, _ = parser.parse_known_args()

    data_dir = Path(args.data_dir)
    st.title("\U0001f4c8 Data Collection Progress")
    st.caption(f"Monitoring: {data_dir}")

    if not data_dir.exists():
        st.error(f"Cannot access dataset directory: {data_dir}")
        return

    with st.sidebar:
        auto_refresh = st.checkbox("Auto-refresh", value=True)
        refresh_interval = st.slider("Refresh interval (s)", 5, 60, 30)

    files = gather_latest_sensor_files(data_dir)
    if not files:
        st.info("No sensor data found in dataset yet.")
    else:
        num_cols = min(3, len(files))
        cols = st.columns(num_cols)
        for idx, (sensor, file_path) in enumerate(sorted(files.items())):
            with cols[idx % num_cols]:
                st.subheader(sensor)
                display_sensor_file(sensor, file_path, data_dir)

    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()


if __name__ == "__main__":
    main()
