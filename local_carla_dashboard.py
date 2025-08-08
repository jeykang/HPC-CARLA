#!/usr/bin/env python3
import os
import time
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
from PIL import Image
import streamlit as st
import plotly.express as px
import pandas as pd

st.set_page_config(page_title="CARLA Agent Dashboard", page_icon="🤖", layout="wide", initial_sidebar_state="expanded")

class AgentCentricDashboard:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.cache_timeout = 30
        self.last_scan = 0
        self.cached_agents = None

    def scan_agents(self, force_refresh=False):
        if not force_refresh and self.cached_agents and (time.time() - self.last_scan) < self.cache_timeout:
            return self.cached_agents

        agents = {}
        with st.spinner('Scanning for agents...'):
            agent_dirs = sorted(self.data_dir.glob("agent-*"))
            for agent_dir in agent_dirs:
                agent_name = agent_dir.name.replace("agent-", "")
                agents[agent_name] = {
                    'path': agent_dir,
                    'runs': [],
                    'total_frames': 0,
                    'active_runs': 0,
                    'weather_conditions': set(),
                    'routes': set()
                }

                for weather_dir in sorted(agent_dir.glob("weather-*")):
                    weather_id = weather_dir.name.replace("weather-", "")
                    for route_dir in sorted(weather_dir.glob("routes_*")):
                        route_name = route_dir.name

                        sample_files = list(route_dir.glob("rgb*/????.png"))[:5]
                        if not sample_files:
                            sample_files = list(route_dir.glob("rgb*/?????.png"))[:5]

                        if sample_files:
                            latest_file = max(sample_files, key=lambda x: x.stat().st_mtime)
                            time_since_update = time.time() - latest_file.stat().st_mtime

                            rgb_dirs = [d for d in route_dir.iterdir() if d.is_dir() and 'rgb' in d.name]
                            frame_count = 0
                            if rgb_dirs:
                                sample_dir = rgb_dirs[0]
                                frame_files = list(sample_dir.glob("????.png"))
                                if not frame_files:
                                    frame_files = list(sample_dir.glob("?????.png"))
                                frame_count = len(frame_files)

                            is_active = time_since_update < 120

                            run_info = {
                                'path': route_dir,
                                'weather': weather_id,
                                'route': route_name,
                                'frame_count': frame_count,
                                'active': is_active,
                                'last_update': datetime.fromtimestamp(latest_file.stat().st_mtime)
                            }

                            agents[agent_name]['runs'].append(run_info)
                            agents[agent_name]['total_frames'] += frame_count
                            agents[agent_name]['weather_conditions'].add(weather_id)
                            agents[agent_name]['routes'].add(route_name.split('_')[1])

                            if is_active:
                                agents[agent_name]['active_runs'] += 1

        self.cached_agents = agents
        self.last_scan = time.time()
        return agents

    def load_latest_frame(self, route_path, camera='front'):
        camera_dirs = {
            'front': ['rgb', 'rgb_1', 'rgb_front'],
            'left': ['rgb_left', 'rgb_0'],
            'right': ['rgb_right', 'rgb_2'],
            'bev': ['bev']
        }

        for cam_dir in camera_dirs.get(camera, [camera]):
            cam_path = route_path / cam_dir
            if cam_path.exists():
                frame_files = sorted(cam_path.glob("????.png"))
                if not frame_files:
                    frame_files = sorted(cam_path.glob("?????.png"))
                if frame_files:
                    try:
                        img = Image.open(frame_files[-1])
                        img.thumbnail((320, 240), Image.Resampling.LANCZOS)
                        return img
                    except:
                        continue
        return None

    def create_agent_grid(self, agent_data, camera_view='front'):
        runs = agent_data['runs']
        if not runs:
            st.warning("No runs found for this agent")
            return

        weather_groups = {}
        for run in runs:
            weather = f"Weather {run['weather']}"
            weather_groups.setdefault(weather, []).append(run)

        for weather, weather_runs in sorted(weather_groups.items()):
            st.subheader(weather)
            cols = st.columns(min(4, len(weather_runs)))
            for idx, run in enumerate(weather_runs[:4]):
                with cols[idx % len(cols)]:
                    route_short = run['route'].replace('routes_', '')
                    status = "🟢" if run['active'] else "⭕"
                    st.markdown(f"**{route_short}** {status}")
                    img = self.load_latest_frame(run['path'], camera_view)
                    if img:
                        st.image(img, use_container_width=True)
                        st.caption(f"{run['frame_count']} frames")
                        st.caption(f"Updated: {run['last_update'].strftime('%H:%M:%S')}")
                    else:
                        st.info("No image data")

    def create_comparison_view(self, agent_data):
        runs = agent_data['runs']
        if len(runs) < 2:
            st.info("Need at least 2 runs for comparison")
            return

        run_labels = [f"W{r['weather']}/{r['route'].replace('routes_', '')}" for r in runs]

        col1, col2 = st.columns(2)
        with col1:
            idx1 = st.selectbox("First Run", range(len(runs)), format_func=lambda x: run_labels[x])
        with col2:
            idx2 = st.selectbox("Second Run", range(len(runs)), format_func=lambda x: run_labels[x], index=min(1, len(runs)-1))

        if idx1 is not None and idx2 is not None:
            run1, run2 = runs[idx1], runs[idx2]
            camera_view = st.select_slider("Camera View", options=['left', 'front', 'right'], value='front')

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"### {run_labels[idx1]}")
                img1 = self.load_latest_frame(run1['path'], camera_view)
                if img1:
                    st.image(img1, use_container_width=True)
                st.metric("Frames", run1['frame_count'])
                st.metric("Status", "Active" if run1['active'] else "Inactive")
            with col2:
                st.markdown(f"### {run_labels[idx2]}")
                img2 = self.load_latest_frame(run2['path'], camera_view)
                if img2:
                    st.image(img2, use_container_width=True)
                st.metric("Frames", run2['frame_count'])
                st.metric("Status", "Active" if run2['active'] else "Inactive")

    def create_agent_analytics(self, agent_data):
        runs = agent_data['runs']
        if not runs:
            st.info("No data available for analytics")
            return

        df = pd.DataFrame(runs)
        if df.empty:
            st.warning("No data to analyze")
            return

        df['weather'] = 'W' + df['weather']
        df['route_short'] = df['route'].str.replace('routes_', '')
        df['town'] = df['route_short'].str.extract(r'(town\d+)')[0]

        col1, col2 = st.columns(2)
        with col1:
            fig1 = px.bar(df, x='weather', y='frame_count', color='route_short',
                          title='Frames by Weather Condition',
                          labels={'frame_count': 'Frame Count', 'weather': 'Weather'})
            st.plotly_chart(fig1, use_container_width=True)
        with col2:
            fig2 = px.bar(df, x='route_short', y='frame_count', color='weather',
                          title='Frames by Route',
                          labels={'frame_count': 'Frame Count', 'route_short': 'Route'})
            fig2.update_xaxes(tickangle=45)
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Collection Progress Matrix")
        try:
            pivot = df.pivot_table(values='frame_count', index='weather', columns='route_short', fill_value=0)
            if not pivot.empty:
                fig3 = px.imshow(pivot, labels=dict(x="Route", y="Weather", color="Frames"),
                                 color_continuous_scale="Viridis", aspect="auto")
                fig3.update_layout(height=400)
                st.plotly_chart(fig3, use_container_width=True)
            else:
                st.info("Not enough data for progress matrix")
        except Exception as e:
            st.warning(f"Could not create progress matrix: {str(e)}")

        col1, col2, col3 = st.columns(3)
        with col1:
            avg_frames = df['frame_count'].mean()
            st.metric("Average Frames/Run", f"{avg_frames:.0f}" if not pd.isna(avg_frames) else "N/A")
        with col2:
            completion_rate = (df['frame_count'] > 100).mean() * 100
            st.metric("Completion Rate", f"{completion_rate:.0f}%" if not pd.isna(completion_rate) else "N/A")
        with col3:
            active_rate = df['active'].mean() * 100
            st.metric("Active Rate", f"{active_rate:.0f}%" if not pd.isna(active_rate) else "N/A")

    def create_timeline_view(self, agent_data):
        runs = agent_data['runs']
        if not runs:
            st.info("No timeline data available")
            return

        timeline_data = [{
            'Run': f"W{r['weather']}/{r['route'].replace('routes_', '')}",
            'Last Update': r['last_update'],
            'Frames': r['frame_count'],
            'Active': r['active']
        } for r in runs]

        df = pd.DataFrame(timeline_data).sort_values('Last Update')
        if df.empty:
            st.info("No timeline data to display")
            return

        fig = px.scatter(df, x='Last Update', y='Run', size='Frames', color='Active',
                         color_discrete_map={True: 'green', False: 'gray'},
                         title='Data Collection Timeline',
                         labels={'Last Update': 'Time', 'Run': 'Run ID'})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    def render_dashboard(self):
        st.title("🤖 CARLA Agent-Centric Dashboard")
        st.caption(f"Data source: {self.data_dir}")

        if not self.data_dir.exists():
            st.error(f"Cannot access data directory: {self.data_dir}")
            st.info("Please check that the HPC filesystem is properly mounted")
            return

        agents = self.scan_agents()
        if not agents:
            st.warning("No agents found in the data directory")
            st.info("Make sure data collection has started and data is being saved to the mounted directory")
            return

        with st.sidebar:
            st.header("Agent Selection")
            agent_options = []
            for name, data in agents.items():
                active_indicator = "🟢" if data['active_runs'] > 0 else "⭕"
                agent_options.append(f"{active_indicator} {name} ({len(data['runs'])} runs, {data['total_frames']} frames)")

            selected_idx = st.selectbox("Select Agent", range(len(agent_options)), format_func=lambda x: agent_options[x])
            selected_agent = list(agents.keys())[selected_idx]
            agent_data = agents[selected_agent]

            st.subheader(f"Agent: {selected_agent}")
            st.metric("Total Runs", len(agent_data['runs']))
            st.metric("Active Runs", agent_data['active_runs'])
            st.metric("Total Frames", agent_data['total_frames'])
            st.metric("Weather Conditions", len(agent_data['weather_conditions']))
            st.metric("Towns", len(agent_data['routes']))

            st.divider()
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Refresh"):
                    self.cached_agents = None
                    st.rerun()
            with col2:
                auto_refresh = st.checkbox("Auto-refresh", key="auto_refresh")

            if auto_refresh:
                st.slider("Interval (s)", 10, 60, 30, key="refresh_interval")

            st.divider()
            st.subheader("Performance")
            self.cache_timeout = st.slider("Cache timeout (s)", 10, 120, 30)

        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📸 Grid View", "🔍 Comparison", "📊 Analytics", "⏱️ Timeline", "📋 Details"])

        with tab1:
            st.header(f"All Runs: {selected_agent}")
            camera_view = st.radio("Camera View", ['front', 'left', 'right'], horizontal=True)
            self.create_agent_grid(agent_data, camera_view)

        with tab2:
            st.header("Run Comparison")
            self.create_comparison_view(agent_data)

        with tab3:
            st.header(f"Analytics: {selected_agent}")
            self.create_agent_analytics(agent_data)

        with tab4:
            st.header("Collection Timeline")
            self.create_timeline_view(agent_data)

        with tab5:
            st.header("Detailed Run Information")
            run_details = [{
                'Weather': f"W{r['weather']}",
                'Route': r['route'].replace('routes_', ''),
                'Frames': r['frame_count'],
                'Status': '🟢 Active' if r['active'] else '⭕ Inactive',
                'Last Update': r['last_update'].strftime('%Y-%m-%d %H:%M:%S'),
                'Path': str(r['path'].relative_to(self.data_dir))
            } for r in agent_data['runs']]
            df = pd.DataFrame(run_details)
            st.dataframe(df, use_container_width=True, hide_index=True)
            csv = df.to_csv(index=False)
            st.download_button(label="📥 Download Run Details as CSV", data=csv,
                               file_name=f"{selected_agent}_runs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                               mime="text/csv")

        if st.session_state.get('auto_refresh', False):
            time.sleep(st.session_state.get('refresh_interval', 30))
            st.rerun()

def main():
    parser = argparse.ArgumentParser(description='Agent-Centric CARLA Dashboard')
    default_dir = os.environ.get('CARLA_DATA_DIR', './dataset')
    parser.add_argument('--data-dir', type=str, default=default_dir, help='Path to mounted data directory')
    args, _ = parser.parse_known_args()
    dashboard = AgentCentricDashboard(args.data_dir)
    dashboard.render_dashboard()

if __name__ == "__main__":
    main()
