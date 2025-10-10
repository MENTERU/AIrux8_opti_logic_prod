#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
All Settings Dashboard - Single Comprehensive View
================================================
Creates a single dashboard showing all settings for each area in line graphs.
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime
import os
import argparse
import numpy as np


def load_optimization_data(csv_path: str) -> pd.DataFrame:
    """Load and preprocess optimization results CSV"""
    print(f"Loading data from: {csv_path}")
    
    # Read CSV
    df = pd.read_csv(csv_path)
    df['Date Time'] = pd.to_datetime(df['Date Time'])
    df = df.sort_values('Date Time').reset_index(drop=True)
    
    # Convert fan speed strings to numeric values
    fan_speed_mapping = {
        'Auto': 0,
        'LOW': 1, 
        'MEDIUM': 2,
        'HIGH': 3,
        'Top': 4
    }
    
    # Apply fan speed mapping to all fan speed columns
    for col in df.columns:
        if 'FanSpeed' in col:
            df[col] = df[col].map(fan_speed_mapping)
            print(f"Converted {col} from strings to numeric values")
    
    print(f"Loaded {len(df)} records from {df['Date Time'].min()} to {df['Date Time'].max()}")
    return df


def get_area_columns(df: pd.DataFrame) -> dict:
    """Extract column names for each area"""
    areas = {}
    
    # Get all unique area names from column headers
    for col in df.columns:
        if '_' in col and col != 'Date Time' and col != 'outside_temp':
            area_name = col.split('_')[0]
            if area_name not in areas:
                areas[area_name] = {}
            areas[area_name][col.split('_')[1]] = col
    
    return areas


def create_all_settings_dashboard(df: pd.DataFrame, areas: dict, output_dir: str = "all_settings_dashboard"):
    """Create a single comprehensive dashboard showing all settings for all areas"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a large subplot with many rows (one for each metric)
    # We'll have: Outside Temp, Set Temp, Pred Temp, Power, Mode, Fan Speed, ON/OFF for each area
    num_areas = len(areas)
    num_metrics = 7  # Outside Temp, Set Temp, Pred Temp, Power, Mode, Fan Speed, ON/OFF
    
    # Create subplots - one row per metric, one column
    fig = make_subplots(
        rows=num_metrics, cols=1,
        subplot_titles=(
            'Outside Temperature',
            'Set Temperature - All Areas',
            'Predicted Temperature - All Areas', 
            'Power Consumption - All Areas',
            'AC Mode - All Areas',
            'Fan Speed - All Areas',
            'ON/OFF Status - All Areas'
        ),
        vertical_spacing=0.05,
        specs=[[{"secondary_y": False}]] * num_metrics
    )
    
    # Color palette for areas
    colors = px.colors.qualitative.Set1
    
    # 1. Outside Temperature (row=1)
    fig.add_trace(
        go.Scatter(x=df['Date Time'], y=df['outside_temp'], 
                  name='Outside Temperature', line=dict(color='black', width=3)),
        row=1, col=1
    )
    
    # 2. Set Temperature for all areas (row=2)
    for i, (area_name, area_cols) in enumerate(areas.items()):
        fig.add_trace(
            go.Scatter(x=df['Date Time'], y=df[area_cols['SetTemp']], 
                      name=f'{area_name} Set Temp', line=dict(color=colors[i % len(colors)], width=2)),
            row=2, col=1
        )
    
    # 3. Predicted Temperature for all areas (row=3)
    for i, (area_name, area_cols) in enumerate(areas.items()):
        fig.add_trace(
            go.Scatter(x=df['Date Time'], y=df[area_cols['PredTemp']], 
                      name=f'{area_name} Pred Temp', line=dict(color=colors[i % len(colors)], width=2)),
            row=3, col=1
        )
    
    # 4. Power Consumption for all areas (row=4)
    for i, (area_name, area_cols) in enumerate(areas.items()):
        fig.add_trace(
            go.Scatter(x=df['Date Time'], y=df[area_cols['PredPower']], 
                      name=f'{area_name} Power', line=dict(color=colors[i % len(colors)], width=2)),
            row=4, col=1
        )
    
    # 5. AC Mode for all areas (row=5)
    for i, (area_name, area_cols) in enumerate(areas.items()):
        # Convert mode strings to numbers for plotting
        mode_numeric = df[area_cols['Mode']].map({'OFF': 0, 'COOL': 1, 'HEAT': 2, 'FAN': 3})
        fig.add_trace(
            go.Scatter(x=df['Date Time'], y=mode_numeric, 
                      name=f'{area_name} Mode', line=dict(color=colors[i % len(colors)], width=2),
                      mode='lines+markers', marker=dict(size=4)),
            row=5, col=1
        )
    
    # 6. Fan Speed for all areas (row=6)
    for i, (area_name, area_cols) in enumerate(areas.items()):
        fig.add_trace(
            go.Scatter(x=df['Date Time'], y=df[area_cols['FanSpeed']], 
                      name=f'{area_name} Fan', line=dict(color=colors[i % len(colors)], width=2),
                      mode='lines+markers', marker=dict(size=4)),
            row=6, col=1
        )
    
    # 7. ON/OFF Status for all areas (row=7)
    for i, (area_name, area_cols) in enumerate(areas.items()):
        onoff_numeric = df[area_cols['OnOFF']].map({'ON': 1, 'OFF': 0})
        fig.add_trace(
            go.Scatter(x=df['Date Time'], y=onoff_numeric, 
                      name=f'{area_name} ON/OFF', line=dict(color=colors[i % len(colors)], width=2),
                      mode='lines+markers', marker=dict(size=4)),
            row=7, col=1
        )
    
    # Update layout
    fig.update_layout(
        title='All Settings Dashboard - Complete Overview',
        height=2000,  # Large height to accommodate all subplots
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        )
    )
    
    # Update y-axes for each subplot
    fig.update_yaxes(title_text="Temperature (°C)", row=1, col=1)
    fig.update_yaxes(title_text="Set Temperature (°C)", row=2, col=1)
    fig.update_yaxes(title_text="Predicted Temperature (°C)", row=3, col=1)
    fig.update_yaxes(title_text="Power (W)", row=4, col=1)
    fig.update_yaxes(title_text="Mode", row=5, col=1,
                    tickmode='array', tickvals=[0, 1, 2, 3], 
                    ticktext=['OFF', 'COOL', 'HEAT', 'FAN'])
    fig.update_yaxes(title_text="Fan Speed", row=6, col=1,
                    tickmode='array', tickvals=[0, 1, 2, 3, 4], 
                    ticktext=['Auto', 'Low', 'Medium', 'High', 'Top'])
    fig.update_yaxes(title_text="ON/OFF", row=7, col=1,
                    tickmode='array', tickvals=[0, 1], 
                    ticktext=['OFF', 'ON'])
    
    # Update x-axes for all subplots
    for i in range(1, num_metrics + 1):
        fig.update_xaxes(title_text="Date Time", row=i, col=1)
    
    # Save the dashboard
    output_file = os.path.join(output_dir, 'all_settings_complete_dashboard.html')
    fig.write_html(output_file)
    print(f"Saved complete all-settings dashboard: {output_file}")
    
    return output_file


def create_compact_settings_dashboard(df: pd.DataFrame, areas: dict, output_dir: str = "all_settings_dashboard"):
    """Create a more compact dashboard with fewer subplots but more data per subplot"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a 2x2 subplot layout with more data per subplot
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Temperature Overview (Outside, Set, Predicted)',
            'Power Consumption - All Areas',
            'Control Settings (Mode & Fan Speed)',
            'System Status (ON/OFF & Mode Changes)'
        ),
        specs=[[{"secondary_y": True}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    colors = px.colors.qualitative.Set1
    
    # 1. Temperature Overview (row=1, col=1) - Multiple y-axes
    # Outside temperature on secondary y-axis
    fig.add_trace(
        go.Scatter(x=df['Date Time'], y=df['outside_temp'], 
                  name='Outside Temp', line=dict(color='black', width=3, dash='dash')),
        row=1, col=1, secondary_y=True
    )
    
    # Set temperatures on primary y-axis
    for i, (area_name, area_cols) in enumerate(areas.items()):
        fig.add_trace(
            go.Scatter(x=df['Date Time'], y=df[area_cols['SetTemp']], 
                      name=f'{area_name} Set', line=dict(color=colors[i % len(colors)], width=2)),
            row=1, col=1, secondary_y=False
        )
    
    # Predicted temperatures on primary y-axis (dashed)
    for i, (area_name, area_cols) in enumerate(areas.items()):
        fig.add_trace(
            go.Scatter(x=df['Date Time'], y=df[area_cols['PredTemp']], 
                      name=f'{area_name} Pred', line=dict(color=colors[i % len(colors)], width=2, dash='dot')),
            row=1, col=1, secondary_y=False
        )
    
    # 2. Power Consumption (row=1, col=2)
    for i, (area_name, area_cols) in enumerate(areas.items()):
        fig.add_trace(
            go.Scatter(x=df['Date Time'], y=df[area_cols['PredPower']], 
                      name=f'{area_name} Power', line=dict(color=colors[i % len(colors)], width=2)),
            row=1, col=2
        )
    
    # 3. Control Settings (row=2, col=1) - Mode and Fan Speed combined
    # Mode (left y-axis)
    for i, (area_name, area_cols) in enumerate(areas.items()):
        mode_numeric = df[area_cols['Mode']].map({'OFF': 0, 'COOL': 1, 'HEAT': 2, 'FAN': 3})
        fig.add_trace(
            go.Scatter(x=df['Date Time'], y=mode_numeric, 
                      name=f'{area_name} Mode', line=dict(color=colors[i % len(colors)], width=2),
                      mode='lines+markers', marker=dict(size=3)),
            row=2, col=1
        )
    
    # Fan Speed (offset by 5 to separate from mode)
    for i, (area_name, area_cols) in enumerate(areas.items()):
        fan_offset = df[area_cols['FanSpeed']] + 5  # Offset fan speed by 5
        fig.add_trace(
            go.Scatter(x=df['Date Time'], y=fan_offset, 
                      name=f'{area_name} Fan', line=dict(color=colors[i % len(colors)], width=2, dash='dot'),
                      mode='lines+markers', marker=dict(size=3)),
            row=2, col=1
        )
    
    # 4. System Status (row=2, col=2) - ON/OFF and mode changes
    # ON/OFF status
    for i, (area_name, area_cols) in enumerate(areas.items()):
        onoff_numeric = df[area_cols['OnOFF']].map({'ON': 1, 'OFF': 0})
        fig.add_trace(
            go.Scatter(x=df['Date Time'], y=onoff_numeric, 
                      name=f'{area_name} ON/OFF', line=dict(color=colors[i % len(colors)], width=2),
                      mode='lines+markers', marker=dict(size=3)),
            row=2, col=2
        )
    
    # Mode changes (offset by 2)
    for i, (area_name, area_cols) in enumerate(areas.items()):
        mode_changes = []
        prev_mode = None
        for mode in df[area_cols['Mode']]:
            if prev_mode is not None and mode != prev_mode:
                mode_changes.append(3)  # Mode change indicator
            else:
                mode_changes.append(2)  # No change
            prev_mode = mode
        
        fig.add_trace(
            go.Scatter(x=df['Date Time'], y=mode_changes, 
                      name=f'{area_name} Mode Change', line=dict(color=colors[i % len(colors)], width=2, dash='dot'),
                      mode='lines+markers', marker=dict(size=3)),
            row=2, col=2
        )
    
    # Update layout
    fig.update_layout(
        title='Compact All Settings Dashboard',
        height=1000,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        )
    )
    
    # Update y-axes
    fig.update_yaxes(title_text="Temperature (°C)", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Outside Temperature (°C)", row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Power (W)", row=1, col=2)
    fig.update_yaxes(title_text="Control Value", row=2, col=1,
                    tickmode='array', tickvals=[0, 1, 2, 3, 5, 6, 7, 8, 9], 
                    ticktext=['OFF', 'COOL', 'HEAT', 'FAN', 'Auto+5', 'Low+5', 'Med+5', 'High+5', 'Top+5'])
    fig.update_yaxes(title_text="Status", row=2, col=2,
                    tickmode='array', tickvals=[0, 1, 2, 3], 
                    ticktext=['OFF', 'ON', 'No Change', 'Mode Change'])
    
    # Update x-axes
    for i in range(1, 3):
        for j in range(1, 3):
            fig.update_xaxes(title_text="Date Time", row=i, col=j)
    
    # Save the compact dashboard
    output_file = os.path.join(output_dir, 'compact_all_settings_dashboard.html')
    fig.write_html(output_file)
    print(f"Saved compact all-settings dashboard: {output_file}")
    
    return output_file


def create_single_line_dashboard(df: pd.DataFrame, areas: dict, output_dir: str = "all_settings_dashboard"):
    """Create a single line graph with all settings normalized to 0-1 scale"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a single plot with all settings normalized
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set1
    
    # Normalize all values to 0-1 scale for comparison
    def normalize_series(series, min_val=None, max_val=None):
        if min_val is None:
            min_val = series.min()
        if max_val is None:
            max_val = series.max()
        if max_val == min_val:
            return pd.Series([0.5] * len(series), index=series.index)
        return (series - min_val) / (max_val - min_val)
    
    # Add outside temperature (normalized)
    outside_norm = normalize_series(df['outside_temp'])
    fig.add_trace(
        go.Scatter(x=df['Date Time'], y=outside_norm, 
                  name='Outside Temp (norm)', line=dict(color='black', width=3, dash='dash'))
    )
    
    # Add all area settings
    for i, (area_name, area_cols) in enumerate(areas.items()):
        color = colors[i % len(colors)]
        
        # Set Temperature (normalized)
        set_temp_norm = normalize_series(df[area_cols['SetTemp']])
        fig.add_trace(
            go.Scatter(x=df['Date Time'], y=set_temp_norm, 
                      name=f'{area_name} Set Temp', line=dict(color=color, width=2))
        )
        
        # Predicted Temperature (normalized)
        pred_temp_norm = normalize_series(df[area_cols['PredTemp']])
        fig.add_trace(
            go.Scatter(x=df['Date Time'], y=pred_temp_norm, 
                      name=f'{area_name} Pred Temp', line=dict(color=color, width=2, dash='dot'))
        )
        
        # Power (normalized)
        power_norm = normalize_series(df[area_cols['PredPower']])
        fig.add_trace(
            go.Scatter(x=df['Date Time'], y=power_norm, 
                      name=f'{area_name} Power', line=dict(color=color, width=1.5, dash='dashdot'))
        )
        
        # Mode (already 0-3, normalize to 0-1)
        mode_numeric = df[area_cols['Mode']].map({'OFF': 0, 'COOL': 1, 'HEAT': 2, 'FAN': 3})
        mode_norm = mode_numeric / 3.0
        fig.add_trace(
            go.Scatter(x=df['Date Time'], y=mode_norm, 
                      name=f'{area_name} Mode', line=dict(color=color, width=1),
                      mode='lines+markers', marker=dict(size=2))
        )
        
        # Fan Speed (already 0-4, normalize to 0-1)
        fan_norm = df[area_cols['FanSpeed']] / 4.0
        fig.add_trace(
            go.Scatter(x=df['Date Time'], y=fan_norm, 
                      name=f'{area_name} Fan', line=dict(color=color, width=1, dash='dot'),
                      mode='lines+markers', marker=dict(size=2))
        )
        
        # ON/OFF (already 0-1)
        onoff_numeric = df[area_cols['OnOFF']].map({'ON': 1, 'OFF': 0})
        fig.add_trace(
            go.Scatter(x=df['Date Time'], y=onoff_numeric, 
                      name=f'{area_name} ON/OFF', line=dict(color=color, width=1, dash='dash'),
                      mode='lines+markers', marker=dict(size=2))
        )
    
    # Update layout
    fig.update_layout(
        title='All Settings - Single Normalized View (0-1 Scale)',
        xaxis_title='Date Time',
        yaxis_title='Normalized Value (0-1)',
        height=600,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        )
    )
    
    # Add horizontal reference lines
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_hline(y=0.25, line_dash="dash", line_color="gray", opacity=0.3)
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_hline(y=0.75, line_dash="dash", line_color="gray", opacity=0.3)
    fig.add_hline(y=1, line_dash="dash", line_color="gray", opacity=0.5)
    
    # Save the single line dashboard
    output_file = os.path.join(output_dir, 'single_line_all_settings_dashboard.html')
    fig.write_html(output_file)
    print(f"Saved single line all-settings dashboard: {output_file}")
    
    return output_file


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Create all-settings dashboards')
    parser.add_argument('--csv', type=str, 
                       default='data/04_PlanningData/Clea/control_type_schedule_20251009.csv',
                       help='Path to the optimization results CSV file')
    parser.add_argument('--output-dir', type=str, default='all_settings_dashboard',
                       help='Output directory for dashboard files')
    parser.add_argument('--type', type=str, choices=['complete', 'compact', 'single', 'all'], 
                       default='all', help='Type of dashboard to create')
    
    args = parser.parse_args()
    
    # Load data
    df = load_optimization_data(args.csv)
    
    # Get area information
    areas = get_area_columns(df)
    print(f"Found {len(areas)} areas: {list(areas.keys())}")
    
    # Create dashboards based on type
    if args.type in ['complete', 'all']:
        create_all_settings_dashboard(df, areas, args.output_dir)
    
    if args.type in ['compact', 'all']:
        create_compact_settings_dashboard(df, areas, args.output_dir)
    
    if args.type in ['single', 'all']:
        create_single_line_dashboard(df, areas, args.output_dir)
    
    print(f"\nDashboard creation complete! Check the '{args.output_dir}' directory for HTML files.")
    print("Open the HTML files in your web browser to view the interactive dashboards.")


if __name__ == "__main__":
    main()
