#!/usr/bin/env python3
"""
Create a single comprehensive HTML dashboard with all optimization analysis.
"""

import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def load_optimization_results():
    """Load the optimization results with historical weather data."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Try to load the enhanced results first, fallback to original
    enhanced_path = os.path.join(
        current_dir,
        "..",
        "data",
        "04_PlanningData",
        "Clea",
        "optimized_results_with_historical_weather.csv",
    )
    original_path = os.path.join(
        current_dir,
        "..",
        "data",
        "04_PlanningData",
        "Clea",
        "optimized_results_20250929_20251004.csv",
    )

    if os.path.exists(enhanced_path):
        print(f"Loading enhanced results from: {enhanced_path}")
        df = pd.read_csv(enhanced_path)
    elif os.path.exists(original_path):
        print(f"Loading original results from: {original_path}")
        df = pd.read_csv(original_path)
    else:
        raise FileNotFoundError("No optimization results file found")

    # Convert datetime
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["date"] = df["datetime"].dt.date
    df["hour"] = df["datetime"].dt.hour

    print(
        f"Loaded {len(df)} records from {df['datetime'].min()} to {df['datetime'].max()}"
    )
    return df


def create_comprehensive_dashboard(df):
    """Create a single comprehensive dashboard with all analysis."""

    zones = ["Area 1", "Area 2", "Area 3", "Area 4", "Break Room", "Meeting Room"]

    # Define consistent colors for all data types
    color_scheme = {
        "forecast_temp": "blue",
        "hist_outdoor_temp": "orange",
        "indoor_temp": "green",
        "ac_set_temp": "red",
        "power": "brown",
    }

    # Create a large subplot layout - one graph per row
    fig = make_subplots(
        rows=6,
        cols=1,
        subplot_titles=[
            "Optimization Coverage Summary",
            "Area 1 - Weather & Optimization",
            "Area 2 - Weather & Optimization",
            "Area 3 - Weather & Optimization",
            "Area 4 - Weather & Optimization",
            "Break Room - Weather & Optimization",
        ],
        vertical_spacing=0.08,
        horizontal_spacing=0.1,
        specs=[
            [{"secondary_y": False}],
            [{"secondary_y": True}],
            [{"secondary_y": True}],
            [{"secondary_y": True}],
            [{"secondary_y": True}],
            [{"secondary_y": True}],
        ],
    )

    # 1. Coverage Summary (row 1, col 1)
    coverage_data = []
    for zone in zones:
        zone_set_temp_col = f"{zone}_set_temp"
        if zone_set_temp_col in df.columns:
            total_hours = len(df)
            filled_hours = df[zone_set_temp_col].notna().sum()
            coverage_pct = (filled_hours / total_hours) * 100
            coverage_data.append({"Zone": zone, "Coverage %": coverage_pct})

    coverage_df = pd.DataFrame(coverage_data)

    fig.add_trace(
        go.Bar(
            x=coverage_df["Zone"],
            y=coverage_df["Coverage %"],
            name="Coverage %",
            marker_color=coverage_df["Coverage %"],
            marker_colorscale="RdYlGn",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # 2-6. Individual zone plots (rows 2-6)
    for i, zone in enumerate(zones):
        # Skip Meeting Room - only showing first 5 areas
        if i == 5:  # Meeting Room
            continue

        # Row is i+2 (row 1 is coverage summary, row 2 is first zone)
        row = i + 2
        col = 1

        zone_set_temp_col = f"{zone}_set_temp"
        if zone_set_temp_col not in df.columns:
            continue

        zone_data = df[df[zone_set_temp_col].notna()].copy()

        if len(zone_data) == 0:
            continue

        # Add forecast temperature
        fig.add_trace(
            go.Scatter(
                x=zone_data["datetime"],
                y=zone_data["forecast_outdoor_temp"],
                mode="lines+markers",
                name=f"{zone} Forecast Outdoor Temp",
                line=dict(color=color_scheme["forecast_temp"], width=2),
                marker=dict(size=4),
                showlegend=True,
            ),
            row=row,
            col=col,
        )

        # Add historical outdoor temp if available
        hist_temp_col = f"{zone}_hist_outdoor_temp"
        if hist_temp_col in df.columns:
            hist_data = zone_data[zone_data[hist_temp_col].notna()]
            if len(hist_data) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=hist_data["datetime"],
                        y=hist_data[hist_temp_col],
                        mode="lines+markers",
                        name=f"{zone} Hist Outdoor Temp",
                        line=dict(
                            color=color_scheme["hist_outdoor_temp"],
                            width=2,
                            dash="dash",
                        ),
                        marker=dict(size=4),
                        showlegend=True,
                    ),
                    row=row,
                    col=col,
                )

        # Add indoor temperature if available
        indoor_temp_col = f"{zone}_indoor_temp"
        if indoor_temp_col in df.columns:
            indoor_data = zone_data[zone_data[indoor_temp_col].notna()]
            if len(indoor_data) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=indoor_data["datetime"],
                        y=indoor_data[indoor_temp_col],
                        mode="lines+markers",
                        name=f"{zone} Indoor Temp",
                        line=dict(color=color_scheme["indoor_temp"], width=2),
                        marker=dict(size=4),
                        showlegend=True,
                    ),
                    row=row,
                    col=col,
                )

        # Add AC set temperature
        fig.add_trace(
            go.Scatter(
                x=zone_data["datetime"],
                y=zone_data[zone_set_temp_col],
                mode="lines+markers",
                name=f"{zone} AC Set Temp",
                line=dict(color=color_scheme["ac_set_temp"], width=2),
                marker=dict(size=4),
                showlegend=True,
            ),
            row=row,
            col=col,
        )

        # Add power consumption on secondary y-axis
        power_col = f"{zone}_power"
        if power_col in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=zone_data["datetime"],
                    y=zone_data[power_col],
                    mode="lines+markers",
                    name=f"{zone} Power",
                    line=dict(color=color_scheme["power"], width=2),
                    marker=dict(size=4),
                    showlegend=True,
                ),
                row=row,
                col=col,
                secondary_y=True,
            )

    # Update layout
    fig.update_layout(
        title="Comprehensive Optimization Analysis Dashboard",
        height=2400,
        showlegend=True,
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            font=dict(size=10),
        ),
    )

    # Update axes labels
    fig.update_xaxes(title_text="Zone", row=1, col=1, gridcolor="lightgray")
    fig.update_yaxes(title_text="Coverage (%)", row=1, col=1, gridcolor="lightgray")

    # Update axes for zone plots (rows 2-6)
    for row in range(2, 7):
        fig.update_xaxes(title_text="DateTime", row=row, col=1, gridcolor="lightgray")
        fig.update_yaxes(
            title_text="Temperature (°C)", row=row, col=1, gridcolor="lightgray"
        )
        fig.update_yaxes(
            title_text="Power (W)",
            secondary_y=True,
            row=row,
            col=1,
            gridcolor="lightgray",
        )

    return fig


def main():
    """Main function to generate the single comprehensive dashboard."""

    print("=== COMPREHENSIVE OPTIMIZATION DASHBOARD ===")

    # Load data
    df = load_optimization_results()

    # Create output directory
    output_dir = "analysis/output"
    os.makedirs(output_dir, exist_ok=True)

    # Create single comprehensive dashboard
    print("\nCreating comprehensive dashboard...")
    fig = create_comprehensive_dashboard(df)

    # Save single HTML file
    output_file = f"{output_dir}/optimization_plots.html"
    fig.write_html(output_file)
    print(f"  Saved: {output_file}")

    print(f"\n=== VISUALIZATION COMPLETE ===")
    print(f"Single comprehensive dashboard saved to: {output_file}")
    print(f"Open the HTML file in your browser to view the interactive dashboard.")


if __name__ == "__main__":
    main()
