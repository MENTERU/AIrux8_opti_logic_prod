#!/usr/bin/env python3
"""
AC Power Analysis Visualization Script

This script creates interactive HTML plots to visualize power consumption patterns
when AC is ON vs OFF (count-based system).

Usage:
    python visualization/plot_ac_power_analysis.py

Output:
    - Interactive HTML file with all visualizations
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.offline as pyo
from plotly.subplots import make_subplots


def load_features_data():
    """Load the features CSV data"""
    features_path = Path("data/02_PreprocessedData/Clea/features_processed_Clea.csv")

    if not features_path.exists():
        print(f"❌ Error: Features file not found at {features_path}")
        sys.exit(1)

    df = pd.read_csv(features_path)
    df["Datetime"] = pd.to_datetime(df["Datetime"])

    # Add derived columns
    df["Hour"] = df["Datetime"].dt.hour
    df["Date"] = df["Datetime"].dt.date
    df["DayOfWeek"] = df["Datetime"].dt.day_name()
    df["AC_Status"] = df["A/C ON/OFF"].apply(lambda x: "OFF" if x == 0 else "ON")
    df["Power_kW"] = df["adjusted_power"] / 1000  # Convert to kW

    print(f"✅ Loaded features data: {len(df):,} rows")
    return df


def create_output_directory():
    """Create output directory for plots"""
    output_dir = Path("visualization/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def create_power_distribution_plots(df):
    """Create power distribution plots"""

    # 1. Box plot by zone and AC status
    fig1 = px.box(
        df,
        x="zone",
        y="Power_kW",
        color="AC_Status",
        title="Power Distribution by Zone and AC Status",
        labels={"Power_kW": "Power (kW)", "zone": "Zone"},
    )
    fig1.update_layout(xaxis_tickangle=-45)

    # 2. Histogram comparison
    fig2 = go.Figure()

    ac_off_data = df[df["AC_Status"] == "OFF"]["Power_kW"]
    ac_on_data = df[df["AC_Status"] == "ON"]["Power_kW"]

    fig2.add_trace(go.Histogram(x=ac_off_data, name="AC OFF", opacity=0.7, nbinsx=50))
    fig2.add_trace(go.Histogram(x=ac_on_data, name="AC ON", opacity=0.7, nbinsx=50))

    fig2.update_layout(
        title="Power Distribution: AC OFF vs AC ON",
        xaxis_title="Power (kW)",
        yaxis_title="Count",
        barmode="overlay",
    )

    # 3. Violin plot by zone
    fig3 = px.violin(
        df,
        x="zone",
        y="Power_kW",
        color="AC_Status",
        title="Power Distribution Shape by Zone",
        labels={"Power_kW": "Power (kW)", "zone": "Zone"},
    )
    fig3.update_layout(xaxis_tickangle=-45)

    # 4. Power ratio by zone
    zone_stats = df.groupby(["zone", "AC_Status"])["Power_kW"].mean().unstack()
    power_ratios = zone_stats["ON"] / zone_stats["OFF"]

    fig4 = go.Figure(
        data=[
            go.Bar(
                x=power_ratios.index,
                y=power_ratios.values,
                text=[f"{ratio:.1f}x" for ratio in power_ratios.values],
                textposition="auto",
                marker_color="green",
            )
        ]
    )

    fig4.update_layout(
        title="Power Ratio (ON/OFF) by Zone",
        xaxis_title="Zone",
        yaxis_title="Power Ratio (ON/OFF)",
        xaxis_tickangle=-45,
    )

    return [fig1, fig2, fig3, fig4]


def create_power_vs_units_plots(df):
    """Create power vs units count plots"""

    # 1. Scatter plot: Power vs Units ON
    # Clean data for scatter plot
    df_clean = df.dropna(subset=["A/C ON/OFF", "Power_kW"])
    fig1 = px.scatter(
        df_clean,
        x="A/C ON/OFF",
        y="Power_kW",
        color="Power_kW",
        size="Power_kW",
        title="Power vs Number of Units ON (All Zones)",
        labels={"A/C ON/OFF": "Number of Units ON", "Power_kW": "Power (kW)"},
        color_continuous_scale="viridis",
    )

    # 2. Power per unit analysis
    df_with_units = df[df["A/C ON/OFF"] > 0].copy()
    df_with_units["Power_per_unit"] = (
        df_with_units["Power_kW"] / df_with_units["A/C ON/OFF"]
    )

    fig2 = px.box(
        df_with_units,
        x="zone",
        y="Power_per_unit",
        title="Power per Unit ON by Zone",
        labels={"Power_per_unit": "Power per Unit (kW)", "zone": "Zone"},
    )
    fig2.update_layout(xaxis_tickangle=-45)

    # 3. Average power by units count
    units_power_avg = (
        df.groupby("A/C ON/OFF")["Power_kW"].agg(["mean", "std", "count"]).reset_index()
    )
    units_power_avg = units_power_avg[units_power_avg["count"] >= 10]

    fig3 = go.Figure()
    fig3.add_trace(
        go.Scatter(
            x=units_power_avg["A/C ON/OFF"],
            y=units_power_avg["mean"],
            error_y=dict(type="data", array=units_power_avg["std"]),
            mode="markers+lines",
            name="Average Power",
            marker=dict(size=8),
        )
    )

    fig3.update_layout(
        title="Average Power vs Number of Units ON",
        xaxis_title="Number of Units ON",
        yaxis_title="Average Power (kW)",
    )

    # 4. Power efficiency by zone
    zone_efficiency = (
        df_with_units.groupby("zone")["Power_per_unit"]
        .agg(["mean", "std"])
        .reset_index()
    )

    fig4 = go.Figure(
        data=[
            go.Bar(
                x=zone_efficiency["zone"],
                y=zone_efficiency["mean"],
                error_y=dict(type="data", array=zone_efficiency["std"]),
                marker_color="orange",
                text=[f"{mean:.1f}" for mean in zone_efficiency["mean"]],
                textposition="auto",
            )
        ]
    )

    fig4.update_layout(
        title="Power Efficiency (kW per Unit) by Zone",
        xaxis_title="Zone",
        yaxis_title="Power per Unit (kW)",
        xaxis_tickangle=-45,
    )

    return [fig1, fig2, fig3, fig4]


def create_time_series_plots(df):
    """Create time series analysis plots"""

    # 1. Power by hour of day
    hourly_power = df.groupby(["Hour", "AC_Status"])["Power_kW"].mean().unstack()

    fig1 = go.Figure()
    fig1.add_trace(
        go.Scatter(
            x=hourly_power.index,
            y=hourly_power["OFF"],
            mode="lines+markers",
            name="AC OFF",
            line=dict(width=3),
        )
    )
    fig1.add_trace(
        go.Scatter(
            x=hourly_power.index,
            y=hourly_power["ON"],
            mode="lines+markers",
            name="AC ON",
            line=dict(width=3),
        )
    )

    fig1.update_layout(
        title="Average Power by Hour of Day",
        xaxis_title="Hour of Day",
        yaxis_title="Average Power (kW)",
        xaxis=dict(tickmode="linear", tick0=0, dtick=2),
    )

    # 2. Power by day of week
    daily_power = df.groupby(["DayOfWeek", "AC_Status"])["Power_kW"].mean().unstack()
    day_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    daily_power = daily_power.reindex(day_order)

    fig2 = go.Figure()
    fig2.add_trace(
        go.Bar(x=daily_power.index, y=daily_power["OFF"], name="AC OFF", opacity=0.7)
    )
    fig2.add_trace(
        go.Bar(x=daily_power.index, y=daily_power["ON"], name="AC ON", opacity=0.7)
    )

    fig2.update_layout(
        title="Average Power by Day of Week",
        xaxis_title="Day of Week",
        yaxis_title="Average Power (kW)",
        barmode="group",
    )

    # 3. AC ON/OFF ratio by hour
    hourly_counts = df.groupby(["Hour", "AC_Status"]).size().unstack(fill_value=0)
    hourly_ratio = hourly_counts["ON"] / (hourly_counts["ON"] + hourly_counts["OFF"])

    fig3 = go.Figure(
        data=[
            go.Scatter(
                x=hourly_ratio.index,
                y=hourly_ratio.values,
                mode="lines+markers",
                line=dict(width=3, color="green"),
            )
        ]
    )

    fig3.update_layout(
        title="AC ON Ratio by Hour of Day",
        xaxis_title="Hour of Day",
        yaxis_title="AC ON Ratio",
        yaxis=dict(range=[0, 1]),
        xaxis=dict(tickmode="linear", tick0=0, dtick=2),
    )

    # 4. Sample time series for one zone
    sample_zone = "Area 1"
    sample_data = df[df["zone"] == sample_zone].copy()
    sample_data = sample_data.sort_values("Datetime")
    sample_data = sample_data.head(168)  # One week of hourly data

    fig4 = make_subplots(specs=[[{"secondary_y": True}]])

    fig4.add_trace(
        go.Scatter(
            x=sample_data["Datetime"],
            y=sample_data["Power_kW"],
            name="Power (kW)",
            line=dict(color="blue", width=2),
        ),
        secondary_y=False,
    )

    fig4.add_trace(
        go.Scatter(
            x=sample_data["Datetime"],
            y=sample_data["A/C ON/OFF"],
            name="Units ON",
            line=dict(color="red", width=2),
        ),
        secondary_y=True,
    )

    fig4.update_xaxes(title_text="Date")
    fig4.update_yaxes(title_text="Power (kW)", secondary_y=False)
    fig4.update_yaxes(title_text="Number of Units ON", secondary_y=True)

    fig4.update_layout(title_text=f"Power and AC Status - {sample_zone} (Sample Week)")

    return [fig1, fig2, fig3, fig4]


def create_zone_comparison_plots(df):
    """Create detailed zone comparison plots"""

    # 1. Power statistics by zone
    zone_stats = (
        df.groupby(["zone", "AC_Status"])["Power_kW"]
        .agg(["mean", "median", "std"])
        .unstack()
    )

    fig1 = go.Figure()
    fig1.add_trace(
        go.Bar(
            x=zone_stats.index,
            y=zone_stats[("mean", "OFF")],
            name="AC OFF Mean",
            opacity=0.7,
        )
    )
    fig1.add_trace(
        go.Bar(
            x=zone_stats.index,
            y=zone_stats[("mean", "ON")],
            name="AC ON Mean",
            opacity=0.7,
        )
    )
    fig1.add_trace(
        go.Bar(
            x=zone_stats.index,
            y=zone_stats[("median", "OFF")],
            name="AC OFF Median",
            opacity=0.7,
        )
    )

    fig1.update_layout(
        title="Power Statistics by Zone",
        xaxis_title="Zone",
        yaxis_title="Power (kW)",
        barmode="group",
        xaxis_tickangle=-45,
    )

    # 2. Power range by zone
    zone_ranges = (
        df.groupby(["zone", "AC_Status"])["Power_kW"].agg(["min", "max"]).unstack()
    )

    fig2 = go.Figure()

    for i, zone in enumerate(zone_ranges.index):
        off_min, off_max = (
            zone_ranges.loc[zone, ("min", "OFF")],
            zone_ranges.loc[zone, ("max", "OFF")],
        )
        on_min, on_max = (
            zone_ranges.loc[zone, ("min", "ON")],
            zone_ranges.loc[zone, ("max", "ON")],
        )

        fig2.add_trace(
            go.Scatter(
                x=[i, i],
                y=[off_min, off_max],
                mode="lines",
                line=dict(width=6, color="blue"),
                name="AC OFF Range" if i == 0 else None,
                showlegend=(i == 0),
            )
        )
        fig2.add_trace(
            go.Scatter(
                x=[i, i],
                y=[on_min, on_max],
                mode="lines",
                line=dict(width=6, color="red"),
                name="AC ON Range" if i == 0 else None,
                showlegend=(i == 0),
            )
        )

    fig2.update_layout(
        title="Power Range by Zone",
        xaxis_title="Zone",
        yaxis_title="Power (kW)",
        xaxis=dict(
            tickmode="array",
            tickvals=list(range(len(zone_ranges))),
            ticktext=zone_ranges.index,
            tickangle=-45,
        ),
    )

    # 3. Units distribution by zone
    units_dist = df.groupby("zone")["A/C ON/OFF"].value_counts().unstack(fill_value=0)
    units_dist_pct = units_dist.div(units_dist.sum(axis=1), axis=0) * 100

    fig3 = go.Figure()

    for col in units_dist_pct.columns:
        fig3.add_trace(
            go.Bar(
                x=units_dist_pct.index,
                y=units_dist_pct[col],
                name=f"{col} units",
                opacity=0.7,
            )
        )

    fig3.update_layout(
        title="Units ON Distribution by Zone (%)",
        xaxis_title="Zone",
        yaxis_title="Percentage",
        barmode="stack",
        xaxis_tickangle=-45,
    )

    # 4. Power efficiency heatmap
    efficiency_data = []
    for zone in df["zone"].unique():
        zone_data = df[df["zone"] == zone]
        for units_count in sorted(zone_data["A/C ON/OFF"].unique()):
            if units_count > 0:
                subset = zone_data[zone_data["A/C ON/OFF"] == units_count]
                if len(subset) > 0:
                    avg_power = subset["Power_kW"].mean()
                    power_per_unit = avg_power / units_count
                    efficiency_data.append(
                        {
                            "Zone": zone,
                            "Units_ON": units_count,
                            "Power_per_Unit": power_per_unit,
                        }
                    )

    if efficiency_data:
        efficiency_df = pd.DataFrame(efficiency_data)
        pivot_table = efficiency_df.pivot(
            index="Zone", columns="Units_ON", values="Power_per_Unit"
        )

        fig4 = px.imshow(
            pivot_table,
            title="Power Efficiency Heatmap (kW per Unit)",
            labels=dict(x="Number of Units ON", y="Zone", color="Power per Unit (kW)"),
        )
    else:
        fig4 = go.Figure()
        fig4.add_annotation(
            text="No efficiency data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )

    return [fig1, fig2, fig3, fig4]


def create_baseline_power_plots(df):
    """Create baseline power analysis plots"""

    ac_off_data = df[df["AC_Status"] == "OFF"].copy()

    # 1. Baseline power by zone
    baseline_stats = (
        ac_off_data.groupby("zone")["Power_kW"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    fig1 = go.Figure(
        data=[
            go.Bar(
                x=baseline_stats["zone"],
                y=baseline_stats["mean"],
                error_y=dict(type="data", array=baseline_stats["std"]),
                text=[f"{mean:.1f}" for mean in baseline_stats["mean"]],
                textposition="auto",
                marker_color="lightblue",
            )
        ]
    )

    fig1.update_layout(
        title="Baseline Power by Zone (AC OFF)",
        xaxis_title="Zone",
        yaxis_title="Baseline Power (kW)",
        xaxis_tickangle=-45,
    )

    # 2. Baseline power vs total units
    units_baseline = (
        ac_off_data.groupby(["zone", "Total Units"])["Power_kW"].mean().reset_index()
    )

    fig2 = go.Figure()

    for zone in units_baseline["zone"].unique():
        zone_data = units_baseline[units_baseline["zone"] == zone]
        fig2.add_trace(
            go.Scatter(
                x=zone_data["Total Units"],
                y=zone_data["Power_kW"],
                mode="markers",
                name=zone,
                marker=dict(size=10),
            )
        )

    fig2.update_layout(
        title="Baseline Power vs Total Units",
        xaxis_title="Total Units in Zone",
        yaxis_title="Baseline Power (kW)",
    )

    # 3. Baseline power distribution
    fig3 = go.Figure(
        data=[
            go.Histogram(
                x=ac_off_data["Power_kW"],
                nbinsx=50,
                marker_color="lightgreen",
                opacity=0.7,
            )
        ]
    )

    fig3.add_vline(
        x=ac_off_data["Power_kW"].mean(),
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean: {ac_off_data['Power_kW'].mean():.1f} kW",
    )
    fig3.add_vline(
        x=ac_off_data["Power_kW"].median(),
        line_dash="dash",
        line_color="orange",
        annotation_text=f"Median: {ac_off_data['Power_kW'].median():.1f} kW",
    )

    fig3.update_layout(
        title="Baseline Power Distribution (AC OFF)",
        xaxis_title="Power (kW)",
        yaxis_title="Frequency",
    )

    # 4. Baseline power by hour
    hourly_baseline = (
        ac_off_data.groupby("Hour")["Power_kW"].agg(["mean", "std"]).reset_index()
    )

    fig4 = go.Figure(
        data=[
            go.Scatter(
                x=hourly_baseline["Hour"],
                y=hourly_baseline["mean"],
                error_y=dict(type="data", array=hourly_baseline["std"]),
                mode="lines+markers",
                line=dict(width=3),
            )
        ]
    )

    fig4.update_layout(
        title="Baseline Power by Hour of Day",
        xaxis_title="Hour of Day",
        yaxis_title="Baseline Power (kW)",
        xaxis=dict(tickmode="linear", tick0=0, dtick=2),
    )

    return [fig1, fig2, fig3, fig4]


def create_html_dashboard(df, output_dir):
    """Create comprehensive HTML dashboard"""

    print("📊 Creating interactive HTML dashboard...")

    # Create all plot groups
    power_dist_plots = create_power_distribution_plots(df)
    power_units_plots = create_power_vs_units_plots(df)
    time_series_plots = create_time_series_plots(df)
    zone_comparison_plots = create_zone_comparison_plots(df)
    baseline_plots = create_baseline_power_plots(df)

    # Combine all plots into a single figure with subplots
    from plotly.subplots import make_subplots

    # Create a comprehensive dashboard with all plots
    fig = make_subplots(
        rows=5,
        cols=4,
        subplot_titles=[
            "Power Distribution by Zone",
            "Power Distribution Histogram",
            "Power Distribution Shape",
            "Power Ratio by Zone",
            "Power vs Units ON",
            "Power per Unit by Zone",
            "Average Power vs Units",
            "Power Efficiency by Zone",
            "Power by Hour",
            "Power by Day",
            "AC ON Ratio by Hour",
            "Sample Time Series",
            "Power Statistics by Zone",
            "Power Range by Zone",
            "Units Distribution",
            "Power Efficiency Heatmap",
            "Baseline Power by Zone",
            "Baseline vs Total Units",
            "Baseline Distribution",
            "Baseline by Hour",
        ],
        specs=[
            [
                {"type": "box"},
                {"type": "histogram"},
                {"type": "violin"},
                {"type": "bar"},
            ],
            [
                {"type": "scatter"},
                {"type": "box"},
                {"type": "scatter"},
                {"type": "bar"},
            ],
            [
                {"type": "scatter"},
                {"type": "bar"},
                {"type": "scatter"},
                {"type": "scatter"},
            ],
            [
                {"type": "bar"},
                {"type": "scatter"},
                {"type": "bar"},
                {"type": "heatmap"},
            ],
            [
                {"type": "bar"},
                {"type": "scatter"},
                {"type": "histogram"},
                {"type": "scatter"},
            ],
        ],
        vertical_spacing=0.08,
        horizontal_spacing=0.05,
    )

    # Add traces from each plot group
    # Power Distribution plots
    for trace in power_dist_plots[0].data:
        fig.add_trace(trace, row=1, col=1)
    for trace in power_dist_plots[1].data:
        fig.add_trace(trace, row=1, col=2)
    for trace in power_dist_plots[2].data:
        fig.add_trace(trace, row=1, col=3)
    for trace in power_dist_plots[3].data:
        fig.add_trace(trace, row=1, col=4)

    # Power vs Units plots
    for trace in power_units_plots[0].data:
        fig.add_trace(trace, row=2, col=1)
    for trace in power_units_plots[1].data:
        fig.add_trace(trace, row=2, col=2)
    for trace in power_units_plots[2].data:
        fig.add_trace(trace, row=2, col=3)
    for trace in power_units_plots[3].data:
        fig.add_trace(trace, row=2, col=4)

    # Time Series plots
    for trace in time_series_plots[0].data:
        fig.add_trace(trace, row=3, col=1)
    for trace in time_series_plots[1].data:
        fig.add_trace(trace, row=3, col=2)
    for trace in time_series_plots[2].data:
        fig.add_trace(trace, row=3, col=3)
    for trace in time_series_plots[3].data:
        fig.add_trace(trace, row=3, col=4)

    # Zone Comparison plots
    for trace in zone_comparison_plots[0].data:
        fig.add_trace(trace, row=4, col=1)
    for trace in zone_comparison_plots[1].data:
        fig.add_trace(trace, row=4, col=2)
    for trace in zone_comparison_plots[2].data:
        fig.add_trace(trace, row=4, col=3)
    for trace in zone_comparison_plots[3].data:
        fig.add_trace(trace, row=4, col=4)

    # Baseline plots
    for trace in baseline_plots[0].data:
        fig.add_trace(trace, row=5, col=1)
    for trace in baseline_plots[1].data:
        fig.add_trace(trace, row=5, col=2)
    for trace in baseline_plots[2].data:
        fig.add_trace(trace, row=5, col=3)
    for trace in baseline_plots[3].data:
        fig.add_trace(trace, row=5, col=4)

    # Update layout
    fig.update_layout(
        height=2000,
        title_text="🔌 AC Power Analysis Dashboard - Interactive Analysis of Power Consumption Patterns",
        title_x=0.5,
        showlegend=False,
    )

    # Save as HTML
    html_file = output_dir / "ac_power_analysis_dashboard.html"
    fig.write_html(html_file)

    return html_file


def main():
    """Main function to generate HTML dashboard"""

    print("🚀 Starting AC Power Analysis Visualization...")

    # Load data
    df = load_features_data()

    # Create output directory
    output_dir = create_output_directory()

    # Generate HTML dashboard
    html_file = create_html_dashboard(df, output_dir)

    print(f"\n✅ Interactive HTML dashboard created successfully!")
    print(f"📁 Output file: {html_file}")
    print(
        f"🌐 Open the HTML file in your web browser to view the interactive dashboard"
    )
    print(f"📊 The dashboard includes:")
    print(f"   • Power distribution analysis")
    print(f"   • Power vs units count analysis")
    print(f"   • Time series analysis")
    print(f"   • Zone comparison analysis")
    print(f"   • Baseline power analysis")


if __name__ == "__main__":
    main()
