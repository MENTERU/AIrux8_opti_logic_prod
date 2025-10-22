#!/usr/bin/env python3
"""
Create individual HTML files for each analysis section
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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


def create_power_distribution_html(df, output_dir):
    """Create power distribution analysis HTML"""

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

    # 3. Power ratio by zone
    zone_stats = df.groupby(["zone", "AC_Status"])["Power_kW"].mean().unstack()
    power_ratios = zone_stats["ON"] / zone_stats["OFF"]

    fig3 = go.Figure(
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

    fig3.update_layout(
        title="Power Ratio (ON/OFF) by Zone",
        xaxis_title="Zone",
        yaxis_title="Power Ratio (ON/OFF)",
        xaxis_tickangle=-45,
    )

    # Combine into subplots
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "Power Distribution by Zone",
            "Power Distribution Histogram",
            "Power Ratio by Zone",
            "Summary Statistics",
        ],
        specs=[
            [{"type": "box"}, {"type": "histogram"}],
            [{"type": "bar"}, {"type": "table"}],
        ],
    )

    # Add traces
    for trace in fig1.data:
        fig.add_trace(trace, row=1, col=1)
    for trace in fig2.data:
        fig.add_trace(trace, row=1, col=2)
    for trace in fig3.data:
        fig.add_trace(trace, row=2, col=1)

    # Add summary table
    summary_data = []
    for zone in sorted(df["zone"].unique()):
        zone_data = df[df["zone"] == zone]
        off_data = zone_data[zone_data["AC_Status"] == "OFF"]
        on_data = zone_data[zone_data["AC_Status"] == "ON"]

        summary_data.append(
            [
                zone,
                f"{off_data['Power_kW'].mean():.1f}",
                f"{on_data['Power_kW'].mean():.1f}",
                f"{on_data['Power_kW'].mean() / off_data['Power_kW'].mean():.1f}x",
            ]
        )

    fig.add_trace(
        go.Table(
            header=dict(
                values=["Zone", "AC OFF (kW)", "AC ON (kW)", "Ratio"],
                fill_color="lightblue",
            ),
            cells=dict(values=list(zip(*summary_data)), fill_color="white"),
        ),
        row=2,
        col=2,
    )

    fig.update_layout(height=800, title_text="Power Distribution Analysis")

    # Save HTML
    html_file = output_dir / "power_distribution_analysis.html"
    fig.write_html(html_file)
    print(f"✅ Created: {html_file}")

    return html_file


def create_baseline_analysis_html(df, output_dir):
    """Create baseline power analysis HTML"""

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

    # 2. Baseline power distribution
    fig2 = go.Figure(
        data=[
            go.Histogram(
                x=ac_off_data["Power_kW"],
                nbinsx=50,
                marker_color="lightgreen",
                opacity=0.7,
            )
        ]
    )

    fig2.add_vline(
        x=ac_off_data["Power_kW"].mean(),
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean: {ac_off_data['Power_kW'].mean():.1f} kW",
    )
    fig2.add_vline(
        x=ac_off_data["Power_kW"].median(),
        line_dash="dash",
        line_color="orange",
        annotation_text=f"Median: {ac_off_data['Power_kW'].median():.1f} kW",
    )

    fig2.update_layout(
        title="Baseline Power Distribution (AC OFF)",
        xaxis_title="Power (kW)",
        yaxis_title="Frequency",
    )

    # 3. Baseline power by hour
    hourly_baseline = (
        ac_off_data.groupby("Hour")["Power_kW"].agg(["mean", "std"]).reset_index()
    )

    fig3 = go.Figure(
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

    fig3.update_layout(
        title="Baseline Power by Hour of Day",
        xaxis_title="Hour of Day",
        yaxis_title="Baseline Power (kW)",
        xaxis=dict(tickmode="linear", tick0=0, dtick=2),
    )

    # Combine into subplots
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "Baseline Power by Zone",
            "Baseline Power Distribution",
            "Baseline Power by Hour",
            "Baseline Statistics",
        ],
        specs=[
            [{"type": "bar"}, {"type": "histogram"}],
            [{"type": "scatter"}, {"type": "table"}],
        ],
    )

    # Add traces
    for trace in fig1.data:
        fig.add_trace(trace, row=1, col=1)
    for trace in fig2.data:
        fig.add_trace(trace, row=1, col=2)
    for trace in fig3.data:
        fig.add_trace(trace, row=2, col=1)

    # Add summary table
    summary_data = [
        ["Total AC OFF Records", f"{len(ac_off_data):,}"],
        ["Mean Baseline Power", f"{ac_off_data['Power_kW'].mean():.1f} kW"],
        ["Median Baseline Power", f"{ac_off_data['Power_kW'].median():.1f} kW"],
        ["Min Baseline Power", f"{ac_off_data['Power_kW'].min():.1f} kW"],
        ["Max Baseline Power", f"{ac_off_data['Power_kW'].max():.1f} kW"],
        ["Std Baseline Power", f"{ac_off_data['Power_kW'].std():.1f} kW"],
    ]

    fig.add_trace(
        go.Table(
            header=dict(values=["Metric", "Value"], fill_color="lightgreen"),
            cells=dict(values=list(zip(*summary_data)), fill_color="white"),
        ),
        row=2,
        col=2,
    )

    fig.update_layout(height=800, title_text="Baseline Power Analysis (AC OFF)")

    # Save HTML
    html_file = output_dir / "baseline_power_analysis.html"
    fig.write_html(html_file)
    print(f"✅ Created: {html_file}")

    return html_file


def main():
    """Main function"""

    print("🚀 Creating individual HTML analysis files...")

    # Load data
    df = load_features_data()

    # Create output directory
    output_dir = Path("visualization/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create individual HTML files
    create_power_distribution_html(df, output_dir)
    create_baseline_analysis_html(df, output_dir)

    print(f"\n✅ Individual HTML files created successfully!")
    print(f"📁 Output directory: {output_dir}")
    print(f"📊 Created files:")
    print(f"   • power_distribution_analysis.html")
    print(f"   • baseline_power_analysis.html")


if __name__ == "__main__":
    main()
