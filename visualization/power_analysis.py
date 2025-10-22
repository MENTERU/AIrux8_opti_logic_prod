import os

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_power_analysis(data_file, output_dir):
    """
    Generates a series of plots to analyze the relationship between AC ON/OFF counts and power consumption.

    Args:
        data_file (str): Path to the features_processed_Clea.csv file.
        output_dir (str): Directory to save the HTML plot file.
    """
    df = pd.read_csv(data_file)
    df["Datetime"] = pd.to_datetime(df["Datetime"])

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "power_analysis.html")

    with open(output_file, "w") as f:
        f.write("<html><head><title>Power Analysis</title></head><body>")
        f.write("<h1>Power Analysis based on AC ON/OFF Count</h1>")

        # Plot 1: Scatter plot of adjusted_power vs. A/C ON/OFF
        fig1 = go.Figure()
        for zone in df["zone"].unique():
            zone_df = df[df["zone"] == zone]
            fig1.add_trace(
                go.Scatter(
                    x=zone_df["A/C ON/OFF"],
                    y=zone_df["adjusted_power"],
                    mode="markers",
                    name=zone,
                )
            )
        fig1.update_layout(
            title="Adjusted Power vs. A/C ON/OFF Count",
            xaxis_title="A/C ON/OFF (Count of Units)",
            yaxis_title="Adjusted Power",
            height=600,
        )
        f.write(fig1.to_html(full_html=False, include_plotlyjs="cdn"))

        # Plot 2: Box plot of adjusted_power for each A/C ON/OFF count
        fig2 = go.Figure()
        for zone in df["zone"].unique():
            zone_df = df[df["zone"] == zone]
            fig2.add_trace(
                go.Box(x=zone_df["A/C ON/OFF"], y=zone_df["adjusted_power"], name=zone)
            )
        fig2.update_layout(
            title="Distribution of Adjusted Power for each A/C ON/OFF Count",
            xaxis_title="A/C ON/OFF (Count of Units)",
            yaxis_title="Adjusted Power",
            boxmode="group",
            height=600,
        )
        f.write(fig2.to_html(full_html=False, include_plotlyjs=False))

        # Plot 3: Time series of adjusted_power and A/C ON/OFF for a sample period
        f.write("<h2>Time Series Analysis (Sample Period)</h2>")
        sample_df = df[
            (df["Datetime"] >= "2024-10-18") & (df["Datetime"] <= "2024-10-21")
        ]
        for zone in sample_df["zone"].unique():
            zone_df = sample_df[sample_df["zone"] == zone]
            fig3 = make_subplots(specs=[[{"secondary_y": True}]])
            fig3.add_trace(
                go.Scatter(
                    x=zone_df["Datetime"],
                    y=zone_df["adjusted_power"],
                    name="Adjusted Power",
                ),
                secondary_y=False,
            )
            fig3.add_trace(
                go.Scatter(
                    x=zone_df["Datetime"],
                    y=zone_df["A/C ON/OFF"],
                    name="A/C ON/OFF Count",
                ),
                secondary_y=True,
            )
            fig3.update_layout(
                title_text=f"Time Series of Power and AC Count for {zone} (October 18-21, 2024)",
                height=400,
            )
            fig3.update_yaxes(title_text="Adjusted Power", secondary_y=False)
            fig3.update_yaxes(title_text="A/C ON/OFF Count", secondary_y=True)
            f.write(fig3.to_html(full_html=False, include_plotlyjs=False))

        # Plot 4: Analysis of adjusted_power when A/C ON/OFF is 0
        zero_power_df = df[df["A/C ON/OFF"] == 0]
        fig4 = go.Figure()
        for zone in zero_power_df["zone"].unique():
            zone_df = zero_power_df[zero_power_df["zone"] == zone]
            fig4.add_trace(go.Box(y=zone_df["adjusted_power"], name=zone))
        fig4.update_layout(
            title="Distribution of Adjusted Power when A/C ON/OFF is 0",
            yaxis_title="Adjusted Power",
            height=600,
        )
        f.write(fig4.to_html(full_html=False, include_plotlyjs=False))

        f.write("</body></html>")

    print(f"Generated power analysis plot: {output_file}")


if __name__ == "__main__":
    data_file = "/Users/hussain/Menteru-Github/AIrux8_opti_logic/data/02_PreprocessedData/Clea/features_processed_Clea.csv"
    output_dir = "/Users/hussain/Menteru-Github/AIrux8_opti_logic/analysis/output/"
    plot_power_analysis(data_file, output_dir)
