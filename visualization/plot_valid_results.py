import os

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_validation_results(plot_validation_only=True, validation_start_dict=None):
    """
    Reads all validation result CSVs from a directory, and for each area, generates plots
    for predicted vs. actual power and temperature, including set temperature and outdoor temperature.
    The plots are saved as a single HTML file.

    Args:
        plot_validation_only (bool): If True, only plot validation period for each area.
        validation_start_dict (dict): Dictionary mapping area_name to validation start datetime (as pd.Timestamp or str).
    """
    results_dir = "/Users/hussain/Menteru-Github/AIrux8_opti_logic/data/05_ValidationResults/Clea/"
    output_html_file = (
        "/Users/hussain/Menteru-Github/AIrux8_opti_logic/validation_plots.html"
    )

    csv_files = [f for f in os.listdir(results_dir) if f.endswith(".csv")]

    if not csv_files:
        print(f"No CSV files found in {results_dir}")
        return

    with open(output_html_file, "w") as f:
        f.write("<html><head><title>Validation Results</title></head><body>")
        f.write("<h1>Validation Results</h1>")

        for i, file_name in enumerate(sorted(csv_files)):
            area_name = os.path.splitext(file_name)[0].replace("valid_results_", "")
            file_path = os.path.join(results_dir, file_name)
            df = pd.read_csv(file_path)
            df["Datetime"] = pd.to_datetime(df["Datetime"])

            # Filter for validation period if flag is set
            if plot_validation_only:
                validation_start = pd.to_datetime("2025-06-30 15:00:00")
                df = df[df["Datetime"] >= validation_start]

            temp_pred_col = f"{area_name}_temp_pred"
            power_pred_col = f"{area_name}_power_pred"

            fig = make_subplots(
                rows=2,
                cols=1,
                subplot_titles=(f"{area_name} - Temperature", f"{area_name} - Power"),
            )

            # Add Temperature Plot
            fig.add_trace(
                go.Scatter(
                    x=df["Datetime"],
                    y=df["Indoor Temp."],
                    name="Indoor Temp (True)",
                    showlegend=True,
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=df["Datetime"],
                    y=df[temp_pred_col],
                    name="Indoor Temp (Pred)",
                    showlegend=True,
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=df["Datetime"],
                    y=df["A/C Set Temperature"],
                    name="Set Temp",
                    showlegend=True,
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=df["Datetime"],
                    y=df["Outdoor Temp."],
                    name="Outdoor Temp",
                    showlegend=True,
                ),
                row=1,
                col=1,
            )

            # Add Power Plot
            fig.add_trace(
                go.Scatter(
                    x=df["Datetime"],
                    y=df["adjusted_power"],
                    name="Power (True)",
                    showlegend=True,
                ),
                row=2,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=df["Datetime"],
                    y=df[power_pred_col],
                    name="Power (Pred)",
                    showlegend=True,
                ),
                row=2,
                col=1,
            )

            fig.update_layout(height=800, title_text=f"{area_name} Validation")

            f.write(
                fig.to_html(
                    full_html=False, include_plotlyjs="cdn" if i == 0 else False
                )
            )

        f.write("</body></html>")

    print(f"Plots saved to {output_html_file}")


if __name__ == "__main__":
    # Example usage:
    # To plot only validation period starting from 2025-06-30 15:00:00 for all zones:
    plot_validation_results(plot_validation_only=True)
