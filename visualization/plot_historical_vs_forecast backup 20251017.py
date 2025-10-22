import json

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_summary_table(hist_df, fore_df, all_areas):
    """
    Generates an HTML table summarizing historical and forecast power statistics.
    """
    summary_data = []
    for area in all_areas:
        hist_power_col = f"{area}_Power"
        fore_power_col = f"{area}_PredPower"

        hist_total_power = (
            hist_df[hist_power_col].sum() if hist_power_col in hist_df.columns else 0
        )
        fore_total_power = (
            fore_df[fore_power_col].sum() if fore_power_col in fore_df.columns else 0
        )

        hist_mean_power = (
            hist_df[hist_power_col].mean() if hist_power_col in hist_df.columns else 0
        )
        fore_mean_power = (
            fore_df[fore_power_col].mean() if fore_power_col in fore_df.columns else 0
        )

        abs_diff = fore_total_power - hist_total_power
        percent_diff = (
            (abs_diff / hist_total_power * 100) if hist_total_power != 0 else 0
        )

        summary_data.append(
            {
                "Area": area,
                "Historical Total Power": f"{hist_total_power:.2f}",
                "Forecast Total Power": f"{fore_total_power:.2f}",
                "Absolute Difference": f"{abs_diff:.2f}",
                "Percentage Difference": f"{percent_diff:.2f}%",
                "Historical Mean Power": f"{hist_mean_power:.2f}",
                "Forecast Mean Power": f"{fore_mean_power:.2f}",
            }
        )

    # Add overall totals
    overall_hist_total = sum([float(d["Historical Total Power"]) for d in summary_data])
    overall_fore_total = sum([float(d["Forecast Total Power"]) for d in summary_data])
    overall_abs_diff = overall_fore_total - overall_hist_total
    overall_percent_diff = (
        (overall_abs_diff / overall_hist_total * 100) if overall_hist_total != 0 else 0
    )

    overall_hist_mean = (
        sum([float(d["Historical Mean Power"]) for d in summary_data])
        / len(summary_data)
        if summary_data
        else 0
    )
    overall_fore_mean = (
        sum([float(d["Forecast Mean Power"]) for d in summary_data]) / len(summary_data)
        if summary_data
        else 0
    )

    summary_data.append(
        {
            "Area": "Overall",
            "Historical Total Power": f"{overall_hist_total:.2f}",
            "Forecast Total Power": f"{overall_fore_total:.2f}",
            "Absolute Difference": f"{overall_abs_diff:.2f}",
            "Percentage Difference": f"{overall_percent_diff:.2f}%",
            "Historical Mean Power": f"{overall_hist_mean:.2f}",
            "Forecast Mean Power": f"{overall_fore_mean:.2f}",
        }
    )

    summary_df = pd.DataFrame(summary_data)

    # Convert DataFrame to HTML table with styling
    html_table = summary_df.to_html(
        index=False,
        classes="table table-striped table-hover",
        decimal=".",
        float_format="%.2f",
    )

    # Add some basic CSS for better appearance
    css_style = """
    <style>
        .table-container {
            display: flex;
            justify-content: center;
            margin-top: 20px;
            margin-bottom: 20px;
        }
        .table {
            width: 70%; /* Reduced width */
            margin-bottom: 1rem;
            color: #333;
            border-collapse: collapse;
            border-radius: 8px; /* Soften corners */
            overflow: hidden; /* Ensures border-radius is applied */
            box-shadow: 0 0 15px rgba(0, 0, 0, 0.1);
        }
        .table th,
        .table td {
            padding: 12px 15px; /* Increased padding */
            vertical-align: middle;
            border: 1px solid #e0e0e0; /* Lighter border */
            text-align: center; /* Center align text */
        }
        .table thead th {
            background-color: #2196F3; /* Blue header */
            color: white;
            font-weight: bold;
            border-bottom: 2px solid #1976D2; /* Darker blue border */
        }
        .table tbody tr:nth-of-type(odd) {
            background-color: #f9f9f9; /* Light grey for odd rows */
        }
        .table tbody tr:nth-of-type(even) {
            background-color: #ffffff; /* White for even rows */
        }
        .table tbody tr:hover {
            background-color: #E3F2FD; /* Light blue on hover */
            color: #333;
        }
        .table tfoot tr {
            font-weight: bold;
            background-color: #e0e0e0;
        }
    </style>
    <div class="table-container">
    """
    return css_style + html_table + "</div>"


def plot_area_graph(
    fig,
    df,
    area,
    row,
    is_historical,
    power_range=None,
    temp_range=None,
    fan_speed_mapping=None,
    mode_mapping=None,
):
    """
    Adds a single area's graph (either historical or forecast) to a subplot figure.
    """
    if is_historical:
        power_col, temp_col, mode_col, set_temp_col, fan_speed_col = (
            f"{area}_Power",
            f"{area}_IndoorTemp",
            f"{area}_Mode",
            f"{area}_SetTemp",
            f"{area}_FanSpeed",
        )
        power_name, temp_name = f"{area} Power", f"{area} Indoor Temp"
        outdoor_temp_col = "OutdoorTemp"
        date_col = "Datetime"
    else:
        power_col, temp_col, mode_col, set_temp_col, fan_speed_col = (
            f"{area}_PredPower",
            f"{area}_PredTemp",
            f"{area}_Mode",
            f"{area}_SetTemp",
            f"{area}_FanSpeed",
        )
        power_name, temp_name = f"{area} PredPower", f"{area} PredTemp"
        outdoor_temp_col = "outside_temp"
        date_col = "Date Time"

    fig.add_trace(
        go.Scatter(
            x=df[date_col],
            y=df[power_col],
            mode="lines",
            name=power_name,
        ),
        row=row,
        col=1,
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=df[date_col],
            y=df[temp_col],
            mode="lines",
            name=temp_name,
        ),
        row=row,
        col=1,
        secondary_y=True,
    )

    # Add FanSpeed, Mode, and SetTemp as dotted lines
    for col, name, mapping in zip(
        [fan_speed_col, mode_col, set_temp_col],
        ["Fan Speed", "Mode", "Set Temp"],
        [fan_speed_mapping, mode_mapping, None],
    ):
        if col in df.columns:
            # Create customdata for hover display with labels
            if mapping:
                # Map numeric values to labels for hover display
                customdata = df[col].map(mapping).fillna(df[col].astype(str))
                hovertemplate = (
                    f"<b>{name}</b>: %{{customdata}} (%{{y}})<br><extra></extra>"
                )
            else:
                customdata = None
                hovertemplate = f"<b>{name}</b>: %{{y}}<br><extra></extra>"

            # Add offset to fan speed and mode values to be more visible on the temperature axis
            if name in ["Fan Speed", "Mode"]:
                # Add offset to make values more visible (don't show numeric values in hover)
                if name == "Fan Speed":
                    offset_y = (
                        df[col] + 15
                    )  # Add 15 offset so 0 becomes 15, 1 becomes 16, etc.
                    hovertemplate = f"<b>{name}</b>: %{{customdata}}<br><extra></extra>"
                else:  # Mode
                    offset_y = (
                        df[col] + 25
                    )  # Add 25 offset so 0 becomes 25, 1 becomes 26, etc.
                    hovertemplate = f"<b>{name}</b>: %{{customdata}}<br><extra></extra>"

                fig.add_trace(
                    go.Scatter(
                        x=df[date_col],
                        y=offset_y,
                        mode="lines",
                        name=f"{area} {name}",
                        line=dict(dash="dot"),
                        customdata=customdata,
                        hovertemplate=hovertemplate,
                    ),
                    row=row,
                    col=1,
                    secondary_y=True,
                )
            else:
                # SetTemp uses normal scaling
                fig.add_trace(
                    go.Scatter(
                        x=df[date_col],
                        y=df[col],
                        mode="lines",
                        name=f"{area} {name}",
                        line=dict(dash="dot"),
                        customdata=customdata,
                        hovertemplate=hovertemplate,
                    ),
                    row=row,
                    col=1,
                    secondary_y=True,
                )

    fig.add_trace(
        go.Scatter(
            x=df[date_col],
            y=df[outdoor_temp_col],
            mode="lines",
            name="Outdoor Temp",
            legendgroup="OutdoorTemp",
            showlegend=(row == 1),
            line=dict(color="black"),
        ),
        row=row,
        col=1,
        secondary_y=True,
    )

    fig.update_yaxes(title_text="Power", row=row, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Temperature", row=row, col=1, secondary_y=True)

    # Apply unified ranges if provided
    if power_range is not None:
        fig.update_yaxes(range=power_range, row=row, col=1, secondary_y=False)
    if temp_range is not None:
        fig.update_yaxes(range=temp_range, row=row, col=1, secondary_y=True)


def plot_historical_vs_forecast(
    historical_file, forecast_file, mapping_file, output_file
):
    """
    Generates a single HTML report comparing historical and forecast data for each area.
    """
    # --- Load and process forecast data first to get date range ---
    fore_df = pd.read_csv(forecast_file)
    fore_df["Date Time"] = pd.to_datetime(fore_df["Date Time"])

    # Extract date range from forecast data
    forecast_start = fore_df["Date Time"].min().date()
    forecast_end = fore_df["Date Time"].max().date()

    # Calculate historical date range (one year before forecast)
    historical_start = pd.Timestamp(forecast_start) - pd.Timedelta(days=365)
    historical_end = pd.Timestamp(forecast_end) - pd.Timedelta(days=365)

    print(f"Forecast period: {forecast_start} to {forecast_end}")
    print(f"Historical period: {historical_start.date()} to {historical_end.date()}")

    # --- Load and process historical data ---
    hist_df = pd.read_csv(historical_file)
    hist_df["Datetime"] = pd.to_datetime(hist_df["Datetime"])
    hist_df = hist_df[
        (hist_df["Datetime"] >= historical_start.strftime("%Y-%m-%d"))
        & (hist_df["Datetime"] <= historical_end.strftime("%Y-%m-%d"))
    ]
    hist_df.rename(
        columns={
            "A/C Set Temperature": "SetTemp",
            "Indoor Temp.": "IndoorTemp",
            "A/C ON/OFF": "OnOFF",
            "A/C Mode": "Mode",
            "A/C Fan Speed": "FanSpeed",
            "adjusted_power": "Power",
            "Outdoor Temp.": "OutdoorTemp",
        },
        inplace=True,
    )

    with open(mapping_file, "r") as f:
        category_mapping = json.load(f)

    # Create mappings from numeric values to labels for hover display
    fan_speed_mapping = {
        v: k.upper() for k, v in category_mapping["A/C Fan Speed"].items()
    }
    mode_mapping = {v: k.upper() for k, v in category_mapping["A/C Mode"].items()}

    # Create reverse mappings for forecast data (string to number)
    fan_speed_mapping_rev = {
        k.upper(): v for k, v in category_mapping["A/C Fan Speed"].items()
    }
    mode_mapping_rev = {k.upper(): v for k, v in category_mapping["A/C Mode"].items()}

    hist_pivot = hist_df.pivot_table(
        index="Datetime",
        columns="zone",
        values=["Power", "IndoorTemp", "Mode", "SetTemp", "FanSpeed", "OnOFF"],
    )
    hist_pivot.columns = [f"{col[1]}_{col[0]}" for col in hist_pivot.columns]
    hist_pivot.reset_index(inplace=True)
    outdoor_temp_df = (
        hist_df[["Datetime", "OutdoorTemp"]].drop_duplicates().set_index("Datetime")
    )
    hist_pivot = hist_pivot.set_index("Datetime").join(outdoor_temp_df).reset_index()

    all_areas = sorted(
        [zone for zone in hist_df["zone"].unique() if zone.startswith("Area")]
    ) + sorted(
        [zone for zone in hist_df["zone"].unique() if not zone.startswith("Area")]
    )

    # --- Map forecast data string columns to numeric using reverse mapping ---
    for area in all_areas:
        mode_col = f"{area}_Mode"
        if mode_col in fore_df.columns and fore_df[mode_col].dtype == "object":
            # Convert string values to numeric using reverse mapping
            fore_df[mode_col] = fore_df[mode_col].str.upper().map(mode_mapping_rev)
            # Fill any NaN values with 0 (OFF mode)
            fore_df[mode_col] = fore_df[mode_col].fillna(0)

        fan_speed_col = f"{area}_FanSpeed"
        if (
            fan_speed_col in fore_df.columns
            and fore_df[fan_speed_col].dtype == "object"
        ):
            # Convert string values to numeric using reverse mapping
            fore_df[fan_speed_col] = (
                fore_df[fan_speed_col].str.upper().map(fan_speed_mapping_rev)
            )
            # Fill any NaN values with 0 (AUTO fan speed)
            fore_df[fan_speed_col] = fore_df[fan_speed_col].fillna(0)

    # --- Calculate global ranges for unified y-axis scales ---
    power_values = []
    temp_values = []

    # Collect all power and temperature values from both datasets
    for area in all_areas:
        # Historical data
        if f"{area}_Power" in hist_pivot.columns:
            power_values.extend(hist_pivot[f"{area}_Power"].dropna().tolist())
        if f"{area}_IndoorTemp" in hist_pivot.columns:
            temp_values.extend(hist_pivot[f"{area}_IndoorTemp"].dropna().tolist())

        # Forecast data
        if f"{area}_PredPower" in fore_df.columns:
            power_values.extend(fore_df[f"{area}_PredPower"].dropna().tolist())
        if f"{area}_PredTemp" in fore_df.columns:
            temp_values.extend(fore_df[f"{area}_PredTemp"].dropna().tolist())

    # Calculate ranges with some padding
    if power_values:
        power_min, power_max = min(power_values), max(power_values)
        power_padding = (power_max - power_min) * 0.05  # 5% padding
        power_range = [max(0, power_min - power_padding), power_max + power_padding]
    else:
        power_range = None

    if temp_values:
        temp_min, temp_max = min(temp_values), max(temp_values)
        temp_padding = (temp_max - temp_min) * 0.05  # 5% padding
        temp_range = [temp_min - temp_padding, temp_max + temp_padding]
    else:
        temp_range = None

    # Print calculated ranges for debugging
    print(f"Unified Power Range: {power_range}")
    print(f"Unified Temperature Range: {temp_range}")

    # --- Generate plots ---
    with open(output_file, "w") as f:
        f.write("<html><head><title>Historical vs Forecast</title></head><body>")
        f.write('<div style="text-align: center;"><h1>過去実績と予測の比較</h1></div>')
        f.write(
            '<div style="text-align: center; margin-bottom: 20px; color: #666; font-size: 14px;">'
        )
        f.write("</div>")

        for area in all_areas:
            # Historical plot
            fig_hist = make_subplots(
                rows=1,
                cols=1,
                specs=[[{"secondary_y": True}]],
                subplot_titles=[
                    f"{area} - 実績 ({historical_start.date()} to {historical_end.date()})"
                ],
            )
            plot_area_graph(
                fig_hist,
                hist_pivot,
                area,
                1,
                is_historical=True,
                power_range=power_range,
                temp_range=temp_range,
                fan_speed_mapping=fan_speed_mapping,
                mode_mapping=mode_mapping,
            )
            fig_hist.update_layout(height=400)
            f.write(fig_hist.to_html(full_html=False, include_plotlyjs="cdn"))

            # Forecast plot
            fig_fore = make_subplots(
                rows=1,
                cols=1,
                specs=[[{"secondary_y": True}]],
                subplot_titles=[f"{area} - 予測 ({forecast_start} to {forecast_end})"],
            )
            plot_area_graph(
                fig_fore,
                fore_df,
                area,
                1,
                is_historical=False,
                power_range=power_range,
                temp_range=temp_range,
                fan_speed_mapping=fan_speed_mapping,
                mode_mapping=mode_mapping,
            )
            fig_fore.update_layout(height=400)
            f.write(fig_fore.to_html(full_html=False, include_plotlyjs=False))

        # --- Generate bar chart comparison ---
        total_power_hist = {
            area: hist_pivot[f"{area}_Power"].sum() for area in all_areas
        }
        total_power_fore = {
            area: fore_df[f"{area}_PredPower"].sum() for area in all_areas
        }

        bar_fig = go.Figure()
        bar_fig.add_trace(
            go.Bar(
                x=all_areas,
                y=[total_power_hist.get(area, 0) for area in all_areas],
                name="Historical",
            )
        )
        bar_fig.add_trace(
            go.Bar(
                x=all_areas,
                y=[total_power_fore.get(area, 0) for area in all_areas],
                name="Forecast",
            )
        )
        bar_fig.update_layout(
            barmode="group",
            title_text=f"消費電力量比較: 実績 ({historical_start.date()} to {historical_end.date()}) vs 予測 ({forecast_start} to {forecast_end})",
            height=800,
        )

        f.write(bar_fig.to_html(full_html=False, include_plotlyjs=False))

        # --- Generate and add summary table ---
        summary_table_html = generate_summary_table(hist_pivot, fore_df, all_areas)
        f.write(
            '<div style="text-align: center; margin-top: 40px;"><h2>サマリー統計 (Summary Statistics)</h2></div>'
        )
        f.write(summary_table_html)

        f.write("</body></html>")

    print(f"Generated plot: {output_file}")


if __name__ == "__main__":
    historical_data_file = "/Users/hussain/Menteru-Github/AIrux8_opti_logic/data/02_PreprocessedData/Clea/features_processed_Clea.csv"
    forecast_data_file = "/Users/hussain/Menteru-Github/AIrux8_opti_logic/data/04_PlanningData/Clea/control_type_schedule_20251018.csv"
    mapping_file = (
        "/Users/hussain/Menteru-Github/AIrux8_opti_logic/config/category_mapping.json"
    )
    output_file = "/Users/hussain/Menteru-Github/AIrux8_opti_logic/visualization/plot_historical_vs_forecast.html"
    plot_historical_vs_forecast(
        historical_data_file, forecast_data_file, mapping_file, output_file
    )
