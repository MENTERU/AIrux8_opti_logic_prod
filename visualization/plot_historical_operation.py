import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json

def plot_historical_operation(file_path, output_path, mapping_path):
    """
    Reads historical processed data, creates interactive plots for a specific period, and saves them as an HTML file.
    """
    df = pd.read_csv(file_path)
    df['Datetime'] = pd.to_datetime(df['Datetime'])

    # Load category mapping
    with open(mapping_path, 'r') as f:
        category_mapping = json.load(f)
    
    fan_speed_mapping = {v: k for k, v in category_mapping['A/C Fan Speed'].items()}

    # Filter for the desired period in 2024
    start_date = '2024-10-16'
    end_date = '2024-10-19'
    df = df[(df['Datetime'] >= start_date) & (df['Datetime'] <= end_date)]

    # Rename columns for easier access
    df.rename(columns={
        'A/C Set Temperature': 'SetTemp',
        'Indoor Temp.': 'IndoorTemp',
        'A/C ON/OFF': 'OnOFF',
        'A/C Mode': 'Mode',
        'A/C Fan Speed': 'FanSpeed',
        'adjusted_power': 'Power',
        'Outdoor Temp.': 'OutdoorTemp'
    }, inplace=True)

    # Pivot the table
    pivot_df = df.pivot_table(index='Datetime', columns='zone', values=['Power', 'IndoorTemp', 'Mode', 'SetTemp', 'FanSpeed', 'OnOFF'])
    
    # Flatten the multi-index columns
    pivot_df.columns = [f'{col[1]}_{col[0]}' for col in pivot_df.columns]
    pivot_df.reset_index(inplace=True)

    # Get outdoor temperature (it's the same for all zones, so we can take it from the original df)
    outdoor_temp_df = df[['Datetime', 'OutdoorTemp']].drop_duplicates().set_index('Datetime')
    pivot_df = pivot_df.set_index('Datetime').join(outdoor_temp_df).reset_index()

    all_areas = sorted([zone for zone in df['zone'].unique() if zone.startswith('Area')]) + sorted([zone for zone in df['zone'].unique() if not zone.startswith('Area')])
    
    # Apply fan speed mapping
    for area in all_areas:
        pivot_df[f'{area}_FanSpeed'] = pivot_df[f'{area}_FanSpeed'].map(fan_speed_mapping)

    fig = make_subplots(rows=len(all_areas), cols=1, shared_xaxes=True, subplot_titles=[f'{area} Schedule' for area in all_areas], specs=[[{"secondary_y": True}]]*len(all_areas))

    for i, area in enumerate(all_areas, 1):
        custom_data = pivot_df[[f'{area}_Mode', f'{area}_SetTemp', f'{area}_FanSpeed']]
        
        hovertemplate_power = ('<b>Date Time</b>: %{x}<br>' +
                               '<b>Power</b>: %{y:.2f}<br>' +
                               '<b>Mode</b>: %{customdata[0]}<br>' +
                               '<b>Set Temp</b>: %{customdata[1]}<br>' +
                               '<b>Fan Speed</b>: %{customdata[2]}<extra></extra>')

        fig.add_trace(go.Scatter(
            x=pivot_df['Datetime'], 
            y=pivot_df[f'{area}_Power'], 
            mode='lines', 
            name=f'{area} Power',
            customdata=custom_data,
            hovertemplate=hovertemplate_power
        ), row=i, col=1, secondary_y=False)
        
        hovertemplate_temp = ('<b>Date Time</b>: %{x}<br>' +
                              '<b>Indoor Temp</b>: %{y:.2f}<br>' +
                              '<b>Mode</b>: %{customdata[0]}<br>' +
                              '<b>Set Temp</b>: %{customdata[1]}<br>' +
                              '<b>Fan Speed</b>: %{customdata[2]}<extra></extra>')

        fig.add_trace(go.Scatter(
            x=pivot_df['Datetime'], 
            y=pivot_df[f'{area}_IndoorTemp'], 
            mode='lines', 
            name=f'{area} Indoor Temp',
            customdata=custom_data,
            hovertemplate=hovertemplate_temp
        ), row=i, col=1, secondary_y=True)
        
        fig.add_trace(go.Scatter(
            x=pivot_df['Datetime'],
            y=pivot_df['OutdoorTemp'],
            mode='lines',
            name='Outdoor Temp',
            legendgroup='OutdoorTemp',
            showlegend=(i==1),
            line=dict(color='black')
        ), row=i, col=1, secondary_y=True)
        
        fig.update_yaxes(title_text="Power", row=i, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Temperature", row=i, col=1, secondary_y=True)

    fig.update_layout(height=300*len(all_areas), title_text="Historical Operation Analysis (2024-10-16 to 2024-10-19)<br><sup>Hover over the lines for more details</sup>")
    
    total_power = {area: pivot_df[f'{area}_Power'].sum() for area in all_areas}
    
    bar_fig = go.Figure(data=[go.Bar(x=list(total_power.keys()), y=list(total_power.values()))])
    bar_fig.update_layout(title_text="Total Power per Area", height=400)

    with open(output_path, 'w') as f:
        f.write(fig.to_html(full_html=True, include_plotlyjs='cdn'))
        f.write(bar_fig.to_html(full_html=False, include_plotlyjs=False))

    print(f"Generated plot: {output_path}")

if __name__ == '__main__':
    historical_data_file = '/Users/hussain/Menteru-Github/AIrux8_opti_logic/data/02_PreprocessedData/Clea/features_processed_Clea.csv'
    mapping_file = '/Users/hussain/Menteru-Github/AIrux8_opti_logic/config/category_mapping.json'
    output_file = '/Users/hussain/Menteru-Github/AIrux8_opti_logic/visualization/historical_operation.html'
    plot_historical_operation(historical_data_file, output_file, mapping_file)