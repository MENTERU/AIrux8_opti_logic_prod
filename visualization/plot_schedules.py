import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_control_type_schedule(file_path, output_path):
    """
    Reads control type schedule data, creates interactive plots, and saves them as an HTML file.
    """
    df = pd.read_csv(file_path)
    df['Date Time'] = pd.to_datetime(df['Date Time'])

    all_areas = [col.split('_')[0] for col in df.columns if '_OnOFF' in col]
    desired_order = ['Area 1', 'Area 2', 'Area 3', 'Area 4', 'Meeting Room', 'Break Room']
    areas = [area for area in desired_order if area in all_areas]

    fig = make_subplots(rows=len(areas), cols=1, shared_xaxes=True, subplot_titles=[f'{area} Schedule' for area in areas], specs=[[{"secondary_y": True}]]*len(areas))

    for i, area in enumerate(areas, 1):
        # Prepare hover data
        custom_data = df[[f'{area}_Mode', f'{area}_SetTemp', f'{area}_FanSpeed']]
        
        hovertemplate_power = ('<b>Date Time</b>: %{x}<br>' +
                               '<b>Predicted Power</b>: %{y:.2f}<br>' +
                               '<b>Mode</b>: %{customdata[0]}<br>' +
                               '<b>Set Temp</b>: %{customdata[1]}<br>' +
                               '<b>Fan Speed</b>: %{customdata[2]}<extra></extra>')

        # Plot PredPower
        fig.add_trace(go.Scatter(
            x=df['Date Time'], 
            y=df[f'{area}_PredPower'], 
            mode='lines', 
            name=f'{area} PredPower',
            customdata=custom_data,
            hovertemplate=hovertemplate_power
        ), row=i, col=1, secondary_y=False)
        
        hovertemplate_temp = ('<b>Date Time</b>: %{x}<br>' +
                              '<b>Predicted Temp</b>: %{y:.2f}<br>' +
                              '<b>Mode</b>: %{customdata[0]}<br>' +
                              '<b>Set Temp</b>: %{customdata[1]}<br>' +
                              '<b>Fan Speed</b>: %{customdata[2]}<extra></extra>')

        # Plot PredTemp on a secondary y-axis
        fig.add_trace(go.Scatter(
            x=df['Date Time'], 
            y=df[f'{area}_PredTemp'], 
            mode='lines', 
            name=f'{area} PredTemp',
            customdata=custom_data,
            hovertemplate=hovertemplate_temp
        ), row=i, col=1, secondary_y=True)
        
        # Plot outside_temp on the secondary y-axis
        fig.add_trace(go.Scatter(
            x=df['Date Time'],
            y=df['outside_temp'],
            mode='lines',
            name='Outside Temp',
            legendgroup='outside_temp',
            showlegend=(i==1), # Only show legend for the first plot
            line=dict(color='black')
        ), row=i, col=1, secondary_y=True)
        
        fig.update_yaxes(title_text="Predicted Power", row=i, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Predicted Temp", row=i, col=1, secondary_y=True)


    fig.update_layout(height=300*len(areas), title_text="Control Type Schedule Analysis<br><sup>Hover over the lines for more details</sup>")
    
    # Calculate total power for each area
    total_power = {area: df[f'{area}_PredPower'].sum() for area in areas}
    
    # Create bar chart
    bar_fig = go.Figure(data=[go.Bar(x=list(total_power.keys()), y=list(total_power.values()))])
    bar_fig.update_layout(title_text="Total Predicted Power per Area", height=400)

    with open(output_path, 'w') as f:
        f.write(fig.to_html(full_html=True, include_plotlyjs='cdn'))
        f.write(bar_fig.to_html(full_html=False, include_plotlyjs=False))

    print(f"Generated plot: {output_path}")


if __name__ == '__main__':
    # Define file paths
    control_schedule_file = '/Users/hussain/Menteru-Github/AIrux8_opti_logic/data/04_PlanningData/Clea/control_type_schedule_20251016.csv'
    
    # Define output paths
    control_output_file = '/Users/hussain/Menteru-Github/AIrux8_opti_logic/visualization/control_type_schedule.html'

    # Generate plots
    plot_control_type_schedule(control_schedule_file, control_output_file)