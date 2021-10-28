# file app.py
import dash
from dash import dcc
from dash import html
import plotly.graph_objects as go
from utils.data_handling import load_data
import numpy as np
import plotly.io as pio
from dash.dependencies import Input, Output
pio.templates.default = 'plotly_dark'

app = dash.Dash(__name__, assets_folder='./assets')

df = load_data('latent_classification')
X_START = 0.02
delta = 0.2
yy, zz = np.meshgrid(np.arange(df['y'].min(), df['y'].max(), delta), np.arange(df['z'].min(), df['z'].max(), delta))
xx = np.ones(yy.shape)

points = go.Scatter3d(
        x=df['x'],
        y=df['y'],
        z=df['z'],
        mode='markers',
        marker=dict(
            size=5,
            color=df['label'],                # set color to an array/list of desired values
            colorscale='Viridis',   # choose a colorscale
            opacity=1)
        )

plane = go.Surface(
            x=xx*X_START,
            y=yy,
            z=zz,
            colorscale=['grey' for i in range(len(xx))],
            opacity=0.8,
            showscale=False
        )

fig = go.Figure(
    data=[points, plane],

)
app.layout = html.Div(
    children=[
        html.H1(children='Latent Visualizer'),
        dcc.Graph(id='graph', figure=fig),
        dcc.Slider(id='plane', min=df['x'].min(), max=df['x'].max(), step=0.01, value=X_START, marks={0.02: 'SVM'}, updatemode='drag')
    ]
)

@app.callback(Output('graph','figure'), Input('plane','value'))
def change_plane_x(slider_value):
    plane=go.Surface(
            x=xx*slider_value,
            y=yy,
            z=zz,
            colorscale=['grey' for i in range(len(xx))],
            opacity=0.8,
            showscale=False
        )
    return go.Figure(data=[points,plane])

if __name__ == '__main__':
    app.run_server(debug=True)