# file app.py
import dash
from dash import dcc
from dash import html
import plotly.graph_objects as go
from utils.data_handling import load_data
import numpy as np
from dash.dependencies import Input, Output, State
from dash_folder.template import assets_folder, colors

app = dash.Dash(__name__, assets_folder=assets_folder)

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
            color=[colors.yellow if label else colors.blue for label in df['label']],
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


def copy_layout(layout, relayout_data):
    # Keep zoom after updating graph
    if relayout_data:
        if 'scene.camera' in relayout_data:
            layout.scene.camera.update(relayout_data['scene.camera'])
    return layout

@app.callback(Output('graph','figure'), Input('plane','value'), State("graph", "relayoutData"))
def change_plane_x(slider_value, relayout_data):
    plane=go.Surface(
            x=xx*slider_value,
            y=yy,
            z=zz,
            colorscale=['grey' for i in range(len(xx))],
            opacity=0.8,
            showscale=False
        )
    layout = go.Layout()
    layout = copy_layout(layout, relayout_data)
    print(relayout_data)
    return go.Figure(data=[points,plane], layout=layout)

if __name__ == '__main__':
    app.run_server(debug=True)