# file app.py
import dash
from dash import dcc
from dash import html
import plotly.graph_objects as go
from utils.data_handling import load_data
import numpy as np
from dash.dependencies import Input, Output, State
from dash_folder.template import assets_folder, colors
from sklearn.metrics import accuracy_score, precision_score, recall_score
from utils.metric_evaluation import evaluate_metric_quadratic

app = dash.Dash(__name__, assets_folder=assets_folder)

df = load_data('latent_classification')
X_START = 0.02
delta = 0.1
x = df['x'].to_numpy()
y = df['y'].to_numpy()
z = df['z'].to_numpy()
labels = df['label'].to_numpy()

anger_mask = df['label']==0
sad_mask = df['label']==1

yy, zz = np.meshgrid(np.arange(y.min(), y.max(), delta), np.array([z.min(), z.max()]))
xx = np.ones(yy.shape)

anger_samples = go.Scatter3d(
    x=df[anger_mask]['x'],
    y=df[anger_mask]['y'],
    z=df[anger_mask]['z'],
    name='Anger',
    mode='markers',
    marker=dict(
        size=5,
        color=colors.yellow,
        opacity=1
    )
)

sad_samples = go.Scatter3d(
    x=df[sad_mask]['x'],
    y=df[sad_mask]['y'],
    z=df[sad_mask]['z'],
    name='Sad',
    mode='markers',
    marker=dict(
        size=5,
        color=colors.blue,
        opacity=1
    )
)

plane = go.Surface(
    x=xx*X_START,
    y=yy,
    z=zz,
    colorscale=['grey' for i in range(len(xx))],
    opacity=0.8,
    showscale=False,
    name='Classification Plane',
    showlegend=True
)


graph_layout_default = go.Layout()
graph_layout_default_settings = {'scene': {
    'camera': {
        'up': {'x': 0, 'y': 0, 'z': 1},
        'center': {'x': -0.07214567600111127, 'y': -0.08497201560137722, 'z': -0.27943759806176177},
        'eye': {'x': -0.5135394958253185, 'y': -1.9748036183855688, 'z': 0.7264046470993168},
        'projection': {'type': 'perspective'},
    },
    'xaxis_title': 'Anger(-),Sad(+)-Dim',
    'yaxis_title': 'PCA 1',
    'zaxis_title': 'PCA 2',
}}
graph_layout_default.update(graph_layout_default_settings)

graph_fig = go.Figure(
    data=[anger_samples, sad_samples, plane],
    layout=graph_layout_default

)
app.layout = html.Div(
    children=[
        html.Div([
            html.Div([
                html.Div(id='accuracy'),
                html.Div(id='precision'),
                html.Div(id='recall'),
            ],
                id='metrics',
                style={
                    'display':'flex',
                    'flex-direction':'column',
                    'justify-content':'space-evenly',
                    'align-items': 'center',
                    'width':'20vw',
                }
            ),
            dcc.Graph(
                id='graph',
                figure=graph_fig,
                style={
                    'width': '80vw'
                }
            ),
        ],
            style={
                'display':'flex',
                'height':'90vh',
            }
        ),


        html.Div([
            html.Div([
                dcc.Slider(
                    id='plane',
                    min=x.min(),
                    max=x.max(),
                    step=delta,
                    value=X_START,
                    marks={
                        X_START: f"{X_START:.2f} (SVM)",
                        x.min(): f"{x.min():.2f} (Min)",
                        x.max(): f"{x.max():.2f} (Max)"
                    },
                    tooltip={
                        'always_visible': False,
                        'placement': 'top'
                    },
                    updatemode='drag'
                )
            ], style={
                'height':'10vh',
                'align-items':'center',
                'width':'60vw',
            })
        ], style={
            'display':'flex',
            'justify-content':'center'
        })

    ]
)



def copy_relayout_data(layout, relayout_data):
    # Keep zoom after updating graph
    if relayout_data:
        if 'scene.camera' in relayout_data:
            layout.scene.camera.update(relayout_data['scene.camera'])
    return layout

@app.callback(Output('graph','figure'),
              Output('accuracy','children'),
              Output('precision','children'),
              Output('recall','children'),
              Output('accuracy','style'),
              Output('precision','style'),
              Output('recall','style'),
              Input('plane','value'),
              State('graph', 'relayoutData'),
              State('graph','figure'))
def change_plane_x(slider_value, relayout_data, prev_graph):
    plane=go.Surface(
            x=xx*slider_value,
            y=yy,
            z=zz,
            colorscale=['grey' for i in range(len(xx))],
            opacity=0.8,
            showscale=False
        )
    predicted = np.where(x < slider_value, 0,1)
    accuracy = accuracy_score(labels, predicted)
    precision = precision_score(labels, predicted)
    recall = recall_score(labels, predicted)
    accuracy_label = f"Accuracy: {accuracy:.2f}"
    precision_label = f"Precision: {precision:.2f}"
    recall_label = f"Recall: {recall:.2f}"
    accuracy_style = {'border-color':f"rgb(166,255,198,{evaluate_metric_quadratic(accuracy)})"}
    precision_style = {'border-color': f"rgb(166,255,198,{evaluate_metric_quadratic(precision)})"}
    recall_style = {'border-color': f"rgb(166,255,198,{evaluate_metric_quadratic(recall)})"}
    data = prev_graph['data'][:2] + [plane]
    layout = graph_layout_default
    layout = copy_relayout_data(layout, relayout_data)
    return go.Figure(data=data, layout=layout), accuracy_label, precision_label, recall_label, accuracy_style, precision_style, recall_style

if __name__ == '__main__':
    app.run_server(debug=True)