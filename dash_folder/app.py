# file app.py
import dash
import pandas as pd
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

anger_mask = df['label'] == 0
sad_mask = df['label'] == 1

yy, zz = np.meshgrid(np.arange(y.min(), y.max(), delta), np.array([z.min(), z.max()]))
xx = np.ones(yy.shape)


def scatter(x, y, z, name, color):
    return go.Scatter3d(
        x=x,
        y=y,
        z=z,
        name=name,
        mode='markers',
        marker=dict(
            size=5,
            color=color,
            opacity=1
        )
    )


anger_samples = scatter(df[anger_mask]['x'], df[anger_mask]['y'], df[anger_mask]['z'], 'Anger',colors.yellow)
sad_samples = scatter(df[sad_mask]['x'], df[sad_mask]['y'], df[sad_mask]['z'], 'Sad',colors.blue)

plane = go.Surface(
    x=xx * X_START,
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
                    'display': 'flex',
                    'flex-direction': 'column',
                    'justify-content': 'space-evenly',
                    'align-items': 'center',
                    'width': '20vw',
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
                'display': 'flex',
                'height': '90vh',
            }
        ),

        html.Div([
            html.Div([
                dcc.Dropdown(
                    id='classification-type',
                    options=[
                        {'label': 'Classes', 'value': 'classes'},
                        {'label': 'Correct Classifications', 'value': 'cor'},
                        {'label': 'True/False Negatives/Positives', 'value': 'tfnp'}
                    ],
                    value='classes',
                    style={
                        'width': '100%',
                        'display': 'flex',
                        'flex-direction': 'column',
                    }
                )
            ],
                style={
                    'width': '30vw',
                }
            ),
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
                'height': '10vh',
                'align-items': 'center',
                'justify-content': 'center',
                'width': '50vw',
            })
        ], style={
            'display': 'flex',
            'justify-content': 'space-evenly'
        })

    ]
)


def copy_relayout_data(layout, relayout_data):
    # Keep zoom after updating graph
    if relayout_data:
        if 'scene.camera' in relayout_data:
            layout.scene.camera.update(relayout_data['scene.camera'])
    return layout

def get_trigger():
    ctx = dash.callback_context
    if ctx.triggered:
        return ctx.triggered[0]["prop_id"].split(".")[0]
    else:
        return None

@app.callback(Output('graph', 'figure'),
              Output('accuracy', 'children'),
              Output('precision', 'children'),
              Output('recall', 'children'),
              Output('accuracy', 'style'),
              Output('precision', 'style'),
              Output('recall', 'style'),
              Input('plane', 'value'),
              Input('classification-type','value'),
              State('graph', 'relayoutData'),
              State('graph','figure'))
def change_plane_x(slider_value, classification_type, relayout_data, prev_graph):
    trigger = get_trigger()

    predicted = np.where(x < slider_value, 0, 1)
    if trigger == 'classification-type':
        plane = prev_graph['data'][-1]
        if classification_type == 'tfnp':
            correct = df['label']==pd.Series(predicted)
            positive = df['label'] == True
            fp = df[(correct==False) & (positive)]
            fn = df[(correct==False) &(positive==False)]
            tp = df[(correct) & (positive)]
            tn = df[(correct) & (positive==False)]
            c = [tp, tn, fp, fn]
            cc = [colors.turquoise, colors.blue, colors.pink, colors.purple]
            cn = ['True Positive', 'True Negative', 'False Positive', 'False Negative']
            data = [scatter(c[i]['x'],c[i]['y'],c[i]['z'],cn[i], cc[i]) for i in range(len(c))]
            data.append(plane)
        elif classification_type == 'cor':
            correct_mask = df['label']==pd.Series(predicted)
            cs = df[correct_mask]
            ws = df[correct_mask== False]
            correct_scatter = scatter(cs['x'],cs['y'],cs['z'],'Correct classification', colors.turquoise)
            wrong_scatter = scatter(ws['x'], ws['y'], ws['z'], 'Wrong classification', colors.pink)
            data = [correct_scatter, wrong_scatter, plane]
        else:
            data =[anger_samples, sad_samples, plane]
    else:
        plane = go.Surface(
            x=xx * slider_value,
            y=yy,
            z=zz,
            colorscale=['grey' for i in range(len(xx))],
            opacity=0.8,
            showscale=False
        )
        data = prev_graph['data'][:-1] + [plane]
    accuracy = accuracy_score(labels, predicted)
    precision = precision_score(labels, predicted)
    recall = recall_score(labels, predicted)
    accuracy_label = f"Accuracy: {accuracy:.2f}"
    precision_label = f"Precision: {precision:.2f}"
    recall_label = f"Recall: {recall:.2f}"
    accuracy_style = {'border-color': f"rgb(166,255,198,{evaluate_metric_quadratic(accuracy)})"}
    precision_style = {'border-color': f"rgb(166,255,198,{evaluate_metric_quadratic(precision)})"}
    recall_style = {'border-color': f"rgb(166,255,198,{evaluate_metric_quadratic(recall)})"}

    layout = graph_layout_default
    layout = copy_relayout_data(layout, relayout_data)
    return go.Figure(data=data,
                     layout=layout), accuracy_label, precision_label, recall_label, accuracy_style, precision_style, recall_style


if __name__ == '__main__':
    app.run_server(debug=True)
